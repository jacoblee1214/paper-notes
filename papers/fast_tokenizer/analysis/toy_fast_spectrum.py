"""토이 T1: 주파수에 따른 DCT 계수 희소성 (논문 Table I과 Fig. 3 주장의 CPU 점검).

논문 주장(§V-B, Table I 아래 문단): FAST의 토큰 수는 제어 주파수에 거의 무관하고(1초당 행동 차원당 약 30개),
naive 토큰 수는 D × Hz로 늘어난다. 이 토이는 BPE를 빼고 '양자화 후 0이 아닌 DCT 계수 수'를 센다.
이는 BPE 이전의 기호 수이므로 논문의 실제 토큰 수와 같은 양이 아니다(상한 성격의 대리 지표).

또 논문 §V-B의 스케일 γ=10에서 재구성 오차 RMSE를 측정하고, 직교 DCT에서 나오는 해석 값
1/(sqrt(12)·γ)와 비교한다(Parseval 항등식으로 유도).

신호: 1초 구간에서 주파수 상한이 있는 무작위 부드러운 궤적(사인 합, 최대 3 사이클/초). 차원 D = 7.
정규화: 연속 신호의 1·99 분위수로 [-1, 1]에 맞춤 (샘플 수 n과 무관).
실행: python3 analysis/toy_fast_spectrum.py   (numpy, scipy; 수 초)
출력: analysis/toy_fast_spectrum_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy import stats

from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
OUT = HERE / "toy_fast_spectrum_results.json"

SEED = 0
D = 7
N_TRAJ = 300
SCALE = 10.0  # 논문 γ
RATES = [5, 10, 15, 20, 50, 100]  # Hz (1초 청크의 샘플 수)
MAX_FREQ = 3  # 사이클/초
rng = np.random.default_rng(SEED)


def make_traj(rng, n_samples, n_dims):
    """부드러운 궤적: 주파수 0..MAX_FREQ 사인 몇 개의 합. 반환 shape (n_dims, n_samples)."""
    t = np.linspace(0.0, 1.0, n_samples)
    out = np.zeros((n_dims, n_samples))
    for d in range(n_dims):
        for _ in range(3):
            f = rng.uniform(0.0, MAX_FREQ)
            ph = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.2, 1.0)
            out[d] += amp * np.sin(2 * np.pi * f * t + ph)
    return out


def make_spline(rng, n_samples, n_dims):
    """논문 §IV 소형 예시와 같은 종류: 1초 구간에서 무작위 4점을 지나는 3차 스플라인(차원별)."""
    t = np.linspace(0.0, 1.0, n_samples)
    out = np.zeros((n_dims, n_samples))
    knots = np.linspace(0.0, 1.0, 4)
    for d in range(n_dims):
        out[d] = CubicSpline(knots, rng.uniform(-1, 1, 4))(t)
    return out


SIGNALS = {"sinusoid": make_traj, "cubic_spline": make_spline}
# 차원별 고정 정규화 계수: 매우 조밀한 샘플의 1%/99% 분위수 (n과 무관한 연속 신호 근사)
dense = np.stack([make_traj(rng, 2000, D) for _ in range(40)])  # (40, D, 2000)
q01 = np.quantile(dense.transpose(1, 0, 2).reshape(D, -1), 0.01, axis=1)
q99 = np.quantile(dense.transpose(1, 0, 2).reshape(D, -1), 0.99, axis=1)


def normalize(x):
    # (D, n) -> [-1, 1] 근사 (논문 식의 분위수 정규화, 클리핑은 하지 않음)
    return (x - q01[:, None]) / (q99 - q01)[:, None] * 2.0 - 1.0


def fast_proxy(a):
    """a: (n, D) 정규화된 청크. 논문 Alg. 1의 DCT(시간축, 직교)·양자화(γ)까지 수행하고 0이 아닌 계수 수와 복원값을 돌려준다."""
    c = dct(a, axis=0, norm="ortho")
    q = np.around(SCALE * c)
    nnz = int(np.count_nonzero(q))
    rec = idct(q / SCALE, axis=0, norm="ortho")
    return nnz, rec


res = {"seed": SEED, "n_traj": N_TRAJ, "scale_gamma": SCALE, "dims": D, "rates_hz": RATES, "signals": {}}
for sig_name, gen in SIGNALS.items():
    # 차원별 고정 정규화 계수: 매우 조밀한 샘플의 1%/99% 분위수 (n과 무관한 연속 신호 근사)
    dense = np.stack([gen(rng, 2000, D) for _ in range(40)])  # (40, D, 2000)
    flat = dense.transpose(1, 0, 2).reshape(D, -1)
    q01 = np.quantile(flat, 0.01, axis=1)
    q99 = np.quantile(flat, 0.99, axis=1)

    def normalize(x, q01=q01, q99=q99):
        # (D, n) -> [-1, 1] 근사 (분위수 정규화, 클리핑 없음)
        return (x - q01[:, None]) / (q99 - q01)[:, None] * 2.0 - 1.0

    by_rate = {}
    for hz in RATES:
        nnz_list, rmse_list, naive = [], [], hz * D
        for _ in range(N_TRAJ):
            a = normalize(gen(rng, hz, D)).T  # (n, D)
            nnz, rec = fast_proxy(a)
            nnz_list.append(nnz)
            rmse_list.append(float(np.sqrt(np.mean((rec - a) ** 2))))
        nnz = np.array(nnz_list, dtype=float)
        rm = np.array(rmse_list)
        tcrit = stats.t.ppf(0.975, N_TRAJ - 1)
        by_rate[str(hz)] = {
            "naive_tokens_D_x_hz": naive,
            "nonzero_dct_mean": round(float(nnz.mean()), 2),
            "nonzero_dct_ci95_halfwidth": round(float(tcrit * nnz.std(ddof=1) / np.sqrt(N_TRAJ)), 2),
            "compression_naive_over_nonzero": round(naive / nnz.mean(), 2),
            "rmse_mean": round(float(rm.mean()), 5),
            "rmse_ci95_halfwidth": round(float(tcrit * rm.std(ddof=1) / np.sqrt(N_TRAJ)), 6),
        }
    res["signals"][sig_name] = by_rate

# 해석 값: 직교 변환에서 계수마다 균일 잡음 (1/γ)·U(-1/2, 1/2)이면 시간축 RMSE = 1/(sqrt(12)·γ)
res["analytic_uniform_rmse_1_over_sqrt12_gamma"] = round(1 / (np.sqrt(12) * SCALE), 5)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
