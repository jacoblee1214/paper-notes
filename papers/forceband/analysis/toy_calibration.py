"""토이: 사용자별 이득·오프셋 불일치와 짧은 보정, 그리고 직전 힘 입력(teacher forcing)의 이득 범위.

공식 저장소 README는 체크포인트가 "모양은 옮겨지지만 절대 크기는 옮겨지지 않는다"고 적고, 제로샷 평가가
Pearson r 0.59인데 R² -4.94라고 보고한다. 이 토이는 합성 신호에서 그런 상황이 이득(gain)과 오프셋 불일치로
설명될 수 있는지, 그리고 두 모수(a, b)만 맞추는 보정에 몇 초의 짝 데이터가 필요한지 본다.
실제 sEMG 모델과 데이터는 쓰지 않는다. 가정이 결과를 정하므로 논문 수치의 검증이 아니라 사고 실험이다.

T1: 접촉 on/off 힘 신호 f(t), 모델 출력 p = g f + 오프셋 + 평활 잡음. 보정 길이별 시험 R².
T2: 10 s 창에서 직전 힘 한 점(teacher forcing)이 줄일 수 있는 오차의 상한: 접촉 지속 시간별로
    "직전 창 마지막 값 유지" 예측의 MAE 를 0 예측과 평균 예측에 견준다.

실행: python3 analysis/toy_calibration.py   (numpy; 수 초). 결과: analysis/toy_calibration_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_calibration_results.json"
FS = 100  # Hz (토이)


def telegraph(rng, n, mean_on, mean_off):
    f = np.zeros(n)
    t = 0
    on = rng.random() < 0.4
    while t < n:
        dur = max(1, int(rng.exponential((mean_on if on else mean_off)) * FS))
        if on:
            f[t:t + dur] = rng.uniform(5, 40)
        t += dur
        on = not on
    return f


def smooth_noise(rng, n, tau_s=0.3):
    a = np.exp(-1 / (tau_s * FS))
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + np.sqrt(1 - a * a) * e[i]
    return x


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def t1(reps=200, total_s=600, gain=1.73, offset=0.0, snr_corr=0.6):
    rng = np.random.default_rng(0)
    cal_lens = [5, 10, 20, 40, 80, 160]
    out = {l: [] for l in cal_lens}
    zs, rr = [], []
    for _ in range(reps):
        n = total_s * FS
        f = telegraph(rng, n, 2.0, 3.0)
        # 상관이 약 snr_corr 가 되도록 잡음 크기를 고른다 (모델이 모양은 맞히는 상황)
        noise = smooth_noise(rng, n)
        sig = f.std()
        k = sig * np.sqrt(1 / snr_corr ** 2 - 1)
        shape = f + k * noise
        p = gain * shape + offset
        test = slice(int(300 * FS), n)  # 뒤 300 s 는 시험
        zs.append(r2(f[test], p[test]))
        rr.append(np.corrcoef(f[test], p[test])[0, 1])
        for l in cal_lens:
            cal = slice(0, l * FS)
            A = np.vstack([p[cal], np.ones(l * FS)]).T
            a, b = np.linalg.lstsq(A, f[cal], rcond=None)[0]
            out[l].append(r2(f[test], a * p[test] + b))
    res = {
        "zero_shot_r2_median": float(np.median(zs)), "pearson_r_median": float(np.median(rr)),
        "pearson_r_squared": float(np.median(rr) ** 2),
        "calibrated_r2_by_cal_seconds": {str(l): dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                                                        p90=float(np.percentile(v, 90))) for l, v in out.items()},
        "assumptions": dict(gain=gain, offset=offset, target_corr=snr_corr, on_mean_s=2.0, off_mean_s=3.0, reps=reps),
    }
    return res


def t2(reps=100, win_s=10, total_s=600):
    rng = np.random.default_rng(1)
    res = {}
    W = win_s * FS
    for mean_on in (0.5, 1.0, 2.0, 5.0):
        zero, mean_p, hold = [], [], []
        for _ in range(reps):
            n = total_s * FS
            f = telegraph(rng, n, mean_on, 3.0)
            m = f.mean()
            e0, e1, e2 = [], [], []
            for s in range(W, n - W, W):
                w = f[s:s + W]
                e0.append(np.abs(w).mean())
                e1.append(np.abs(w - m).mean())
                e2.append(np.abs(w - f[s - 1]).mean())
            zero.append(np.mean(e0)); mean_p.append(np.mean(e1)); hold.append(np.mean(e2))
        res[str(mean_on)] = dict(mae_zero=float(np.mean(zero)), mae_global_mean=float(np.mean(mean_p)),
                                 mae_hold_prev_last_value=float(np.mean(hold)))
    return res


if __name__ == "__main__":
    out = {"T1_calibration": t1(), "T2_hold_previous_value": t2()}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
