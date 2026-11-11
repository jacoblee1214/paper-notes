"""추가 검증 (KineFuse): 고유감각 누수 대 촉각 정보, 제로화 대조 (CPU, numpy, 단일 스레드, 수 초).

질문: 촉각 제로화(추론 시 촉각 입력만 0)는 "촉각 정보"를 빼는가, 아니면 고유감각이 물체 자세를
이미 알려 주므로 "구조 효과"와 섞이는가? 고유감각 제로화와 둘 다 제로화를 같이 본다.

가정 모델 (논문 실험이 아님. 논문에 없는 값은 모두 가정):
  - 1자유도 손가락 관절 각 q (도), 프레임마다 U(-45, 45).
  - 물체 회전 theta = q + delta. delta는 파지마다 N(0, sig_d^2) (파지 오프셋·미끄럼). sig_d가 작을수록
    고유감각 q가 theta를 많이 알려 준다 (고유감각 누수의 크기).
  - 시각 추정: v = theta + N(0, sig_v^2), sig_v = 2 + 20 * occ (occ = 가림 비율 0, 0.3, 0.6, 0.9).
  - 촉각: 접촉 c ~ Bernoulli(0.7). 접촉 시 h = delta + N(0, 3^2), 비접촉 시 h = 0 (c = 0).
  - 파지(에피소드) 400개 x 프레임 50개. 파지 절반은 학습, 절반은 평가. 가림 수준은 파지마다 무작위.
  - 추정기: 선형 ridge (정규화 1e-3), 특징 표준화. 구조(손가락 토큰)는 없음. 정보량만 본다.
  - 제로화: 추론 시 해당 채널의 원시 입력을 0으로 두고 표준화한다 (논문 "영 입력" 실험과 같은 방식).

실행: python3 analysis/extra_proprio_zero.py -> analysis/extra_proprio_zero_results.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_proprio_zero_results.json"
OCC_LEVELS = np.array([0.0, 0.3, 0.6, 0.9])
T_EP = 50
N_EP = 400
P_CONTACT = 0.7
SIG_H = 3.0
LAM = 1e-3
SIG_DELTAS = [0.0, 2.0, 8.0, 20.0]


def make_data(sig_d, n_ep, rng):
    n = n_ep * T_EP
    delta = np.repeat(rng.normal(0.0, sig_d, n_ep), T_EP)
    occ = np.repeat(rng.choice(OCC_LEVELS, n_ep), T_EP)
    q = rng.uniform(-45.0, 45.0, n)
    theta = q + delta
    v = theta + (2.0 + 20.0 * occ) * rng.standard_normal(n)
    c = (rng.random(n) < P_CONTACT).astype(float)
    h = c * (delta + SIG_H * rng.standard_normal(n))
    return {"v": v, "h": h, "c": c, "q": q, "theta": theta, "occ": occ}


FEATURES = {"V": ["v"], "H": ["h", "c"], "P": ["q"]}
MODELS = {"V": ["V"], "VH": ["V", "H"], "VP": ["V", "P"], "VHP": ["V", "H", "P"]}


def design(d, groups, zero=()):
    cols = []
    for g in groups:
        for name in FEATURES[g]:
            x = d[name].copy()
            if g in zero:
                x = np.zeros_like(x)
            cols.append(x)
    return np.stack(cols, axis=1)


def fit(d_tr, groups):
    X = design(d_tr, groups)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    ym = d_tr["theta"].mean()
    A = Z.T @ Z + LAM * len(Z) * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (d_tr["theta"] - ym))
    return {"mu": mu, "sd": sd, "w": w, "ym": ym, "groups": groups}


def predict(m, d, zero=()):
    X = design(d, m["groups"], zero=zero)
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["ym"]


def rmse(pred, d, mask=None):
    e = pred - d["theta"]
    if mask is not None:
        e = e[mask]
    return float(np.sqrt(np.mean(e ** 2)))


def run_one(sig_d, seed):
    rng = np.random.default_rng(seed)
    tr = make_data(sig_d, N_EP // 2, rng)
    te = make_data(sig_d, N_EP // 2, rng)
    models = {k: fit(tr, v) for k, v in MODELS.items()}
    res = {}
    res["V"] = rmse(predict(models["V"], te), te)
    res["VH"] = rmse(predict(models["VH"], te), te)
    res["VP"] = rmse(predict(models["VP"], te), te)
    res["VHP"] = rmse(predict(models["VHP"], te), te)
    # 추론 시 제로화: 학습은 VHP, 입력만 0
    res["VHP_zeroH_at_test"] = rmse(predict(models["VHP"], te, zero=("H",)), te)
    res["VHP_zeroP_at_test"] = rmse(predict(models["VHP"], te, zero=("P",)), te)
    res["VHP_zeroH_and_P_at_test"] = rmse(predict(models["VHP"], te, zero=("H", "P")), te)
    # 학습 단계에서 H를 뺀 VP 와 비교: 제로화 대 학습 제외
    res["VP_trained_without_H"] = res["VP"]
    # 가림 수준별 (VHP, 제로화 포함)
    per_occ = {}
    for o in OCC_LEVELS:
        mask = te["occ"] == o
        per_occ[f"occ_{o:g}"] = {
            "V": round(rmse(predict(models["V"], te), te, mask), 2),
            "VHP": round(rmse(predict(models["VHP"], te), te, mask), 2),
            "VHP_zeroH": round(rmse(predict(models["VHP"], te, zero=("H",)), te, mask), 2),
            "VHP_zeroP": round(rmse(predict(models["VHP"], te, zero=("P",)), te, mask), 2),
        }
    return {k: round(v, 3) for k, v in res.items()}, per_occ


def main():
    out = {"settings": {"occlusion_levels": OCC_LEVELS.tolist(), "frames_per_episode": T_EP,
                        "episodes_train": N_EP // 2, "episodes_test": N_EP // 2,
                        "p_contact": P_CONTACT, "haptic_noise_deg": SIG_H, "vision_sigma_deg": "2 + 20*occ",
                        "ridge_lambda": LAM, "seeds": 5,
                        "note": "가정 모델. 논문 수치를 재현하지 않는다. 선형 추정기, 구조 없음."},
           "sig_delta_sweep": {}}
    per_occ_8 = None
    for sig_d in SIG_DELTAS:
        runs = []
        for seed in range(5):
            r, po = run_one(sig_d, seed)
            runs.append(r)
            if sig_d == 8.0 and seed == 0:
                per_occ_8 = po
        keys = runs[0].keys()
        out["sig_delta_sweep"][f"sig_delta_{sig_d:g}"] = {
            k: {"mean": round(float(np.mean([r[k] for r in runs])), 3),
                "sd": round(float(np.std([r[k] for r in runs], ddof=1)), 3)} for k in keys}
    out["per_occlusion_sig_delta_8_seed0"] = per_occ_8
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
