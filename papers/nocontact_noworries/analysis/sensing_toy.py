"""NoContactNoWorries 감각 toy (CPU, numpy·scipy, 수 초). 실행: python3 analysis/sensing_toy.py

질문 1: 접촉 정보는 관절 추종 오차(proprioception)와 가려진 시각 중 어디에서 얼마나 나오는가?
질문 2: 관절 오차의 크기와 접촉의 Spearman 상관(논문 Fig. 6)은 접촉 감지 때문인가, 속도(운동 강도) 때문인가?

모델 (1 손가락, 30 Hz 프레임):
- 접촉 c_t: 두 상태 Markov 사슬, 정상 분포 접촉 비율 π = 0.4 (가정; 논문은 접촉 비율을 보고하지 않음)
- 명령 관절 q_com: 무작위 보행, 단계 크기 s_t = s0 (1 + κ c_t). κ > 0은 접촉 중 더 빨리 움직임, κ < 0은 더 느림
- 관절: q_{t+1} = q_t + α (q_com_t − q_t) − β c_t + 잡음. 정상상태 접촉 편차 δ = β/α (가정 0.05 rad)
- 오차 e_t = q_com_t − q_t (속도 때문에 생기는 추종 지연 + 접촉 편차). 관측 ê = e + σ_e·n
- 시각: 가려짐 o_t ~ Bernoulli(p_occ(c_t)), p_occ(1)=0.8, p_occ(0)=0.45 (전체 가림 비율 약 0.6, 논문 61%와 비슷하게 맞춤).
  점수 s = μ c + N(0,1), 보이면 μ_v = 1.5, 가려지면 μ_o = 0.2
- 특징: 두 입력 모두 길이 8 프레임 창(논문 T = 8). 분류기: 로지스틱 회귀(numpy IRLS), 앞 절반 학습·뒤 절반 평가

결과: analysis/sensing_toy_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
T_FRAMES = 60000
WIN = 8
PI_ON = 0.08   # 접촉 시작 확률 (접촉 아닐 때)
PI_OFF = 0.12  # 접촉 해제 확률 → π = 0.08 / 0.20 = 0.4


def window_mean(x, w=WIN):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def simulate(rng, sigma_e, kappa, delta=0.05, alpha=0.2, s0=0.01, mu_v=1.5, mu_o=0.2, p_occ=(0.45, 0.8)):
    c = np.zeros(T_FRAMES, int)
    u = rng.random(T_FRAMES)
    s = 0
    for t in range(T_FRAMES):
        if s == 0 and u[t] < PI_ON:
            s = 1
        elif s == 1 and u[t] < PI_OFF:
            s = 0
        c[t] = s
    beta = alpha * delta
    qc, q = 0.0, 0.0
    e = np.zeros(T_FRAMES)
    for t in range(T_FRAMES):
        step = s0 * (1 + kappa * c[t])
        qc_next = qc + step * rng.standard_normal()
        q = q + alpha * (qc - q) - beta * c[t] + 0.001 * rng.standard_normal()
        e[t] = qc - q
        qc = qc_next
    e_obs = e + sigma_e * rng.standard_normal(T_FRAMES)
    occ = (rng.random(T_FRAMES) < np.where(c == 1, p_occ[1], p_occ[0])).astype(int)
    score = np.where(occ == 1, mu_o, mu_v) * c + rng.standard_normal(T_FRAMES)
    return c, e_obs, occ, score


def logit_fit(X, y, iters=25):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        w = w + np.linalg.solve(H, g)
    return w


def logit_pred(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


def auc(y, s):
    r = stats.rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f1_at(y, p, thr=0.5):
    yhat = p >= thr
    tp = np.sum(yhat & (y == 1))
    fp = np.sum(yhat & (y == 0))
    fn = np.sum(~yhat & (y == 1))
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def evaluate(c, e_obs, occ, score):
    ok = np.isfinite(window_mean(e_obs))
    y = c[ok].astype(float)
    o = occ.astype(float)
    # 시각: 보이는 점수, 가려진 점수, 가림 여부를 따로 넣어 가림에 따른 상호작용을 허용
    vis = np.column_stack([window_mean(score * (1 - o))[ok], window_mean(score * o)[ok], window_mean(o)[ok]])
    prop = np.column_stack([window_mean(np.abs(e_obs))[ok]])
    prop2 = np.column_stack([window_mean(np.abs(e_obs))[ok], window_mean(e_obs)[ok]])
    feats = {"vision (occlusion-aware)": vis, "proprio |e|": prop, "proprio |e| + signed e": prop2,
             "fusion (vision + |e|)": np.column_stack([vis, prop])}
    half = len(y) // 2
    out = {}
    models = {}
    for name, X in feats.items():
        mu, sd = X[:half].mean(0), X[:half].std(0) + 1e-12
        Z = (X - mu) / sd
        w = logit_fit(Z[:half], y[:half])
        p = logit_pred(w, Z[half:])
        models[name] = p
        out[name] = {"auc": round(auc(y[half:], p), 3), "f1_at_0.5": round(f1_at(y[half:], p), 3)}
    occ_ok = o[ok][half:] == 1
    pv = models["vision (occlusion-aware)"]
    # Table II 방식(가림 조건별 F1)은 가림 조건마다 접촉 비율이 달라 F1이 흔들린다. AUC와 접촉 비율을 함께 본다.
    for tag, m in (("occluded", occ_ok), ("visible", ~occ_ok)):
        out[f"vision_{tag}_f1"] = round(f1_at(y[half:][m], pv[m]), 3)
        out[f"vision_{tag}_auc"] = round(auc(y[half:][m], pv[m]), 3)
        out[f"vision_{tag}_contact_rate"] = round(float(y[half:][m].mean()), 3)
        # 핵심 질문: 가려진 프레임에서 관절 특징을 더하면 AUC가 오르는가 (논문 Table II의 "Full > Vision" 가림 칸)
        out[f"fusion_{tag}_auc"] = round(auc(y[half:][m], models["fusion (vision + |e|)"][m]), 3)
        out[f"proprio_{tag}_auc"] = round(auc(y[half:][m], models["proprio |e|"][m]), 3)
    return out


def spearman_err_contact(c, e_obs):
    ok = np.isfinite(window_mean(e_obs))
    x = window_mean(np.abs(e_obs))[ok]
    rho, _ = stats.spearmanr(x, c[ok])
    return round(float(rho), 3)


def main():
    rng = np.random.default_rng(0)
    res = {"settings": {"T_frames": T_FRAMES, "pi_contact": round(PI_ON / (PI_ON + PI_OFF), 3), "window": WIN,
                        "delta_rad": 0.05, "alpha": 0.2, "occlusion_rate": None}}
    scen = []
    for sigma_e in (0.005, 0.02):
        for kappa in (0.0, -0.5, 0.5):
            c, e_obs, occ, score = simulate(rng, sigma_e, kappa)
            r = evaluate(c, e_obs, occ, score)
            r["spearman_absErr_vs_contact"] = spearman_err_contact(c, e_obs)
            r["occlusion_rate_measured"] = round(float(occ.mean()), 3)
            r["contact_rate_measured"] = round(float(c.mean()), 3)
            scen.append({"sigma_e_rad": sigma_e, "kappa_speed_in_contact": kappa, **r})
            print(f"sigma={sigma_e} kappa={kappa:+.1f} | {json.dumps(r, ensure_ascii=False)}")
    res["scenarios"] = scen
    # 대조: 접촉이 관절 편차를 만들지 않는데(δ = 0) 정책이 접촉 중 속도를 바꾸는 경우(κ ≠ 0)
    # → 접촉과 무관한 "운동 강도"만으로 생기는 Spearman. 논문의 시간 셔플 대조로는 이것을 걸러내지 못한다.
    res["null_delta0"] = {}
    for kappa in (0.0, -0.5, 0.5, -1.0, 1.0):
        c, e_obs, occ, score = simulate(rng, 0.005, kappa, delta=0.0)
        res["null_delta0"][f"kappa={kappa:+.1f}"] = {
            "spearman_absErr_vs_contact": spearman_err_contact(c, e_obs),
            "proprio_auc": evaluate(c, e_obs, occ, score)["proprio |e|"]["auc"]}
    print("null delta=0:", res["null_delta0"])
    (ROOT / "analysis" / "sensing_toy_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
