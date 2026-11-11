"""추가 검증: 레이블 없는 전이에서 예측-관찰 보상 GRPO로 움직임 토큰을 복원하는지 (numpy, CPU).

실행: python3 analysis/extra_grpo_recovery.py  → analysis/extra_grpo_recovery_results.json

세계: 잠재 관측 o ∈ R^6, 이산 움직임 토큰 a ∈ {0..5}(진짜 토큰). 전이는 o' = A o + E_a + B_a o + 잡음이다.
순방향 모델 f(o, a)는 레이블 있는 전이(3,000개)로 teacher forcing 최소제곱 학습한다(논문에서 VLM이 정답 CoD로 학습되는 것에 대응).
정책은 선형 소프트맥스 π(a|o)이며 출력은 K+1개이고, 마지막 출력은 형식 오류(malformed)다.
RL은 레이블 없는 전이(3,000개)에서 GRPO로 학습한다: 상태마다 G = 4 후보를 샘플하고,
보상 r = cos(f(o, a), o') + 잡음(σ_r)이다. 형식 오류 후보의 보상은 그룹 최저 유효 보상 − 벌칙이다.
그룹 표준화 advantage를 쓰고, 기울기는 log π 기울기를 Adam으로 적용한다.

비교: (1) 레이블 SFT(상한), (2) 의사 레이블(Δo에 k-means, 군집 번호는 레이블로 헝가리안 매칭), (3) 무작위(1/K).
측정: 검증 세트(레이블 있음, 학습에 쓰지 않음)에서 argmax 토큰 정확도, 형식 오류 선택률,
그리고 03 §3.1의 "정상 후보 advantage 폭 축소"를 같은 설정에서 측정한다.
실험 1: 보상 잡음 σ_r 스캔(형식 오류 없음). 실험 2: 형식 오류 초기 확률 0.3에서 벌칙 0과 0.05 비교.
시드 5개(세계와 학습 모두 시드별로 새로 뽑음).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
D, K, G = 6, 6, 4
N_LAB, N_UNL, N_TEST = 3000, 3000, 1000
ITERS, BATCH, LR_RL, LR_SFT = 800, 64, 0.01, 0.05
EARLY_IT = 100  # 학습 초기(100 step) 정확도: 형식 오류 벌칙이 학습 속도에 주는 영향 확인용
SEEDS = (0, 1, 2, 3, 4)
NOISE_SCAN = (0.0, 0.05, 0.1, 0.2, 0.5)
PENALTY = 0.05
PENALTY_SCAN = (0.0, 0.05, 0.5)
P_BAD_INIT = 0.3


def make_world(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    A = 0.9 * Q
    E = rng.normal(0, 0.8, (K, D))
    Bm = rng.normal(0, 0.2, (K, D, D))
    Wexp = rng.normal(0, 1, (D + 1, K))  # 전문가 토큰 규칙: 관찰 o의 함수 (상태에서 예측 가능해야 SFT가 의미가 있음)
    return A, E, Bm, Wexp


def transitions(rng, world, n):
    A, E, Bm, Wexp = world
    o = rng.normal(0, 1, (n, D))
    a = (feats(o) @ Wexp).argmax(1)
    o2 = o @ A.T + E[a] + np.einsum("nij,nj->ni", Bm[a], o) + 0.1 * rng.normal(size=(n, D))
    return o, a, o2


def feats(o, a=None):
    """policy/forward 공용 특징: [o, 1]. 순방향 모델은 여기에 onehot(a) ⊗ [o, 1]을 더한다."""
    base = np.concatenate([o, np.ones((len(o), 1))], 1)
    if a is None:
        return base
    oh = np.eye(K)[a]
    inter = (oh[:, :, None] * base[:, None, :]).reshape(len(o), -1)
    return np.concatenate([base, inter], 1)


def fit_forward(o, a, o2):
    X = feats(o, a)
    W, *_ = np.linalg.lstsq(X, o2, rcond=None)
    return W


def predict_all(W, o):
    """모든 후보 a에 대한 예측 (n, K, D)."""
    return np.stack([feats(o, np.full(len(o), k)) @ W for k in range(K)], 1)


def cos(u, v):
    return (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m, self.v = np.zeros(shape), np.zeros(shape)

    def step(self, W, g):  # 상승 방향 (g를 최대화)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return W + self.lr * mh / (np.sqrt(vh) + self.eps)


def accuracy(W, o, a):
    z = feats(o) @ W
    pred_full = z.argmax(1)
    pred_valid = z[:, :K].argmax(1)
    return {"acc_full": float((pred_full == a).mean()), "acc_valid_only": float((pred_valid == a).mean()),
            "malformed_rate": float((pred_full == K).mean())}


def sft(rng, o, a, o_test, a_test):
    W = np.zeros((D + 1, K + 1))
    opt = Adam(W.shape, LR_SFT)
    X = feats(o)
    Y = np.eye(K + 1)[a]
    for _ in range(ITERS):
        idx = rng.integers(0, len(o), 256)
        P = softmax(X[idx] @ W)
        g = X[idx].T @ (Y[idx] - P) / len(idx)
        g[:, K] = 0  # 형식 오류 열은 SFT에서 쓰지 않는다
        W = opt.step(W, g)
    W[:, K] = -50.0  # 형식 오류 출력 억제 (SFT 목표에 없음)
    return accuracy(W, o_test, a_test)


def pseudo_label(rng, o, o2, a, o_test, a_test):
    delta = o2 - o
    centers = delta[rng.choice(len(delta), K, replace=False)].copy()
    for _ in range(50):
        d2 = ((delta[:, None] - centers[None]) ** 2).sum(-1)
        lab = d2.argmin(1)
        for k in range(K):
            if (lab == k).any():
                centers[k] = delta[lab == k].mean(0)
    # 군집 번호를 진짜 토큰에 매칭 (레이블 있는 학습 세트로 헝가리안, 오라클 매칭)
    d2 = ((delta[:, None] - centers[None]) ** 2).sum(-1)
    lab = d2.argmin(1)
    C = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            C[k, j] = np.sum((lab == k) & (a == j))
    row, col = linear_sum_assignment(-C)
    mapping = dict(zip(row, col))
    y_pseudo = np.array([mapping[k] for k in lab])
    W = np.zeros((D + 1, K + 1))
    opt = Adam(W.shape, LR_SFT)
    X = feats(o)
    Y = np.eye(K + 1)[y_pseudo]
    for _ in range(ITERS):
        idx = rng.integers(0, len(o), 256)
        P = softmax(X[idx] @ W)
        g = X[idx].T @ (Y[idx] - P) / len(idx)
        g[:, K] = 0
        W = opt.step(W, g)
    W[:, K] = -50.0
    res = accuracy(W, o_test, a_test)
    return res


def grpo(rng, o_unl, o2_unl, W_fwd, p_bad, noise, penalty):
    W = np.zeros((D + 1, K + 1))
    W[D, K] = np.log(p_bad * K / (1 - p_bad))  # 초기 형식 오류 확률 = p_bad (유효 토큰은 균등)
    pred_all = predict_all(W_fwd, o_unl)           # (N, K, D), 전이마다 고정
    opt = Adam(W.shape, LR_RL)
    adv_stats, W_early = [], None
    for it in range(ITERS):
        if it == EARLY_IT:
            W_early = W.copy()
        idx = rng.integers(0, len(o_unl), BATCH)
        X = feats(o_unl[idx])
        P = softmax(X @ W)
        cum = P.cumsum(1)
        u = rng.random((BATCH, G))
        acts = np.minimum((u[:, :, None] > cum[:, None, :]).sum(-1), K)   # (B, G) 샘플 토큰
        valid = acts < K
        sel = pred_all[idx[:, None], np.minimum(acts, K - 1)]              # (B, G, D)
        rewards = cos(sel, o2_unl[idx][:, None, :]) + noise * rng.normal(size=(BATCH, G))
        floor = np.where(valid, rewards, np.inf).min(1)
        floor = np.where(np.isfinite(floor), floor, -1.0)
        rewards = np.where(valid, rewards, floor[:, None] - penalty)
        mu, sd = rewards.mean(1, keepdims=True), rewards.std(1, keepdims=True)
        adv = (rewards - mu) / (sd + 1e-6)
        if it % 50 == 0 or it == ITERS - 1:
            adv_stats.append(float(np.mean(sd)))
        Y = np.eye(K + 1)[acts]                      # (B, G, K+1)
        grad_logp = Y - P[:, None, :]                # (B, G, K+1)
        g = np.einsum("bg,bgk,bd->dk", adv, grad_logp, X) / (BATCH * G)
        W = opt.step(W, g)
    return W, adv_stats, W_early


def adv_shrink(rng, reward_scale):
    """03 §3.1 재현: 그룹 4개 보상의 표준편차 s, 형식 오류 1개 유무에 따른 정상 후보 advantage 폭."""
    widths_clean, widths_mal = [], []
    for _ in range(4000):
        r4 = rng.normal(0, reward_scale, G)
        a = (r4 - r4.mean()) / (r4.std() + 1e-8)
        widths_clean.append(a.max() - a.min())
        r4m = r4.copy()
        r4m[-1] = r4[:-1].min() - PENALTY
        am = (r4m - r4m.mean()) / (r4m.std() + 1e-8)
        widths_mal.append(am[:-1].max() - am[:-1].min())
    return float(np.mean(widths_clean)), float(np.mean(widths_mal))


def main():
    out = {"settings": {"D": D, "K": K, "G": G, "N_lab": N_LAB, "N_unl": N_UNL, "N_test": N_TEST, "iters": ITERS,
                        "batch": BATCH, "lr_rl": LR_RL, "lr_sft": LR_SFT, "seeds": list(SEEDS),
                        "penalty_scan": list(PENALTY_SCAN), "p_bad_init": P_BAD_INIT, "noise_scan": list(NOISE_SCAN)}}
    exp1 = {str(n): {"grpo": [], "reward_std_within_group": []} for n in NOISE_SCAN}
    base = {"sft": [], "pseudo_label": [], "chance": 1.0 / K, "forward_model_cos_true_vs_other": []}
    exp2 = {str(p): {"grpo": [], "mal_rate_end": [], "reward_std": []} for p in PENALTY_SCAN}
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        world = make_world(rng)
        o_l, a_l, o2_l = transitions(rng, world, N_LAB)
        o_u, _, o2_u = transitions(rng, world, N_UNL)
        o_t, a_t, o2_t = transitions(rng, world, N_TEST)
        W_fwd = fit_forward(o_l, a_l, o2_l)
        # 순방향 모델 점검: 참 토큰 예측이 다른 토큰 예측보다 관찰에 더 가까운 비율
        pred_t = predict_all(W_fwd, o_t)
        true_c = cos(pred_t[np.arange(len(a_t)), a_t], o2_t)
        other = np.stack([cos(pred_t[np.arange(len(a_t)), (a_t + j) % K], o2_t) for j in range(1, K)], 1)
        base["forward_model_cos_true_vs_other"].append(float((true_c[:, None] > other).mean()))
        base["sft"].append(sft(rng, o_l, a_l, o_t, a_t)["acc_valid_only"])
        base["pseudo_label"].append(pseudo_label(rng, o_l, o2_l, a_l, o_t, a_t)["acc_valid_only"])
        # 보상 폭 (진짜 토큰 대비 다른 후보 사이의 cos 차이 표준편차)
        rstd = float(np.mean(np.std(np.stack([cos(pred_t[:, k], o2_t) for k in range(K)], 1), axis=1)))
        for n in NOISE_SCAN:
            W, sds, _ = grpo(np.random.default_rng(seed * 10 + 1), o_u, o2_u, W_fwd, 1e-6, n, 0.0)
            acc = accuracy(W, o_t, a_t)
            exp1[str(n)]["grpo"].append(acc["acc_valid_only"])
            exp1[str(n)]["reward_std_within_group"].append(sds[-1])
        base["reward_std_across_candidates"] = base.get("reward_std_across_candidates", []) + [rstd]
        for pen in PENALTY_SCAN:
            W, sds, W_early = grpo(np.random.default_rng(seed * 10 + 2), o_u, o2_u, W_fwd, P_BAD_INIT, 0.0, pen)
            acc = accuracy(W, o_t, a_t)
            v = exp2[str(pen)]
            v.setdefault("acc_at_100", []).append(accuracy(W_early, o_t, a_t)["acc_valid_only"])
            v.setdefault("mal_rate_at_100", []).append(accuracy(W_early, o_t, a_t)["malformed_rate"])
            v["grpo"].append(acc["acc_valid_only"])
            v["mal_rate_end"].append(acc["malformed_rate"])
            v["reward_std"].append(sds[-1])
    for n in NOISE_SCAN:
        v = exp1[str(n)]
        v["grpo_mean"] = round(float(np.mean(v["grpo"])), 3)
        v["grpo_sd"] = round(float(np.std(v["grpo"])), 3)
        v["reward_std_within_group_mean"] = round(float(np.mean(v["reward_std_within_group"])), 4)
    for k in exp2:
        v = exp2[k]
        v["grpo_mean"] = round(float(np.mean(v["grpo"])), 3)
        v["grpo_sd"] = round(float(np.std(v["grpo"])), 3)
        v["mal_rate_end_mean"] = round(float(np.mean(v["mal_rate_end"])), 3)
        v["acc_at_100_mean"] = round(float(np.mean(v["acc_at_100"])), 3)
        v["acc_at_100_sd"] = round(float(np.std(v["acc_at_100"])), 3)
        v["mal_rate_at_100_mean"] = round(float(np.mean(v["mal_rate_at_100"])), 3)
        v["reward_std_mean"] = round(float(np.mean(v["reward_std"])), 4)
    base_sum = {"sft_mean": round(float(np.mean(base["sft"])), 3), "sft_sd": round(float(np.std(base["sft"])), 3),
                "pseudo_label_mean": round(float(np.mean(base["pseudo_label"])), 3),
                "pseudo_label_sd": round(float(np.std(base["pseudo_label"])), 3),
                "chance": round(1 / K, 3),
                "forward_model_true_beats_other_fraction": round(float(np.mean(base["forward_model_cos_true_vs_other"])), 3),
                "reward_std_across_candidates_mean": round(float(np.mean(base["reward_std_across_candidates"])), 4)}
    shrink_clean, shrink_mal = adv_shrink(np.random.default_rng(9), 0.031)
    out["baselines"] = base_sum
    out["exp1_reward_noise"] = exp1
    out["exp2_malformed_penalty"] = exp2
    out["exp3_adv_width_shrink_03_sec3_1"] = {"reward_scale": 0.031, "clean_width": round(shrink_clean, 3),
                                              "with_1_malformed_width": round(shrink_mal, 3),
                                              "shrink_ratio": round(shrink_mal / shrink_clean, 3)}
    path = ROOT / "analysis" / "extra_grpo_recovery_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
