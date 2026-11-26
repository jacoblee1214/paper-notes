"""정리 1과 정리 2를 작은 CPU 토이로 확인한다. 결과: analysis/toy_theory_results.json

토이 A (정리 1의 범위): 1차원 선형-가우시안 환경 s' = s + a + eta, eta ~ N(0, 0.01^2).
  데이터 정책: a = -0.5 s + sigma_a * xi (sigma_a를 바꿔 가며 데이터의 행동 영향 I_D를 조절).
  모델 두 가지를 같은 데이터로 학습: 선형 회귀(외삽함), k-최근접 이웃(외삽 안 함).
  평가 두 가지: (i) 평가 분포 = 데이터 분포 (정리 1이 말하는 경우), (ii) 평가 정책 a ~ N(0, 0.5^2) (다른 학습자).
  추정량은 식 (A.1) 그대로: I_hat = E_p [ log p_hat(s'|s,a) - log p_hat_pi(s'|s) ], p_hat_pi는 행동을 평가 정책으로 섞은 분포.
  확인: (i)에서 I_hat <= I_D (정리 1), (ii)에서는 I_hat이 I_D보다 클 수 있음(정리 1의 우변은 평가 분포의 I이지 데이터의 I가 아님).

토이 B (정리 2): 표 형태 문맥 밴딧(상태 4, 행동 3, 다음 상태 5)에서 pi* ∝ beta exp(E[log q - log beta])가 I를 올리는지 2000개 무작위 환경에서 확인.
  q를 진짜 사후분포로 쓰면 항상 오른다. q를 유한 표본(n/상태)으로 추정하면 오르지 않는 경우가 생기는 비율을 잰다.

실행: python3 analysis/toy_theory.py   (numpy, scipy; 약 1분)
"""
import json
from math import log
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_theory_results.json"
rng = np.random.default_rng(7)

# ------------------------------------------------------------------ 토이 A
SIG_ETA = 0.01
SIG_E = 0.5     # 평가 정책 a ~ N(0, 0.5^2)


def make_data(n, sig_a):
    s = rng.normal(0, 1, n)
    a = -0.5 * s + sig_a * rng.normal(0, 1, n)
    sp = s + a + SIG_ETA * rng.normal(0, 1, n)
    return s, a, sp


def fit_linear(s, a, sp):
    X = np.c_[s, a, np.ones_like(s)]
    w = np.linalg.lstsq(X, sp, rcond=None)[0]
    return lambda ss, aa: w[0] * ss + w[1] * aa + w[2], w


def fit_knn(s, a, sp, k=10):
    tree = cKDTree(np.c_[s, a])

    def f(ss, aa):
        _, idx = tree.query(np.c_[ss, aa], k=k)
        return sp[idx].mean(1)
    return f


def gauss_logpdf(x, mu, sig):
    return -0.5 * ((x - mu) / sig) ** 2 - np.log(sig) - 0.5 * np.log(2 * np.pi)


def explained_info(f, sig_hat, mean_pi, sd_pi, n=1500, G=1500):
    """I_hat = E_p[log p_hat(s'|s,a) - log p_hat_pi(s'|s)] (식 A.1).
    행동 정책 pi(.|s) = N(mean_pi(s), sd_pi^2). p_hat_pi는 a에 대한 구적(G개 격자, 정책 밀도 가중)으로 계산해 몬테카를로 편향을 피한다."""
    s = rng.normal(0, 1, n)
    a = mean_pi(s) + sd_pi * rng.normal(0, 1, n)
    sp = s + a + SIG_ETA * rng.normal(0, 1, n)
    lp = gauss_logpdf(sp, f(s, a), sig_hat)
    z = np.linspace(-6, 6, G)                       # 표준화 격자
    wz = np.exp(-0.5 * z ** 2); wz /= wz.sum()
    ss = np.repeat(s, G); aa = (mean_pi(s)[:, None] + sd_pi * z[None, :]).ravel()
    mu = f(ss, aa).reshape(n, G)
    lps = gauss_logpdf(sp[:, None], mu, sig_hat) + np.log(wz)[None, :]
    lpi = logsumexp(lps, axis=1)
    return float(np.mean(lp - lpi))


size = {"linear": (3000, 3000), "knn": (700, 900)}   # (평가 점 수, 행동 격자 수). kNN은 질의가 비싸서 줄임
rows = []
for sig_a in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    s, a, sp = make_data(4000, sig_a)
    sv, av, spv = make_data(2000, sig_a)             # 검증(데이터 분포)으로 sigma_hat 추정
    I_D = 0.5 * log(1 + sig_a ** 2 / SIG_ETA ** 2)   # 데이터의 조건부 상호정보 (nats)
    I_eval_true = 0.5 * log(1 + SIG_E ** 2 / SIG_ETA ** 2)
    row = {"sigma_a": sig_a, "I_D": round(I_D, 3), "I_eval_true": round(I_eval_true, 3)}
    f_lin, w = fit_linear(s, a, sp)
    f_knn = fit_knn(s, a, sp)
    for name, f in (("linear", f_lin), ("knn", f_knn)):
        sig_hat = max(float(np.std(spv - f(sv, av))), 1e-4)
        row[name] = {
            "sigma_hat": round(sig_hat, 4),
            "I_hat_on_data_distribution": round(explained_info(f, sig_hat, lambda ss: -0.5 * ss, max(sig_a, 1e-6), *size[name]), 3),
            "I_hat_on_eval_policy": round(explained_info(f, sig_hat, lambda ss: np.zeros_like(ss), SIG_E, *size[name]), 3),
        }
    row["linear_w_s_w_a"] = [round(float(w[0]), 3), round(float(w[1]), 3)]
    rows.append(row)
    print(row)

A = {"setting": {"sig_eta": SIG_ETA, "sig_e": SIG_E, "n_train": 4000, "knn_k": 10, "eval": "선형 3000점 x 격자 3000, kNN 700점 x 격자 900 (행동 구적)"},
     "rows": rows}
# 검사: 데이터 분포에서 I_hat <= I_D (MC 잡음 0.05 허용)
A["check_bound_on_data_distribution"] = all(
    r[m]["I_hat_on_data_distribution"] <= r["I_D"] + 0.05 for r in rows for m in ("linear", "knn"))
A["linear_exceeds_I_D_on_eval_policy_rows"] = [r["sigma_a"] for r in rows if r["linear"]["I_hat_on_eval_policy"] > r["I_D"] + 0.05]
A["knn_exceeds_I_D_on_eval_policy_rows"] = [r["sigma_a"] for r in rows if r["knn"]["I_hat_on_eval_policy"] > r["I_D"] + 0.05]
A["check_bound_vs_true_eval_MI"] = all(
    r[m]["I_hat_on_eval_policy"] <= r["I_eval_true"] + 0.1 for r in rows for m in ("linear", "knn"))

# ------------------------------------------------------------------ 토이 B
NS, NA, NSP = 4, 3, 5


def mutual_info(rho, pi, p):
    """I(rho, pi) = E_rho pi p [ log p(s'|s,a) - log p_pi(s'|s) ]"""
    ppi = np.einsum("sa,sak->sk", pi, p)
    return float(np.sum(rho[:, None, None] * pi[:, :, None] * p * (np.log(p) - np.log(ppi)[:, None, :])))


def lower_bound(rho, pi, p, q):
    """L(pi) = E_rho pi p [ log q(a|s,s') - log pi(a|s) ]"""
    return float(np.sum(rho[:, None, None] * pi[:, :, None] * p * (np.log(q) .transpose(0, 1, 2) - np.log(pi)[:, :, None])))


def posterior(beta, p):
    joint = beta[:, :, None] * p                  # [s,a,s']
    return joint / joint.sum(1, keepdims=True)    # q(a|s,s')


def tilt(beta, p, q):
    r = np.sum(p * (np.log(q) - np.log(beta)[:, :, None]), axis=2)    # E_{s'}[log q - log beta] (s,a)
    w = beta * np.exp(r)
    return w / w.sum(1, keepdims=True)


rho = np.full(NS, 1 / NS)
exact_ok = 0; tight_ok = 0; N_INST = 2000
for _ in range(N_INST):
    p = rng.dirichlet(np.full(NSP, 0.4), size=(NS, NA))
    p = np.clip(p, 1e-6, None); p /= p.sum(2, keepdims=True)
    beta = rng.dirichlet(np.full(NA, 1.0), size=NS)
    q = posterior(beta, p)
    pistar = tilt(beta, p, q)
    I_b, I_s = mutual_info(rho, beta, p), mutual_info(rho, pistar, p)
    L_b, L_s = lower_bound(rho, beta, p, q), lower_bound(rho, pistar, p, q)
    exact_ok += (I_s >= I_b - 1e-12) and (L_s <= I_s + 1e-12)
    tight_ok += abs(L_b - I_b) < 1e-9

B = {"instances": N_INST, "exact_q_I_increases_and_L_below_I": exact_ok, "L_beta_equals_I_beta": tight_ok}
noisy = {}
for n_per_state in (5, 20, 100, 1000):
    worse = 0; deltas = []; EPS = 0.5
    for _ in range(N_INST):
        p = rng.dirichlet(np.full(NSP, 0.4), size=(NS, NA))
        p = np.clip(p, 1e-6, None); p /= p.sum(2, keepdims=True)
        beta = rng.dirichlet(np.full(NA, 1.0), size=NS)
        # beta로 상태마다 n개 표본을 뽑아 q를 세어서 추정
        qh = np.zeros((NS, NA, NSP)) + EPS * 1e-3
        for s_ in range(NS):
            acts = rng.choice(NA, n_per_state, p=beta[s_])
            for a_ in acts:
                sp_ = rng.choice(NSP, p=p[s_, a_])
                qh[s_, a_, sp_] += 1
        # 세어진 (a, s') 결합 빈도를 s'별로 정규화한 q_hat(a|s,s')
        qh = qh / qh.sum(1, keepdims=True)
        pistar_h = tilt(beta, p, qh)               # E_{s'}는 진짜 p로, q만 추정 (낙관적 설정)
        d = mutual_info(rho, pistar_h, p) - mutual_info(rho, beta, p)
        deltas.append(d); worse += d < 0
    noisy[str(n_per_state)] = {"fraction_I_decreases": worse / N_INST, "mean_delta_I_nats": round(float(np.mean(deltas)), 4),
                               "p05_delta": round(float(np.percentile(deltas, 5)), 4)}
B["noisy_q_hat"] = noisy
print(json.dumps(B, indent=1))

OUT.write_text(json.dumps({"A_theorem1_scope": A, "B_theorem2_tabular": B}, indent=1), encoding="utf-8")
print("wrote", OUT)
