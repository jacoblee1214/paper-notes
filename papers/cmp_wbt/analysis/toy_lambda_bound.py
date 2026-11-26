"""토이 T1: 논문 부록 C의 "TD(lambda) 목표는 엄밀한 하한" 주장을 작은 표 형식 MDP에서 수치로 확인한다.

- 상태 30개: 위험(I=0) 6개, 어떤 명령에서도 안전 상태에 머무는 닫힌 안전 집합 8개, 나머지 16개는 과도 상태.
  잠재 명령 z는 6개이고 전이는 무작위(시드 고정). 과도 상태에서 위험 상태로 빠지는 확률이 명령마다 다르다.
- V*는 식 2의 고정점(영속 안전 확률의 최댓값)을 값 반복으로 구한다. W(s,z) = I(s) * E[V*(s')].
- 행동 정책(인코더)은 균등 무작위 z라고 둔다. P^(n)과 G^lambda를 정확히 계산한다(표본 없이 행렬곱).
확인 항목:
  (a) P^(n)이 n에 대해 단조 비증가이고 G^lambda <= W 인가 (V* 사용, 부록 C.3-C.4)
  (b) timeout=1 규칙(식 3)을 쓰면 에피소드 끝 k 스텝 전에서 목표가 하한을 넘는가 (부록 C.4의 증명이 덮지 않는 경우)
  (c) V*를 W(s, 0)(고정 명령 하나)로 바꾸면(식 3의 근사) 하한이 유지되는가, 추정 잡음이 섞이면 어떤가
논문은 이 MDP를 다루지 않으므로 이 토이는 증명의 논리만 검사하며, 논문 수치와는 무관하다.

실행: python3 analysis/toy_lambda_bound.py   (numpy, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
nS, nF, nZ = 30, 6, 6
I = np.ones(nS)
I[:nF] = 0.0
P = np.zeros((nZ, nS, nS))  # P[z, s, s']
SAFE_CLOSED = np.arange(nF, nF + 8)
TRANS = np.arange(nF + 8, nS)
for z in range(nZ):
    for s in range(nS):
        if s < nF:
            P[z, s] = rng.dirichlet(np.ones(nS))  # 위험 상태 (W=0 이므로 값은 쓰이지 않음)
        elif s in SAFE_CLOSED:
            P[z, s, SAFE_CLOSED] = rng.dirichlet(np.ones(8))
        else:
            q_fail = rng.beta(1.0, 6.0) + 0.03 * z   # 명령마다 위험 확률이 다름
            q_home = rng.beta(1.0, 5.0)              # 닫힌 안전 집합으로 들어갈 확률
            q_home = min(q_home, 1 - q_fail - 0.05)
            P[z, s, :nF] = q_fail * rng.dirichlet(np.ones(nF))
            P[z, s, SAFE_CLOSED] = q_home * rng.dirichlet(np.ones(8))
            P[z, s, TRANS] = (1 - q_fail - q_home) * rng.dirichlet(np.ones(len(TRANS)))


def value_iteration():
    V = np.ones(nS)
    for _ in range(5000):
        W = I[None, :] * np.einsum("zst,t->zs", P, V)  # (z, s)
        Vn = W.max(axis=0)
        if np.abs(Vn - V).max() < 1e-13:
            break
        V = Vn
    return V


Vstar = value_iteration()
W = I[:, None] * np.einsum("zst,t->sz", P, Vstar)  # W[s, z]
assert np.allclose(W.max(axis=1), Vstar, atol=1e-9)
Tb = P.mean(axis=0)  # 균등 정책의 전이 행렬 T_b[s, s']


def n_step(v_term, N):
    """P^(n)[s, z], n = 1..N: 처음 n 스텝은 롤아웃, 이후 v_term(s_{t+n}) 로 마감."""
    G = v_term.copy()  # G_0 = v_term
    out = []
    # P^(n)(s,z) = I(s) sum_{s1} P(s1|s,z) G_{n-1}(s1);  G_k(s) = I(s) sum_{s'} Tb(s,s') G_{k-1}(s')
    for n in range(1, N + 1):
        out.append(I[:, None] * np.einsum("zst,t->sz", P, G))
        G = I * (Tb @ G)
    return np.stack(out)  # (N, s, z)


def lam_target(Pn, lam):
    n = np.arange(1, Pn.shape[0] + 1)
    w = (1 - lam) * lam ** (n - 1)
    return np.tensordot(w, Pn, axes=(0, 0)) + (lam ** Pn.shape[0]) * Pn[-1]  # 잔여 질량은 마지막 항으로


res = {"states": nS, "unsafe_states": nF, "latents": nZ, "mean_Vstar_safe": float(Vstar[I > 0].mean())}
safe = I > 0

# (a) V* 사용
N = 400
Pn = n_step(Vstar, N)
res["a_P1_equals_W"] = bool(np.allclose(Pn[0], W))
res["a_monotone_nonincreasing_max_violation"] = float((Pn[1:] - Pn[:-1]).max())
a = {}
for lam in (0.5, 0.8, 0.95):
    G = lam_target(Pn, lam)
    a[str(lam)] = {"max(G-W)": float((G - W).max()), "mean_gap(W-G)_safe": float((W - G)[safe].mean())}
res["a_lambda_target_vs_W"] = a

# (b) timeout=1: 끝 k 스텝 전 기대 목표 H_k(s) 를 timeout 없는 무한 지평 목표 H_inf(s), 균등 정책 평균 W_b(s) = E_z W(s,z) 와 비교
Wb = W.mean(axis=1)
b = {}
for lam in (0.8, 0.95):
    Hinf = np.zeros(nS)
    for _ in range(3000):
        Hinf = I * (Tb @ ((1 - lam) * Vstar + lam * Hinf))
    H = I.copy()  # H_0: 타임아웃 시점에 안전하면 1
    rows = []
    for k in range(1, 201):
        H = I * (Tb @ ((1 - lam) * Vstar + lam * H))
        if k in (1, 2, 5, 10, 20, 40, 100):
            rows.append({"k": k, "lambda^(k-1)": round(lam ** (k - 1), 4),
                         "mean(H_k - H_inf)_safe": float((H - Hinf)[safe].mean()),
                         "share_states_H_k_exceeds_W_b": float(((H - Wb)[safe] > 1e-12).mean()),
                         "max(H_k - W_b)_safe": float((H - Wb)[safe].max())})
    b[str(lam)] = {"mean(W_b - H_inf)_safe": float((Wb - Hinf)[safe].mean()), "rows": rows}
res["b_timeout1_bias"] = b

# (c) V* 를 W(s, z0) 로 대체 (z0 = 고정 명령 하나), 그리고 추정 잡음
z0 = int(np.argmax(W[safe].mean(axis=0)))  # 평균적으로 가장 좋은 명령을 "원점"으로 둔다 (낙관적 선택)
z_bad = int(np.argmin(W[safe].mean(axis=0)))
c = {"origin_choice": {"best_avg_z": z0, "worst_avg_z": z_bad}}
for name, z in (("origin=best_avg_z", z0), ("origin=worst_avg_z", z_bad)):
    Vo = W[:, z]
    c[name] = {"mean(Vstar - W(s,z0))_safe": float((Vstar - Vo)[safe].mean()),
               "max(W(s,z0) - Vstar)": float((Vo - Vstar).max())}
    Pn_o = n_step(Vo, N)
    G = lam_target(Pn_o, 0.8)
    c[name]["max(G-W)"] = float((G - W).max())
    c[name]["mean_gap(W-G)_safe"] = float((W - G)[safe].mean())
noise = {}
for sd in (0.02, 0.05, 0.1):
    viol, gaps = [], []
    for _ in range(200):
        Vhat = np.clip(W[:, z0] + rng.normal(0, sd, nS), 0, 1)
        G = lam_target(n_step(Vhat, N), 0.8)
        viol.append(float(((G - W)[safe] > 1e-12).mean()))
        gaps.append(float((G - W)[safe].max()))
    noise[str(sd)] = {"mean_share_of_(s,z)_with_G>W": float(np.mean(viol)), "mean_max(G-W)": float(np.mean(gaps))}
c["estimator_noise_sd_on_origin_value"] = noise
bias = {}
for dlt in (0.05, 0.1, 0.2, 0.3):
    Vhat = np.clip(W[:, z0] + dlt, 0, 1)
    G = lam_target(n_step(Vhat, N), 0.8)
    bias[str(dlt)] = {"share_of_(s,z)_with_G>W": float(((G - W)[safe] > 1e-12).mean()), "max(G-W)": float((G - W)[safe].max())}
c["estimator_constant_overestimate_delta"] = bias
res["c_origin_substitution_and_noise"] = c

out = Path(__file__).with_name("toy_lambda_bound_results.json")
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
