"""Foresight 곱셈 보상, PBRS, 희소 보상을 3단계 체인 MDP toy에서 비교 (CPU, numpy 1스레드).

모형 (tabular, 행동 4개, 단계 3개: grasp, insert, rotate)
- 단계 k의 행동 a는 기하 성공 확률 SIG[k][a]와 핸드오프 품질 Q[k][a]를 가진다.
- 단계 k+1의 성공 확률은 SIG[k+1][b] * Q[k][h] (h = 직전 단계 행동). 실패하면 에피소드가 끝난다.
- 정책은 단계별 softmax 표이고, 상태는 직전 행동이다.

두 실험
1) 단계별 학습 (논문 Alg. 1 형식, 뒤에서부터, 하류 정책 고정). 기대 기울기를 열거로 정확히 계산한다.
   보상: 상수 x_k(subtask 기준선), x_k * V_hat[a_k](foresight 곱셈; V_hat에 추정기 오차를 넣음), 전역 희소 R.
2) 끝까지 학습(joint). 표본 기울기(배치 32 에피소드, REINFORCE)로 PBRS 형태 F를 넣은 희소 보상과 비교.
   Φ에 편향·과대확신·악용 오차를 넣어 최종 정책이 바뀌는지 본다.

실행: python3 analysis/shaping_toy.py → analysis/shaping_toy_results.json (약 1분)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "shaping_toy_results.json"
A = 4
SIG = np.array([[0.98, 0.92, 0.85, 0.75],   # grasp: 행동 0이 기하 성공은 가장 높음
                [0.95, 0.90, 0.85, 0.70],   # insert
                [0.99, 0.97, 0.90, 0.80]])  # rotate (마지막 단계)
Q = np.array([[0.35, 0.60, 0.80, 0.95],     # grasp 핸드오프 품질 (행동 0은 가장 나쁨: "불안정한 잡기")
              [0.40, 0.65, 0.85, 0.97]])    # insert 핸드오프 품질
LR_EXACT, ITERS_STAGE = 5.0, 800
LR_JOINT, B_JOINT, ITERS_JOINT, N_INIT = 1.0, 32, 1000, 20
N_NOISE = 3200   # 추정기 표본 수 (핸드오프 4개에 나눈 Binomial 표본)

P = np.zeros((3, A, A))          # P[k][s, a]: 단계 k, 직전 상태 s, 행동 a의 성공 확률
P[0] = SIG[0][None, :]
P[1] = SIG[1][None, :] * Q[0][:, None]
P[2] = SIG[2][None, :] * Q[1][:, None]


def build_leaves():
    """에피소드 경로 열거. 각 경로는 단계별 (상태, 행동, 성공, 실행 여부)."""
    rows = []
    for a0 in range(A):
        for x0 in (0, 1):
            if x0 == 0:
                rows.append(([0, -1, -1], [a0, -1, -1], [0, -1, -1], [1, 0, 0])); continue
            for a1 in range(A):
                for x1 in (0, 1):
                    if x1 == 0:
                        rows.append(([0, a0, -1], [a0, a1, -1], [1, 0, -1], [1, 1, 0])); continue
                    for a2 in range(A):
                        for x2 in (0, 1):
                            rows.append(([0, a0, a1], [a0, a1, a2], [1, 1, x2], [1, 1, 1]))
    st, ac, xs, tk = (np.array(v) for v in zip(*rows))
    return st, ac, xs, tk.astype(bool)


ST, AC, XS, TK = build_leaves()
L = len(ST)
STc, ACc = np.where(TK, ST, 0), np.where(TK, AC, 0)
XSc = np.where(TK, XS, 0)
X_LEAF = XSc.astype(float)                       # (L,3) 단계 성공 지표 (실행 안 된 단계는 0)
R_LEAF = (TK.all(axis=1) & (XSc == 1).all(axis=1)).astype(float)   # 전역 성공
OH = np.eye(A)[ACc]                              # (L,3,A)
PSUCC = np.stack([P[k][STc[:, k], ACc[:, k]] for k in range(3)], axis=1)
RTG_R = np.repeat(R_LEAF[:, None], 3, axis=1)    # 전역 희소 보상의 반환


def softmax_rows(th):
    z = np.exp(th - th.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def leaf_prob(pi):
    pol = np.stack([pi[k][STc[:, k], ACc[:, k]] for k in range(3)], axis=1)
    out = np.where(XSc == 1, PSUCC, 1 - PSUCC)
    return np.prod(np.where(TK, pol * out, 1.0), axis=1)


def full_success(pi):
    return float(np.sum(leaf_prob(pi) * R_LEAF))


def value_from(pi, k, s):
    """단계 k 시작, 직전 상태 s에서 k 이후 전부 성공할 확률 (하류 정책 고정)."""
    if k == 3:
        return 1.0
    return sum(pi[k][s, a] * P[k][s, a] * value_from(pi, k + 1, a) for a in range(A))


def grad(pi, G, w=None):
    """기대 기울기 (w 없으면 경로 확률). G: (L,3) 경로·단계별 반환. 반환: 단계별 (A,A) 기울기."""
    if w is None:
        w = leaf_prob(pi)
    out = []
    for k in range(3):
        g = np.zeros((A, A))
        m = TK[:, k]
        contrib = (w * G[:, k])[m, None] * (OH[m, k, :] - pi[k][STc[m, k]])
        np.add.at(g, STc[m, k], contrib)
        out.append(g)
    return out


def estimators(V, rng, perm):
    """추정기 오차 모형. V: 참값 벡터(길이 A). 키 = 실험 이름."""
    lg = np.log(np.clip(V, 1e-6, 1 - 1e-6) / (1 - np.clip(V, 1e-6, 1 - 1e-6)))
    trap = V.copy(); trap[0] = 0.99   # 불안정한 잡기(행동 0)를 높게 예측
    return {
        "exact": V.copy(),
        "noisy": rng.binomial(N_NOISE, V) / N_NOISE,
        "bias+0.15": np.clip(V + 0.15, 0, 1),
        "bias-0.15": np.clip(V - 0.15, 0, 1),
        "overconf_x3": 1 / (1 + np.exp(-3 * lg)),
        "trap_exploit": trap,
        "scale_matched_const": np.full(A, V.mean()),
        "shuffled": V[perm],
    }


def optimum():
    """참 최적 값과 단계별 최적 행동 (역방향 DP)."""
    V = {3: np.ones(A)}
    act = {}
    for k in (2, 1, 0):
        states = range(A) if k > 0 else [0]
        vals = np.zeros(A)
        for s in states:
            c = [P[k][s, a] * V[k + 1][a] for a in range(A)]
            vals[s], act[(k, s)] = max(c), int(np.argmax(c))
        V[k] = vals
    return V[0][0], act


def phi_table(Vstar_dp):
    """Φ_k(s): 최적 DP 값(단계 k 시작, 상태 s). Φ_3 = 0. 행 0은 상태가 하나."""
    Phi = np.zeros((3, A))
    for k in range(3):
        Phi[k] = Vstar_dp[k]
    return Phi


def pbrs_rtg(Phi):
    """PBRS 보상 F_k = x_k Φ_{k+1}(a_k) - Φ_k(s_k) (실행한 단계만)의 반환 합 sum_{j>=k} F_j."""
    F = np.zeros((L, 3))
    for k in range(3):
        nxt = np.zeros(L) if k == 2 else Phi[k + 1][ACc[:, k]]
        F[:, k] = np.where(TK[:, k], X_LEAF[:, k] * nxt - Phi[k][STc[:, k]], 0.0)
    return np.cumsum(F[:, ::-1], axis=1)[:, ::-1]


def staged(scheme, est, init, rng, perm, P_star):
    """Alg. 1: 단계 2 → 1 → 0. 각 단계는 하류가 고정된 상태에서 기대 기울기로 학습."""
    theta = [t.copy() for t in init]
    hist = []
    for k in (2, 1, 0):
        for _ in range(ITERS_STAGE):
            pi = [softmax_rows(t) for t in theta]
            if scheme == "myopic":
                G = X_LEAF
            elif scheme == "sparse_global":
                G = RTG_R
            else:
                if k == 2:
                    Vh = np.ones(A)
                else:
                    Vtrue = np.array([value_from(pi, k + 1, h) for h in range(A)])
                    Vh = estimators(Vtrue, rng, perm)[est]
                G = np.repeat((X_LEAF[:, k] * Vh[ACc[:, k]])[:, None], 3, axis=1)
            theta[k] = theta[k] + LR_EXACT * grad(pi, G)[k]
            hist.append(full_success([softmax_rows(t) for t in theta]))
    return full_success([softmax_rows(t) for t in theta]), hist


def joint(G_rtg, init, rng):
    """끝까지 학습. 배치 B_JOINT 에피소드를 표본으로 뽑아 REINFORCE (기대 대신 표본 기울기)."""
    theta = [t.copy() for t in init]
    hist = []
    for _ in range(ITERS_JOINT):
        pi = [softmax_rows(t) for t in theta]
        pl = leaf_prob(pi)
        idx = rng.choice(L, size=B_JOINT, p=pl / pl.sum())
        w = np.bincount(idx, minlength=L) / B_JOINT
        g = grad(pi, G_rtg, w=w)
        theta = [theta[k] + LR_JOINT * g[k] for k in range(3)]
        hist.append(full_success([softmax_rows(t) for t in theta]))
    return full_success([softmax_rows(t) for t in theta]), hist


def summarize(finals, hits, P_star):
    hit = [h for h in hits if h is not None]
    return {"final_mean": round(float(np.mean(finals)), 4), "final_sd": round(float(np.std(finals, ddof=1)), 4),
            "final_minus_optimum": round(float(np.mean(finals) - P_star), 4),
            "reached_95pct_of_optimum": f"{len(hit)}/{len(hits)}",
            "iters_to_95pct_median": float(np.median(hit)) if hit else None}


def main():
    P_star, act = optimum()
    Vdp = {0: np.full(A, P_star), 1: np.array([max(P[1][s, b] * max(P[2][b, c] for c in range(A)) for b in range(A)) for s in range(A)]),
           2: np.array([max(P[2][s, c] for c in range(A)) for s in range(A)])}
    Phi_star = phi_table(Vdp)
    rng0 = np.random.default_rng(0)
    inits = [[rng0.normal(0, 0.1, (A, A)) for _ in range(3)] for _ in range(N_INIT)]
    perms = [rng0.permutation(A) for _ in range(N_INIT)]
    out = {"_meta": {"A": A, "SIG": SIG.tolist(), "Q": Q.tolist(), "P_star": round(float(P_star), 4),
                     "optimal_actions": {f"phase{k},state{s}": v for (k, s), v in act.items()},
                     "myopic_all_action0": round(full_success([np.tile(np.eye(A)[0], (A, 1))] * 3), 4),
                     "n_init": N_INIT, "n_leaves": int(L),
                     "staged": f"exact expected gradient, lr {LR_EXACT}, {ITERS_STAGE} it/stage",
                     "joint": f"sampled REINFORCE, batch {B_JOINT}, lr {LR_JOINT}, {ITERS_JOINT} it"}}

    curves = {}
    stage_list = [("myopic_constant", "myopic", None), ("sparse_global_staged", "sparse_global", None),
                  ("foresight_exact", "foresight", "exact"), ("foresight_noisy", "foresight", "noisy"),
                  ("foresight_bias+0.15", "foresight", "bias+0.15"), ("foresight_bias-0.15", "foresight", "bias-0.15"),
                  ("foresight_overconf_x3", "foresight", "overconf_x3"),
                  ("foresight_trap_exploit", "foresight", "trap_exploit"),
                  ("foresight_scale_matched_const", "foresight", "scale_matched_const"),
                  ("foresight_shuffled", "foresight", "shuffled")]
    for label, scheme, est in stage_list:
        fin, hits = [], []
        for i, init in enumerate(inits):
            f, h = staged(scheme, est, init, np.random.default_rng(100 + i), perms[i], P_star)
            fin.append(f)
            hh = np.where(np.array(h) >= 0.95 * P_star)[0]
            hits.append(int(hh[0]) if len(hh) else None)
        out[label] = summarize(fin, hits, P_star)
        curves[label] = np.mean(np.array([staged(scheme, est, init, np.random.default_rng(100 + i), perms[i], P_star)[1]
                                          for i, init in enumerate(inits[:4])]), axis=0)[::10].round(4).tolist()
        print(label, out[label], flush=True)

    joint_list = [("joint_sparse", None), ("joint_pbrs_exact", Phi_star),
                  ("joint_pbrs_bias+0.15", np.clip(Phi_star + 0.15, 0, 1) * (np.arange(3)[:, None] < 3)),
                  ("joint_pbrs_bias-0.15", np.clip(Phi_star - 0.15, 0, 1)),
                  ("joint_pbrs_trap_exploit", None)]
    trap_phi = Phi_star.copy(); trap_phi[1, 0] = 0.99; trap_phi[2, 0] = 0.99
    joint_list[4] = ("joint_pbrs_trap_exploit", trap_phi)
    for label, Phi in joint_list:
        Grt = RTG_R if Phi is None else pbrs_rtg(Phi) + R_LEAF[:, None]
        fin, hits, cur = [], [], []
        for i, init in enumerate(inits):
            f, h = joint(Grt, init, np.random.default_rng(200 + i))
            fin.append(f)
            hh = np.where(np.array(h) >= 0.95 * P_star)[0]
            hits.append(int(hh[0]) if len(hh) else None)
            if i < 4:
                cur.append(h)
        out[label] = summarize(fin, hits, P_star)
        curves[label] = np.mean(np.array(cur), axis=0)[::20].round(4).tolist()
        print(label, out[label], flush=True)

    # 검증 1: 정확한 V에서 단계 0의 foresight 기대 기울기 = 전역 희소 보상의 기대 기울기
    pi_chk = [softmax_rows(t) for t in inits[0]]
    Vt = np.array([value_from(pi_chk, 1, h) for h in range(A)])
    g_s = grad(pi_chk, RTG_R)[0]
    g_f = grad(pi_chk, np.repeat((X_LEAF[:, 0] * Vt[ACc[:, 0]])[:, None], 3, axis=1))[0]
    out["_check_foresight_equals_sparse_expected_grad"] = {"max_abs_diff_phase0": float(np.max(np.abs(g_s - g_f)))}
    # 검증 2: PBRS 반환 합이 모든 경로에서 R - Φ_0 (telescoping)
    tot = R_LEAF + pbrs_rtg(Phi_star)[:, 0]
    out["_check_pbrs_telescoping"] = {"max_abs_dev_from_R_minus_Phi0": float(np.max(np.abs(tot - (R_LEAF - Phi_star[0, 0]))))}
    # 검증 3: 정확한 기대 기울기에서 PBRS 항은 상태만의 함수 (기준값)라 기울기를 바꾸지 않는다
    g_r = grad(pi_chk, RTG_R)
    g_p = grad(pi_chk, pbrs_rtg(Phi_star) + R_LEAF[:, None])
    out["_check_pbrs_exact_gradient_unchanged"] = {"max_abs_diff_all_phases": float(max(np.max(np.abs(a - b)) for a, b in zip(g_r, g_p)))}
    out["_curves_every_10_or_20"] = curves
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
