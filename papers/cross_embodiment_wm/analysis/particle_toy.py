"""논문의 핵심 메커니즘을 2차원 toy로 확인한다 (CPU, numpy 1스레드). 실행: python3 analysis/particle_toy.py → analysis/particle_toy_results.json

이 toy는 논문 모델을 재현하지 않는다. 가정을 명시한 장난감 세계에서 다음만 본다.

  실험 A (손 교체와 집계 방식): 손 6개(링크 길이·입자 간격이 다름). 한 손을 빼고 나머지 x개 손의 데이터로
      "입자 변위 예측기"를 학습한 뒤 빠진 손에서 시험한다. 예측기는 상수 하나(κ)만 가진 선형 모형이고,
      집계 방식만 다르다.
        sum  : 접촉 이웃의 손 변위 가중합 Σ_j w_ij δ_j  (그래프 차수에 따라 값이 커짐)
        mean : 가중 평균 Σ_j w_ij δ_j / Σ_j w_ij  (차수에 무관)
      참 세계는 평균형(차수 불변)이다. 따라서 "입자 밀도가 다른 손"에서 sum은 틀릴 것으로 예상하고, 이를 확인한다.
      비교 기준은 빠진 손 자신의 데이터로만 학습한 손별 모형(논문의 "Ours (Target Only)" 역할)이다.
  실험 B (MPC의 모형 오차 민감도): 참 세계의 게인에 오차 ε를 넣은 모형으로 샘플링 MPC를 돌린다.
      논문 설정과 맞춘 값: 후보 500개, 최적화 반복 10회, 계획 길이 H ∈ {1,2,4}, 앞 2 단계 실행 후 재계획.
      성공률과 ε의 관계, H의 효과를 본다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "particle_toy_results.json"
R_CONTACT = 0.05     # 접촉 반경 (m)
GATE = 0.5           # 접촉 가중합이 이 값에 닿으면 게이트 1
M_OBJ = 30           # 물체 입자 수
BASE = np.array([0.0, 0.0])


def fk_particles(q, lengths, h):
    """q: (K,2) 관절각 → 손 입자 위치 (K,N,2). 링크를 간격 h로 샘플(입자 수 ∝ 링크 길이)."""
    l1, l2 = lengths
    s1 = np.arange(0.0, l1 + 1e-9, h)
    s2 = np.arange(0.0, l2 + 1e-9, h)
    th1, th2 = q[:, 0:1], q[:, 1:2]
    p1 = np.stack([s1[None, :] * np.cos(th1), s1[None, :] * np.sin(th1)], -1)           # (K,n1,2)
    j2 = np.stack([l1 * np.cos(th1), l1 * np.sin(th1)], -1)                              # (K,1,2)
    ang = th1 + th2
    p2 = j2 + np.stack([s2[None, :] * np.cos(ang), s2[None, :] * np.sin(ang)], -1)     # (K,n2,2)
    return np.concatenate([p1, p2], axis=1)


def contact_weights(obj, hand):
    """obj (K,M,2), hand (K,N,2) → w (K,M,N) = (1 - r/R)^2 for r < R."""
    r = np.linalg.norm(obj[:, :, None, :] - hand[:, None, :, :], axis=-1)
    return np.clip(1.0 - r / R_CONTACT, 0.0, None) ** 2


def true_step(obj, hand, dhand, gain=0.8):
    """참 세계: 평균형(차수 불변) 접촉 전달 + 게이트."""
    w = contact_weights(obj, hand)                       # (K,M,N)
    sw = w.sum(-1)                                       # (K,M)
    num = np.einsum("kmn,knd->kmd", w, dhand)
    mean = num / np.maximum(sw, 1e-12)[..., None]
    gate = np.minimum(1.0, sw / GATE)[..., None]
    return gain * gate * mean


def sum_features(obj, hand, dhand):
    w = contact_weights(obj, hand)
    return np.einsum("kmn,knd->kmd", w, dhand)


def mean_features(obj, hand, dhand):
    w = contact_weights(obj, hand)
    sw = w.sum(-1)
    num = np.einsum("kmn,knd->kmd", w, dhand)
    mean = num / np.maximum(sw, 1e-12)[..., None]
    return mean * np.minimum(1.0, sw / GATE)[..., None]


def fit_gain(feat, target):
    """최소제곱 κ: target ≈ κ·feat (스칼라, 모든 입자·방향 공유)."""
    f, t = feat.ravel(), target.ravel()
    return float(f @ t / max(f @ f, 1e-12))


def sample_hand(rng, lengths, h, T, u_std=0.12):
    """한 손의 무작위 관절 궤적 (K=1) → 손 위치 (T,N,2)와 변위 (T-1,N,2)."""
    q = np.zeros((1, 2))
    q[0] = [rng.uniform(-0.6, 0.6), rng.uniform(-0.8, 0.8)]
    qs = [q.copy()]
    for _ in range(T - 1):
        q = q + rng.normal(0, u_std, size=(1, 2))
        q = np.clip(q, -1.6, 1.6)
        qs.append(q.copy())
    qs = np.concatenate(qs, 0)                           # (T,2)
    P = fk_particles(qs, lengths, h)[:, :, :]            # (T,N,2)
    return qs, P


def make_episodes(rng, lengths, h, n_ep, T=40):
    """다수의 짧은 에피소드: 물체 입자를 손 근처에 두고 무작위로 민다. 반환: (obj_t, hand_t, dhand_t, obj_next)."""
    data = []
    for _ in range(n_ep):
        qs, P = sample_hand(rng, lengths, h, T)
        # 물체: 손 끝 근처 원판(반경 0.05), 입자 위치는 피보나치 격자로 고정 배치 + 작은 잡음
        ang = np.pi * (3 - np.sqrt(5))
        idx = np.arange(M_OBJ)
        rr = 0.05 * np.sqrt((idx + 0.5) / M_OBJ)
        base_obj = np.stack([rr * np.cos(idx * ang), rr * np.sin(idx * ang)], -1)
        center = P[0, -1] + rng.uniform(-0.03, 0.03, 2) + np.array([0.02, 0.0])
        obj = (base_obj + center)[None]                  # (1,M,2)
        for t in range(T - 1):
            dh = P[t + 1] - P[t]                          # (N,2)
            dobj = true_step(obj, P[t][None], dh[None])[0]
            data.append((obj[0].copy(), P[t].copy(), dh.copy(), dobj.copy()))
            obj = obj + dobj[None]
    return data


def features_for(data, kind):
    feats, targs = [], []
    for obj, hand, dh, dobj in data:
        o, hd, dd = obj[None], hand[None], dh[None]
        f = sum_features(o, hd, dd) if kind == "sum" else mean_features(o, hd, dd)
        feats.append(f[0])
        targs.append(dobj)
    return np.concatenate(feats), np.concatenate(targs)


def mse_of(kappa, data, kind):
    f, t = features_for(data, kind)
    return float(np.mean((kappa * f - t) ** 2))


def stack_hand(data):
    """한 손의 전이 목록 → 배열 (n,M,2) 물체, (n,N,2) 손, (n,N,2) 손 변위, (n,M,2) 물체 변위."""
    obj = np.stack([d[0] for d in data])
    hand = np.stack([d[1] for d in data])
    dh = np.stack([d[2] for d in data])
    dobj = np.stack([d[3] for d in data])
    return obj, hand, dh, dobj


def feats(arr, kind):
    obj, hand, dh, _ = arr
    f = sum_features(obj, hand, dh) if kind == "sum" else mean_features(obj, hand, dh)
    return f.reshape(-1, 2)


def fit_k(F_list, T_list):
    f = np.concatenate(F_list).ravel()
    t = np.concatenate(T_list).ravel()
    return float(f @ t / max(f @ f, 1e-12))


def experiment_A(seed=0):
    rng = np.random.default_rng(seed)
    # 6개 손: (링크 길이 l1, l2), 입자 간격 h. 입자 밀도 = 1/h, 길이에 따라 총 입자 수가 다름
    hands = {
        "H1 (짧은 손, 촘촘)": ((0.14, 0.12), 0.015),
        "H2 (짧은 손, 보통)": ((0.14, 0.12), 0.025),
        "H3 (중간, 보통)": ((0.20, 0.16), 0.030),
        "H4 (중간, 성김)": ((0.20, 0.16), 0.045),
        "H5 (긴 손, 성김)": ((0.26, 0.22), 0.050),
        "H6 (긴 손, 보통)": ((0.26, 0.22), 0.035),
    }
    names = list(hands)
    n_ep = 40
    A = {}
    for nm in names:
        lengths, h = hands[nm]
        arr = stack_hand(make_episodes(rng, lengths, h, n_ep))
        A[nm] = {"arr": arr, "F": {k: feats(arr, k) for k in ("sum", "mean")}, "T": arr[3].reshape(-1, 2)}
    # 입자 수와 반경 내 평균 이웃 수(차수)
    deg = {}
    for nm in names:
        lengths, h = hands[nm]
        s1 = np.arange(0.0, lengths[0] + 1e-9, h)
        s2 = np.arange(0.0, lengths[1] + 1e-9, h)
        obj, hand, _, _ = A[nm]["arr"]
        r = np.linalg.norm(obj[:20, :, None, :] - hand[:20, None, :, :], axis=-1)
        deg[nm] = {"N_particles": int(len(s1) + len(s2)), "spacing": h,
                   "mean_neighbors_in_R_per_object_particle": float(np.mean(np.sum(r < R_CONTACT, axis=-1)))}

    def test_mse(k, held, kind):
        arr = A[held]["arr"]
        pred = k * A[held]["F"][kind]
        return float(np.mean((pred - A[held]["T"]) ** 2))

    rows = []
    for held in names:
        train_names = [n for n in names if n != held]
        out = {"held": held}
        for kind in ("sum", "mean"):
            k = fit_k([A[n]["F"][kind] for n in train_names], [A[n]["T"] for n in train_names])
            out[f"pooled_{kind}_kappa"] = k
            out[f"pooled_{kind}_mse"] = test_mse(k, held, kind)
            # 빠진 손 자신의 데이터 절반으로 학습한 손별 모형, 나머지 절반으로 시험
            half = n_ep * 39 // 2
            Fh, Th = A[held]["F"][kind], A[held]["T"]
            kown = float(Fh[:half].ravel() @ Th[:half].ravel() / max(Fh[:half].ravel() @ Fh[:half].ravel(), 1e-12))
            out[f"own_{kind}_mse"] = float(np.mean((kown * Fh[half:] - Th[half:]) ** 2))
        rows.append(out)
    diffs = np.array([r["pooled_sum_mse"] / r["pooled_mean_mse"] for r in rows])
    w = stats.wilcoxon(np.array([r["pooled_sum_mse"] for r in rows]), np.array([r["pooled_mean_mse"] for r in rows]))
    # 손 수 x별 곡선: Fig. 2처럼 가능한 부분집합을 열거하되 최대 8개만 (CPU 시간 제한)
    import itertools
    curve = {}
    for kind in ("sum", "mean"):
        curve[kind] = {}
        for x in range(1, 6):
            per_target = []
            for held in names:
                pool = [n for n in names if n != held]
                subsets = list(itertools.combinations(pool, x))
                if len(subsets) > 8:
                    pick = np.random.default_rng(100 + x).choice(len(subsets), 8, replace=False)
                    subsets = [subsets[i] for i in pick]
                errs = []
                for sub in subsets:
                    k = fit_k([A[n]["F"][kind] for n in sub], [A[n]["T"] for n in sub])
                    errs.append(test_mse(k, held, kind))
                per_target.append(float(np.mean(errs)))
            curve[kind][x] = {"per_target_mean_mse": per_target, "avg": float(np.mean(per_target)),
                              "n_subsets_used_max": 8}
    return {"hands": {n: {"lengths": hands[n][0], "h": hands[n][1], **deg[n]} for n in names},
            "leave_one_out": rows,
            "ratio_sum_over_mean_pooled": {"values": diffs.round(3).tolist(), "median": float(np.median(diffs)),
                                           "wilcoxon_p": float(w.pvalue)},
            "scaling_curve": curve,
            "n_episodes_per_hand": n_ep}


# ---------- 실험 B: MPC ----------

def mpc_rollout_predict(obj, hand_q0, u_seq, gain, lengths, h):
    """u_seq: (C,H,2) 후보 관절 증분. 모형으로 H 단계 예측. 반환 obj_H (C,M,2) 와 H단계 누적 비용 계산용 궤적."""
    C, H, _ = u_seq.shape
    q = np.repeat(hand_q0[None], C, 0)
    o = np.repeat(obj[None], C, 0)
    for t in range(H):
        q_new = q + u_seq[:, t]
        P0 = fk_particles(q, lengths, h)
        P1 = fk_particles(q_new, lengths, h)
        dP = P1 - P0
        o = o + true_step(o, P0, dP, gain=gain)
        q = q_new
    return o, q


def chamfer(a, b):
    """a (C,M,2), b (M,2): 대칭 Chamfer 평균 거리."""
    d = np.linalg.norm(a[:, :, None, :] - b[None, None, :, :], axis=-1)
    return d.min(2).mean(1) + d.min(1).mean(1)


def run_episode(rng, gain_true, gain_model, H, lengths, h, target, obj0, q0, n_replan=6, C=500, iters=10, n_exec=2, sigma=0.15):
    obj = obj0.copy()
    q = q0.copy()
    for _ in range(n_replan):
        mu = np.zeros((H, 2))
        sd = np.full((H, 2), sigma)
        best_seq = None
        for it in range(iters):
            seq = mu[None] + sd[None] * rng.standard_normal((C, H, 2))
            seq = np.clip(seq, -0.25, 0.25)
            oH, _ = mpc_rollout_predict(obj, q, seq, gain_model, lengths, h)
            cost = chamfer(oH, target)
            elite = seq[np.argsort(cost)[:50]]
            mu = elite.mean(0)
            sd = elite.std(0) + 0.02
            best_seq = seq[np.argmin(cost)]
        # 실제 실행(참 세계)
        for t in range(min(n_exec, H)):
            q_new = q + best_seq[t]
            P0 = fk_particles(q[None], lengths, h)[0]
            P1 = fk_particles(q_new[None], lengths, h)[0]
            obj = obj + true_step(obj[None], P0[None], (P1 - P0)[None], gain=gain_true)[0]
            q = q_new
    final = float(chamfer(obj[None], target)[0])
    return final


def experiment_B(seed=1, trials=12):
    lengths, h = (0.20, 0.16), 0.03
    eps_list = [-0.4, -0.2, 0.0, 0.2, 0.4]
    H_list = [1, 2, 4]
    rng_t = np.random.default_rng(seed)
    # 공통 시험 조건(목표·초기 상태)을 모든 (ε, H)에 같이 씀
    cases = []
    for _ in range(trials):
        ang = np.pi * (3 - np.sqrt(5))
        idx = np.arange(M_OBJ)
        rr = 0.05 * np.sqrt((idx + 0.5) / M_OBJ)
        base_obj = np.stack([rr * np.cos(idx * ang), rr * np.sin(idx * ang)], -1)
        q0 = np.array([rng_t.uniform(-0.3, 0.3), rng_t.uniform(-0.3, 0.3)])
        P = fk_particles(q0[None], lengths, h)[0]
        center = P[-1] + np.array([0.03, 0.0]) + rng_t.uniform(-0.01, 0.01, 2)
        obj0 = base_obj + center
        # 목표: 물체를 무작위 방향으로 0.04~0.08 m 옮긴 모양(평행이동만)
        d = rng_t.uniform(0.04, 0.08)
        th = rng_t.uniform(0, 2 * np.pi)
        target = obj0 + d * np.array([np.cos(th), np.sin(th)])
        cases.append((obj0, q0, target, int(rng_t.integers(1 << 30))))
    results = {}
    t0 = time.time()
    for H in H_list:
        results[H] = {}
        for eps in eps_list:
            succ_dist = []
            for obj0, q0, target, s in cases:
                rng = np.random.default_rng(s)
                dist = run_episode(rng, 0.8, 0.8 * (1 + eps), H, lengths, h, target, obj0, q0)
                succ_dist.append(dist)
            succ_dist = np.array(succ_dist)
            results[H][eps] = {"final_chamfer_mean": float(succ_dist.mean()),
                               "final_chamfer_median": float(np.median(succ_dist)),
                               "dist": succ_dist.round(5).tolist()}
    # 성공 기준: 참 세계 ε=0, H=4의 최종 Chamfer 중앙값을 기준선으로 쓴다(사후 정의, 보고에 명시)
    thr = float(np.median(results[4][0.0]["dist"]))
    for H in H_list:
        for eps in eps_list:
            d = np.array(results[H][eps]["dist"])
            results[H][eps]["success"] = int(np.sum(d <= thr))
            results[H][eps]["n"] = len(d)
    return {"threshold_chamfer": thr, "threshold_rule": "참 세계(ε=0), H=4 최종 Chamfer 중앙값 (사후 정의, 기준 자체가 결과에 의존)",
            "results": {str(H): {str(e): v for e, v in results[H].items()} for H in H_list},
            "trials": trials, "elapsed_s": round(time.time() - t0, 1),
            "settings": {"candidates": 500, "iterations": 10, "horizon": H_list, "execute": 2, "replans": 6,
                         "elite": 50, "gain_true": 0.8, "lengths": lengths, "h": h}}


def main():
    t0 = time.time()
    A = experiment_A(seed=0)
    tA = time.time() - t0
    B = experiment_B(seed=1, trials=8)
    res = {"experiment_A_generalization": A, "elapsed_A_s": round(tA, 1), "experiment_B_mpc": B,
           "caveat": "2차원 toy. 논문 모델(DPI-Net, 학습된 MLP)과 다른 단순 모형이며, 가정(참 세계 평균형, 게이트) 아래의 결과다."}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print("A elapsed", round(tA, 1), "s")
    for r in A["leave_one_out"]:
        print(f"{r['held']:<20} pooled sum {r['pooled_sum_mse']:.3e} mean {r['pooled_mean_mse']:.3e} own(sum) {r['own_sum_mse']:.3e} own(mean) {r['own_mean_mse']:.3e} k_sum {r['pooled_sum_kappa']:.3f}")
    print("neighbors:", {k: round(v["mean_neighbors_in_R_per_object_particle"], 2) for k, v in A["hands"].items()})
    print("ratio sum/mean median", round(A["ratio_sum_over_mean_pooled"]["median"], 3), "wilcoxon p", round(A["ratio_sum_over_mean_pooled"]["wilcoxon_p"], 4))
    for kind in ("sum", "mean"):
        print(kind, {x: round(v["avg"], 6) for x, v in A["scaling_curve"][kind].items()})
    print("B elapsed", B["elapsed_s"], "thr", round(B["threshold_chamfer"], 4))
    for H in B["results"]:
        print("H", H, {e: (v["success"], v["n"]) for e, v in B["results"][H].items()})


if __name__ == "__main__":
    main()
