"""ManipTrans (arXiv:2503.21860 v1, CVPR 2025) 표 재계산과 일관성 검사.
표준 라이브러리만 사용한다. 결과는 recompute_results.json에 저장한다."""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

def pct_ok(k, n, shown, dec=1):
    """k/n을 소수 dec 자리(%)로 반올림한 값이 shown과 같은지."""
    return n > 0 and abs(round(100.0 * k / n, dec) - shown) < 1e-9

def feasible_counts(shown, n_range):
    """표기된 백분율을 정확히 만드는 (k, n) 쌍을 찾는다."""
    res = []
    for n in n_range:
        ks = [k for k in range(0, n + 1) if pct_ok(k, n, shown)]
        if ks:
            res.append({"n": n, "k": ks})
    return res

# 1) 표 1 SR: 단일/양손 열. 각 셀에서 가능한 분모(n)를 찾는다.
t1 = {"Retarget-Only": (4.6, 0.0), "RL-Only": (34.3, 12.1),
      "Retarget+Residual": (47.8, 13.9), "ManipTrans": (58.1, 39.5)}
n_rng = range(10, 201)
t1_single = {m: [c["n"] for c in feasible_counts(v[0], n_rng)] for m, v in t1.items()}
t1_bi = {m: [c["n"] for c in feasible_counts(v[1], n_rng)] for m, v in t1.items() if v[1] > 0}
out["table1_feasible_denominators"] = {"single": t1_single, "bimanual": t1_bi}
# 단일과 양손 분모가 모든 행에서 동시에 가능한 (n_s, n_b) 쌍 (n_b는 약 절반 규모 가정 없이 탐색)
common_s = set(t1_single["ManipTrans"]) & set(t1_single["RL-Only"]) & set(t1_single["Retarget+Residual"])
common_b = set(t1_bi["ManipTrans"]) & set(t1_bi["RL-Only"]) & set(t1_bi["Retarget+Residual"])
all_s = set(t1_single["Retarget-Only"]) & common_s
out["table1_common_denominators"] = {"single_all4_rows": sorted(all_s), "single_3rows": sorted(common_s)[:40],
                                     "bimanual_3rows": sorted(common_b)[:40]}
pairs = []
for N in range(40, 161):
    for ns in sorted(all_s):
        nb = N - ns
        if nb in common_b and nb > 0:
            pairs.append([ns, nb])
out["table1_pairs_sum_40_160"] = pairs[:60]

# 2) 표 2 (리 배치 정책, 4개 방법): 4개 값을 모두 만드는 최소 분모
t2 = {"IBC": 4.69, "BET": 9.69, "DP-UNet": 18.44, "DP-Trans": 14.69}
cands = []
for N in range(20, 2001):
    if all(any(pct_ok(k, N, v, 2) for k in range(0, N + 1)) for v in t2.values()):
        cands.append(N)
out["table2_common_denominators_20_to_2000"] = cands[:30]
# 각 값의 정확한 분수 후보 (N=320, 420 등 명시 검사)
t2_check = {}
for N in (320, 420, 300, 280, 140, 21 * 20):
    t2_check[str(N)] = {m: [k for k in range(N + 1) if pct_ok(k, N, v, 2)] for m, v in t2.items()}
out["table2_check"] = t2_check

# 3) 표 4 노이즈 실험: 단일 SR이 기준(표 1 단일 58.1)보다 낮은지, 양손 SR 단조성
t4_single = {"sigma0 (표 1)": 58.1, "0.5cm": 55.1, "1.0cm": 55.3, "1.5cm": 46.7}
t4_bi = {"sigma0 (표 1)": 39.5, "0.5cm": 30.1, "1.0cm": 19.5, "1.5cm": 39.2}
out["table4_nonmonotone_bimanual"] = [t4_bi["1.0cm"] < t4_bi["1.5cm"]]

# 4) 이항 신뢰구간(Wilson 95%)과 두 비율의 Fisher 정확 검정(양측)
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def hyper_pmf(k, K, n, N):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)

def fisher_two_sided(a, n1, b, n2):
    """2x2 표 [[a, n1-a], [b, n2-b]]의 양측 Fisher 검정 p값."""
    K = a + b
    N = n1 + n2
    lo, hi = max(0, K - n2), min(n1, K)
    p_obs = hyper_pmf(a, K, n1, N)
    p = 0.0
    for x in range(lo, hi + 1):
        px = hyper_pmf(x, K, n1, N)
        if px <= p_obs * (1 + 1e-7):
            p += px
    return min(1.0, p)

def k_from(shown, n, dec=1):
    return [k for k in range(0, n + 1) if pct_ok(k, n, shown, dec)]

# 단일 과제 n은 표에서 역산 가능한 값 중 공통으로 맞는 값을 사용
S_N = None
for n in range(20, 201):
    if all(k_from(v, n) for v in (58.1, 47.8, 34.3, 4.6)):
        S_N = n
        break
out["single_task_n_first_match"] = S_N

# 5) 비교: 단일 과제에서 ManipTrans 대 Retarget+Residual 일부 가정 (n=S_N)
comp = {}
if S_N:
    a = k_from(58.1, S_N)[0]
    b = k_from(47.8, S_N)[0]
    comp["single_ManipTrans_vs_RetResidual"] = {
        "n_per_method": S_N, "k_ManipTrans": a, "k_RetRes": b,
        "wilson_ManipTrans": wilson(a, S_N), "wilson_RetRes": wilson(b, S_N),
        "fisher_p_two_sided": fisher_two_sided(a, S_N, b, S_N)}
out["comparisons_assumed_n"] = comp

# 6) 양손 과제: 분모가 명시되지 않아 가정 n(예: 40)으로 민감도 확인
for n_b in (40, 43):
    a = k_from(39.5, n_b)
    b = k_from(13.9, n_b)
    if a and b:
        out[f"bimanual_sensitivity_n{n_b}"] = {
            "k_ManipTrans": a[0], "k_RetRes": b[0],
            "fisher_p_two_sided": fisher_two_sided(a[0], n_b, b[0], n_b)}

# 7) 데이터셋 통계 검산
frames, eps, fps = 1.34e6, 3300, 60
out["dataset_avg_frames_per_episode"] = frames / eps
out["dataset_avg_seconds_per_episode"] = frames / eps / fps
out["bimanual_seq_fraction_claim"] = 600 / 3300

# 8) 시간 비교 텍스트 검산: 본문 QuasiSim ~40 h, 부록 C 단계당 10-20 h(3단계)
out["quasisim_total_h_from_stages"] = [3 * 10, 3 * 20]
out["maniptrans_pretrain_h"] = 1.5 * 24
out["maniptrans_residual_min"] = 15

# 9) 관측·행동 차원 (부록 F Table 7, 재정책 정책)
obs = {"hand_joint_q": 12, "wrist_state": 13, "object_state": 13, "obj_pose_goal": 7, "wrist_pose_goal": 7}
act = {"joint_aq": 12, "wrist_pose_aw": 7}
out["table7_obs_dim"] = sum(obs.values())
out["table7_act_dim"] = sum(act.values())


# 10) 유의성: 분모가 확정되지 않아, 두 방법이 같은 n을 쓴다고 가정하고
#     가능한 모든 n(20-200)에 대한 Fisher p값의 범위를 보고한다.
def p_range(a_shown, b_shown, dec=1, n_lo=20, n_hi=200):
    ps, ns = [], []
    for n in range(n_lo, n_hi + 1):
        ka = k_from(a_shown, n, dec)
        kb = k_from(b_shown, n, dec)
        for a in ka:
            for b in kb:
                ps.append(fisher_two_sided(a, n, b, n))
                ns.append(n)
    if not ps:
        return None
    return {"n_feasible_count": len(set(ns)), "p_min": min(ps), "p_max": max(ps)}

sig = {}
sig["single_ManipTrans_58.1_vs_RetRes_47.8"] = p_range(58.1, 47.8)
sig["bimanual_ManipTrans_39.5_vs_RetRes_13.9"] = p_range(39.5, 13.9)
sig["bimanual_ManipTrans_39.5_vs_RL_12.1"] = p_range(39.5, 12.1)
out["sig_table1_range_over_feasible_n"] = sig

# 표 2: 분모 320으로 고정(유일한 공통 분모 중 최소)
k2 = {m: k_from(v, 320, 2)[0] for m, v in t2.items()}
out["table2_counts_n320"] = k2
pairs2 = {}
for a, b in [("DP-UNet", "BET"), ("DP-UNet", "IBC"), ("DP-Trans", "BET"), ("DP-UNet", "DP-Trans")]:
    pairs2[f"{a}_vs_{b}"] = {"p_fisher": fisher_two_sided(k2[a], 320, k2[b], 320),
                              "wilson_a": wilson(k2[a], 320), "wilson_b": wilson(k2[b], 320)}
out["table2_pairwise_n320"] = pairs2

# 11) 윌슨 신뢰구간 (표 1 ManipTrans 단일 58.1, 양손 39.5, 분모 가정 n=43 예시)
out["wilson_example_n43"] = {"single_25of43": wilson(25, 43), "bimanual_17of43": wilson(17, 43)}

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])
