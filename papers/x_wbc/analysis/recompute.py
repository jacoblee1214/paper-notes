"""X-WBC (arXiv:2609.15213 v2) 표 수치 재계산과 일관성 검사.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (수 초, numpy·scipy)
논문이 시도 수를 밝히지 않은 값은 '가정'이라고 표시하고, 가정에 따라 결론이 어떻게 바뀌는지만 계산한다.
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def feasible_n(values, decimals, nmax=3_000_000, pct=True):
    """values(퍼센트)가 모두 k/n 을 decimals 자리로 반올림한 값이 되는 가장 작은 n과, 값별 가능한 n의 개수 요약."""
    scale = 10 ** decimals
    v = np.array(values, dtype=float)
    lo = (v - 0.5 / scale) / 100.0
    hi = (v + 0.5 / scale) / 100.0
    found = []
    for n in range(1, nmax + 1):
        kmin = np.ceil(lo * n - 1e-9)
        kmax = np.floor(hi * n + 1e-9)
        if np.all(kmin <= kmax):
            found.append(n)
            if len(found) >= 5:
                break
    return found


def half_up(x, d):
    f = Fraction(x).limit_denominator(10 ** 9)
    s = 10 ** d
    return float(math.floor(f * s + Fraction(1, 2)) / s)


# ---------------- 표 2: 100STYLE 800 클립 ----------------
N = 800
t2 = {"X-WBC": dict(sr=91.13, comp=91.66, mpkpe=0.0649, vel=0.5400, k=729),
      "SONIC": dict(sr=93.00, comp=93.35, mpkpe=0.0687, vel=0.3419, k=744),
      "TWIST": dict(sr=75.62, comp=78.18, mpkpe=0.0565, vel=0.5261, k=605)}
rows = {}
for name, d in t2.items():
    k = d["k"]
    exact = 100 * k / N
    # 실패 클립의 평균 완료율 (Comp = (k + (N-k) c) / N)
    c = (d["comp"] / 100 * N - k) / (N - k)
    c_lo = ((d["comp"] - 0.005) / 100 * N - k) / (N - k)
    c_hi = ((d["comp"] + 0.005) / 100 * N - k) / (N - k)
    # 활성 걸음 수 중 실패 클립이 차지하는 비율 (클립 길이가 같다고 가정)
    active = k + (N - k) * c
    frac_failed_steps = (N - k) * c / active
    rows[name] = dict(k=k, exact_sr=exact, half_even=round(exact, 2), half_up=half_up(exact, 2),
                      reported_sr=d["sr"], wilson95=wilson(k, N),
                      failed_clips=N - k, mean_completion_of_failed=c,
                      mean_completion_of_failed_range=[c_lo, c_hi],
                      failed_share_of_active_steps=frac_failed_steps,
                      seconds_into_clip_at_failure=c * 133.02 * 60 / N)
R["table2"] = rows
R["table2_clip_len_s"] = 133.02 * 60 / N

def fisher(a, b, n=N):
    ta = [[a, n - a], [b, n - b]]
    return stats.fisher_exact(ta)[1]

def newcombe(a, b, n=N):
    l1, u1 = [x / 100 for x in wilson(a, n)]
    l2, u2 = [x / 100 for x in wilson(b, n)]
    p1, p2 = a / n, b / n
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return 100 * d, 100 * lo, 100 * hi

cmp = {}
for a, b in [("X-WBC", "SONIC"), ("X-WBC", "TWIST"), ("SONIC", "TWIST")]:
    ka, kb = t2[a]["k"], t2[b]["k"]
    cmp[f"{a} vs {b}"] = dict(diff_newcombe_pp=newcombe(ka, kb), fisher_unpaired_p=fisher(ka, kb))
R["table2_unpaired"] = cmp

# 짝지은(McNemar) 범위: 같은 800 클립이므로 짝지은 검정이 맞다. 불일치 쌍 수는 논문에 없다.
def mcnemar_range(ka, kb, n=N):
    fa, fb = n - ka, n - kb          # 실패 수
    d = fa - fb                       # b - c  (a만 실패 - b만 실패)
    out = []
    for c in range(0, fb + 1):
        b = c + d
        if b < 0 or b > fa or c > fb:
            continue
        out.append((b + c, stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0, b, c))
    p_min = min(x[1] for x in out)
    p_max = max(x[1] for x in out)
    disc_max = max(x[0] for x in out)
    disc_min = min(x[0] for x in out)
    return dict(failed_a=fa, failed_b=fb, discordant_range=[disc_min, disc_max], p_min=p_min, p_max=p_max)

R["table2_paired_bounds"] = {
    "X-WBC vs SONIC": mcnemar_range(729, 744),
    "X-WBC vs TWIST": mcnemar_range(729, 605),
}

# 100STYLE는 스타일 100개 x 동작 8개. 군집(스타일) 효과가 있다면 유효 표본이 줄어든다: 설계효과 가정별 p
dep = {}
for deff in (1, 2, 4, 8):
    # 설계효과를 비율 검정에 반영: 유효 n = 800/deff, 개수도 같은 비율로 축소
    ne = N / deff
    ka, kb = 729 / deff, 744 / deff
    z = (ka / ne - kb / ne) / math.sqrt((ka / ne * (1 - ka / ne) + kb / ne * (1 - kb / ne)) / ne)
    dep[str(deff)] = 2 * stats.norm.sf(abs(z))
R["table2_xwbc_vs_sonic_unpaired_z_by_design_effect"] = dep
dep2 = {}
for deff in (1, 2, 4, 8):
    ne = N / deff
    p1, p2 = 729 / N, 605 / N
    z = (p1 - p2) / math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / ne)
    dep2[str(deff)] = 2 * stats.norm.sf(abs(z))
R["table2_xwbc_vs_twist_unpaired_z_by_design_effect"] = dep2

# 외부 평가와 학습 분포 평가의 차이 (G1, VR)
R["generalization_gap_g1"] = dict(
    sr_pp=91.13 - 98.60,
    mpkpe_ratio=0.0649 / 0.0412,
    vel_ratio=0.5400 / 0.2120)
R["table2_rel"] = dict(
    mpkpe_xwbc_vs_twist=0.0649 / 0.0565 - 1, mpkpe_xwbc_vs_sonic=0.0649 / 0.0687 - 1,
    vel_xwbc_vs_sonic=0.5400 / 0.3419 - 1, vel_xwbc_vs_twist=0.5400 / 0.5261 - 1,
    sr_gap_twist_exact=100 * (729 - 605) / N, sr_gap_twist_reported=91.13 - 75.62)

# ---------------- 표 1 ----------------
G1 = {"X-WBC/VR": (98.60, .0412, .2120, .9936, .4354), "MLP/VR": (97.70, .0411, .2272, .9433, .4465),
      "Single/VR": (97.69, .0412, .2195, .9756, .4431), "X-WBC/Robot": (98.21, .0409, .2164, 1.1591, .3554),
      "X-WBC/Human": (98.66, .0397, .2098, .9562, .4053), "noHR/VR": (95.34, .0492, .2494, 1.1032, .5093),
      "noH/VR": (95.17, .0500, .2508, 1.1437, .5125), "noR/VR": (95.33, .0503, .2461, 1.1760, .5046),
      "noAlign/VR": (95.41, .0496, .2469, 1.1128, .5029), "noRS/VR": (96.54, .0474, .2241, 1.0611, .4793)}
H2 = {"X-WBC/VR": (93.22, .0629, .4172, 1.4716, .4535), "MLP/VR": (91.33, .0652, .4542, 1.4316, .4833),
      "Single/VR": (92.22, .0698, .4417, 1.7709, .4978), "X-WBC/Robot": (92.08, .0634, .4274, 1.6865, .3758),
      "X-WBC/Human": (93.02, .0610, .4124, 1.4069, .4270), "noHR/VR": (85.71, .0722, .5016, 1.6970, .5299),
      "noH/VR": (83.84, .0763, .5244, 1.7784, .5342), "noR/VR": (85.21, .0757, .5106, 1.7719, .5229),
      "noAlign/VR": (85.68, .0735, .5100, 1.6512, .5297), "noRS/VR": (88.65, .0724, .4575, 1.5091, .4875)}
R["table1_min_common_n"] = {
    "G1_all10": feasible_n([v[0] for v in G1.values()], 2),
    "H2_all10": feasible_n([v[0] for v in H2.values()], 2),
    "G1_and_H2_all20": feasible_n([v[0] for v in G1.values()] + [v[0] for v in H2.values()], 2),
    "G1_arch3": feasible_n([G1[k][0] for k in ("X-WBC/VR", "MLP/VR", "Single/VR")], 2),
    "H2_arch3": feasible_n([H2[k][0] for k in ("X-WBC/VR", "MLP/VR", "Single/VR")], 2)}

eff = {}
for rob, T in (("G1", G1), ("H2", H2)):
    x = T["X-WBC/VR"]
    e = {}
    for other in ("MLP/VR", "Single/VR"):
        o = T[other]
        e[other] = dict(sr_gap_pp=x[0] - o[0],
                        fail_rate_xwbc=100 - x[0], fail_rate_other=100 - o[0],
                        fail_reduction_pct=100 * (1 - (100 - x[0]) / (100 - o[0])),
                        mpkpe_change_pct=100 * (x[2 - 1] / o[1] - 1),
                        vel_change_pct=100 * (x[2] / o[2] - 1),
                        root_pos_change_pct=100 * (x[3] / o[3] - 1),
                        root_rot_change_pct=100 * (x[4] / o[4] - 1))
    eff[rob] = e
R["table1_joint_training_effect"] = eff

# 표본 잡음만 고려한 필요 표본 수: 두 비율 차이가 양측 p < .05가 되려면 팔당 n
def n_needed(p1, p2, power=None):
    pb = (p1 + p2) / 2
    return 2 * pb * (1 - pb) * (1.959964 / abs(p1 - p2)) ** 2
R["table1_n_for_significance_unpaired"] = {
    "G1 98.60 vs 97.70": n_needed(.9860, .9770), "G1 98.60 vs 97.69": n_needed(.9860, .9769),
    "H2 93.22 vs 91.33": n_needed(.9322, .9133), "H2 93.22 vs 92.22": n_needed(.9322, .9222),
    "G1 VR 98.60 vs Human 98.66": n_needed(.9860, .9866), "H2 VR 93.22 vs Robot 92.08": n_needed(.9322, .9208)}

# 명령 경로 간 SR 범위와 공동 학습 효과의 비교
R["table1_route_spread_pp"] = dict(
    G1=max(G1[k][0] for k in ("X-WBC/VR", "X-WBC/Robot", "X-WBC/Human")) - min(G1[k][0] for k in ("X-WBC/VR", "X-WBC/Robot", "X-WBC/Human")),
    H2=max(H2[k][0] for k in ("X-WBC/VR", "X-WBC/Robot", "X-WBC/Human")) - min(H2[k][0] for k in ("X-WBC/VR", "X-WBC/Robot", "X-WBC/Human")))

abl = {}
for rob, T in (("G1", G1), ("H2", H2)):
    base = T["X-WBC/VR"][0]
    abl[rob] = {k: dict(sr_drop_pp=base - T[k][0], mpkpe_rise_pct=100 * (T[k][1] / T["X-WBC/VR"][1] - 1))
                for k in ("noHR/VR", "noH/VR", "noR/VR", "noAlign/VR", "noRS/VR")}
R["table1_ablation_drops"] = abl
R["table1_ablation_sr_spread_pp"] = {
    rob: max(T[k][0] for k in ("noHR/VR", "noH/VR", "noR/VR", "noAlign/VR")) - min(T[k][0] for k in ("noHR/VR", "noH/VR", "noR/VR", "noAlign/VR"))
    for rob, T in (("G1", G1), ("H2", H2))}

# 본문 문장 검사
chk = {}
chk["lowest_vel_among_arch_rows_G1"] = G1["X-WBC/VR"][2] < min(G1["MLP/VR"][2], G1["Single/VR"][2])
chk["lowest_vel_among_arch_rows_H2"] = H2["X-WBC/VR"][2] < min(H2["MLP/VR"][2], H2["Single/VR"][2])
chk["robot_route_lowest_root_rot_G1"] = G1["X-WBC/Robot"][4] < min(G1["X-WBC/VR"][4], G1["X-WBC/Human"][4])
chk["robot_route_lowest_root_rot_H2"] = H2["X-WBC/Robot"][4] < min(H2["X-WBC/VR"][4], H2["X-WBC/Human"][4])
chk["all_ablations_lower_SR_G1"] = all(G1[k][0] < G1["X-WBC/VR"][0] for k in G1 if k.startswith("no"))
chk["all_ablations_lower_SR_H2"] = all(H2[k][0] < H2["X-WBC/VR"][0] for k in H2 if k.startswith("no"))
chk["mlp_better_root_pos_than_xwbc_G1"] = G1["MLP/VR"][3] < G1["X-WBC/VR"][3]
chk["mlp_better_root_pos_than_xwbc_H2"] = H2["MLP/VR"][3] < H2["X-WBC/VR"][3]
chk["single_better_root_pos_G1"] = G1["Single/VR"][3] < G1["X-WBC/VR"][3]
chk["single_better_root_pos_H2"] = H2["Single/VR"][3] < H2["X-WBC/VR"][3]
chk["H2_drop_order"] = sorted(((round(H2["X-WBC/VR"][0] - H2[k][0], 2), k) for k in H2 if k.startswith("no")), reverse=True)
chk["H2_noHR_less_harmful_than_noR"] = H2["noHR/VR"][0] > H2["noR/VR"][0]
chk["H2_noAlign_vs_noHR_pp"] = H2["noAlign/VR"][0] - H2["noHR/VR"][0]
R["table1_text_checks"] = chk


# 평가 표본 잡음만 보는 하한: 공통 n의 최솟값(G1 6268, H2 5620)을 쓰고 서로 독립인 두 비율로 검정한다(보수적).
# 같은 모션을 모든 행이 평가했다면 짝지은 검정이 더 예리하다. 학습 시드 변동은 여기에 들어 있지 않다.
def two_prop_p(p1, p2, n):
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    return 2 * stats.norm.sf(abs(p1 - p2) / se)
R["table1_p_at_n8000_eval_noise_only"] = {
    "G1 X-WBC vs MLP": two_prop_p(.9860, .9770, 8000), "G1 X-WBC vs Single": two_prop_p(.9860, .9769, 8000),
    "H2 X-WBC vs MLP": two_prop_p(.9322, .9133, 8000), "H2 X-WBC vs Single": two_prop_p(.9322, .9222, 8000),
    "G1 VR vs Human": two_prop_p(.9860, .9866, 8000), "H2 VR vs Robot route": two_prop_p(.9322, .9208, 8000)}
R["table1_p_at_min_n_eval_noise_only"] = {
    "G1 X-WBC vs MLP (n=6268)": two_prop_p(.9860, .9770, 6268),
    "G1 X-WBC vs Single (n=6268)": two_prop_p(.9860, .9769, 6268),
    "H2 X-WBC vs MLP (n=5620)": two_prop_p(.9322, .9133, 5620),
    "H2 X-WBC vs Single (n=5620)": two_prop_p(.9322, .9222, 5620),
    "G1 VR vs Human (n=6268)": two_prop_p(.9860, .9866, 6268),
    "H2 VR vs Robot route (n=5620)": two_prop_p(.9322, .9208, 5620)}

# ---------------- 부록 표 A7/A8 정합성과 매개변수 수 ----------------
dims = dict(human=22 * 12, vr=5 * 12, robot_cmd=14 * 12, proprio=32 * 3 + 3 + 3, action=32)
priv = 3 + 14 * 9
dims["critic_if_actor_plus_priv"] = dims["proprio"] + dims["robot_cmd"] + priv
R["tableA7_dims"] = dims
R["tableA7_reported"] = dict(human=264, vr=60, robot_cmd=168, proprio=102, critic=399, action=32)

def mlp_params(sizes):
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

def tf_layer(d, heads, hd, ffn_mult):
    inner = heads * hd
    return d * 3 * inner + 3 * inner + inner * d + d + 2 * d * ffn_mult * d + ffn_mult * d + d + 4 * d

enc = lambda i: mlp_params([i, 256, 256, 64])
P = {}
for ffn in (2, 4):
    per_layer = tf_layer(224, 8, 32, ffn)
    backbone = 3 * per_layer
    actor_shared = enc(264) + enc(60)
    actor_per_robot = enc(168) + enc(102) + mlp_params([224, 32])
    fuse = mlp_params([128, 224])
    actor = actor_shared + 9 * actor_per_robot + fuse + backbone
    critic = mlp_params([399, 224]) + backbone + mlp_params([224, 1])
    P[f"ffn_x{ffn}"] = dict(transformer_layer=per_layer, backbone_3layers=backbone, actor_total=actor,
                           critic_total=critic, per_robot_extra=actor_per_robot)
mlp_bb = mlp_params([128, 256]) + mlp_params([256, 1024, 768, 384]) + mlp_params([384, 32])
P["mlp_backbone_actor_only_core"] = mlp_bb
R["param_estimates"] = P
R["kv_cache_flops_per_step_ffn4"] = 2 * 3 * (224 * 3 * 256 + 256 * 224 + 2 * 4 * 224 * 224) + 2 * 3 * 2 * 32 * 256

# ---------------- 배치 크기와 샘플링 부록 ----------------
envs = 1024 * 9
R["batch"] = dict(envs_total=envs, transitions_per_iter=envs * 24, per_minibatch=envs * 24 // 4,
                  iters_per_workingset_refresh=1e9 / (envs * 24),
                  warmup_transitions=500 * envs * 24,
                  env_seconds_per_refresh_per_env=1e9 / envs / 50)
R["data"] = dict(bones_seed_sequences=142220, hours=288, mean_len_s=288 * 3600 / 142220,
                 est_sequences_200h=142220 * 200 / 288, frames_200h_50hz=200 * 3600 * 50,
                 mean_bins_per_motion=200 * 3600 / (142220 * 200 / 288))

# ---------------- 검색(Recall@1) ----------------
ret = {"same_phase_source": 86.0, "same_motion_source": 94.2, "same_phase_robot": 61.3, "same_motion_robot": 76.1}
R["retrieval_min_n_one_decimal"] = {
    "all_four": feasible_n(list(ret.values()), 1),
    "source_pair": feasible_n([86.0, 94.2], 1), "robot_pair": feasible_n([61.3, 76.1], 1)}
R["retrieval_chance_candidates"] = dict(
    n_for_104=[n for n in range(2, 2000) if round(100 / n, 2) == 1.04],
    n_for_017=[n for n in range(2, 2000) if round(100 / n, 2) == 0.17],
    n_96x6=100 / 576, n_96=100 / 96)
R["retrieval_wilson_if_n"] = {str(n): {k: wilson(round(v / 100 * n), n) for k, v in ret.items()} for n in (96, 576, 1000)}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("wrote", OUT)
