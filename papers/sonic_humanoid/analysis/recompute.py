"""논문 본문과 표·그림에 적힌 수치를 다시 계산하고 검정한다 (CPU, 수 초).

입력: 논문 본문, 표 1·2·3, 그림 2 (arXiv:2511.07820 v4). 모든 입력 값은 아래 상수에 논문에서 옮겨 적었다.
출력: analysis/recompute_results.json 과 표준출력 요약.

검정 방법
- 비율: Wilson 95% 구간, 두 비율 차이는 Newcombe(Wilson 기반) 95% 구간, 양측 두 비율 z 검정(합동 분산),
  작은 표본은 Fisher 정확 검정.
- 쌍체 실제-시뮬: 논문이 쓴 쌍체 t 검정은 0/1 결과에 적용되며, 불일치 쌍 수는 공개되지 않는다.
  가능한 최댓값 범위만 계산한다.
- 논문은 n 개수가 적힌 셀만 검정 대상으로 삼는다. 가정이 필요한 셀은 결과에 "가정"이라고 표시한다.
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
Z = 1.959963984540054


# ---------- 기본 도구 ----------

def wilson(k, n, z=Z):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def newcombe_diff(k1, n1, k2, n2):
    """두 비율 차이 p1 - p2 의 Newcombe(Wilson) 95% 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return z, 2 * stats.norm.sf(abs(z))


def welch_binary(k1, n1, k2, n2):
    """0/1 결과에 Welch t 검정을 적용한 값 (논문 방법과 같은 검정)."""
    x1 = [1] * k1 + [0] * (n1 - k1)
    x2 = [1] * k2 + [0] * (n2 - k2)
    t, p = stats.ttest_ind(x1, x2, equal_var=False)
    return float(t), float(p)


def pct_to_count(pct, n):
    return int(round(pct / 100.0 * n))


R = {}

# ---------- V1. 실제 로봇 (그림 2 K–L, 본문 2.1) ----------
real_k, real_n = 123, 124
sim_k, sim_n = 124, 124
lo, hi = wilson(real_k, real_n)
sim_lo, sim_hi = wilson(sim_k, sim_n)
# 불일치 쌍 수는 공개되지 않는다. 시뮬이 전부 성공이므로 불일치 쌍은 실제 실패 1개뿐이다.
# 쌍체 검정의 가능한 최소·최대 p는 McNemar 정확 검정으로 본다 (b = 1, c = 0).
mcnemar_p_if_one_discordant = stats.binomtest(1, 1, 0.5).pvalue  # b=1, c=0 의 양측 정확 p
R["V1_real_world"] = {
    "real_success": [real_k, real_n],
    "real_rate": real_k / real_n,
    "real_wilson95": [lo, hi],
    "sim_success": [sim_k, sim_n],
    "sim_wilson95": [sim_lo, sim_hi],
    "note": "시뮬 전부 성공이므로 쌍체 불일치는 최대 1쌍입니다. 1쌍이면 McNemar 정확 p = 1.0 입니다(정보 부족).",
    "mcnemar_p_one_discordant": mcnemar_p_if_one_discordant,
}

# ---------- V6. 기준선 비교 (그림 2 D, 본문 2.1의 수치) ----------
N = {"test_content": 7016, "test_repetition": 9395, "phuma": 68326}
rates = {
    "SONIC": {"test_content": 98.5, "test_repetition": 99.2, "phuma": 97.2},
    "BeyondMimic": {"test_content": 82.0, "test_repetition": 85.4, "phuma": 73.8},
    "Any2Track": {"test_content": 61.6, "test_repetition": 69.4, "phuma": 78.5},
}
K1 = {}
for split, n in N.items():
    k_s = pct_to_count(rates["SONIC"][split], n)
    for base in ("BeyondMimic", "Any2Track"):
        k_b = pct_to_count(rates[base][split], n)
        z, p = two_prop_z(k_s, n, k_b, n)
        d, dlo, dhi = newcombe_diff(k_s, n, k_b, n)
        K1[f"{split}__{base}"] = {
            "n": n, "sonic_k": k_s, "base_k": k_b,
            "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * dlo, 1), round(100 * dhi, 1)],
            "z": round(z, 2), "p": p,
        }
R["V6_baselines_two_prop"] = K1
R["V6_note"] = ("백분율은 소수 첫째 자리로 반올림된 값이라 개수는 최대 ±0.5%p×N 만큼 어긋날 수 있습니다. "
                "p 값은 모두 0.001보다 훨씬 작아 논문의 P < 0.001과 맞습니다.")

# 본문 "42% 감소": (40.9 - 23.7) / 40.9
R["V6_mpjpe_reduction_pct"] = round(100 * (40.9 - 23.7) / 40.9, 2)

# ---------- V7. 속도 추종 생존율 (그림 2 I, 본문 2.1) ----------
s_k, s_n = 197, 200
o_k, o_n = 86, 200
fp = stats.fisher_exact([[s_k, s_n - s_k], [o_k, o_n - o_k]])[1]
z, zp = two_prop_z(s_k, s_n, o_k, o_n)
t, tp = welch_binary(s_k, s_n, o_k, o_n)
d, dlo, dhi = newcombe_diff(s_k, s_n, o_k, o_n)
R["V7_survival_openhomie"] = {
    "sonic": [s_k, s_n], "openhomie": [o_k, o_n],
    "fisher_p": fp, "z": round(z, 2), "z_p": zp,
    "welch_t_on_binary": round(t, 2), "welch_p": tp,
    "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * dlo, 1), round(100 * dhi, 1)],
    "note": "분모 200 회가 어떤 조건인지 본문에 없습니다. 부록 S2.5의 OpenHomie 평가는 n = 80(20 속도 × 4 방향)입니다.",
}

# ---------- V2. VLA 과제 (표 1 A) ----------
vla = [
    ("Apple to plate", 18, 20),
    ("Object pickup (carrot)", 15, 20),
    ("Object pickup (scrub)", 19, 20),
    ("Open trash can (foot)", 7, 10),
    ("Soda can to trash can", 6, 10),
    ("Drill and box relocation", 7, 10),
]
V2 = {"tasks": []}
for name, k, n in vla:
    lo, hi = wilson(k, n)
    V2["tasks"].append({"task": name, "k": k, "n": n, "rate": round(k / n, 3),
                        "wilson95": [round(lo, 3), round(hi, 3)]})
# 논문의 "5-task 평균 75%"는 과제 비율의 단순 평균이다 (물체 집기 두 행은 75%와 95%의 평균 85%).
task_rates = [90, (75 + 95) / 2, 70, 60, 70]
V2["unweighted_mean_5tasks_pct"] = round(sum(task_rates) / len(task_rates), 1)
pooled_k = sum(k for _, k, _ in vla)
pooled_n = sum(n for _, _, n in vla)
plo, phi = wilson(pooled_k, pooled_n)
V2["pooled_all_6_rows"] = {"k": pooled_k, "n": pooled_n, "rate": round(pooled_k / pooled_n, 3),
                           "wilson95": [round(plo, 3), round(phi, 3)]}
R["V2_vla_table1A"] = V2

# ---------- V3. FSQ 대 SMPL (표 1 B) ----------
# 표 1 B에는 시도 수가 없다. 가정: 표 1 A의 시도 수(당근 20, 트래시 10, 소다 10)와 같다고 둔다.
fsq = {"carrot": (15, 20), "trash": (7, 10), "soda": (6, 10)}       # 75%, 70%, 60%
smpl = {"carrot": (12, 20), "trash": (2, 10), "soda": (0, 10)}      # 60%, 20%, 0%
V3 = {"assumption": "SMPL 시도 수 = 표 1 A 시도 수 (논문에 없음)", "tasks": {}}
for task in fsq:
    fk, fn = fsq[task]
    sk, sn = smpl[task]
    fisher_p = stats.fisher_exact([[fk, fn - fk], [sk, sn - sk]])[1]
    d, dlo, dhi = newcombe_diff(fk, fn, sk, sn)
    V3["tasks"][task] = {"fsq": [fk, fn], "smpl": [sk, sn], "fisher_p": round(fisher_p, 4),
                         "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * dlo, 1), round(100 * dhi, 1)]}
fsq_pool = (sum(v[0] for v in fsq.values()), sum(v[1] for v in fsq.values()))
smpl_pool = (sum(v[0] for v in smpl.values()), sum(v[1] for v in smpl.values()))
V3["pooled_fsq"] = [fsq_pool[0], fsq_pool[1], round(fsq_pool[0] / fsq_pool[1], 3)]
V3["pooled_smpl"] = [smpl_pool[0], smpl_pool[1], round(smpl_pool[0] / smpl_pool[1], 3)]
V3["paper_avg_fsq_pct"] = round((75 + 70 + 60) / 3, 1)
V3["paper_avg_smpl_pct"] = round((60 + 20 + 0) / 3, 1)
V3["paper_avg_delta_pp"] = round(((75 - 60) + (70 - 20) + (60 - 0)) / 3, 1)
V3["pooled_diff_pp"] = round(100 * (fsq_pool[0] / fsq_pool[1] - smpl_pool[0] / smpl_pool[1]), 1)
R["V3_fsq_vs_smpl_table1B"] = V3

# ---------- V4. 표 3 (A) FSQ 대 VQ-VAE ----------
V4 = {}
for split, n in (("test_content", 7016), ("test_repetition", 9395)):
    key = {"test_content": (99.3, 98.7), "test_repetition": (99.6, 99.3)}[split]
    k1 = pct_to_count(key[0], n)
    k2 = pct_to_count(key[1], n)
    z, p = two_prop_z(k1, n, k2, n)
    V4[split] = {"fsq": key[0], "vq": key[1], "n": n, "z": round(z, 2), "p": p,
                 "note": "단일 평가, 시드 변동 없음"}
V4["mpjpe_gap_test_content_mm"] = round(35.3 - 26.6, 1)
# 표 3 내부 일관성: '기본값' FSQ-32-32 가 세 번 다른 값으로 나온다
V4["default_config_reported"] = {
    "table3A_fsq_128gpu": {"sr": 99.3, "mpjpe": 26.6},
    "table3B_fsq_32_32_32gpu": {"sr": 98.8, "mpjpe": 27.5},
    "table3C_robot_encoder_128gpu": {"sr": 99.6, "mpjpe": 23.8},
    "fig2_largest_model_128gpu": {"sr": 99.6, "mpjpe": 23.8},
}
R["V4_ablation_table3"] = V4

# 표 3 (B): 차원과 레벨 효과 (단일 평가). 같은 레벨 기준으로 차원을 16→32로 올린 효과와
# 같은 차원 기준으로 레벨을 16→32로 올린 효과를 비교한다.
V5 = {
    "dims_effect_at_16_levels_mm": round(29.7 - 35.7, 1),   # FSQ-16-16 -> FSQ-16-32
    "levels_effect_at_16_dims_mm": round(30.3 - 35.7, 1),   # FSQ-16-16 -> FSQ-32-16
    "dims_effect_at_32_levels_mm": round(27.5 - 30.3, 1),   # FSQ-32-16 -> FSQ-32-32
    "levels_effect_at_32_dims_mm": round(27.5 - 29.7, 1),   # FSQ-16-32 -> FSQ-32-32
}
R["V5_quantizer_capacity_table3B"] = V5

# ---------- K1. 표 2 행 합계 점검 ----------
rows = {
    "train": {"Locomotion": 53255, "Gestures": 37939, "Acting": 68742, "Combat": 50162, "Props": 14513,
              "Dance": 9689, "Injured": 9386, "Action": 9920, "Others": 63583},
    "test_content": {"Locomotion": 2481, "Gestures": 1488, "Acting": 0, "Combat": 0, "Props": 701,
                     "Dance": 504, "Injured": 1167, "Action": 228, "Others": 429},
    "test_repetition": {"Locomotion": 2683, "Gestures": 1125, "Acting": 20, "Combat": 0, "Props": 253,
                        "Dance": 485, "Injured": 528, "Action": 322, "Others": 890},
}
clip_totals = {"train": 317189, "test_content": 7016, "test_repetition": 9395}
K3 = {}
for split, r in rows.items():
    s = sum(r.values())
    K3[split] = {"sum_of_main_categories": s, "reported_clips": clip_totals[split],
                 "unassigned": clip_totals[split] - s}
R["K1_table2_category_sums"] = K3

# ---------- K2. 데이터 규모: 프레임 수와 클립 수 ----------
hours_train = 611
frames_per_s = 50
frames_total = hours_train * 3600 * frames_per_s
mean_clip_s = hours_train * 3600 / 317189
K4 = {
    "train_frames_from_611h_50hz_M": round(frames_total / 1e6, 1),
    "mean_clip_seconds_full": round(mean_clip_s, 2),
    "subsets": [],
}
for frames_m, clips in ((4, 20000), (10, 50000), (22, 110000), (100, 317189)):
    implied_mean_s = frames_m * 1e6 / frames_per_s / clips
    K4["subsets"].append({"frames_M": frames_m, "clips_reported": clips,
                          "mean_clip_s_implied": round(implied_mean_s, 2)})
K4["note"] = ("전체 데이터의 평균 클립 길이는 약 6.9 s입니다. 부분집합은 같은 프레임 수에 클립이 더 많아 "
              "평균 길이가 약 4 s로 짧아집니다. 부분집합 추출 방식(하위 범주 균등 추출)이 이를 설명할 수 있지만 "
              "본문에 설명이 없습니다.")
R["K2_data_scale_frames_clips"] = K4

# ---------- V8. 계산량 ----------
R["V8_compute_check"] = {
    "128_gpus_7_days_gpu_h": 128 * 24 * 7,
    "reported_gpu_h": 21000,
    "note": "128 GPU × 7일 = 21,504 GPU 시간으로 본문 21k와 맞습니다. 2k, 9k 값의 벽시계 시간은 본문에 없습니다.",
}

# ---------- K3. 초록 대 본문 데이터 양 ----------
R["K3_abstract_hours"] = {"source_hours": 700, "training_hours_after_filter": 611,
                         "note": "초록은 '100M+ frames from 700 hours'라고 쓰지만 학습에 쓴 것은 611시간입니다."}

# ---------- K9. 사용한 통계 방식의 독립성 ----------
R["K9_checkpoint_independence"] = {
    "note": ("그림 2 A–C 의 n = 6은 '평가 체크포인트' 6개입니다. 같은 학습 실행의 서로 다른 시점이므로 독립 반복이 아닙니다. "
             "Welch 검정의 독립 가정이 맞지 않으며, 시드 변동이 빠져 있습니다. P ≤ 0.011 값은 Data S1 없이는 재현할 수 없습니다.")
}

if __name__ == "__main__":
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("wrote", OUT)
    print("V1 real rate", round(R["V1_real_world"]["real_rate"], 4), "wilson", [round(x, 3) for x in R["V1_real_world"]["real_wilson95"]])
    for k, v in K1.items():
        print("V6", k, v["diff_pp"], v["newcombe95_pp"], "p=%.2e" % v["p"])
    print("V7 survival p(fisher)=%.2e welch p=%.2e z p=%.2e" % (fp, tp, zp))
    print("V2 pooled", R["V2_vla_table1A"]["pooled_all_6_rows"], "unweighted", R["V2_vla_table1A"]["unweighted_mean_5tasks_pct"])
    print("V3 pooled fsq", V3["pooled_fsq"], "smpl", V3["pooled_smpl"], "paper avg delta", V3["paper_avg_delta_pp"], "pooled diff", V3["pooled_diff_pp"])
    for task, v in V3["tasks"].items():
        print("V3", task, "fisher p=%.4f" % v["fisher_p"], "diff", v["diff_pp"])
    print("V4", {k: (v["z"], "%.3g" % v["p"]) for k, v in V4.items() if isinstance(v, dict) and "p" in v})
    print("V5", V5)
    print("K1", K3)
    print("K2", K4["train_frames_from_611h_50hz_M"], K4["mean_clip_seconds_full"], [s["mean_clip_s_implied"] for s in K4["subsets"]])
