"""GRAIL (arXiv:2606.05160 v2) 본문 수치 재계산과 일관성 검사. CPU만 사용, 수 초.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (표준 출력에 요약)
입력은 논문 표 값을 코드에 옮겨 적은 것이다(paper/paper.txt 대조). 분모(시도 수)가 논문에 없는 곳은
가정을 명시하고 민감도를 같이 적는다.
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---- V1: Table 3 (실로봇 pick-up, 객체당 10회) ----
seen = {"Cube": 10, "Apple": 6, "Tea Box": 10, "Carrot": 7, "Wet Wipes": 9}
unseen = {"Spray Can": 10, "Lint Roller": 5, "Peach": 9, "Flashlight": 8, "Medicine Bottle": 8}
n_per = 10
pct_seen = [100 * k / n_per for k in seen.values()]
pct_unseen = [100 * k / n_per for k in unseen.values()]
ks, ku = sum(seen.values()), sum(unseen.values())
R["V1_table3"] = {
    "mean_of_seen_per_object_pct": round(sum(pct_seen) / 5, 2),
    "mean_of_unseen_per_object_pct": round(sum(pct_unseen) / 5, 2),
    "reported_avg_seen": 84, "reported_avg_unseen": 80,
    "pooled_seen": f"{ks}/{5*n_per}", "pooled_unseen": f"{ku}/{5*n_per}",
    "pooled_seen_pct": round(100 * ks / 50, 1), "pooled_unseen_pct": round(100 * ku / 50, 1),
    "seen_CP95_pct": [round(100 * x, 1) for x in clopper_pearson(ks, 50)],
    "seen_Wilson95_pct": [round(100 * x, 1) for x in wilson(ks, 50)],
    "unseen_CP95_pct": [round(100 * x, 1) for x in clopper_pearson(ku, 50)],
    "unseen_Wilson95_pct": [round(100 * x, 1) for x in wilson(ku, 50)],
    "per_object_CP95_pct": {name: [round(100 * x, 1) for x in clopper_pearson(k, 10)]
                            for name, k in {**seen, **unseen}.items()},
    "seen_vs_unseen_fisher_p": round(stats.fisher_exact([[ks, 50 - ks], [ku, 50 - ku]])[1], 3),
}
# 계단: 본문은 90%만 적고 시도 수를 쓰지 않았다. 10회(9/10)를 가정한 민감도.
R["V1_stairs_sensitivity"] = {
    n: {"k": round(0.9 * n), "CP95_pct": [round(100 * x, 1) for x in clopper_pearson(round(0.9 * n), n)]}
    for n in (10, 20, 30)
}

# ---- V2: Table 2 (124 motion 벤치마크, 1회 평가 가정) ----
t2 = {"HDMI": 48.5, "ResMimic": 49.2, "Ours w/o SONIC": 45.0, "Ours w/o pi_phi": 39.7,
      "Ours w/o Rel. Obs.": 57.9, "Ours (Full)": 81.4}
N2 = 124
feasible = {}
for name, p in t2.items():
    feasible[name] = [k for k in range(N2 + 1) if round(100 * k / N2, 1) == p]
R["V2_denominator_check_N124"] = feasible
# 분모가 하나라면 여섯 행이 모두 같은 N에서 나와야 한다. N ≤ 500 전체에서 확인.
common = []
for N in range(1, 501):
    if all(any(round(100 * k / N, 1) == p for k in range(N + 1)) for p in t2.values()):
        common.append(N)
R["V2_common_denominator_N_le_500"] = common


def fisher_and_newcombe(k1, n1, k2, n2):
    p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {"diff_pct": round(100 * d, 1), "newcombe95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
            "fisher_p": float(f"{p:.2e}")}


k_full = round(0.814 * N2)  # 101/124 = 81.45 → 본문 81.4 (반올림 차이)
R["V2_pairwise_N124_assumed"] = {
    "Full_vs_HDMI": fisher_and_newcombe(101, N2, 60, N2),
    "Full_vs_ResMimic": fisher_and_newcombe(101, N2, 61, N2),
    "Full_vs_noPi": fisher_and_newcombe(101, N2, 49, N2),
}
R["V2_full_wilson95_pct"] = [round(100 * x, 1) for x in wilson(101, N2)]

# Table 8 ablation SR: 같은 124 분모 가정 시 행별 정수 개수
R["V8_SR_nearest_counts_N124"] = {str(p): round(p * N2 / 100, 2) for p in (41.6, 42.6, 53.3, 81.4)}

# Table 1 (20 objects): 분모 미기재. 1자리 반올림으로 맞는 N
t1 = {"HOIDiff": 15.8, "CHOIS": 10.5, "DAViD": 24.0, "Ours": 88.9}
R["V1_table1_common_N_le_400"] = [N for N in range(1, 401)
                                   if all(any(round(100 * k / N, 1) == p for k in range(N + 1))
                                          for p in t1.values())]

# ---- T1 순위: Table 1, 낮을수록 좋은 지표 여부 ----
t1_rows = {
    "Contact": ({"HOIDiff": .012, "CHOIS": .034, "DAViD": .246, "Ours": .008}, "low"),
    "Pen": ({"HOIDiff": 2.07, "CHOIS": 3.74, "DAViD": 1.46, "Ours": .90}, "low"),
    "InterScore": ({"HOIDiff": 1.79, "CHOIS": 2.47, "DAViD": 2.74, "Ours": 3.58}, "high"),
    "HumanSmo": ({"HOIDiff": .0043, "CHOIS": .0055, "DAViD": .0024, "Ours": .0033}, "low"),
    "ObjSmo": ({"HOIDiff": .0118, "CHOIS": .0062, "DAViD": .0605, "Ours": .0022}, "low"),
    "SR": ({"HOIDiff": 15.8, "CHOIS": 10.5, "DAViD": 24.0, "Ours": 88.9}, "high"),
    "BodyDev": ({"HOIDiff": .2120, "CHOIS": .2564, "DAViD": .4723, "Ours": .0913}, "low"),
    "ObjDev": ({"HOIDiff": .3352, "CHOIS": .3642, "DAViD": .5826, "Ours": .0851}, "low"),
}
R["T1_ours_rank"] = {}
for m, (vals, d) in t1_rows.items():
    srt = sorted(vals, key=lambda x: vals[x], reverse=(d == "high"))
    R["T1_ours_rank"][m] = srt.index("Ours") + 1
R["T1_best_on_count"] = sum(1 for v in R["T1_ours_rank"].values() if v == 1)

# ---- 표 7 사용자 연구 ----
aff = [2.0, 12.2, 11.2, 74.7]
phy = [1.9, 16.8, 10.4, 70.9]
R["T7_sums"] = {"aff": round(sum(aff), 1), "phys": round(sum(phy), 1), "ceiling_pct": 75.0,
                "ours_share_of_ceiling_aff": round(74.7 / 75 * 100, 1),
                "ours_share_of_ceiling_phys": round(70.9 / 75 * 100, 1)}

# ---- 표 4 시간 분해 ----
stages = {"video": 1, "human": 2, "object": 1, "prep": 2, "joint": 8}
R["T4_total_min"] = sum(stages.values())
R["T4_video_plus_init_min"] = stages["video"] + stages["human"] + stages["object"]

# ---- B.3 학습 비용 ----
hours, gpus = 30, 64
R["B3_gpu_hours_full"] = hours * gpus
R["B3_wallclock_min_per_motion"] = [round(hours * 60 / n, 2) for n in (4000, 2000)]
R["B3_gpu_min_per_motion"] = [round(hours * gpus * 60 / n, 1) for n in (4000, 2000)]
R["B3_warmstart_hours_check"] = round(hours / 5, 1)
R["B3_env_steps_total"] = 1024 * 64 * 24 * 30000  # 24 steps/env는 B.1 설정 기준(근사)

# ---- 데이터셋 규모: HF 데이터 카드 표 ----
cards = {"pickup_table": (2991, 10), "pickup_ground": (1736, 15), "sitting": (1748, 5),
         "slope": (1880, 10), "curb": (1769, 10), "stair_p1-p4": (24700, None),
         "manip_ground": (600, 5), "manip_table": (673, 5)}
R["DS_total_motions"] = sum(v[0] for v in cards.values())
R["DS_frame_check_pickup_table"] = 2991 * 10 * 25 == 747750
R["DS_frame_check_manip_ground_126"] = 600 * 126 == 75600
R["DS_frame_check_manip_table_126"] = 673 * 126 == 84798

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1))
