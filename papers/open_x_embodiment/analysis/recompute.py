"""Open X-Embodiment 논문의 보고 수치를 다시 계산한다 (CPU, 표준 라이브러리 + scipy).

입력: 논문 보고값(아래 상수, 본문·그림 라벨 기준), 데이터셋 메타데이터 스프레드시트 스냅샷
      analysis/meta/oxe_dataset_sheet_2026-11-22.csv (공개 Google Sheet를 CSV로 내려받은 것)
출력: analysis/recompute_results.json

학습·추론은 하지 않습니다. 셀별 시행 수 n은 논문에 없으므로, 보고된 백분율을 설명하는 n 후보를 찾아 가정으로만 씁니다.
실행: python3 analysis/recompute.py
"""
import csv
import itertools
import json
import math
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
SHEET = HERE / "meta" / "oxe_dataset_sheet_2026-11-22.csv"
OUT = HERE / "recompute_results.json"
RES = {}

# ---------------------------------------------------------------- 보고값 (논문 v9 PDF, 그림·표 라벨)
FIG4 = {  # 그림 4 막대 라벨 (5개 도메인 + 평균)
    "domains": ["Kitchen Manipulation", "Cable Routing", "NYU Door Opening", "Autolab UR5", "Task-Agnostic Play"],
    "Original": [43, 24, 53, 53, 33],
    "RT-1": [48, 18, 65, 25, 68],
    "RT-1-X": [63, 56, 80, 45, 72],
    "printed_mean": {"Original": 41, "RT-1": 44, "RT-1-X": 63},
}
TABLE1 = {  # 표 I (성공률 %, 열: IRIS / RAIL / RT-1 paper 6 skills)
    "cols": ["Bridge IRIS (WidowX)", "Bridge RAIL (WidowX)", "RT-1 paper 6 skills (Google Robot)"],
    "Original": [13, 13, None],
    "RT-1": [40, 30, 92],
    "RT-1-X": [27, 27, 73],
    "RT-2-X (55B)": [50, 30, 91],
}
TABLE2 = {  # 표 II: (행, 모델, 크기, 이력, 데이터, 공동학습, 초기, 창발 %, 일반화 %)
    "rows": {
        "1": ("RT-2 55B", "Google Robot action", 27.3, 62),
        "2": ("RT-2-X 55B", "Robotics data", 75.8, 61),
        "3": ("RT-2-X 55B", "Robotics data except Bridge", 42.8, 54),
        "4": ("RT-2-X 5B, history 2", "Robotics data", 44.4, 52),
        "5": ("RT-2-X 5B, history none", "Robotics data", 14.5, 30),
        "6": ("RT-2-X 5B, history 2, from scratch", "Robotics data (no web co-train)", 0.0, 1),
        "7": ("RT-2-X 5B, history 2, fine-tune", "Robotics data (no web co-train)", 48.7, 47),
    },
}
PRINT_DECIMALS = {"emergent": 1, "general": 0}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def candidates(pct, decimals, nmax=100, rule="round"):
    """보고된 백분율(pct, 소수 자릿수 decimals)을 설명하는 (k, n) 후보.
    rule='round': 반올림, rule='floor': 버림(내림). 두 규칙을 따로 본다."""
    out = []
    for n in range(1, nmax + 1):
        for k in range(0, n + 1):
            x = 100.0 * k / n
            if rule == "round":
                ok = round(x, decimals) == round(pct, decimals)
            else:
                ok = math.floor(x * 10 ** decimals + 1e-9) / 10 ** decimals == round(pct, decimals)
            if ok:
                out.append((k, n))
    return out


def fisher_p(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table)[1])


# ---------------------------------------------------------------- V1: 그림 4 평균과 "50% 높다"
v1 = {}
for m in ["Original", "RT-1", "RT-1-X"]:
    vals = FIG4[m]
    v1[m] = {"mean_of_5_domains": round(sum(vals) / 5, 2), "printed_mean": FIG4["printed_mean"][m]}
rel_orig = 63 / 41 - 1
rel_rt1 = 63 / 44 - 1
v1["claim_50pct"] = {
    "RT-1-X over Original (printed means 63 vs 41)": round(100 * rel_orig, 1),
    "RT-1-X over RT-1 (printed means 63 vs 44)": round(100 * rel_rt1, 1),
    "RT-1-X over Original (unrounded mean of 5 = 63.2 vs 41.2)": round(100 * (316 / 5) / (206 / 5) - 100, 1),
    "RT-1-X over RT-1 (unrounded mean of 5 = 63.2 vs 44.8)": round(100 * (316 / 5) / (224 / 5) - 100, 1),
}
wins = [a > b for a, b in zip(FIG4["RT-1-X"], FIG4["Original"])]
v1["wins_vs_original_per_domain"] = {"count": sum(wins), "of": 5,
                                     "two_sided_sign_test_p": float(stats.binomtest(sum(wins), 5, 0.5).pvalue)}
v1["per_domain_ratio_RT1X_over_Original"] = [round(a / b, 2) for a, b in zip(FIG4["RT-1-X"], FIG4["Original"])]
v1["per_domain_ratio_RT1X_over_RT1"] = [round(a / b, 2) for a, b in zip(FIG4["RT-1-X"], FIG4["RT-1"])]
v1["note"] = ("그림 4 캡션 '50% higher'는 평균 기준으로 Original 대비 +54%, RT-1 대비 +43%입니다. "
              "Autolab UR5에서는 RT-1-X(45)가 Original(53)보다 낮습니다. 막대 라벨은 정수이고 n은 없습니다.")
RES["V1_fig4_means_and_claim"] = v1

# ---------------------------------------------------------------- V2: 표 I 캡션·본문 대조
v2 = {}
v2["caption_claim"] = "캡션: 'RT-1-X underfits and performs worse than the Original Method and RT-1'"
v2["RT-1-X_vs_Original"] = {
    TABLE1["cols"][0]: TABLE1["RT-1-X"][0] - TABLE1["Original"][0],
    TABLE1["cols"][1]: TABLE1["RT-1-X"][1] - TABLE1["Original"][1],
}
v2["RT-1-X_vs_RT-1"] = {
    TABLE1["cols"][0]: TABLE1["RT-1-X"][0] - TABLE1["RT-1"][0],
    TABLE1["cols"][1]: TABLE1["RT-1-X"][1] - TABLE1["RT-1"][1],
    TABLE1["cols"][2]: TABLE1["RT-1-X"][2] - TABLE1["RT-1"][2],
}
v2["RT-2-X_vs_RT-1"] = {
    TABLE1["cols"][0]: TABLE1["RT-2-X (55B)"][0] - TABLE1["RT-1"][0],
    TABLE1["cols"][1]: TABLE1["RT-2-X (55B)"][1] - TABLE1["RT-1"][1],
    TABLE1["cols"][2]: TABLE1["RT-2-X (55B)"][2] - TABLE1["RT-1"][2],
}
v2["reading"] = ("표 I에서 RT-1-X는 두 Bridge 열에서 Original(13%)보다 높습니다(+14, +14). "
                 "캡션의 'worse than the Original Method'는 이 두 열과 맞지 않습니다. "
                 "본문 'RT-2-X outperforms both the Original Method and RT-1'는 IRIS 열에서만 성립합니다. "
                 "RAIL 열은 30 대 30으로 같고, RT-1 논문 열은 91 대 92로 낮습니다. "
                 "Original Method의 13%가 두 Bridge 열에 같은 값으로 들어 있어 복사 오류일 수 있습니다.")
RES["V2_table1_caption_text_check"] = v2

# ---------------------------------------------------------------- V3: 표 II 배수와 n 후보
v3 = {}
v3["ratio_emergent_row2_over_row1"] = round(75.8 / 27.3, 3)
v3["ratio_emergent_row2_over_row1_note"] = "본문 '~3x' 또는 project page '3x'는 2.78배를 올림한 값입니다."
v3["diff_emergent_row2_minus_row3"] = round(75.8 - 42.8, 1)
v3["diff_emergent_row4_minus_row5"] = round(44.4 - 14.5, 1)
v3["diff_emergent_row4_minus_row7_codft_vs_ft"] = round(44.4 - 48.7, 1)
v3["diff_general_row4_minus_row5"] = 52 - 30
cand = {}
for r, (model, data, em, ge) in TABLE2["rows"].items():
    cand[r] = {"emergent_pct": em,
               "emergent_n_round_le200": [n for k, n in candidates(em, 1, nmax=200)] if em > 0 else "any (k=0)",
               "emergent_n_floor_le200": [n for k, n in candidates(em, 1, nmax=200, rule="floor")] if em > 0 else "any (k=0)",
               "general_pct": ge}
v3["note_42_8"] = ("42.8%는 반올림 기준으로 n <= 137인 k/n이 없습니다(가장 작은 n = 138). "
                   "버림 기준이면 3/7 = 42.857이 42.8로 적힙니다. 다른 셀은 반올림과 맞습니다. "
                   "따라서 이 셀은 다른 n이거나 다른 집계(예: 과제 평균)일 가능성이 있습니다.")
v3["candidate_n"] = cand
# 모든 창발 행을 동시에 설명하는 공통 n이 있는지
emergent_rows = ["1", "2", "3", "4", "5", "7"]
common = []
for n in range(2, 201):
    ok = True
    for r in emergent_rows:
        em = TABLE2["rows"][r][2]
        if not any(nn == n for _, nn in candidates(em, 1, nmax=200)):
            ok = False
            break
    if ok:
        common.append(n)
v3["common_n_for_all_emergent_rows_le200"] = common
RES["V3_table2_ratios_and_n"] = v3

# ---------------------------------------------------------------- V4: 표 I 열별 n 추정과 검정 (가정 n)
v4 = {}
bridge_vals = {"Original": [13, 13], "RT-1": [40, 30], "RT-1-X": [27, 27], "RT-2-X (55B)": [50, 30]}
common_bridge = None
for n in range(2, 201):
    ok = True
    for m, vs in bridge_vals.items():
        for v in vs:
            if not any(nn == n for _, nn in candidates(v, 0, nmax=200)):
                ok = False
    if ok:
        common_bridge = (common_bridge or []) + [n]
v4["bridge_common_n_candidates_le200"] = common_bridge
N = 30  # 가정: Bridge 열 셀당 30회 (모든 Bridge 값이 k/30로 설명됨)
pairs = [("RT-1-X vs Original (IRIS)", (8, N), (4, N)),
         ("RT-1-X vs RT-1 (IRIS)", (8, N), (12, N)),
         ("RT-2-X vs RT-1 (IRIS)", (15, N), (12, N)),
         ("RT-2-X vs Original (IRIS)", (15, N), (4, N)),
         ("RT-1-X vs Original (RAIL)", (8, N), (4, N)),
         ("RT-2-X vs RT-1 (RAIL)", (9, N), (9, N))]
v4_rows = []
for name, (k1, n1), (k2, n2) in pairs:
    lo1, hi1 = wilson(k1, n1)
    lo2, hi2 = wilson(k2, n2)
    v4_rows.append({"comparison": name, "a": f"{k1}/{n1}", "b": f"{k2}/{n2}",
                    "a_wilson95": [round(lo1, 1), round(hi1, 1)], "b_wilson95": [round(lo2, 1), round(hi2, 1)],
                    "diff_pp": round(100 * (k1 / n1 - k2 / n2), 1),
                    "fisher_p_two_sided": round(fisher_p(k1, n1, k2, n2), 3)})
v4["assumed_n30_pairs"] = v4_rows
RES["V4_table1_assumed_n_tests"] = v4

# ---------------------------------------------------------------- V5: 표 II 행 쌍의 검정 (n 후보 전부)
def pair_range(r1, r2, nmax=100):
    e1, e2 = TABLE2["rows"][r1][2], TABLE2["rows"][r2][2]
    c1 = [(k, n) for k, n in candidates(e1, 1, nmax)]
    c2 = [(k, n) for k, n in candidates(e2, 1, nmax=200)]
    ps = []
    for (k1, n1), (k2, n2) in itertools.product(c1, c2):
        ps.append(fisher_p(k1, n1, k2, n2))
    if not ps:
        return None
    return {"p_min": round(min(ps), 4), "p_max": round(max(ps), 4), "pairs_tested": len(ps)}


v5 = {
    "row2_vs_row1_emergent (RT-2-X 55B vs RT-2 55B)": pair_range("2", "1"),
    "row2_vs_row3_emergent (Bridge removal; row3 n>=138 under rounding)": pair_range("2", "3"),
    "row4_vs_row5_emergent (history 2 vs none, 5B)": pair_range("4", "5"),
    "row4_vs_row7_emergent (co-fine-tune vs fine-tune, 5B)": pair_range("4", "7"),
    "row2_vs_row4_emergent (55B vs 5B; history confounded)": pair_range("2", "4"),
}
v5_note = ("n이 논문에 없으므로 각 셀의 k/n 후보를 모두 넣고 p의 범위를 봅니다. "
           "범위에 0.05가 들어가면 그 비교는 시행 수 가정 없이 결론을 내릴 수 없습니다.")
RES["V5_table2_pairwise_fisher_over_candidate_n"] = {"note": v5_note, "results": v5}

# ---------------------------------------------------------------- V6: 데이터셋 스프레드시트 재집계
rows = list(csv.reader(open(SHEET, encoding="utf-8")))
data = [r for r in rows[15:] if any(c.strip() for c in r)]
data = [r + [""] * (22 - len(r)) for r in data]


def num(s):
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


recs = []
ver = "v1.0"
for r in data:
    if r[0].strip():
        ver = r[0].strip()
    recs.append({"ver": ver, "name": r[1].strip(), "robot": r[2].strip(),
                 "ep": num(r[3]) or 0.0, "gb": num(r[4])})
v10 = [x for x in recs if x["ver"] == "v1.0"]
tot_all = sum(x["ep"] for x in recs)
tot10 = sum(x["ep"] for x in v10)
robots10 = {x["robot"] for x in v10 if x["robot"] != "Multi-Robot"}
MIX = ["RT-1 Robot Action", "QT-Opt", "Berkeley Bridge", "Freiburg Franka Play", "USC Jaco Play",
       "Berkeley Cable Routing", "Roboturk", "NYU VINN", "Austin VIOLA", "Berkeley Autolab UR5",
       "TOTO Benchmark", "Language Table"]  # 논문 IV-C의 12개 출처 (TACO play = Freiburg Franka Play)
mix = [x for x in v10 if x["name"] in MIX]
mix_tot = sum(x["ep"] for x in mix)
fam = lambda r: "xArm" if r.startswith("xArm") else r
by_fam = {}
for x in v10:
    f = fam(x["robot"])
    by_fam.setdefault(f, [0, 0.0])
    by_fam[f][0] += 1
    by_fam[f][1] += x["ep"]
ranked = sorted(by_fam.items(), key=lambda kv: -kv[1][1])
v6 = {
    "sheet_total_rows_datasets": len(recs),
    "sheet_v1_0_datasets": len(v10),
    "sheet_v1_1_added_datasets": len(recs) - len(v10),
    "sheet_total_episodes_all": int(tot_all),
    "sheet_total_episodes_v1_0": int(tot10),
    "paper_claim_v1_0": "60 datasets, 1M+ trajectories (paper III-A)",
    "distinct_embodiment_strings_v1_0_excl_multirobot": len(robots10),
    "paper_claim_embodiments": 22,
    "franka_datasets_v1_0": sum(1 for x in v10 if x["robot"] == "Franka"),
    "top_embodiments_by_episodes_v1_0": [{"robot": k, "datasets": v[0], "episodes": int(v[1]),
                                          "share_pct": round(100 * v[1] / tot10, 1)} for k, v in ranked[:6]],
    "top5_datasets_share_pct": round(100 * sum(sorted([x["ep"] for x in v10], reverse=True)[:5]) / tot10, 1),
    "mixture_sources_matched": len(mix),
    "mixture_embodiments": len({x["robot"] for x in mix}),
    "mixture_episodes": int(mix_tot),
    "mixture_share_of_v1_0_pct": round(100 * mix_tot / tot10, 1),
    "mixture_QTOpt_plus_LanguageTable_share_of_mixture_pct": round(
        100 * sum(x["ep"] for x in mix if x["name"] in ("QT-Opt", "Language Table")) / mix_tot, 1),
    "mixture_bridge_share_of_mixture_pct": round(
        100 * sum(x["ep"] for x in mix if x["name"] == "Berkeley Bridge") / mix_tot, 2),
    "mixture_robots": sorted({x["robot"] for x in mix}),
    "sheet_header_claim_total_gb": 8964.94,
    "sheet_sum_gb_all": round(sum(x["gb"] or 0 for x in recs), 2),
    "gb_missing_rows_v1_0": [x["name"] for x in v10 if x["gb"] is None],
    "colab_DATASETS_list_count": 53,
}
# 그림 2(c) 대 스프레드시트: 논문 그림에서 xArm이 가장 큰 조각으로 보이지만 스프레드시트 기준 1위는 Kuka iiwa
v6["fig2c_check"] = ("논문 그림 2(c) 파이 조각은 눈으로 읽었을 때 xArm이 가장 큽니다. "
                     "스프레드시트 v1.0 기준 1위는 Kuka iiwa(QT-Opt 580,392, 41.6%)이고 xArm은 31.7%입니다. "
                     "스냅샷 시점 차이이거나 집계 기준 차이일 수 있으며, 논문은 집계 방법을 밝히지 않습니다.")
RES["V6_dataset_sheet_recount"] = v6

# ---------------------------------------------------------------- V7: 혼합 샘플링 산술 (학습 없음)
w_ep = {x["name"]: x["ep"] for x in mix}
w_gb = {x["name"]: (x["gb"] or 0.0) for x in mix}
uni = {x["name"]: 1.0 for x in mix}


def shares(w):
    s = sum(w.values())
    return {k: round(100 * v / s, 2) for k, v in sorted(w.items(), key=lambda kv: -kv[1])}


sh_ep = shares(w_ep)
sh_uni = shares(uni)
RES["V7_mixture_weighting_arithmetic"] = {
    "note": ("논문은 혼합 가중치와 샘플링 방식을 밝히지 않습니다(본문·표·부록 모두 없음). "
             "아래는 세 가지 가정 가중치에서 데이터셋별 점유율의 산술입니다. 학습 없음."),
    "episode_proportional_share_pct": sh_ep,
    "uniform_over_datasets_share_pct": {k: round(100 * 1 / len(uni), 2) for k in uni},
    "bridge_share_episode_pct": sh_ep.get("Berkeley Bridge"),
    "bridge_share_uniform_pct": round(100 / len(uni), 2),
}

# 공식 저장소 콜랩 Minimal_Training_Example.ipynb의 DATASET_NAME_TO_WEIGHTS (예제용, 활성 10개; kuka 20, language_table 30은 주석)
EXAMPLE_W = {"rt_1": 150, "bridge": 50, "taco_play": 5, "jaco_play": 20, "berkeley_cable_routing": 20,
             "roboturk": 10, "nyu_door_opening_surprising_effectiveness": 5, "viola": 3,
             "berkeley_autolab_ur5": 5, "toto": 5}
ew_sum = sum(EXAMPLE_W.values())
RES["V7b_public_example_weights_arithmetic"] = {
    "note": ("공식 저장소 학습 예제(콜랩)의 가중치입니다. 주석에 'for example'이라고 되어 있고 학습 스텝도 1,000입니다. "
             "논문 혼합의 가중치가 이것이라는 근거는 없습니다. 활성 10개이며 QT-Opt와 Language Table은 빠져 있습니다."),
    "sum": ew_sum,
    "share_pct": {k: round(100 * v / ew_sum, 1) for k, v in EXAMPLE_W.items()},
}

# ---------------------------------------------------------------- V8: 그림 2(a) 눈으로 읽은 값과 스프레드시트 비교
fig2a_reading = {"Franka": 23, "xArm": 6, "Sawyer": 4, "Google Robot": 3, "Kuka iiwa": 3, "UR5": 3,
                 "WidowX": 2, "Hello Stretch": 2, "PR2": 2, "DLR SARA": 2, "Jaco 2": 1, "Unitree A1": 1,
                 "xArm Bimanual": 1, "Cobotta": 1, "DLR EDAN": 1, "PAMY2": 1, "Kinova Gen3": 1,
                 "Fanuc Mate": 1, "Jackal": 1, "RC Car": 1, "TurtleBot 2": 1, "Baxter": 1}
sheet_cnt = {}
for x in v10:
    if x["robot"] == "Multi-Robot":
        continue
    sheet_cnt[x["robot"]] = sheet_cnt.get(x["robot"], 0) + 1
diff = {}
for k in sorted(set(fig2a_reading) | set(sheet_cnt)):
    a = fig2a_reading.get(k, 0)
    b = sheet_cnt.get(k, 0)
    if a != b:
        diff[k] = {"figure_reading": a, "sheet_v1_0": b}
RES["V8_fig2a_vs_sheet"] = {
    "figure_reading_sum": sum(fig2a_reading.values()),
    "sheet_sum_excl_multirobot": sum(sheet_cnt.values()),
    "mismatches_eyeballed": diff,
    "note": "그림 값은 막대 높이를 눈으로 읽은 것이라 ±1 오차가 있을 수 있습니다. 합이 60이 아니라 62로 읽힙니다.",
}

# ---------------------------------------------------------------- V9: 전체 시행 수 3600의 산술 상한
RES["V9_trial_budget"] = {
    "paper_claim": "3600 evaluation trials across 6 robots (V 절 서두)",
    "reported_cells": {"Table I cells x trials": "셀당 n 미보고 (Bridge 열 n=30 가정 시 8셀 x 30 = 240)",
                       "Table II cells": "7행 x 2지표, 행마다 n 후보가 다름 (V3, V5)",
                       "Fig 4": "5도메인 x 3모델, n 미보고"},
    "verdict": "셀별 시행 수가 없어서 3600을 재계산할 수 없습니다. 보고된 셀 수로 보면 n이 수백에서 수천 사이여야 합니다.",
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print(json.dumps({k: RES[k] for k in ["V1_fig4_means_and_claim"]}, ensure_ascii=False, indent=1)[:1500])
    print("V2 reading:", RES["V2_table1_caption_text_check"]["reading"][:120])
    print("V3 common n:", RES["V3_table2_ratios_and_n"]["common_n_for_all_emergent_rows_le200"])
    print("V4 bridge common n:", RES["V4_table1_assumed_n_tests"]["bridge_common_n_candidates_le200"])
    for r in RES["V4_table1_assumed_n_tests"]["assumed_n30_pairs"]:
        print("  V4", r["comparison"], r["diff_pp"], "p=", r["fisher_p_two_sided"])
    for k, v in RES["V5_table2_pairwise_fisher_over_candidate_n"]["results"].items():
        print("  V5", k, v)
    print("V6", {k: v for k, v in RES["V6_dataset_sheet_recount"].items() if k not in ("fig2c_check",)})
    print("V7 bridge share (episodes):", RES["V7_mixture_weighting_arithmetic"]["bridge_share_episode_pct"])
    print("V8 mismatches:", RES["V8_fig2a_vs_sheet"]["mismatches_eyeballed"])
