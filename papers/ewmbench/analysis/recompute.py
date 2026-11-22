"""EWMBench 보고 수치 재계산 (CPU, 표준 라이브러리만 사용).

입력: 논문 Table 2(arXiv v2, 4자리 반올림 값), Figure 4(B)·6(B)의 막대 라벨(그림에 인쇄된 값).
출력: analysis/recompute_results.json

검사
  1. Table 2의 Motion Avg, Semantics Avg, Overall이 구성 요소의 합과 맞는지
  2. Logics 열의 분모(모든 값이 1/d의 정수배인지)
  3. 순위 일치: Table 2 Overall, Figure 6(B) EWMBench, Figure 6(B) VBench, 사람 순위
  4. Kendall tau-b와 n = 4에서 완전 일치가 우연히 나올 확률
  5. 시료 수 n = 100(10 과제 x 10 에피소드)을 가정한 최악 표준오차와 쌍별 차이의 구분 가능성
"""
import itertools
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Table 2 (논문 p.6). 열: SceneC, HSD, Dyn, nDTW, Avg(motion), Diversity, BLEU, CLIP, Logics, Avg(sem), Overall
T2 = {
    "EnerVerse_FT": dict(scene=0.9427, hsd=0.5356, dyn=0.5363, ndtw=0.5957, mot=1.6676, div=0.0691, bleu=0.1800, clip=0.8638, logics=0.9778, sem=2.0907, overall=4.7010),
    "LTX_FT":       dict(scene=0.9436, hsd=0.4758, dyn=0.6197, ndtw=0.5208, mot=1.6163, div=0.0162, bleu=0.1740, clip=0.8548, logics=0.9444, sem=1.9894, overall=4.5493),
    "Kling":        dict(scene=0.8888, hsd=0.3231, dyn=0.3047, ndtw=0.3162, mot=0.9440, div=0.0493, bleu=0.1675, clip=0.8535, logics=0.9667, sem=2.0370, overall=3.8698),
    "Hailuo":       dict(scene=0.8577, hsd=0.2229, dyn=0.1344, ndtw=0.1789, mot=0.5362, div=0.0370, bleu=0.1848, clip=0.8857, logics=0.9111, sem=2.0186, overall=3.4125),
    "COSMOS":       dict(scene=0.7963, hsd=0.2500, dyn=0.2052, ndtw=0.2533, mot=0.7085, div=0.0803, bleu=0.1230, clip=0.8458, logics=0.7333, sem=1.7824, overall=3.2872),
    "OpenSora":     dict(scene=0.9210, hsd=0.1548, dyn=0.0474, ndtw=0.1420, mot=0.3442, div=0.0415, bleu=0.1598, clip=0.8505, logics=0.8222, sem=1.8739, overall=3.1392),
    "LTX":          dict(scene=0.9156, hsd=0.1575, dyn=0.1002, ndtw=0.1425, mot=0.4002, div=0.0174, bleu=0.0687, clip=0.8324, logics=0.7333, sem=1.6518, overall=2.9676),
}

# Figure 6(B) 막대 라벨 (논문 p.8). 그림 4 쌍 순서와 같은 4개 모델만 있음.
FIG6B_EWM = {"LTX_FT": 5.49, "Kling": 4.75, "Hailuo": 4.50, "OpenSora": 4.05}
FIG6B_VBENCH = {"LTX_FT": 68.3, "Kling": 69.7, "Hailuo": 71.0, "OpenSora": 59.8}
# Figure 6(A) 사람 순위 (1 = 최고). 점수 3/2/0 체계이며 3위 점수는 본문에 없음.
HUMAN_RANK = {"LTX_FT": 1, "Kling": 2, "Hailuo": 3, "OpenSora": 4}


def check_sums(tol=0.0005):
    out = {}
    worst = 0.0
    for m, r in T2.items():
        mot = r["hsd"] + r["dyn"] + r["ndtw"]
        sem = r["div"] + r["bleu"] + r["clip"] + r["logics"]
        ov = r["scene"] + r["mot"] + r["sem"]
        dev = {"mot": mot - r["mot"], "sem": sem - r["sem"], "overall": ov - r["overall"]}
        worst = max(worst, *[abs(v) for v in dev.values()])
        out[m] = {k: round(v, 6) for k, v in dev.items()}
    out["_max_abs_dev"] = round(worst, 6)
    out["_all_within_tol"] = worst <= tol
    return out


def logics_denominator(max_d=200):
    """Logics 값(4자리 반올림)이 모두 1/d의 정수배가 되는 가장 작은 d."""
    vals = [r["logics"] for r in T2.values()]
    for d in range(1, max_d + 1):
        ok = all(abs(v * d - round(v * d)) <= 0.00005 * d + 1e-9 for v in vals)
        if ok:
            nums = {m: round(r["logics"] * d) for m, r in T2.items()}
            return {"smallest_d": d, "numerators": nums}
    return {"smallest_d": None}


def kendall_tau_b(x, y):
    """x, y: 같은 모델 순서의 dict. 동점 보정 tau-b."""
    keys = list(x)
    conc = disc = tx = ty = 0
    for a, b in itertools.combinations(keys, 2):
        dx = (x[a] - x[b])
        dy = (y[a] - y[b])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx * dy > 0:
            conc += 1
        else:
            disc += 1
    n0 = len(keys) * (len(keys) - 1) / 2
    denom = math.sqrt((n0 - tx) * (n0 - ty))
    return (conc - disc) / denom if denom else None, conc, disc


def ranks_from_scores(scores, higher_better=True):
    order = sorted(scores, key=lambda m: scores[m], reverse=higher_better)
    return {m: i + 1 for i, m in enumerate(order)}


def ranking_checks():
    overall = {m: T2[m]["overall"] for m in FIG6B_EWM}
    r_t2 = ranks_from_scores(overall)
    r_fig_ewm = ranks_from_scores(FIG6B_EWM)
    r_vb = ranks_from_scores(FIG6B_VBENCH)
    tau_t2 = kendall_tau_b(HUMAN_RANK, r_t2)
    tau_fig = kendall_tau_b(HUMAN_RANK, r_fig_ewm)
    tau_vb = kendall_tau_b(HUMAN_RANK, r_vb)
    # n = 4에서 사람 순위와 완전 일치(tau = 1)가 임의 순위로 나올 확률 = 1/4!
    p_perfect = 1 / math.factorial(4)
    mismatch = {m: round(FIG6B_EWM[m] - overall[m], 4) for m in FIG6B_EWM}
    return {
        "table2_overall_ranks": r_t2,
        "fig6b_ewm_ranks": r_fig_ewm,
        "fig6b_vbench_ranks": r_vb,
        "human_ranks": HUMAN_RANK,
        "tau_human_vs_table2_overall": tau_t2[0],
        "tau_human_vs_fig6b_ewm": tau_fig[0],
        "tau_human_vs_fig6b_vbench": tau_vb[0],
        "pairs_concordant_disconcordant_vbench": [tau_vb[1], tau_vb[2]],
        "p_perfect_agreement_random_n4": round(p_perfect, 4),
        "fig6b_minus_table2_overall": mismatch,
    }


def sensitivity(n=100):
    """시료 수 n, 지표 값 범위 [0,1] 최악 가정(SD = 0.5), 독립 가정.
    Overall = SceneC + (HSD+Dyn+nDTW) + (Div+BLEU+CLIP+Logics): 항목 8개."""
    sd_item = 0.5
    se_item = sd_item / math.sqrt(n)
    se_overall = se_item * math.sqrt(8)  # 8개 항목의 합
    half95 = 1.96 * se_overall
    diff_se = math.sqrt(2) * se_overall
    diff_half95 = 1.96 * diff_se
    pairs = []
    order = sorted(T2, key=lambda m: T2[m]["overall"], reverse=True)
    for a, b in zip(order, order[1:]):
        d = T2[a]["overall"] - T2[b]["overall"]
        pairs.append({"pair": f"{a} - {b}", "diff": round(d, 4), "resolvable_worst_case": d > diff_half95})
    return {
        "n_episodes_assumed": n,
        "se_overall_worst_case": round(se_overall, 4),
        "half_width95_single_model": round(half95, 4),
        "half_width95_difference_unpaired": round(diff_half95, 4),
        "adjacent_pairs": pairs,
        "note": "쌍 비교는 같은 에피소드를 쓰므로 실제 표준오차는 이보다 작을 수 있음. 에피소드별 값이 없어 최악 가정으로만 계산.",
    }


def main():
    res = {
        "table2_sum_checks": check_sums(),
        "logics_denominator": logics_denominator(),
        "ranking": ranking_checks(),
        "sensitivity_n100": sensitivity(100),
        "video_count": {"models": 7, "tasks": 10, "episodes_per_task": 10, "trials": 3,
                        "computed": 7 * 10 * 10 * 3, "reported": 2100},
        "yolo_val_recall": {"reported": 0.91667, "as_fraction_guess": "11/12", "value": round(11 / 12, 5)},
    }
    path = os.path.join(HERE, "recompute_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
