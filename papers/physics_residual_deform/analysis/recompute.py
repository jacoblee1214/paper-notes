"""논문 Table I–III 수치를 재계산하고 일관성을 검사한다 (CPU, 수 초).

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json

입력: 논문 본문 표를 pdftotext로 읽은 값(paper/paper.pdf, 2607.13451v1). 값은 아래에 직접 적는다.
가정 (논문이 밝히지 않음, 03_analysis.md §0에 근거):
  E1. ±는 표준편차(SD)이고 평가 에피소드 수는 객체당 20 (repo/train.sh의 eval 구간 100–120).
  E2. 두 방법은 같은 에피소드 집합으로 평가된다 (짝 비교가 아니라 독립 표본 Welch로 보수적으로 본다).
  E3. 민감도: ±가 표준오차(SE)라면 SD = ± × sqrt(20). SD가 커지므로 검정력은 작아진다(보수적 방향).
기호: 검산 라벨 V1–V9, 가정 라벨 E1–E3.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_EVAL = 20  # E1

METHODS = ["Spring", "MPM", "GBND", "PGND", "DiffSM", "Ours"]
OBJECTS = ["Rope", "Paper", "Flag", "Sloth", "Duster", "Teddy"]

# (평균, SD) 쌍. None은 "–" (MPM은 Teddy에서 평가하지 않음).
# Table I (tracking, cm), 행 = 객체, 열 = METHODS
T1 = {
    "MDE": {
        "Rope": [(4.4, 2.0), (7.3, 2.5), (5.5, 1.7), (3.3, 1.8), (3.5, 1.7), (2.6, 1.31)],
        "Paper": [(2.3, 2.0), (15.5, 4.2), (3.0, 1.4), (2.1, 0.5), (1.9, 1.6), (1.7, 0.7)],
        "Flag": [(5.7, 2.5), (23.1, 7.0), (30.9, 6.2), (3.2, 1.4), (4.6, 2.2), (2.8, 1.2)],
        "Sloth": [(3.2, 0.8), (7.4, 1.8), (7.7, 2.6), (4.0, 1.3), (2.9, 0.7), (2.7, 0.4)],
        "Duster": [(3.8, 2.0), (6.0, 2.6), (5.1, 2.2), (3.8, 0.1), (3.5, 1.6), (2.9, 1.8)],
        "Teddy": [(5.8, 3.4), None, (1.5, 0.4), (1.6, 1.3), (4.6, 2.7), (1.3, 0.3)],
    },
    "CD": {
        "Rope": [(4.5, 2.4), (6.1, 2.4), (6.6, 2.4), (2.7, 1.4), (3.4, 1.8), (2.3, 1.3)],
        "Paper": [(1.7, 1.2), (14.8, 6.5), (5.0, 1.7), (2.2, 0.9), (1.5, 1.0), (1.3, 0.6)],
        "Flag": [(9.1, 4.7), (23.6, 10.4), (49.1, 13.0), (4.3, 2.4), (6.8, 3.6), (4.3, 2.2)],
        "Sloth": [(2.9, 0.7), (7.1, 1.7), (6.5, 1.8), (3.2, 0.8), (2.8, 0.7), (2.6, 0.3)],
        "Duster": [(4.7, 2.5), (5.2, 1.7), (6.3, 2.8), (4.0, 0.1), (4.3, 2.1), (3.8, 2.5)],
        "Teddy": [(5.1, 3.0), None, (2.7, 0.3), (1.6, 1.0), (3.8, 2.3), (1.3, 0.2)],
    },
    "EMD": {
        "Rope": [(2.4, 1.3), (3.9, 1.6), (3.0, 1.3), (1.4, 0.6), (1.9, 1.0), (1.2, 0.6)],
        "Paper": [(1.8, 1.2), (11.4, 5.5), (2.1, 0.9), (1.5, 0.5), (1.6, 1.0), (1.4, 0.7)],
        "Flag": [(4.9, 2.6), (19.4, 7.8), (24.5, 6.3), (2.3, 1.3), (3.9, 2.1), (2.2, 1.2)],
        "Sloth": [(1.5, 0.4), (3.9, 0.9), (3.1, 1.1), (1.6, 0.4), (1.6, 0.4), (1.4, 0.2)],
        "Duster": [(1.9, 1.2), (2.8, 1.2), (2.8, 1.4), (1.6, 0.4), (1.6, 1.0), (1.4, 1.1)],
        "Teddy": [(2.9, 1.8), None, (0.8, 0.2), (0.8, 0.6), (2.3, 1.3), (0.6, 0.1)],
    },
}
# Table II (visual), J와 F는 ×10 (높을수록 좋음), LPIPS는 ×100 (낮을수록 좋음)
T2 = {
    "J": {
        "Rope": [(3.17, 1.51), (2.44, 1.45), (0.53, 0.55), (3.18, 2.13), (3.45, 1.43), (3.95, 1.69)],
        "Paper": [(6.82, 1.47), (4.51, 0.94), (5.57, 1.40), (7.08, 0.90), (7.10, 1.37), (7.53, 0.82)],
        "Flag": [(4.60, 1.95), (3.78, 1.12), (5.08, 1.42), (5.47, 1.66), (4.98, 1.82), (5.54, 1.89)],
        "Sloth": [(5.30, 0.87), (5.22, 0.88), (3.70, 1.49), (5.86, 0.93), (5.58, 0.82), (6.21, 0.78)],
        "Duster": [(6.66, 0.82), (5.92, 0.87), (5.13, 1.04), (6.69, 0.46), (6.70, 0.78), (6.77, 0.83)],
        "Teddy": [(5.43, 1.99), None, (7.52, 0.66), (7.13, 1.02), (5.82, 1.85), (7.82, 1.14)],
    },
    "F": {
        "Rope": [(6.59, 1.60), (4.92, 2.45), (1.95, 1.91), (5.87, 2.96), (6.95, 1.50), (7.38, 1.85)],
        "Paper": [(4.71, 2.61), (4.73, 0.82), (4.95, 1.35), (5.39, 1.65), (4.98, 2.45), (5.91, 1.76)],
        "Flag": [(2.08, 1.81), (2.49, 0.97), (2.54, 2.04), (2.65, 1.98), (2.22, 1.70), (2.68, 2.14)],
        "Sloth": [(5.76, 1.12), (5.13, 0.86), (3.59, 1.15), (5.53, 1.11), (6.03, 1.05), (6.30, 1.07)],
        "Duster": [(5.32, 0.97), (5.00, 1.08), (4.11, 1.09), (5.29, 0.97), (5.45, 0.91), (5.51, 1.02)],
        "Teddy": [(4.73, 2.72), None, (7.71, 0.96), (7.27, 1.54), (5.05, 2.50), (8.03, 1.75)],
    },
    "LPIPS": {
        "Rope": [(2.6, 0.9), (3.8, 1.4), (5.2, 1.7), (2.6, 1.3), (2.35, 0.85), (2.3, 1.1)],
        "Paper": [(5.6, 2.2), (6.5, 1.4), (5.7, 1.3), (5.2, 1.1), (5.05, 2.00), (4.9, 1.4)],
        "Flag": [(9.9, 3.0), (9.1, 2.5), (8.8, 1.3), (8.9, 1.9), (8.95, 2.70), (8.5, 2.9)],
        "Sloth": [(5.9, 2.0), (7.7, 1.7), (10.5, 2.0), (6.5, 1.8), (5.45, 1.85), (5.4, 2.1)],
        "Duster": [(7.0, 1.9), (6.8, 1.7), (6.4, 1.3), (6.6, 0.9), (6.35, 1.75), (5.9, 1.9)],
        "Teddy": [(3.6, 1.7), None, (2.2, 0.5), (2.0, 0.8), (3.25, 1.55), (1.7, 0.9)],
    },
}
# Table III (ablation, MDE cm). 행 이름과 (Rope, Sloth)
T3 = {
    "PGRD": (2.6, 2.7),
    "PointNet++ (encoder)": (3.0, 3.2),
    "w/o decoder": (2.8, 3.0),
    "w/o temporal aggregator": (2.8, 3.1),
    "gating -> addition": (2.7, 2.9),
    "w/o internal particles": (None, 3.8),
}
LOWER_IS_BETTER = {"MDE": True, "CD": True, "EMD": True, "J": False, "F": False, "LPIPS": True}
TABLE = {**T1, **T2}


def welch(m1, s1, n1, m2, s2, n2):
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m1 - m2) / se
    df = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    )
    p = 2 * stats.t.sf(abs(t), df)
    return t, p


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def main():
    res = {"assumptions": {"E1_SD_n": N_EVAL}, "V1_best_per_cell": {}, "V2_text_claims": {}}

    # V1: 각 셀에서 PGRD가 최솟값(또는 최댓값)인지, 동률 포함
    not_best = []
    for metric, tab in TABLE.items():
        for obj in OBJECTS:
            row = tab[obj]
            vals = [(m, c[0]) for m, c in zip(METHODS, row) if c is not None]
            lower = LOWER_IS_BETTER[metric]
            best = min(vals, key=lambda x: x[1]) if lower else max(vals, key=lambda x: x[1])
            ours = dict(vals)["Ours"]
            ok = (best[0] == "Ours") or (ours == best[1])
            if not ok:
                not_best.append({"metric": metric, "object": obj, "best": best, "ours": ours})
    res["V1_best_per_cell"]["cells_checked"] = len(TABLE) * len(OBJECTS)
    res["V1_best_per_cell"]["PGRD_not_best_or_tied"] = not_best

    # V2: 본문 주장 "PGND가 GBND보다 모든 객체에서 크게 낫다" (Sec. IV-D)
    gbnd_vs_pgnd = []
    for metric in ["MDE", "CD", "EMD"]:
        for obj in OBJECTS:
            g = TABLE[metric][obj][2]
            p = TABLE[metric][obj][3]
            if g is None:
                continue
            if not (p[0] < g[0]):
                gbnd_vs_pgnd.append({"metric": metric, "object": obj, "PGND": p[0], "GBND": g[0]})
    res["V2_text_claims"]["PGND_better_than_GBND_in_all_objects_MDE_CD_EMD"] = {
        "holds": len(gbnd_vs_pgnd) == 0,
        "exceptions": gbnd_vs_pgnd,
    }

    # V3: 물리 기준선 개선 "DiffSM이 Spring보다 대부분 객체에서 낫다" (Sec. IV-D)
    dsm = []
    for metric in ["MDE", "CD", "EMD"]:
        n_better = sum(TABLE[metric][o][4][0] < TABLE[metric][o][0][0] for o in OBJECTS)
        dsm.append({"metric": metric, "DiffSM_lower_than_Spring_objects": n_better, "of": 6})
    res["V3_DiffSM_vs_Spring"] = dsm

    # V4: 주 비교 PGRD 대 가장 가까운 기준선 (객체·지표별 최선 기준선), Welch, E1 가정
    comps = []
    for metric in ["MDE", "CD", "EMD"]:
        for obj in OBJECTS:
            cells = TABLE[metric][obj]
            ours_m, ours_s = cells[5]
            cands = [(m, c) for m, c in zip(METHODS[:5], cells[:5]) if c is not None]
            bm, bc = min(cands, key=lambda x: x[1][0])
            t, p = welch(bc[0], bc[1], N_EVAL, ours_m, ours_s, N_EVAL)
            comps.append({
                "metric": metric, "object": obj, "best_baseline": bm,
                "baseline": bc[0], "ours": ours_m, "diff_ours_minus_base": round(ours_m - bc[0], 3),
                "t": round(t, 3), "p_welch_E1": round(p, 4),
            })
    pv = [c["p_welch_E1"] for c in comps]
    holm_p = holm(pv)
    for c, hp in zip(comps, holm_p):
        c["p_holm_18"] = round(hp, 4)
    res["V4_PGRD_vs_best_baseline_MDE_CD_EMD"] = comps
    res["V4_summary"] = {
        "n_comparisons": len(comps),
        "p_raw_lt_0.05": int(sum(p < 0.05 for p in pv)),
        "p_holm_lt_0.05": int(sum(p < 0.05 for p in holm_p)),
    }


    # V4b: 표 II(J, F, LPIPS)도 같은 방식으로 PGRD 대 최선 기준선
    comps_vis = []
    for metric in ["J", "F", "LPIPS"]:
        for obj in OBJECTS:
            cells = TABLE[metric][obj]
            ours_m, ours_s = cells[5]
            cands = [(m, c) for m, c in zip(METHODS[:5], cells[:5]) if c is not None]
            if LOWER_IS_BETTER[metric]:
                bm, bc = min(cands, key=lambda x: x[1][0])
            else:
                bm, bc = max(cands, key=lambda x: x[1][0])
            t, p = welch(bc[0], bc[1], N_EVAL, ours_m, ours_s, N_EVAL)
            comps_vis.append({"metric": metric, "object": obj, "best_baseline": bm,
                              "baseline": bc[0], "ours": ours_m, "p_welch_E1": round(p, 4)})
    res["V4b_visual_PGRD_vs_best_baseline"] = comps_vis
    res["V4b_summary"] = {"n_comparisons": len(comps_vis),
                          "p_raw_lt_0.05": int(sum(c["p_welch_E1"] < 0.05 for c in comps_vis))}

    # V5: 같은 비교를 E3 (±가 SE)로 보면
    comps_se = []
    for c in comps:
        obj, metric = c["object"], c["metric"]
        cells = TABLE[metric][obj]
        ours_m, ours_s = cells[5]
        bc = cells[METHODS.index(c["best_baseline"])]
        t, p = welch(bc[0], bc[1] * math.sqrt(N_EVAL), N_EVAL, ours_m, ours_s * math.sqrt(N_EVAL), N_EVAL)
        comps_se.append({"metric": metric, "object": obj, "p_if_pm_is_SE": round(p, 6)})
    res["V5_if_pm_is_SE"] = comps_se

    # V6: 계획 성공률 8/10 대 2/10 (Sec. V-A, Fig. 8)
    from scipy.stats import fisher_exact
    _, p_fisher = fisher_exact([[8, 2], [2, 8]], alternative="two-sided")
    _, p_one = fisher_exact([[8, 2], [2, 8]], alternative="greater")

    def wilson(k, n, z=1.96):
        ph = k / n
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        return round(c - h, 3), round(c + h, 3)

    res["V6_planning_success"] = {
        "PGRD": {"k": 8, "n": 10, "wilson95": wilson(8, 10)},
        "SpringMass": {"k": 2, "n": 10, "wilson95": wilson(2, 10)},
        "fisher_two_sided_p": round(p_fisher, 4),
        "fisher_one_sided_p": round(p_one, 4),
        "note": "시행 10회, 과제 1개(케이블 재배선). 다른 과제는 성공 횟수가 없음.",
    }

    # V7: 표 III 대 표 I 일관성 (PGRD 값이 표 I과 같은가)
    res["V7_ablation_consistency"] = {
        "PGRD_rope_TableIII_vs_TableI": (T3["PGRD"][0], T1["MDE"]["Rope"][5][0]),
        "PGRD_sloth_TableIII_vs_TableI": (T3["PGRD"][1], T1["MDE"]["Sloth"][5][0]),
        "ablation_deltas_rope": {k: round(v[0] - T3["PGRD"][0], 2) for k, v in T3.items() if v[0] is not None},
        "ablation_deltas_sloth": {k: round(v[1] - T3["PGRD"][1], 2) for k, v in T3.items()},
        "note": "표 III에는 SD와 시드 수가 없다. 차이 0.1–0.4 cm는 표 I의 Rope SD 1.31보다 작다.",
    }

    # V8: 표 I의 MDE 최선 기준선 대 Ours 상대 감소율
    rel = {}
    for obj in OBJECTS:
        cells = T1["MDE"][obj]
        cands = [c[0] for c in cells[:5] if c is not None]
        rel[obj] = round(100 * (min(cands) - cells[5][0]) / min(cands), 1)
    res["V8_MDE_rel_reduction_vs_best_baseline_pct"] = rel

    # V9: 평가 지표 정의 (코드 기준은 04_code_verification.md). 여기서는 논문 수치만 검사.
    # 표 II의 J·F는 ×10이고 LPIPS는 ×100이라는 표기가 본문과 맞는지 확인
    res["V9_units_note"] = "J, F는 표에 ×10, LPIPS는 ×100으로 적혀 있다. 본문 표기와 같은 척도로 읽었다."

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["V4b_summary", "V1_best_per_cell", "V2_text_claims", "V3_DiffSM_vs_Spring", "V4_summary", "V6_planning_success", "V8_MDE_rel_reduction_vs_best_baseline_pct"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
