"""ATOM-Bench (arXiv:2606.16826 v1) 표 수치 재계산과 일관성 검사.

실행: python3 analysis/recompute.py   (표준 라이브러리만 사용, 수 초)  → analysis/recompute_results.json

입력은 논문 표 1, 2, 10과 그림 3, 4, 8에서 옮긴 값이다.
- 표 1, 2, 10, 그림 8의 숫자 라벨은 논문에 적힌 값이다.
- 그림 3의 막대 높이는 PDF에 들어 있는 그림 이미지의 픽셀에서 읽은 값이다(눈금 0.0과 1.0 기준, 오차 약 ±0.01).
- 그림 4의 점 좌표는 눈금선에서 읽은 값이며 10 % 단위로 반올림했다(시행 10회라서 값이 10 % 단위).
시행 수는 논문 3.3절의 "과제당 공유 시드 10개"에서 가져왔다(조합 120, 운동 80, 지시 70).
"""
import json
import math
import random
from itertools import combinations
from pathlib import Path

OUT = Path(__file__).with_name("recompute_results.json")
MODELS = ["Pi0.5", "Motus", "LingBot-VLA", "GROOT N1.6", "SmolVLA"]

# ---- 논문 표 1: (운동 SR, 운동 PSR, 지시 SR, 지시 PSR), 플랫폼별
T1 = {
    "Franka": {"Pi0.5": (46.2, 56.2, 94.3, 95.7), "Motus": (36.2, 46.2, 67.1, 78.7),
               "LingBot-VLA": (37.5, 46.5, 54.3, 60.5), "GROOT N1.6": (28.8, 47.1, 57.1, 69.5),
               "SmolVLA": (17.5, 29.8, 11.4, 31.2)},
    "Cobot": {"Pi0.5": (45.0, 72.0, 71.4, 83.2), "Motus": (35.0, 59.3, 50.0, 67.5),
              "LingBot-VLA": (26.2, 42.8, 21.4, 47.1), "GROOT N1.6": (23.8, 41.0, 27.1, 52.1),
              "SmolVLA": (10.0, 32.2, 5.7, 29.3)},
}
# ---- 논문 표 2: (SR, PSR, AS, CFS)
T2 = {
    "Franka": {"Pi0.5": (15.8, 30.4, 83.3, 73.7), "Motus": (10.8, 26.5, 69.3, 49.4),
               "LingBot-VLA": (3.3, 12.1, 60.3, 54.5), "GROOT N1.6": (3.3, 11.6, 66.3, 61.2),
               "SmolVLA": (0.0, 6.6, 33.3, 28.3)},
    "Cobot": {"Pi0.5": (16.7, 42.6, 79.5, 56.8), "Motus": (7.5, 31.5, 68.5, 48.3),
              "LingBot-VLA": (0.0, 24.4, 47.2, 29.3), "GROOT N1.6": (1.7, 13.7, 48.2, 38.8),
              "SmolVLA": (0.0, 6.3, 32.5, 27.3)},
}
N = {"motor": 80, "instr": 70, "comp": 120}  # 과제 수 x 시드 10

# ---- 그림 3에서 읽은 막대(모델별 원자별 SR, 두 플랫폼 합산). 원자별 시행 수: pick_place 40, access 40, 나머지 20.
FIG3_ATOMS = ["pick_place", "stack", "pour", "push", "reorient", "access",
              "color", "goal_destination", "size", "source_relation", "shape", "exclusion", "count"]
FIG3 = {
    "Pi0.5": [.95, .45, .55, .15, 0, .30, .90, .75, .85, .80, .90, .75, .85],
    "Motus": [1.0, .50, .10, .20, 0, 0, 1.0, .50, .70, .30, .60, 0, .10],
    "LingBot-VLA": [.40, .80, .10, .20, .30, .05, .70, .90, .50, .60, .30, .50, .30],
    "GROOT N1.6": [.80, .20, .05, .15, .10, 0, .55, .55, .40, .60, .35, .10, .40],
    "SmolVLA": [.30, 0, 0, 0, .10, .05, .10, 0, .05, .10, .05, 0, 0],
}
FIG3_LABEL = {"pick_place": 69, "stack": 39, "pour": 16, "push": 14, "reorient": 10, "access": 8,
              "color": 65, "goal_destination": 54, "size": 50, "source_relation": 48, "shape": 44,
              "exclusion": 34, "count": 33}
TEXT_MEAN = {"pick_place": 74, "access": 9, "reorient": 11, "push": 16, "pour": 20}  # 4.2절 본문
TRIALS = {a: 20 for a in FIG3_ATOMS}
TRIALS["pick_place"] = 40
TRIALS["access"] = 40

# ---- 조합 과제의 원자(표 5, 8). 단측과 양팔이 같고 dx10만 color가 하나 더 있다.
COMP = {
    "x1": (["pick_place"], ["color", "count"]), "x2": (["pick_place"], ["source_relation", "exclusion"]),
    "x3": (["access", "pick_place"], ["color", "source_relation"]), "x4": (["access", "pick_place"], ["count", "goal_destination"]),
    "x5": (["reorient"], ["count", "size"]), "x6": (["reorient"], ["color", "goal_destination"]),
    "x7": (["push"], ["exclusion", "goal_destination"]), "x8": (["push"], ["shape", "source_relation"]),
    "x9": (["stack"], ["color", "size"]), "x10": (["stack"], ["shape", "exclusion"]),
    "x11": (["pour"], ["size", "goal_destination"]), "x12": (["pour"], ["color", "source_relation"]),
}
COMP_DUAL = dict(COMP)
COMP_DUAL["x10"] = (["stack"], ["color", "shape", "exclusion"])

# ---- 그림 4에서 읽은 TG(%): (Franka, Cobot)
FIG4 = {"m1": (-50, 10), "m2": (-60, -30), "m3": (0, 30), "m4": (-10, 10), "m5": (-60, 30), "m6": (-10, 10),
        "m7": (20, 30), "m8": (0, 70), "i1": (10, -40), "i2": (-30, -30), "i3": (-40, 0), "i4": (-50, -30),
        "i5": (-10, -50), "i6": (-50, -50), "i7": (-30, 30)}

# ---- 그림 8 (Pi0.5, Cobot, 과제당 10회): 기준, Perturbed-Vis, Perturbed-Lan
FIG8 = {"di1": (90, 90, 90), "di2": (90, 100, 90), "dm2": (90, 100, 100), "dm6": (90, 90, 90),
        "dx1": (60, 40, 40), "dx12": (50, 40, 20)}

# ---- 표 10: (플랫폼, 모델) -> (배치, 에폭)
T10 = {("Franka", "Pi0.5"): (256, 3), ("Franka", "LingBot-VLA"): (128, 3), ("Franka", "GROOT N1.6"): (256, 3),
       ("Franka", "SmolVLA"): (256, 9), ("Franka", "Motus"): (64, 6),
       ("Cobot", "Pi0.5"): (256, 3), ("Cobot", "LingBot-VLA"): (256, 3), ("Cobot", "GROOT N1.6"): (256, 6),
       ("Cobot", "SmolVLA"): (256, 5), ("Cobot", "Motus"): (64, 5)}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(a, n1, b, n2):
    """2x2 분할표(성공 a/n1, b/n2)의 Fisher 정확 검정(양측, 확률이 작은 쪽 합)."""
    K = a + b
    tot = math.comb(n1 + n2, K)

    def pm(x):
        return math.comb(n1, x) * math.comb(n2, K - x) / tot
    p0 = pm(a)
    lo, hi = max(0, K - n2), min(n1, K)
    return min(1.0, sum(pm(x) for x in range(lo, hi + 1) if pm(x) <= p0 * (1 + 1e-9)))


def counts_for(value, n):
    """표시 값(소수 한 자리 %)이 나올 수 있는 정수 성공 수."""
    return [k for k in range(n + 1) if abs(100 * k / n - value) <= 0.0501]


def sign_test(neg, pos):
    n = neg + pos
    k = min(neg, pos)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    R = {}

    # V1 정수 실현 가능성
    v1, bad = {}, []
    for plat in T1:
        for m in MODELS:
            ms, mp, is_, ip = T1[plat][m]
            for key, val, n in (("motor_SR", ms, N["motor"]), ("instr_SR", is_, N["instr"])):
                ks = counts_for(val, n)
                v1[f"{plat}/{m}/{key}"] = {"value": val, "n": n, "counts": ks}
                if not ks:
                    bad.append(f"{plat}/{m}/{key}")
    for plat in T2:
        for m in MODELS:
            val = T2[plat][m][0]
            ks = counts_for(val, N["comp"])
            v1[f"{plat}/{m}/comp_SR"] = {"value": val, "n": N["comp"], "counts": ks}
            if not ks:
                bad.append(f"{plat}/{m}/comp_SR")
    R["V1_integer_feasibility"] = {"cells": v1, "infeasible": bad}
    # 합산 성공 수 (표 1 기준)
    tot = {}
    for m in MODELS:
        mo = sum(counts_for(T1[p][m][0], N["motor"])[0] for p in T1)
        ins = sum(counts_for(T1[p][m][2], N["instr"])[0] for p in T1)
        tot[m] = {"motor_succ_of_160": mo, "instr_succ_of_140": ins}
    R["table1_pooled_successes"] = tot

    # V2 롤아웃 수
    R["V2_rollouts"] = {"main": 5 * 2 * 27 * 10, "single_task_pi05": 2 * 15 * 10, "perturbed_extra": 6 * 2 * 10,
                        "lower_bound_total_physical": 5 * 2 * 27 * 10 + 300 + 120,
                        "demos": 30 * 100, "franka_s_per_demo": 9 * 3600 / 1500, "cobot_s_per_demo": 12.5 * 3600 / 1500}

    # V3 PSR >= SR
    viol = []
    for plat in T1:
        for m in MODELS:
            a = T1[plat][m]
            if a[1] < a[0] or a[3] < a[2]:
                viol.append((plat, m, "T1"))
            b = T2[plat][m]
            if b[1] < b[0]:
                viol.append((plat, m, "T2"))
    R["V3_psr_ge_sr_violations"] = viol

    # V4 Wilson 구간과 쌍별 Fisher
    ci, pw = {}, {}
    for plat in T1:
        for m in MODELS:
            ms, _, is_, _ = T1[plat][m]
            for key, val, n in (("motor", ms, N["motor"]), ("instr", is_, N["instr"])):
                k = counts_for(val, n)[0]
                ci[f"{plat}/{m}/{key}"] = {"k": k, "n": n, "wilson": [round(100 * x, 1) for x in wilson(k, n)]}
    for plat in T2:
        for m in MODELS:
            k = counts_for(T2[plat][m][0], N["comp"])[0]
            ci[f"{plat}/{m}/comp"] = {"k": k, "n": N["comp"], "wilson": [round(100 * x, 1) for x in wilson(k, N["comp"])]}
    for plat in T1:
        for key, idx, n in (("motor", 0, N["motor"]), ("instr", 2, N["instr"])):
            for m1, m2 in combinations(MODELS, 2):
                k1 = counts_for(T1[plat][m1][idx], n)[0]
                k2 = counts_for(T1[plat][m2][idx], n)[0]
                pw[f"{plat}/{key}/{m1} vs {m2}"] = [k1, k2, round(fisher_two_sided(k1, n, k2, n), 4)]
    for plat in T2:
        for m1, m2 in combinations(MODELS, 2):
            k1 = counts_for(T2[plat][m1][0], N["comp"])[0]
            k2 = counts_for(T2[plat][m2][0], N["comp"])[0]
            pw[f"{plat}/comp/{m1} vs {m2}"] = [k1, k2, round(fisher_two_sided(k1, N["comp"], k2, N["comp"]), 4)]
    R["V4_wilson"] = ci
    R["V4_fisher_pairs"] = pw

    # V5 CFS 재계산 (평균값에 식을 대입)
    v5 = {}
    for plat in T2:
        for m in MODELS:
            sr, psr, as_, cfs = T2[plat][m]
            f = max(0.0, as_ - psr) / (100 - psr) * 100
            v5[f"{plat}/{m}"] = {"reported": cfs, "formula_on_means": round(f, 1), "diff": round(f - cfs, 1),
                                 "AS_minus_PSR": round(as_ - psr, 1)}
    R["V5_cfs"] = v5

    # V6 AS 추정: 표 1의 군 평균 PSR을 원자 수로 가중
    v6 = {}
    for plat, comp in (("Franka", COMP), ("Cobot", COMP_DUAL)):
        for m in MODELS:
            pm, pi = T1[plat][m][1], T1[plat][m][3]
            vals = []
            for t, (ma, ia) in comp.items():
                vals.append((len(ma) * pm + len(ia) * pi) / (len(ma) + len(ia)))
            est = sum(vals) / len(vals)
            v6[f"{plat}/{m}"] = {"reported_AS": T2[plat][m][2], "estimated_from_table1": round(est, 1),
                                 "diff": round(T2[plat][m][2] - est, 1)}
    R["V6_as_estimate"] = v6

    # V7 곱셈 기준선(그림 3 원자별 SR): 독립이면 곱, 완전 상관이면 최소
    v7 = {}
    for m in MODELS:
        s = dict(zip(FIG3_ATOMS, FIG3[m]))
        prod_f = [math.prod(s[a] for a in ma + ia) for ma, ia in COMP.values()]
        prod_c = [math.prod(s[a] for a in ma + ia) for ma, ia in COMP_DUAL.values()]
        min_f = [min(s[a] for a in ma + ia) for ma, ia in COMP.values()]
        min_c = [min(s[a] for a in ma + ia) for ma, ia in COMP_DUAL.values()]
        obs = (T2["Franka"][m][0] + T2["Cobot"][m][0]) / 2
        v7[m] = {"observed_mean_SR": round(obs, 1),
                 "independent_product": round(100 * (sum(prod_f) + sum(prod_c)) / 24, 1),
                 "perfect_correlation_min": round(100 * (sum(min_f) + sum(min_c)) / 24, 1),
                 "AS_mean": round((T2["Franka"][m][2] + T2["Cobot"][m][2]) / 2, 1)}
    # 원자 SR의 불확실성(Beta(k+1, n-k+1) 표본 4000개)을 반영한 독립 곱 기준선 구간
    rng = random.Random(0)
    for m in MODELS:
        draws = []
        for _ in range(4000):
            s = {}
            for a, v in zip(FIG3_ATOMS, FIG3[m]):
                n = TRIALS[a]
                k = round(v * n)
                s[a] = rng.betavariate(k + 1, n - k + 1)
            tot_p = sum(math.prod(s[a] for a in ma + ia) for ma, ia in COMP.values())
            tot_p += sum(math.prod(s[a] for a in ma + ia) for ma, ia in COMP_DUAL.values())
            draws.append(100 * tot_p / 24)
        draws.sort()
        v7[m]["product_interval_95"] = [round(draws[100], 1), round(draws[3900], 1)]
    R["V7_product_baseline"] = v7

    # V8 그림 3 대 표 1 대 본문
    v8 = {"atom_trials": TRIALS, "per_model": {}, "mean_check": {}}
    for m in MODELS:
        s = dict(zip(FIG3_ATOMS, FIG3[m]))
        mo = sum(s[a] * TRIALS[a] for a in FIG3_ATOMS[:6])
        ins = sum(s[a] * TRIALS[a] for a in FIG3_ATOMS[6:])
        v8["per_model"][m] = {"fig3_motor_succ_of_160": round(mo, 1), "table1_motor_succ_of_160": tot[m]["motor_succ_of_160"],
                              "fig3_instr_succ_of_140": round(ins, 1), "table1_instr_succ_of_140": tot[m]["instr_succ_of_140"]}
    for a in FIG3_ATOMS:
        mean = 100 * sum(FIG3[m][FIG3_ATOMS.index(a)] for m in MODELS) / 5
        v8["mean_check"][a] = {"bar_mean": round(mean, 1), "fig_label": FIG3_LABEL[a], "text": TEXT_MEAN.get(a)}
    v8["motor_overall_from_labels"] = round((4 * 69 + 4 * 8 + 2 * (39 + 16 + 14 + 10)) / 16, 1)
    v8["motor_overall_table1"] = round(sum(T1[p][m][0] for p in T1 for m in MODELS) / 10, 2)
    v8["instr_overall_from_labels"] = round(sum(FIG3_LABEL[a] for a in FIG3_ATOMS[6:]) / 7, 2)
    v8["instr_overall_table1"] = round(sum(T1[p][m][2] for p in T1 for m in MODELS) / 10, 2)
    R["V8_fig3_vs_table1"] = v8

    # V9 지시 < 운동 순서
    cells = []
    for plat in T1:
        for m in MODELS:
            ms, _, is_, _ = T1[plat][m]
            cells.append({"cell": f"{plat}/{m}", "motor": ms, "instr": is_, "instr_gt_motor": is_ > ms})
    R["V9_instruction_easier"] = {"cells": cells, "n_true": sum(c["instr_gt_motor"] for c in cells), "n": len(cells)}

    # V10 TG
    fr = [v[0] for v in FIG4.values()]
    co = [v[1] for v in FIG4.values()]
    v10 = {"franka_neg_pos_zero": [sum(x < 0 for x in fr), sum(x > 0 for x in fr), sum(x == 0 for x in fr)],
           "cobot_neg_pos_zero": [sum(x < 0 for x in co), sum(x > 0 for x in co), sum(x == 0 for x in co)]}
    v10["franka_sign_test_p"] = round(sign_test(v10["franka_neg_pos_zero"][0], v10["franka_neg_pos_zero"][1]), 4)
    v10["cobot_sign_test_p"] = round(sign_test(v10["cobot_neg_pos_zero"][0], v10["cobot_neg_pos_zero"][1]), 4)
    mot = [k for k in FIG4 if k.startswith("m")]
    ins_ = [k for k in FIG4 if k.startswith("i")]
    v10["mean_TG"] = {"franka_motor": sum(FIG4[k][0] for k in mot) / 8, "franka_instr": sum(FIG4[k][0] for k in ins_) / 7,
                      "cobot_motor": sum(FIG4[k][1] for k in mot) / 8, "cobot_instr": sum(FIG4[k][1] for k in ins_) / 7}
    v10["cobot_motor_positive"] = sum(FIG4[k][1] > 0 for k in mot)
    v10["cobot_instr_negative"] = sum(FIG4[k][1] < 0 for k in ins_)
    # 한 점이 유의할 수 있는 최소 |TG| (가장 유리한 분할)
    best = {}
    for tg in range(10, 110, 10):
        pmin = 1.0
        for a in range(0, 11):
            b = a + tg // 10
            if b <= 10:
                pmin = min(pmin, fisher_two_sided(a, 10, b, 10))
        best[tg] = round(pmin, 4)
    v10["best_case_fisher_by_abs_TG"] = best
    v10["points_abs_TG_ge_50"] = {"franka": sum(abs(x) >= 50 for x in fr), "cobot": sum(abs(x) >= 50 for x in co)}
    R["V10_transfer_gap"] = v10

    # V11 섭동
    mean = [sum(v[i] for v in FIG8.values()) / 6 for i in range(3)]
    comp_pool = [sum(FIG8[t][i] for t in ("dx1", "dx12")) / 10 for i in range(3)]
    atom_pool = [sum(FIG8[t][i] for t in ("di1", "di2", "dm2", "dm6")) / 10 for i in range(3)]
    R["V11_perturbation"] = {
        "mean_SR": [round(x, 2) for x in mean], "paper_mean": [78.3, 76.7, 71.7],
        "comp_successes_of_20": comp_pool, "atomic_successes_of_40": atom_pool,
        "fisher_dx1_base_vs_vis": round(fisher_two_sided(6, 10, 4, 10), 3),
        "fisher_dx12_base_vs_lan": round(fisher_two_sided(5, 10, 2, 10), 3),
        "fisher_comp_pool_base_vs_vis": round(fisher_two_sided(int(comp_pool[0]), 20, int(comp_pool[1]), 20), 3),
        "fisher_comp_pool_base_vs_lan": round(fisher_two_sided(int(comp_pool[0]), 20, int(comp_pool[2]), 20), 3),
        "fisher_all_base_vs_lan_60": round(fisher_two_sided(round(mean[0] * 0.6), 60, round(mean[2] * 0.6), 60), 3),
        "fisher_all_vis_vs_lan_60": round(fisher_two_sided(round(mean[1] * 0.6), 60, round(mean[2] * 0.6), 60), 3),
        "successes_of_60": [round(x * 0.6) for x in mean],
    }

    # V12 학습 예산
    frames = {"Franka": 9 * 3600 * 30, "Cobot": 12.5 * 3600 * 30}
    v12 = {}
    for (plat, m), (b, ep) in T10.items():
        v12[f"{plat}/{m}"] = {"batch": b, "epochs": ep, "frame_visits": frames[plat] * ep,
                              "approx_steps": round(frames[plat] * ep / b)}
    R["V12_training_budget"] = {"frames": frames, "cells": v12}

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    print("infeasible:", bad)
    print("V5", {k: (v["reported"], v["formula_on_means"]) for k, v in v5.items()})
    print("V6", {k: (v["reported_AS"], v["estimated_from_table1"]) for k, v in v6.items()})
    print("V7", v7)
    print("V8", json.dumps(v8, ensure_ascii=False))
    print("V9", R["V9_instruction_easier"]["n_true"], "of 10")
    print("V10", v10)
    print("V11", R["V11_perturbation"])
    print("V12", {k: v["approx_steps"] for k, v in v12.items()})
    print("pooled", tot)


if __name__ == "__main__":
    main()
