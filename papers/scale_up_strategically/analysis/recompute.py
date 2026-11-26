"""Scale Up Strategically (arXiv:2607.21582 v1) 표 수치 재계산. 표준 라이브러리만 쓴다(수 초).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json

내용
  V1  표 1(FDR)에서 Copeland 점수를 다시 계산해 표 6, 7, 표 2, 부록 FDH 목록과 비교
  V2  FDR 값이 가질 수 있는 최소 분모(N_f1 + N_f2 <= 400) 확인, 같은 값의 반복
  V3  모델별 쌍 비교가 주장한 순서 color >= object >= spatial >= verb >= size와 맞는 쌍 수
  V4  표 3(실로봇) 비율, 평균 차이, Fisher 정확 검정, Wilson 구간
  V5  본문 +28.7 / +16.7 점 평균과의 비교
  V6  시뮬레이션 부록 수치(48.1 대 33.9 / 28.7 등)의 이항 표준오차와 구분 가능한 최소 차이
  V7  표 5(판정 일치율)의 정수 가능성
  V8  훈련에서 본 칸(seen) 비율: 공개 코드의 지지집합 크기와 그림 8 판독
  K1  본문·부록 구조 수치(6,480 지시, 4,000 영상 등)

시행 수 가정은 각 항목에 적는다. 공식 코드 저장소는 읽기만 했고 실행하지 않았다.
(코드 지지집합 크기는 저장소 https://github.com/yqi19/CoRL2026-Scale-Up-Strategically 의
 커밋 b5b32418bd070b2ce55f3fcde21d742306331ce0 의 data_collection/collection_strategy/lib/*.py 의 리스트를
 이 파일 안에 옮겨 적은 것이다. repo/ 를 import 하지 않는다.)
"""
import json
import math
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"

MODELS = ["pi0.5", "pi0", "OpenVLA-oft", "GR00T-N1.7", "XVLA", "Genie-Envisioner"]
# 표 1 (FDR, %). 쌍 (f1, f2): 양수면 f1이 우세
T1 = {
    ("verb", "color"): [-4.0, -6.0, -6.2, -6.8, -1.0, -22.0],
    ("verb", "object"): [-9.2, -16.7, 16.0, -25.3, -19.0, -25.0],
    ("verb", "size"): [26.3, 41.3, 4.5, 25.1, 38.0, 32.6],
    ("verb", "spatial"): [-7.4, -2.1, 4.5, -12.2, -5.0, -14.2],
    ("color", "object"): [57.3, 42.1, 17.5, 22.8, 31.0, 22.2],
    ("color", "size"): [54.0, 52.4, 33.6, 52.4, 56.0, 30.9],
    ("color", "spatial"): [38.6, 32.6, 39.5, 8.4, 19.0, 36.5],
    ("object", "size"): [22.0, 31.2, 1.6, 52.4, 29.0, 11.4],
    ("object", "spatial"): [-3.8, -2.9, 17.0, 9.5, 9.0, 20.9],
    ("size", "spatial"): [-7.9, -13.0, 19.8, -11.0, -32.0, -23.7],
}
FACTORS = ["color", "object", "spatial", "verb", "size"]
# 표 6 (tau=5%), 표 7 (tau=10%): [color, object, spatial, verb, size]
T6 = {"pi0.5": [3, 2, 2, 1, 0], "pi0": [4, 2, 1, 1, 0], "OpenVLA-oft": [4, 1, 0, 1, 1],
      "GR00T-N1.7": [4, 3, 2, 1, 0], "XVLA": [3, 3, 1, 1, 0], "Genie-Envisioner": [4, 3, 2, 1, 0]}
T7 = {"pi0.5": [3, 1, 0, 1, 0], "pi0": [3, 2, 1, 1, 0], "OpenVLA-oft": [3, 1, 0, 1, 1],
      "GR00T-N1.7": [2, 2, 2, 1, 0], "XVLA": [3, 2, 1, 1, 0], "Genie-Envisioner": [4, 3, 2, 1, 0]}
# 표 2 본문 FDH (색 > 객체 ... 문자열)
T2 = {"pi0.5": "color > object ≈ spatial > verb > size", "pi0": "color > object > spatial ≈ verb > size",
      "OpenVLA-oft": "color > object ≈ verb ≈ size > spatial", "GR00T-N1.7": "color > object > spatial > verb > size",
      "XVLA": "color ≈ object > spatial > verb > size", "Genie-Envisioner": "color > object > spatial > verb > size"}


def copeland(mi, tau):
    sc = {f: 0 for f in FACTORS}
    for (a, b), v in T1.items():
        x = v[mi]
        if x > tau:
            sc[a] += 1
        elif x < -tau:
            sc[b] += 1
    return [sc[f] for f in FACTORS]


def fdh_string(scores):
    order = sorted(range(5), key=lambda i: -scores[i])
    s = FACTORS[order[0]]
    for a, b in zip(order, order[1:]):
        s += (" ≈ " if scores[a] == scores[b] else " > ") + FACTORS[b]
    return s


def v1():
    res = {}
    for tau, tab in ((5.0, T6), (10.0, T7)):
        for mi, m in enumerate(MODELS):
            c = copeland(mi, tau)
            res[f"tau{int(tau)}/{m}"] = {"recomputed": c, "reported": tab[m], "match": c == tab[m],
                                         "fdh_recomputed": fdh_string(c)}
    res["table2_vs_recomputed_tau5"] = {
        m: {"table2": T2[m], "recomputed": fdh_string(copeland(i, 5.0)),
            "same": T2[m].replace("  ", " ") == fdh_string(copeland(i, 5.0))}
        for i, m in enumerate(MODELS)}
    return res


def min_denominator(v, cap=400):
    """round(100*d/m, 1) == v 를 만족하는 정수 d (d와 m의 홀짝 일치)가 있는 최소 m."""
    for m in range(1, cap + 1):
        for d in range(-m, m + 1):
            if (d - m) % 2:
                continue
            if abs(round(100.0 * d / m, 1) - v) < 1e-9:
                return m
    return None


def v2():
    out = {}
    m400 = 0
    total = 0
    mins = {}
    for pair, vals in T1.items():
        for mi, v in enumerate(vals):
            mm = min_denominator(v)
            mins[f"{pair[0]},{pair[1]}/{MODELS[mi]}"] = mm
            total += 1
            # 분모 400(=N_f1+N_f2=400, neither 0건)에서 가능한 값은 d/4 (0.25의 배수)
            if abs(round(v * 4) - v * 4) < 0.02 and any(abs(round(100.0 * d / 400, 1) - v) < 1e-9 for d in range(-400, 401, 2)):
                m400 += 1
    out["min_denominator"] = mins
    out["n_values"] = total
    out["n_values_possible_with_denominator_400"] = m400
    # 같은 값의 반복
    from collections import Counter
    cnt = Counter(abs(v) for vals in T1.values() for v in vals)
    out["repeated_abs_values"] = {str(k): c for k, c in cnt.items() if c > 1}
    out["repeated_positions"] = {
        k: [f"{p[0]},{p[1]}/{MODELS[i]}={vals[i]}" for p, vals in T1.items() for i in range(6) if abs(vals[i]) == float(k)]
        for k in out["repeated_abs_values"]}
    out["note_52.4"] = "52.4 = 11/21 이므로 N_f1+N_f2는 21의 배수여야 정확(최소 21, 400 이하에서는 399까지)"
    out["max_min_denominator"] = max(v for v in mins.values() if v)
    out["n_min_denominator_gt_100"] = sum(1 for v in mins.values() if v and v > 100)
    return out


ORDER_RANK = {f: i for i, f in enumerate(FACTORS)}  # color 0 ... size 4


def v3():
    res = {}
    for mi, m in enumerate(MODELS):
        agree = tie = rev = 0
        revs = []
        for (a, b), vals in T1.items():
            x = vals[mi]
            # 주장 순서: 앞선 인자가 우세. a가 더 앞이면 x>0이 일치
            claim_a_first = ORDER_RANK[a] < ORDER_RANK[b]
            signed = x if claim_a_first else -x  # 양수 = 주장 순서와 일치
            if signed > 5:
                agree += 1
            elif signed < -5:
                rev += 1
                revs.append(f"({a},{b})={x}")
            else:
                tie += 1
        res[m] = {"agree_gt5": agree, "tie_within5": tie, "reversed_gt5": rev, "reversed_pairs": revs}
    # 순환(비추이) 검사: 모델별로 tau=5에서 승리 관계가 순환을 이루는지
    cyc = {}
    for mi, m in enumerate(MODELS):
        beats = {f: set() for f in FACTORS}
        for (a, b), vals in T1.items():
            if vals[mi] > 5:
                beats[a].add(b)
            elif vals[mi] < -5:
                beats[b].add(a)
        found = []
        for x, y, z in combinations(FACTORS, 3):
            for p, q, r in ((x, y, z), (x, z, y)):
                if q in beats[p] and r in beats[q] and p in beats[r]:
                    found.append([p, q, r])
        cyc[m] = found
    res["cycles_tau5"] = cyc
    return res


def v9(m=400, n_sim=4000, seed=0):
    """FDR의 롤아웃 표집 잡음. N_f1 + N_f2 = m(기본 400, 최선의 경우)이라 가정하고 표 1 값을 참값으로 두고
    이항 표집을 반복해 Copeland 순위(tau=5 %)가 보고된 FDH와 같은 비율을 센다. 학습 변동은 포함하지 않는다."""
    import random
    rng = random.Random(seed)
    se_at_0 = round(100 * math.sqrt(1 / m), 2)
    res = {"assumed_m": m, "se_pts_at_fdr0": se_at_0, "n_sim": n_sim,
           "n_cells_abs_le_2se_at_m400": sum(1 for vals in T1.values() for v in vals if abs(v) <= 2 * se_at_0),
           "n_cells_abs_le_10": sum(1 for vals in T1.values() for v in vals if abs(v) <= 10),
           "n_cells_abs_le_5": sum(1 for vals in T1.values() for v in vals if abs(v) <= 5), "models": {}}
    for mi, name in enumerate(MODELS):
        rep = fdh_string(T6[name])
        same = color_top = 0
        for _ in range(n_sim):
            sc = {f: 0 for f in FACTORS}
            for (a, b), vals in T1.items():
                p = (1 + vals[mi] / 100) / 2
                x = sum(rng.random() < p for _ in range(m))
                d = (2 * x - m) / m * 100
                if d > 5:
                    sc[a] += 1
                elif d < -5:
                    sc[b] += 1
            c = [sc[f] for f in FACTORS]
            if fdh_string(c) == rep:
                same += 1
            if c[0] > max(c[1:]):
                color_top += 1
        res["models"][name] = {"reported_fdh": rep, "p_same_fdh": round(same / n_sim, 3), "p_color_strict_top": round(color_top / n_sim, 3)}
    return res


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher_two_sided(a, n1, b, n2):
    """2x2 Fisher 정확 검정 (양측, 확률 합 방식)."""
    k = a + b
    N = n1 + n2

    def pm(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    lo, hi = max(0, k - n2), min(k, n1)
    p0 = pm(a)
    return sum(pm(x) for x in range(lo, hi + 1) if pm(x) <= p0 + 1e-12)


T3 = {"Bun(100)": {"L": 11, "Random": 18, "V": 26}, "Bun(200)": {"L": 18, "Random": 23, "V": 31},
      "Pizza(100)": {"L": 17, "Random": 24, "V": 30}, "Cup(100)": {"L": 18, "Random": 23, "V": 32}}
T3_PCT = {"Bun(100)": {"L": 22.9, "Random": 37.5, "V": 54.2}, "Bun(200)": {"L": 37.5, "Random": 47.9, "V": 64.6},
          "Pizza(100)": {"L": 35.4, "Random": 50.0, "V": 62.5}, "Cup(100)": {"L": 37.5, "Random": 47.9, "V": 66.7}}
N = 48


def v4():
    res = {"percent_matches": {}, "wilson": {}, "fisher": {}}
    for col, d in T3.items():
        for meth, k in d.items():
            res["percent_matches"][f"{col}/{meth}"] = {"k": k, "rate": round(100 * k / N, 1),
                                                       "reported": T3_PCT[col][meth],
                                                       "match": round(100 * k / N, 1) == T3_PCT[col][meth]}
            res["wilson"][f"{col}/{meth}"] = wilson(k, N)
        for other in ("L", "Random"):
            res["fisher"][f"{col}: V vs {other}"] = {
                "diff_pts": round(100 * (d["V"] - d[other]) / N, 1),
                "p": round(fisher_two_sided(d["V"], N, d[other], N), 4)}
    # 합산(네 열을 하나로; 같은 정책·같은 과제에서 나온 열이 섞이므로 독립이 아님, 참고용)
    for other in ("L", "Random"):
        kv = sum(d["V"] for d in T3.values())
        ko = sum(d[other] for d in T3.values())
        res["fisher"][f"pooled4: V vs {other}"] = {"k_V": kv, "k_other": ko, "n_each": 4 * N,
                                                   "diff_pts": round(100 * (kv - ko) / (4 * N), 1),
                                                   "p": round(fisher_two_sided(kv, 4 * N, ko, 4 * N), 6)}
    # 절반 시연 주장: Bun V(100) vs 200 기준선
    res["half_demo_bun"] = {
        "V100 vs L200": {"k": [26, 18], "p": round(fisher_two_sided(26, N, 18, N), 4)},
        "V100 vs Random200": {"k": [26, 23], "p": round(fisher_two_sided(26, N, 23, N), 4)},
        "note": "Pizza와 Cup은 n=100만 보고되어 절반 시연 비교는 Bun 한 과제",
    }
    return res


def v5():
    vl = [100 * (d["V"] - d["L"]) / N for d in T3.values()]
    vr = [100 * (d["V"] - d["Random"]) / N for d in T3.values()]
    vl_r = [round(100 * (d["V"] - d["L"]) / N, 1) for d in T3.values()]
    vr_r = [round(100 * (d["V"] - d["Random"]) / N, 1) for d in T3.values()]
    vl_rep = [round(T3_PCT[c]["V"] - T3_PCT[c]["L"], 1) for c in T3]
    vr_rep = [round(T3_PCT[c]["V"] - T3_PCT[c]["Random"], 1) for c in T3]
    best = [round(T3_PCT[c]["V"] - max(T3_PCT[c]["L"], T3_PCT[c]["Random"]), 1) for c in T3]
    return {
        "V_minus_L_mean_exact": round(sum(vl) / 4, 2), "V_minus_L_mean_of_rounded_table": round(sum(vl_rep) / 4, 2),
        "V_minus_Random_mean_exact": round(sum(vr) / 4, 2), "V_minus_Random_mean_of_rounded_table": round(sum(vr_rep) / 4, 2),
        "text_claims": {"over_L": 28.7, "over_Random": 16.7},
        "per_column_V_minus_L": vl_r, "per_column_V_minus_Random": vr_r,
        "per_column_V_minus_best_baseline": best,
        "note": "본문의 +16.7(랜덤 대비 평균)은 열별 평균 16.2와 다르고, Bun 두 열의 최고 기준선 대비 차(16.7)와 같은 값이다",
    }


def binom_se(p, n):
    return math.sqrt(p * (1 - p) / n)


def v6():
    n = 600  # 3 시드 x 200 롤아웃 (그림 4 캡션). 롤아웃이 독립이라는 가정, 학습 시드는 1개로 가정
    rows = {}
    cases = {
        "SpatialxSize GR00T f18 V vs L": (48.1, 33.9),
        "SpatialxSize GR00T f18 V vs Random": (48.1, 28.7),
        "ColorxSpatial pi0 f18 V vs L": (54.9, 47.8),
        "ColorxSpatial pi0 f18 V vs Random": (54.9, 46.8),
        "VerbxSize pi0.5 f18 V vs Random (본문 51 vs 68)": (51.0, 68.0),
    }
    for k, (a, b) in cases.items():
        pa, pb = a / 100, b / 100
        se = math.sqrt(binom_se(pa, n) ** 2 + binom_se(pb, n) ** 2)
        z = (pa - pb) / se
        p = math.erfc(abs(z) / math.sqrt(2))
        rows[k] = {"diff_pts": round(a - b, 1), "se_pts": round(100 * se, 2), "z": round(z, 2), "p_two_sided_indep": round(p, 4)}
    p = 0.5
    rows["min_detectable_diff_pts_p0.5_n600_vs_600"] = round(100 * 1.96 * math.sqrt(2) * binom_se(p, n), 1)
    rows["per_seed_n200_se_pts_at_0.5"] = round(100 * binom_se(0.5, 200), 1)
    return rows


def v7():
    # 표 5: 성공 90.0, f1 83.0, f2 93.0, 전체 88.7, 주석 롤아웃 80
    def feasible(pct, nmax=200):
        return [n for n in range(1, nmax + 1) if any(abs(round(100.0 * k / n, 1) - pct) < 1e-9 for k in range(n + 1))]
    f_s, f_1, f_2 = feasible(90.0, 80), feasible(83.0, 80), feasible(93.0, 80)
    out = {"n_options_success_90.0": f_s, "n_options_f1_83.0": f_1, "n_options_f2_93.0": f_2}
    best = None
    for a in f_s:
        for b in f_1:
            for c in f_2:
                if a + b + c <= 80:
                    best = (a, b, c)
                    break
    out["exists_split_with_total_le_80"] = best is not None
    out["min_total_over_three_classes"] = min(a + b + c for a in f_s for b in f_1 for c in f_2) if f_s and f_1 and f_2 else None
    out["mean_of_three"] = round((90.0 + 83.0 + 93.0) / 3, 2)
    out["note"] = ("세 클래스 값이 모두 소수 첫째 자리까지 정확하려면 클래스별 주석 수 합이 최소 "
                   f"{out['min_total_over_three_classes']}이어야 한다. 표의 '80'이 전체 수이면 맞지 않고, "
                   "클래스당 80이면 83.0과 93.0이 나오는 정수가 없다(66/80=82.5, 67/80=83.75, 74/80=92.5, 75/80=93.75). "
                   "전체 88.7은 세 값의 단순 평균(88.67)과 같다.")
    return out


def v8():
    # 공개 코드 지지집합 크기 (README와 lib 리스트에서 옮김). 평가는 f에 상관없이 전체 격자(6x6)를 돈다.
    stair = {6: 5, 12: 10, 18: 15}
    lr = {6: 4, 12: 9, 18: 14}
    rnd = {6: 3, 12: 6, 18: 9}
    res = {"pairwise_support_cells": {"V(stair)": stair, "L(Lrandom)": lr, "Random": rnd},
           "eval_grid_cells_6x6": 36,
           "seen_fraction_6x6_eval": {s: {f: round(c / 36, 3) for f, c in d.items()}
                                      for s, d in (("V", stair), ("L", lr), ("Random", rnd))},
           "eval_grid_note": "gr00t_grid_eval.py는 cells=[(c,x) for c in f1vals for x in f2vals] 전체를 돌고 ceil(200/#cells)회 반복(36칸이면 6회, 216 에피소드)"}
    # 실로봇 4x4: 코드의 f12(4x4) 값 대 그림 8 판독
    res["real_robot_4x4_seen_fraction_from_code_f12"] = {"V": round(10 / 16, 3), "L": round(9 / 16, 3), "Random": round(6 / 16, 3)}
    # 같은 seen/unseen 성공률을 가정했을 때 seen 비율 차이만으로 생기는 격차
    gaps = {}
    for s_rate in (0.9, 0.8):
        for u_rate in (0.2, 0.3):
            v = 10 / 16 * s_rate + 6 / 16 * u_rate
            r = 6 / 16 * s_rate + 10 / 16 * u_rate
            gaps[f"seen={s_rate},unseen={u_rate}"] = round(100 * (v - r), 1)
    res["gap_from_seen_fraction_alone_pts_V_minus_Random"] = gaps
    res["gap_formula"] = "(V seen 비율 - Random seen 비율) x (seen 성공률 - unseen 성공률) = 0.25 x (s - u)"
    # 그림 8 판독: 합계가 표 3과 같은지
    rnd_grid = [[2, 0, 3, 1], [3, 3, 1, 0], [0, 3, 0, 1], [3, 0, 0, 3]]
    v_grid = [[3, 3, 1, 3], [3, 3, 2, 3], [3, 2, 0, 0], [3, 2, 0, 0]]
    res["fig8_read_sums"] = {"random": sum(map(sum, rnd_grid)), "V": sum(map(sum, v_grid)),
                             "table3_Bun200": {"Random": 23, "V": 31}}
    res["fig8_note"] = "그림 8 칸 값은 그림 판독 값이다. 합이 표 3 Bun(200)의 23, 31과 일치한다. 훈련 칸(갈색 테두리) 수는 Random 7–8개, V 9–10개로 읽혔다(경계 판독이 어렵다)."
    return res


def k1():
    return {
        "instruction_space": 6 * 6 * 6 * 6 * 5,
        "videos_per_model_10pairs_x_400": 10 * 400,
        "pairwise_grid_cells_6x6": 36, "ood_cells_6x6": 30, "ood_cells_5x6": 25,
        "400_rollouts_over_30_cells": round(400 / 30, 2),
        "complete_cost_pair": 36, "L_cost_formula_2k": 12, "V_cost_formula_3k": 18,
        "all_factor_demo_per_instruction_n400": round(400 / 6480, 3),
        "real_robot_rollouts": 16 * 3,
        "complete_per_cell_demos_n100_n200_16cells": [round(100 / 16, 2), round(200 / 16, 2)],
        "training_instructions_check": {"ColorObject": 6 * 2, "ColorSpatial": 5 * 2, "ColorVerb": 6 * 2, "ColorSize": 6 * 2,
                                        "ObjectSpatial": 5 * 2, "ObjectVerb": 6 * 2, "ObjectSize": 6 * 2,
                                        "SpatialVerb": 5 * 2, "SpatialSize": 5 * 2, "VerbSize": 6 * 2},
    }


def main():
    res = {"V1_copeland": v1(), "V2_fdr_denominator": v2(), "V3_order_agreement": v3(), "V4_table3": v4(),
           "V5_text_averages": v5(), "V6_sim_noise": v6(), "V7_table5": v7(), "V8_seen_fraction": v8(), "V9_fdr_sampling_noise": v9(), "K1_structure": k1()}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    c = res["V1_copeland"]
    print("tau5 match:", [m for m in MODELS if c[f"tau5/{m}"]["match"]])
    print("tau10 match:", [m for m in MODELS if c[f"tau10/{m}"]["match"]])
    print("table2 vs recomputed:", {m: v["same"] for m, v in c["table2_vs_recomputed_tau5"].items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
