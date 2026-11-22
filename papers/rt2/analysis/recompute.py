"""RT-2 보고 수치의 재계산과 일관성 검사 (CPU, 1초 미만, 표준 라이브러리 + scipy).

입력: 논문 본문과 부록 표를 옮긴 값(arXiv v1, 2023-07-28). CoRL 2023 판본(PMLR 229)의 수치도 같은지 대조한다.
출력: analysis/recompute_results.json, 표준출력.

검사 목록 (03 문서의 V 번호와 대응)
  V1  표 4(=CoRL 표 5) Unseen Average가 여섯 셀 평균과 맞는가
  V2  표 6(=CoRL 표 7) Average가 여섯 셀 평균과 맞는가, 표 4와 같은 모델 값이 같은가
  V3  표 5(=CoRL 표 6) 하위 셀이 정수 성공 횟수로 설명되는가 (지시문 수 × 5회)
  V4  표 5 Average 열: 단순 평균과 시행 가중 평균, 보고값과 비교
  V5  표 5 RT-2-PaLI-X 대 RT-1: Fisher 정확검정, 차이의 95% 구간(Newcombe)
  V6  표 4의 셀 크기 추정: 셀당 시행 수 n이 여섯 모델 값을 모두 설명하는 후보
  V7  표 1 Language-Table: ± 값의 의미를 모르므로 두 가정(표준오차, 표준편차)에서 차이 검정
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact, norm

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def half_up(x):
    """반올림(0.5는 올림). 논문 표의 정수 표기 관례에 맞춘다."""
    return math.floor(x + 0.5)




def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def newcombe_diff(k1, n1, k2, n2, z=1.96):
    """두 비율 차이(p1 - p2)의 Newcombe 점수 구간 (퍼센트포인트)."""
    l1, u1 = [x / 100 for x in wilson(k1, n1, z)]
    l2, u2 = [x / 100 for x in wilson(k2, n2, z)]
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return 100 * d, 100 * lo, 100 * hi


# ---------------------------------------------------------------- 보고 값
MODELS_T4 = {  # 표 4 (arXiv v1) = CoRL 표 5. 열: seen, obj E, obj H, bg E, bg H, env E, env H, avg
    "R3M": [45, 32, 14, 13, 9, 0, 2, 12],
    "VC-1": [63, 34, 10, 13, 3, 0, 0, 10],
    "RT-1": [92, 31, 43, 71, 9, 26, 14, 32],
    "MOO": [75, 58, 48, 38, 41, 19, 3, 35],
    "RT-2-PaLI-X-55B": [91, 70, 62, 96, 48, 63, 35, 62],
    "RT-2-PaLM-E-12B": [93, 84, 76, 75, 71, 36, 33, 62],
}
T6 = {  # 표 6 (arXiv v1) = CoRL 표 7. 열: obj E, obj H, bg E, bg H, env E, env H, avg
    "PaLI-X 5B scratch": ([0, 10, 46, 0, 0, 0], 9),
    "PaLI-X 5B fine-tune": ([24, 38, 79, 50, 36, 23], 42),
    "PaLI-X 5B co-fine-tune": ([60, 38, 67, 29, 44, 24], 44),
    "PaLI-X 55B fine-tune": ([60, 62, 75, 38, 57, 19], 52),
    "PaLI-X 55B co-fine-tune": ([70, 62, 96, 48, 63, 35], 63),
}
# 표 5 (arXiv v1) = CoRL 표 6. 값은 (sub1, sub2, sub3, avg), (math, logos, nutrition, color, avg), (celeb, celebA, avg), overall
T5 = {
    "VC-1": dict(sym=[7, 25, 0], sym_avg=11, math=0, logos=8, nut=20, color=13, reas_avg=10,
                 celeb=20, celebA=7, person_avg=13, avg=11),
    "RT-1": dict(sym=[27, 20, 0], sym_avg=16, math=5, logos=0, nut=32, color=28, reas_avg=16,
                 celeb=20, celebA=20, person_avg=20, avg=17),
    "RT-2-PaLI-X-55B": dict(sym=[93, 60, 93], sym_avg=82, math=25, logos=52, nut=48, color=58, reas_avg=46,
                            celeb=53, celebA=53, person_avg=53, avg=60),
    "RT-2-PaLM-E-12B": dict(sym=[67, 20, 20], sym_avg=36, math=35, logos=56, nut=44, color=35, reas_avg=43,
                            celeb=33, celebA=53, person_avg=43, avg=40),
}
# 하위 집단의 지시문 수 (표 3 / CoRL 표 4). 각 지시문 5회 실행(부록 F.2).
N_INSTR = dict(sym1=3, sym2=4, sym3=3, math=4, logos=5, nut=5, color=8, celeb=3, celebA=3)
RUNS = 5

# ---------------------------------------------------------------- V1
v1 = {}
for name, row in MODELS_T4.items():
    cells = row[1:7]
    mean6 = sum(cells) / 6
    v1[name] = dict(cells_mean=round(mean6, 2), reported=row[7], ok=half_up(mean6) == row[7])
res["V1_table4_unseen_average"] = v1
print("V1 표 4 Unseen Average 재계산")
for k, v in v1.items():
    print(f"  {k:18s} 여섯 셀 평균 {v['cells_mean']:6.2f}  보고 {v['reported']:3d}  {'일치' if v['ok'] else '불일치'}")

# ---------------------------------------------------------------- V2
v2 = {}
for name, (cells, rep) in T6.items():
    mean6 = sum(cells) / 6
    v2[name] = dict(cells_mean=round(mean6, 2), reported=rep, ok=half_up(mean6) == rep)
v2["cross_check_55B_co_ft_table4_vs_table6"] = dict(
    table4_avg=MODELS_T4["RT-2-PaLI-X-55B"][7], table6_avg=T6["PaLI-X 55B co-fine-tune"][1],
    cells_identical=MODELS_T4["RT-2-PaLI-X-55B"][1:7] == T6["PaLI-X 55B co-fine-tune"][0])
res["V2_table6_average"] = v2
print("\nV2 표 6 Average 재계산 및 표 4와 교차 검사")
for k, v in v2.items():
    if k.startswith("cross"):
        continue
    print(f"  {k:26s} 여섯 셀 평균 {v['cells_mean']:6.2f}  보고 {v['reported']:3d}  {'일치' if v['ok'] else '불일치'}")
c = v2["cross_check_55B_co_ft_table4_vs_table6"]
print(f"  같은 55B 모델: 표 4 평균 {c['table4_avg']}, 표 6 평균 {c['table6_avg']}, 셀 값 동일 여부 {c['cells_identical']}")

# ---------------------------------------------------------------- V3: 정수 성공 횟수로 설명되는지
def feasible_counts(pct, n):
    """n번 시행에서 반올림해 pct가 되는 성공 횟수 k (0..n) 목록."""
    return [k for k in range(n + 1) if half_up(100 * k / n) == pct]


def infer_k(pct, n):
    ks = feasible_counts(pct, n)
    return ks[0] if ks else None


v3 = {}
groups = {  # 이름: (표 값 키, 지시문 수 키)
    "sym1": ("sym0", "sym1"), "sym2": ("sym1", "sym2"), "sym3": ("sym2", "sym3"),
    "math": ("math", "math"), "logos": ("logos", "logos"), "nut": ("nut", "nut"),
    "color": ("color", "color"), "celeb": ("celeb", "celeb"), "celebA": ("celebA", "celeb"),
}
ninst_map = {"sym1": 3, "sym2": 4, "sym3": 3, "math": 4, "logos": 5, "nut": 5, "color": 8, "celeb": 3, "celebA": 3}
for model, d in T5.items():
    vals = {"sym1": d["sym"][0], "sym2": d["sym"][1], "sym3": d["sym"][2], "math": d["math"],
            "logos": d["logos"], "nut": d["nut"], "color": d["color"], "celeb": d["celeb"], "celebA": d["celebA"]}
    row = {}
    for g, pct in vals.items():
        n = ninst_map[g] * RUNS
        ks = feasible_counts(pct, n)
        row[g] = dict(n=n, pct=pct, feasible_k=ks, ok=len(ks) > 0)
    v3[model] = row
res["V3_table5_integer_check"] = v3
print("\nV3 표 5 하위 셀: n = 지시문 수 × 5 에서 보고값이 정수 횟수로 나오는지")
bad = [(m, g) for m in v3 for g, r in v3[m].items() if not r["ok"]]
for m in v3:
    print("  ", m, " ".join(f"{g}:{r['n']}→{r['feasible_k']}" for g, r in v3[m].items()))
print("  설명되지 않는 셀:", bad if bad else "없음")

# ---------------------------------------------------------------- V4: 평균 열 재계산
def pooled_successes(model):
    d = T5[model]
    k = {g: infer_k(p, ninst_map[g] * RUNS) for g, p in
         {"sym1": d["sym"][0], "sym2": d["sym"][1], "sym3": d["sym"][2], "math": d["math"],
          "logos": d["logos"], "nut": d["nut"], "color": d["color"], "celeb": d["celeb"], "celebA": d["celebA"]}.items()}
    if any(v is None for v in k.values()):  # 정수 횟수로 설명되지 않는 셀은 제외하고 표시한다
        return {g: v for g, v in k.items() if v is not None}, {g: ninst_map[g] * RUNS for g in k if k[g] is not None}
    n = {g: ninst_map[g] * RUNS for g in k}
    return k, n


v4 = {}
for model, d in T5.items():
    k, n = pooled_successes(model)
    sym_mean = (d["sym"][0] + d["sym"][1] + d["sym"][2]) / 3
    reas_mean = (d["math"] + d["logos"] + d["nut"] + d["color"]) / 4
    person_mean = (d["celeb"] + d["celebA"]) / 2
    overall_unw = (d["sym_avg"] + d["reas_avg"] + d["person_avg"]) / 3
    overall_cat = (sym_mean + reas_mean + person_mean) / 3
    sym_k = k["sym1"] + k["sym2"] + k["sym3"]
    sym_n = n["sym1"] + n["sym2"] + n["sym3"]
    reas_k = k["math"] + k["logos"] + k["nut"] + k["color"]
    reas_n = n["math"] + n["logos"] + n["nut"] + n["color"]
    pers_k = k["celeb"] + k["celebA"]
    pers_n = n["celeb"] + n["celebA"]
    total_k = sym_k + reas_k + pers_k
    total_n = sym_n + reas_n + pers_n
    v4[model] = dict(
        sym_avg_reported=d["sym_avg"], sym_avg_mean_of_subcells=round(sym_mean, 2),
        sym_avg_trial_weighted=round(100 * sym_k / sym_n, 2),
        reas_avg_reported=d["reas_avg"], reas_avg_mean_of_subcells=round(reas_mean, 2),
        reas_avg_trial_weighted=round(100 * reas_k / reas_n, 2),
        person_avg_reported=d["person_avg"], person_avg_mean_of_subcells=round(person_mean, 2),
        person_avg_trial_weighted=round(100 * pers_k / pers_n, 2),
        overall_reported=d["avg"], overall_mean_of_category_avgs_reported_parts=round(overall_unw, 2),
        overall_mean_of_categories_recomputed=round(overall_cat, 2),
        overall_trial_weighted=round(100 * total_k / total_n, 2),
        total_successes=total_k, total_trials=total_n,
    )
res["V4_table5_averages"] = v4
print("\nV4 표 5 평균 열")
for m, r in v4.items():
    print(f"  {m:17s} 전체 보고 {r['overall_reported']}, 범주 평균 {r['overall_mean_of_categories_recomputed']}, "
          f"시행 가중 {r['overall_trial_weighted']} ({r['total_successes']}/{r['total_trials']})")
    print(f"  {'':17s} 추론(반올림 무시) 상징 {r['sym_avg_mean_of_subcells']} vs {r['sym_avg_reported']}, "
          f"추론 {r['reas_avg_mean_of_subcells']} vs {r['reas_avg_reported']}, 인물 {r['person_avg_mean_of_subcells']} vs {r['person_avg_reported']}")

# ---------------------------------------------------------------- V5: PaLI 대 RT-1, PaLM 대 PaLI, 대 VC-1
v5 = {}
pairs = [("RT-2-PaLI-X-55B", "RT-1"), ("RT-2-PaLM-E-12B", "RT-1"), ("RT-2-PaLI-X-55B", "VC-1"),
         ("RT-2-PaLI-X-55B", "RT-2-PaLM-E-12B")]
kk = {m: pooled_successes(m)[0] for m in T5}
nn = {m: pooled_successes(m)[1] for m in T5}


def pooled(m):
    k = sum(kk[m].values())
    n = sum(nn[m].values())
    return k, n


for a, b in pairs:
    ka, na = pooled(a)
    kb, nb = pooled(b)
    tab = [[ka, na - ka], [kb, nb - kb]]
    odds, p = fisher_exact(tab)
    diff, lo, hi = newcombe_diff(ka, na, kb, nb)
    v5[f"{a} vs {b}"] = dict(k_a=ka, n_a=na, k_b=kb, n_b=nb, diff_pp=round(diff, 1),
                            newcombe95=[round(lo, 1), round(hi, 1)], fisher_p=float(f"{p:.2g}"),
                            ratio=round((ka / na) / (kb / nb), 2) if kb else None)
res["V5_emergent_tests"] = v5
print("\nV5 표 5 시행 가중 비교 (Fisher 양측, Newcombe 95%)")
for k_, v in v5.items():
    print(f"  {k_:40s} {v['k_a']}/{v['n_a']} vs {v['k_b']}/{v['n_b']}  차이 {v['diff_pp']:+.1f}%p "
          f"[{v['newcombe95'][0]}, {v['newcombe95'][1]}]  p={v['fisher_p']}  비 {v['ratio']}")

# V5b: 9개 하위 집단을 짝지은 비교 (지시문 묶음 단위의 군집을 반영한 보수적 검사). 같은 지시문 집합을 모든 방법에 썼다.
from scipy.stats import ttest_rel, wilcoxon
SUB = ["sym1", "sym2", "sym3", "math", "logos", "nut", "color", "celeb", "celebA"]


def sub_rates(model):
    k, n = pooled_successes(model)
    return [k[g] / n[g] for g in SUB]


v5b = {}
for a, b in [("RT-2-PaLI-X-55B", "RT-1"), ("RT-2-PaLM-E-12B", "RT-1"), ("RT-2-PaLI-X-55B", "RT-2-PaLM-E-12B")]:
    ra, rb = sub_rates(a), sub_rates(b)
    d = [x - y for x, y in zip(ra, rb)]
    t, pt = ttest_rel(ra, rb)
    try:
        w, pw = wilcoxon(ra, rb)
    except ValueError:
        w, pw = None, None
    v5b[f"{a} vs {b}"] = dict(mean_diff_pp=round(100 * sum(d) / len(d), 1), n_pos=sum(x > 0 for x in d),
                              n_neg=sum(x < 0 for x in d), n_zero=sum(x == 0 for x in d),
                              paired_t_p=float(f"{pt:.2g}"), wilcoxon_p=None if pw is None else float(f"{pw:.2g}"))
res["V5b_subgroup_paired"] = v5b
print("\nV5b 9개 하위 집단 짝 비교 (지시문 묶음 수준)")
for k_, v in v5b.items():
    print(f"  {k_:40s} 평균 차이 {v['mean_diff_pp']:+.1f}%p, +{v['n_pos']}/-{v['n_neg']}/0:{v['n_zero']}, "
          f"paired t p={v['paired_t_p']}, Wilcoxon p={v['wilcoxon_p']}")

# ---------------------------------------------------------------- V6: 표 4 셀 크기 후보
# 각 셀의 지시문 수 k (부록 F.2 표 2에서 센 값). 같은 셀은 모든 모델에 같은 n을 쓴다고 가정.
K_CELL = {"obj_E": 10, "obj_H": 21, "bg_E": 12, "bg_H": 17, "env_E": 45, "env_H": 33}
CELL_IDX = {"obj_E": 1, "obj_H": 2, "bg_E": 3, "bg_H": 4, "env_E": 5, "env_H": 6}
v6 = {}
for cell, k in K_CELL.items():
    idx = CELL_IDX[cell]
    vals = {m: row[idx] for m, row in MODELS_T4.items()}
    feas = []
    for n in range(k, 5 * k + 1):
        if all(feasible_counts(p, n) for p in vals.values()):
            feas.append(n)
    v6[cell] = dict(instructions=k, n_range_1_to_5_runs=[k, 5 * k], feasible_common_n=feas)
res["V6_table4_cell_n"] = v6
print("\nV6 표 4 셀: 지시문 수 k에서 1–5회 실행일 때, 여섯 모델 값을 모두 설명하는 공통 n")
for cell, r in v6.items():
    f = r["feasible_common_n"]
    print(f"  {cell:6s} k={r['instructions']:2d}  후보 n: {f if f else '없음'}")

# 셀당 Wilson 구간 예시: RT-2-PaLM-E 대 RT-1, 미관측 셀 크기는 후보 중 최소·최대
v6_ci = {}
for cell in K_CELL:
    idx = CELL_IDX[cell]
    rt2 = MODELS_T4["RT-2-PaLM-E-12B"][idx]
    rt1 = MODELS_T4["RT-1"][idx]
    f = v6[cell]["feasible_common_n"]
    if not f:
        continue
    lo_n, hi_n = min(f), max(f)
    v6_ci[cell] = dict(rt2=rt2, rt1=rt1, n_min=lo_n, n_max=hi_n,
                       rt2_wilson_at_nmin=[round(x, 1) for x in wilson(round(rt2 * lo_n / 100), lo_n)],
                       rt1_wilson_at_nmin=[round(x, 1) for x in wilson(round(rt1 * lo_n / 100), lo_n)])
res["V6_wilson_examples"] = v6_ci

# ---------------------------------------------------------------- V7: 표 1 Language-Table
t1 = {"BC-Zero": (72, 3), "RT-1": (74, 13), "LAVA": (77, 4), "RT-2-PaLI-3B": (90, 10)}
v7 = {}
for other in ("BC-Zero", "RT-1", "LAVA"):
    m1, s1 = t1["RT-2-PaLI-3B"]
    m2, s2 = t1[other]
    diff = m1 - m2
    for label, se in (("±를 표준오차로 볼 때", (s1 ** 2 + s2 ** 2) ** 0.5),
                      ("±를 표준편차로 보고 n을 가정할 때(n=3)", ((s1 ** 2 + s2 ** 2) ** 0.5 / math.sqrt(3)))):
        z = diff / se
        p = 2 * (1 - norm.cdf(abs(z)))
        v7.setdefault(other, {})[label] = dict(diff=diff, se=round(se, 2), z=round(z, 2), p=round(p, 3))
res["V7_language_table"] = v7
print("\nV7 표 1 Language-Table: RT-2-PaLI-3B 대 기준선 (정규근사, n 미보고)")
for other, d in v7.items():
    for label, r in d.items():
        print(f"  vs {other:8s} {label:34s} 차이 {r['diff']:+d}  SE {r['se']}  z {r['z']}  p {r['p']}")

# V8: 표 4 셀별 RT-2-PaLI-X-55B 대 RT-1 (Fisher), 셀 크기는 k(지시문 수, 1회씩)와 5k(5회씩) 두 극단으로 가정
v8 = {}
for cell, k in K_CELL.items():
    idx = CELL_IDX[cell]
    a, b = MODELS_T4["RT-2-PaLI-X-55B"][idx], MODELS_T4["RT-1"][idx]
    row = {}
    for label, n in (("n=k", k), ("n=5k", 5 * k)):
        ka, kb = half_up(a * n / 100), half_up(b * n / 100)
        _, p = fisher_exact([[ka, n - ka], [kb, n - kb]])
        row[label] = dict(n=n, k_rt2=ka, k_rt1=kb, p=float(f"{p:.2g}"))
    v8[cell] = dict(rt2=a, rt1=b, **row)
res["V8_table4_cell_fisher_bounds"] = v8
print("\nV8 표 4 셀별 PaLI-55B 대 RT-1 (Fisher, 셀 크기 가정)")
for cell, r in v8.items():
    print(f"  {cell:6s} {r['rt2']:3d} vs {r['rt1']:3d}  n=k: p={r['n=k']['p']}  n=5k: p={r['n=5k']['p']}")

# V9: 표 6 공동 미세조정 이득 (55B, 5B) 셀별
v9 = {}
for size, (ft, cft) in {"5B": ([24, 38, 79, 50, 36, 23], [60, 38, 67, 29, 44, 24]),
                       "55B": ([60, 62, 75, 38, 57, 19], [70, 62, 96, 48, 63, 35])}.items():
    d = [b - a for a, b in zip(ft, cft)]
    v9[size] = dict(diffs=d, mean=round(sum(d) / 6, 2), n_pos=sum(x > 0 for x in d),
                    n_zero=sum(x == 0 for x in d), n_neg=sum(x < 0 for x in d))
res["V9_cofinetune_gain_cells"] = v9
print("\nV9 공동 미세조정 - 미세조정 (표 6 셀별 %p)")
for s_, r in v9.items():
    print(f"  {s_:4s} {r['diffs']} 평균 {r['mean']:+.2f}  +{r['n_pos']}/0:{r['n_zero']}/-{r['n_neg']}")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote", OUT)
