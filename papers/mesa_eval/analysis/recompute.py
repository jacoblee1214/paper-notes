"""MESA 논문(arXiv:2611.05567 v1)의 표와 본문 수치를 다시 계산한다.

입력: paper/paper.txt (pdftotext -layout). 표 2, 3, 6-15(과제별 성공률), 16-19(Wilson 구간)를 파싱한다.
출력: analysis/recompute_results.json, 콘솔 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

검사 라벨: V1 요약표와 과제별 표, V2 평균 행, V3 Wilson 구간, V4 언어 추종률(LFR) 분모,
V5 그림 3 (ID 대응 과제), V6 GPT 과제 유형, V7 보조 데이터 공동학습 검정, V8 RoboArena 순위,
V9 개수 일관성, V10 과제 부트스트랩, V11 인접 순위 쌍, V12 그림 5, V13 유형 보정, V14 그림 4와 8.
"""
import json
import math
import re
from collections import OrderedDict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
OUT = {}

SM = ["DP", "PG-FM", "GR00T", "pi0", "FAST", "pi0.5"]                       # 단일 팔 6개
BM = ["DP", "PG-Bin", "PG-FM", "GR00T", "pi0", "FAST", "pi0.5", "GPT"]      # 양팔 8개
SUITES = ["ID", "Spatial", "Instance", "Composite", "Category"]
S_NTASK = {"ID": 70, "Spatial": 10, "Instance": 20, "Composite": 20, "Category": 20}
B_NTASK = {"ID": 57, "Spatial": 38, "Instance": 18, "Composite": 15, "Category": 17}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def two_prop_p(k1, n1, k2, n2):
    """비짝지음 두 비율 z 검정(양측). 논문은 짝지은 McNemar를 썼으므로 근사 참고용."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


# ---------------------------------------------------------------- 파싱
def norm(s):
    return re.sub(r"[ \t]+", " ", s.replace("π 0", "pi0").replace("π0", "pi0").replace("π", "pi")).strip()


LINES = [norm(l) for l in TXT.splitlines()]
NUM = r"(?:\d+\.\d|–)"


def parse_summary_tables():
    """표 2와 표 3: 'MESA-ID 1.6 – 30.7 ...' 행. 처음 12행이 표 2, 다음 12행이 표 3."""
    rows = []
    pat = re.compile(r"^((?:Bi)?MESA(?:-\w+| Avg\.)|BiMESA-\w+|BiMESA Avg\.) ((?:%s ){7}%s)$" % (NUM, NUM))
    for l in LINES:
        m = pat.match(l)
        if m:
            rows.append((m.group(1), [None if t == "–" else float(t) for t in m.group(2).split()]))
    assert len(rows) >= 24, len(rows)
    return rows[:12], rows[12:24]


def to_dict(rows):
    d = OrderedDict()
    for name, vals in rows:
        d[name] = dict(zip(["DP", "PG-Bin", "PG-FM", "GR00T", "pi0", "FAST", "pi0.5", "GPT"], vals))
    return d


T2_rows, T3_rows = parse_summary_tables()
T2, T3 = to_dict(T2_rows), to_dict(T3_rows)


def label(prefix, s):
    return f"{prefix}-{s}"


def parse_ci_tables():
    """표 16-19: 'x [lo, hi]' 행."""
    pat = re.compile(r"(\d+\.\d) \[(\d+\.\d), (\d+\.\d)\]")
    out = []
    for l in LINES:
        m = pat.findall(l)
        head = l.split(" ")[0]
        if m and (head.startswith("MESA") or head.startswith("BiMESA") or head == "Average") and len(m) in (6, 8):
            out.append((l.split(" ")[0] if head != "Average" else "Average", [(float(a), float(b), float(c)) for a, b, c in m]))
    return out


CI_ROWS = parse_ci_tables()
# 표 16(6열 6행), 17(8열 6행), 18(6열 6행), 19(8열 6행) 순서
t16 = [r for r in CI_ROWS if len(r[1]) == 6][:6]
t18 = [r for r in CI_ROWS if len(r[1]) == 6][6:12]
t17 = [r for r in CI_ROWS if len(r[1]) == 8][:6]
t19 = [r for r in CI_ROWS if len(r[1]) == 8][6:12]
assert len(t16) == len(t17) == len(t18) == len(t19) == 6, (len(t16), len(t17), len(t18), len(t19))


def parse_task_tables():
    """표 6-15: 과제별 성공률. 숫자 N개로 끝나는 줄을 한 행으로 보고, 뒤따르는 글자 줄을 이름 이어쓰기로 붙인다."""
    tables, cur, ncol = {}, None, None
    rowpat6 = re.compile(r"^(.*?) ((?:\d+\.\d ){5}\d+\.\d)$")
    rowpat8 = re.compile(r"^(.*?) ((?:\d+\.\d ){7}\d+\.\d)$")
    last = None
    for l in LINES:
        m = re.match(r"^Table (\d+): Success rates for all tasks in the (\S+) task set\.", l)
        if m and 6 <= int(m.group(1)) <= 15:
            cur = int(m.group(1))
            tables[cur] = {"suite": m.group(2), "rows": []}
            ncol = 6 if cur <= 10 else 8
            last = None
            continue
        if cur is None:
            continue
        if l.startswith("B.9") or l.startswith("C MESA Details") or l.startswith("Table 16"):
            if l.startswith("C MESA") or l.startswith("Table 16"):
                cur = None
            continue
        pat = rowpat6 if ncol == 6 else rowpat8
        mm = pat.match(l)
        if mm and not l.startswith(("MESA-", "BiMESA-")):
            vals = [float(x) for x in mm.group(2).split()]
            tables[cur]["rows"].append({"name": mm.group(1), "v": vals})
            last = tables[cur]["rows"][-1]
        elif last is not None and l and not re.match(r"^(\d+|GPT-6\.1-|Sol|MESA-\w+ DP.*|BiMESA-\w+ DP.*|B\.9.*|Several.*|annotated.*|for articulated.*|the handoff.*|cross-arm.*|Table.*)$", l) \
                and not l.startswith(("MESA-", "BiMESA-", "Several", "annotated", "for articulated", "the handoff", "cross-arm")):
            last["name"] += " " + l
            last = None if l.endswith(")") and False else last
    return tables


TT = parse_task_tables()
N_ROWS = {k: len(v["rows"]) for k, v in TT.items()}
OUT["V9_rows_per_task_table"] = {f"Table {k}": n for k, n in N_ROWS.items()}


def suite_arrays():
    S, B = {}, {}
    for k, s in zip(range(6, 11), SUITES):
        S[s] = {"names": [r["name"] for r in TT[k]["rows"]], "p": np.array([r["v"] for r in TT[k]["rows"]]).T}  # 6 x T
    for k, s in zip(range(11, 16), SUITES):
        B[s] = {"names": [r["name"] for r in TT[k]["rows"]], "p": np.array([r["v"] for r in TT[k]["rows"]]).T}  # 8 x T
    return S, B


SA, BA = suite_arrays()
for s in SUITES:
    assert SA[s]["p"].shape[1] == S_NTASK[s], (s, SA[s]["p"].shape)
    assert BA[s]["p"].shape[1] == B_NTASK[s], (s, BA[s]["p"].shape)

# ---------------------------------------------------------------- V1 요약표와 과제별 표
v1 = {"single": {}, "bimanual": {}}
maxdiff = 0.0
for s in SUITES:
    for i, m in enumerate(SM):
        rec = T2["MESA-" + s][m]
        calc = SA[s]["p"][i].mean()
        v1["single"][f"{s}/{m}"] = {"table2": rec, "from_tasks": round(calc, 3), "diff": round(calc - rec, 3)}
        maxdiff = max(maxdiff, abs(calc - rec))
    for i, m in enumerate(BM):
        rec = T2["BiMESA-" + s][m]
        calc = BA[s]["p"][i].mean()
        v1["bimanual"][f"{s}/{m}"] = {"table2": rec, "from_tasks": round(calc, 3), "diff": round(calc - rec, 3)}
        maxdiff = max(maxdiff, abs(calc - rec))
v1["max_abs_diff"] = round(maxdiff, 3)
v1["n_cells_over_0.06"] = sum(1 for g in ("single", "bimanual") for c in v1[g].values() if abs(c["diff"]) > 0.06)
v1["cells_over_0.06"] = {f"{g}:{k}": c for g in ("single", "bimanual") for k, c in v1[g].items() if abs(c["diff"]) > 0.06}
OUT["V1_table2_vs_per_task"] = v1

# ---------------------------------------------------------------- V2 평균 행
v2 = {}
for pre, models, tab, key in (("MESA", SM, T2, "table2"), ("BiMESA", BM, T2, "table2")):
    for m in models:
        vals = [tab[f"{pre}-{s}"][m] for s in SUITES]
        if any(v is None for v in vals):
            continue
        mean5 = float(np.mean(vals))
        rec = tab[f"{pre} Avg."][m]
        ntask = S_NTASK if pre == "MESA" else B_NTASK
        w = np.array([ntask[s] for s in SUITES], float)
        pooled = float(np.dot(w, vals) / w.sum())
        v2[f"{pre}/{m}"] = {"table2_avg": rec, "mean_of_5_suites": round(mean5, 2), "task_weighted": round(pooled, 2),
                            "diff_mean5": round(mean5 - rec, 2)}
OUT["V2_average_rows"] = v2

# 표 3(LFR) 평균 행
v2b = {}
for pre, models in (("MESA", SM), ("BiMESA", BM)):
    for m in models:
        vals = [T3[f"{pre}-{s}"][m] for s in SUITES]
        if any(v is None for v in vals):
            continue
        rec = T3[f"{pre} Avg."][m]
        v2b[f"{pre}/{m}"] = {"table3_avg": rec, "mean_of_5": round(float(np.mean(vals)), 2), "diff": round(float(np.mean(vals)) - rec, 2)}
OUT["V2_lfr_average_rows"] = v2b

# ---------------------------------------------------------------- V3 Wilson 구간
v3 = {"single": {}, "bimanual": {}, "worst_abs_err": 0.0}
for si, s in enumerate(SUITES):
    n_s = 50 * S_NTASK[s]
    for i, m in enumerate(SM):
        k = int(round(SA[s]["p"][i].sum() * 0.5))
        lo, hi = wilson(k, n_s)
        rec = t16[si][1][i]
        err = max(abs(lo - rec[1]), abs(hi - rec[2]))
        v3["single"][f"{s}/{m}"] = {"k": k, "n": n_s, "ours": [round(lo, 1), round(hi, 1)], "paper": [rec[1], rec[2]], "err": round(err, 2)}
        v3["worst_abs_err"] = max(v3["worst_abs_err"], err)
    for i, m in enumerate(BM):
        nr = 10 if m == "GPT" else 50
        n_s = nr * B_NTASK[s]
        k = int(round(BA[s]["p"][i].sum() * nr / 100))
        lo, hi = wilson(k, n_s)
        rec = t17[si][1][i]
        err = max(abs(lo - rec[1]), abs(hi - rec[2]))
        v3["bimanual"][f"{s}/{m}"] = {"k": k, "n": n_s, "ours": [round(lo, 1), round(hi, 1)], "paper": [rec[1], rec[2]], "err": round(err, 2)}
        v3["worst_abs_err"] = max(v3["worst_abs_err"], err)
v3["worst_abs_err"] = round(v3["worst_abs_err"], 2)
v3["n_over_0.15"] = sum(1 for g in ("single", "bimanual") for c in v3[g].values() if c["err"] > 0.15)
v3["over_0.15"] = {f"{g}:{k}": c for g in ("single", "bimanual") for k, c in v3[g].items() if c["err"] > 0.15}

# 평균 행: "과제별 이항 분산으로 만든 정규 구간"을 재현해 본다 (5개 묶음의 단순 평균)
def avg_ci(P, nroll, ntask_map):
    """P: 묶음별 (T,) 과제 성공률(%) 배열 5개. 평균 = 묶음 평균들의 평균. 분산 = 각 과제 p(1-p)/n 의 가중합."""
    means, var = [], 0.0
    for s in SUITES:
        p = P[s] / 100.0
        T = len(p)
        means.append(p.mean())
        var += (1 / 25) * (p * (1 - p) / nroll).sum() / T**2
    m = float(np.mean(means))
    h = 1.959964 * math.sqrt(var)
    return 100 * m, 100 * (m - h), 100 * (m + h)


avg_chk = {}
for i, m in enumerate(SM):
    P = {s: SA[s]["p"][i] for s in SUITES}
    mm, lo, hi = avg_ci(P, 50, S_NTASK)
    rec = t16[5][1][i]
    avg_chk[f"MESA/{m}"] = {"ours": [round(mm, 1), round(lo, 1), round(hi, 1)], "paper": list(rec)}
for i, m in enumerate(BM):
    nr = 10 if m == "GPT" else 50
    P = {s: BA[s]["p"][i] for s in SUITES}
    mm, lo, hi = avg_ci(P, nr, B_NTASK)
    rec = t17[5][1][i]
    avg_chk[f"BiMESA/{m}"] = {"ours": [round(mm, 1), round(lo, 1), round(hi, 1)], "paper": list(rec)}
v3["average_row_normal_ci"] = avg_chk
OUT["V3_wilson"] = v3

# ---------------------------------------------------------------- V4 LFR 분모 복원
v4 = {"single": {}, "bimanual": {}}


def implied_denoms(S_cnt, L_pct, ci):
    """L = S/(S+D). n=S+D를 정수로 복원: round(100*S/n,1)==L 이 되는 n과, 그 n의 Wilson 구간이 표와 맞는지."""
    best = []
    for n in range(max(S_cnt, 1), 20000):
        if abs(100 * S_cnt / n - L_pct) <= 0.05 + 1e-9:
            lo, hi = wilson(S_cnt, n)
            best.append((n, lo, hi))
        if 100 * S_cnt / n < L_pct - 0.06:
            break
    ok = [(n, lo, hi) for n, lo, hi in best if abs(lo - ci[1]) <= 0.15 and abs(hi - ci[2]) <= 0.15]
    return best, ok


nofit = 0
for si, s in enumerate(SUITES):
    for i, m in enumerate(SM):
        Sc = int(round(SA[s]["p"][i].sum() * 0.5))
        L = T3["MESA-" + s][m]
        if L is None:
            continue
        best, ok = implied_denoms(Sc, L, t18[si][1][i])
        n = ok[0][0] if ok else (best[0][0] if best else None)
        nofit += 0 if ok else 1
        v4["single"][f"{s}/{m}"] = {"S": Sc, "L": L, "n_S_plus_D": n, "D_implied": (n - Sc) if n else None,
                                    "D_rate_pct_of_rollouts": round(100 * (n - Sc) / (50 * S_NTASK[s]), 1) if n else None,
                                    "ci_consistent": bool(ok)}
for si, s in enumerate(SUITES):
    for i, m in enumerate(BM):
        nr = 10 if m == "GPT" else 50
        Sc = int(round(BA[s]["p"][i].sum() * nr / 100))
        L = T3["BiMESA-" + s][m]
        best, ok = implied_denoms(Sc, L, t19[si][1][i])
        n = ok[0][0] if ok else (best[0][0] if best else None)
        nofit += 0 if ok else 1
        v4["bimanual"][f"{s}/{m}"] = {"S": Sc, "L": L, "n_S_plus_D": n, "D_implied": (n - Sc) if n else None,
                                      "D_rate_pct_of_rollouts": round(100 * (n - Sc) / (nr * B_NTASK[s]), 1) if n else None,
                                      "ci_consistent": bool(ok)}
v4["n_cells_without_consistent_integer_denominator"] = nofit
OUT["V4_lfr_denominators"] = v4

# 단일 팔 π0.5 ID: S=62.8%, D 추정(전체 롤아웃 대비)
OUT["V4_lfr_example_pi05_single_ID"] = v4["single"]["ID/pi0.5"]

# ---------------------------------------------------------------- V5 그림 3 (ID 대응 과제)
FIG3_SPATIAL_ID = {"DP": 5.0, "PG-FM": 40.6, "GR00T": 46.0, "pi0": 64.6, "FAST": 60.0, "pi0.5": 73.0}
FIG3_INST_ID = {"DP": 0.7, "PG-FM": 29.2, "GR00T": 23.9, "pi0": 42.9, "FAST": 49.6, "pi0.5": 64.7}


def base_name(n):
    return n.strip()


v5 = {"spatial": {}, "instance": {}}
id_names = SA["ID"]["names"]
id_p = SA["ID"]["p"]


def find_id(name):
    cands = [j for j, nm in enumerate(id_names) if nm == name]
    return cands


for sname, key, figv in (("Spatial", "spatial", FIG3_SPATIAL_ID), ("Instance", "instance", FIG3_INST_ID)):
    names = SA[sname]["names"]
    # 단일 팔: 같은 지시문 -> ID 표에서 같은 이름
    idx = []
    miss = []
    for nm in names:
        c = find_id(nm)
        if len(c) == 1:
            idx.append(c[0])
        else:
            miss.append((nm, len(c)))
    v5[key]["matched_tasks"] = len(idx)
    v5[key]["unmatched"] = miss
    for i, m in enumerate(SM):
        id_mean = float(id_p[i][idx].mean()) if idx else None
        ood = float(SA[sname]["p"][i].mean())
        n = 50 * len(names)
        k_id = int(round(id_p[i][idx].sum() * 0.5)) if idx else 0
        k_ood = int(round(SA[sname]["p"][i].sum() * 0.5))
        v5[key][m] = {"fig3_id": figv[m], "recomputed_id": round(id_mean, 1) if id_mean is not None else None,
                      "ood": round(ood, 1), "gap": round(ood - id_mean, 1) if id_mean is not None else None,
                      "unpaired_p": round(two_prop_p(k_ood, n, k_id, n), 4) if idx else None}
OUT["V5_fig3_id_counterparts"] = v5

# ---------------------------------------------------------------- V6 GPT 과제 유형 (표 11)
def bi_type(name):
    nm = name.lower()
    if "handoff" in nm:
        return "handoff"
    if nm.startswith("open ") or "and close it" in nm:
        return "articulated"
    return "pick_place"


idn = BA["ID"]["names"]
types_id = [bi_type(n) for n in idn]
cnt_types_id = {t: types_id.count(t) for t in ("pick_place", "handoff", "articulated")}
gpt = BM.index("GPT")
v6 = {"id_type_counts": cnt_types_id}
for t in cnt_types_id:
    ix = [j for j, tt in enumerate(types_id) if tt == t]
    k = int(round(BA["ID"]["p"][gpt][ix].sum() / 10))
    v6[f"gpt_{t}"] = {"k": k, "n": 10 * len(ix), "pct": round(100 * k / (10 * len(ix)), 1)}
v6["paper_claims_pct"] = {"pick_place": 64.1, "handoff": 26.5, "articulated": 13.1}
OUT["V6_gpt_task_types"] = v6

# ---------------------------------------------------------------- V13 유형 보정 (양팔 범주 묶음, 단일 팔 묶음)
v13 = {"bimanual": {}, "single": {}}
for s in ("Spatial", "Instance", "Composite", "Category"):
    tp = [bi_type(n) for n in BA[s]["names"]]
    mix = {t: tp.count(t) for t in ("pick_place", "handoff", "articulated")}
    v13["bimanual"][s] = {"mix": mix, "per_method": {}}
    for i, m in enumerate(BM):
        exp = 0.0
        for t in mix:
            if mix[t] == 0:
                continue
            ix = [j for j, tt in enumerate(types_id) if tt == t]
            exp += mix[t] / len(tp) * BA["ID"]["p"][i][ix].mean()
        obs = BA[s]["p"][i].mean()
        v13["bimanual"][s]["per_method"][m] = {"id_mix_adjusted_expected": round(float(exp), 1), "observed": round(float(obs), 1),
                                               "table2_id": T2["BiMESA-ID"][m],
                                               "gap_vs_expected": round(float(obs - exp), 1)}


def mesa_type(name):
    nm = name.lower()
    if nm.startswith("open ") and "close it" in nm:
        return "open_place_close"
    if nm.startswith("open "):
        return "open_place"
    if "and close it" in nm:
        return "place_close"
    if " and " in nm.replace("and close it", "") and " in the " not in nm.split(" and ")[0] and not nm.startswith("put the") is False and re.search(r"put the \w+( \w+)? and ", nm):
        return "two_step_pp"
    return "pick_place"


id_t = [mesa_type(n) for n in SA["ID"]["names"]]
OUT["V13_single_id_type_counts"] = {t: id_t.count(t) for t in sorted(set(id_t))}
for s in ("Spatial", "Instance", "Composite", "Category"):
    tp = [mesa_type(n) for n in SA[s]["names"]]
    mix = {t: tp.count(t) for t in sorted(set(tp))}
    v13["single"][s] = {"mix": mix, "per_method": {}}
    covered = all(t in id_t for t in mix)
    v13["single"][s]["all_types_in_id"] = covered
    if not covered:
        continue
    for i, m in enumerate(SM):
        exp = 0.0
        for t, c in mix.items():
            ix = [j for j, tt in enumerate(id_t) if tt == t]
            exp += c / len(tp) * SA["ID"]["p"][i][ix].mean()
        obs = SA[s]["p"][i].mean()
        v13["single"][s]["per_method"][m] = {"id_mix_adjusted_expected": round(float(exp), 1), "observed": round(float(obs), 1),
                                             "table2_id": T2["MESA-ID"][m], "gap_vs_expected": round(float(obs - exp), 1)}
OUT["V13_type_adjusted"] = v13

# ---------------------------------------------------------------- V7 보조 데이터 공동학습 (표 4 = 표 20)
T20 = {
    "DP": {"base": [1.6, 1.0, 1.9, 0.3, 0.7], "aux": [2.3, 2.4, 2.2, 0.6, 0.3], "p": [0.03, 0.07, 0.76, 0.51, 0.34]},
    "pi0": {"base": [43.1, 42.2, 34.0, 27.5, 11.6], "aux": [43.4, 54.4, 33.2, 28.8, 8.7], "p": [0.79, 0.001, 0.68, 0.44, 0.02]},
    "pi0.5": {"base": [62.8, 68.6, 55.4, 46.7, 26.4], "aux": [59.0, 69.2, 53.6, 45.6, 23.0], "p": [0.001, 0.86, 0.34, 0.55, 0.04]},
}
v7 = {}
n_sig = 0
for m, d in T20.items():
    v7[m] = {}
    for j, s in enumerate(SUITES):
        n = 50 * S_NTASK[s]
        k0 = round(d["base"][j] * n / 100)
        k1 = round(d["aux"][j] * n / 100)
        p_unp = two_prop_p(k1, n, k0, n)
        v7[m][s] = {"base": d["base"][j], "aux": d["aux"][j], "delta": round(d["aux"][j] - d["base"][j], 1),
                    "paper_mcnemar_p": d["p"][j], "unpaired_z_p": round(p_unp, 4),
                    "significant_at_0.05": d["p"][j] <= 0.05,
                    "significant_bonferroni_15": d["p"][j] <= 0.05 / 15}
        n_sig += int(d["p"][j] <= 0.05)
v7["n_tests"] = 15
v7["n_significant_0.05"] = n_sig
v7["n_significant_bonferroni"] = sum(1 for m in T20 for s in SUITES if v7[m][s]["significant_bonferroni_15"])
v7["pi0.5_nonsignificance_claim_vs_table"] = ("본문 §5.4 'For pi0.5, no statistically significant changes are observed across the five suites' "
                                              "vs 표 20: pi0.5 ID p<0.001, Category p=0.04")
OUT["V7_aux_cotraining"] = v7

# ---------------------------------------------------------------- V8 RoboArena 순위
ELO = {"PG-Bin": 771, "PG-FM": 1527, "pi0": 1449, "FAST": 1564, "pi0.5": 1594}
MESA5 = {"PG-FM": 22.8, "pi0": 31.7, "FAST": 35.3, "pi0.5": 52.0}
BIM5 = {"PG-Bin": 18.3, "PG-FM": 30.4, "pi0": 29.6, "FAST": 39.4, "pi0.5": 48.7}


def rank_stats(a, b):
    ks = list(a)
    x = np.array([a[k] for k in ks])
    y = np.array([b[k] for k in ks])
    tau = stats.kendalltau(x, y)
    rho = stats.spearmanr(x, y)
    # 정확 순열 검정(일측: tau >= 관측)
    obs = tau.statistic
    perm = []
    from itertools import permutations
    for p in permutations(range(len(ks))):
        perm.append(stats.kendalltau(x, y[list(p)]).statistic)
    perm = np.array(perm)
    p_exact_one = float((perm >= obs - 1e-12).mean())
    return {"n": len(ks), "kendall_tau": round(float(tau.statistic), 3), "spearman": round(float(rho.statistic), 3),
            "perm_p_one_sided": round(p_exact_one, 4), "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 3)}


v8 = {"single_vs_roboarena": rank_stats({k: ELO[k] for k in MESA5}, MESA5),
      "bimanual_vs_roboarena": rank_stats({k: ELO[k] for k in BIM5}, BIM5)}
b_no_bin = {k: v for k, v in BIM5.items() if k != "PG-Bin"}
v8["bimanual_without_PGBin"] = rank_stats({k: ELO[k] for k in b_no_bin}, b_no_bin)
# PG-FM 대 pi0 (양팔 평균): 29.6 [28.7,30.5] vs 30.4 [29.5,31.3]
n_b = 50 * sum(B_NTASK.values())
k_fm = round(30.4 / 100 * n_b)
k_p0 = round(29.6 / 100 * n_b)
v8["bimanual_PGFM_vs_pi0_gap_pts"] = 0.8
v8["bimanual_PGFM_vs_pi0_unpaired_p_pooled_approx"] = round(two_prop_p(k_fm, n_b, k_p0, n_b), 3)
v8["elo_gap_PGFM_pi0"] = ELO["PG-FM"] - ELO["pi0"]
OUT["V8_roboarena_rank"] = v8

# ---------------------------------------------------------------- V9 개수 일관성
v9 = {}
v9["single_rollouts_per_model"] = {s: 50 * S_NTASK[s] for s in SUITES}
v9["bimanual_rollouts_per_model"] = {s: 50 * B_NTASK[s] for s in SUITES}
v9["gpt_rollouts"] = {s: 10 * B_NTASK[s] for s in SUITES}
v9["gpt_total_rollouts"] = 10 * sum(B_NTASK.values())
v9["tasks_1021"] = {"single_training_806": 70 + 736, "single_ood_eval": 10 + 20 + 20 + 20, "bimanual": sum(B_NTASK.values()),
                    "sum": 70 + 736 + 70 + sum(B_NTASK.values())}
v9["bimanual_demos"] = {"4690+843+1300": 4690 + 843 + 1300, "tasks 27+17+13": 27 + 17 + 13, "4681+2152": 4681 + 2152}
v9["bimanual_teleop_cross_pp_demos"] = 2152 - 843
v9["bimanual_mg_aligned_pp_demos"] = 4690 - (2152 - 843)
v9["single_demos_80k"] = 806 * 100
v9["aux_added_demos"] = 736 * 10
v9["bimanual_demos_per_task"] = round(6833 / 57, 1)
v9["spatial_38"] = 27 + 8 + 3
v9["instance_18"] = 9 + 5 + 4
v9["category_17"] = 10 + 3 + 4
v9["composite_15"] = 5 + 3 + 4 + 3
v9["composite_single_20"] = 12 + 8
# 표 12(BiMESA-Spatial) 유형 분포
tp12 = [("handoff" if "handoff" in n else "articulated" if ("fixture" in n) else "pick_place") for n in BA["Spatial"]["names"]]
v9["table12_types"] = {t: tp12.count(t) for t in set(tp12)}
tp15 = [bi_type(n) for n in BA["Category"]["names"]]
v9["table15_types"] = {t: tp15.count(t) for t in set(tp15)}
tp13 = [bi_type(n) for n in BA["Instance"]["names"]]
v9["table13_types"] = {t: tp13.count(t) for t in set(tp13)}
tp14 = [bi_type(n) for n in BA["Composite"]["names"]]
v9["table14_types"] = {t: tp14.count(t) for t in set(tp14)}
OUT["V9_counts"] = v9

# ---------------------------------------------------------------- V10 과제 부트스트랩 (짝지은, 과제 단위)
rng = np.random.default_rng(0)
B = 4000


def boot_suite(P, models, names_ord):
    """P: (M, T). 과제를 복원추출해 인접 순위 쌍의 순서 유지 확률과 평균의 95% 구간."""
    M, T = P.shape
    idx = rng.integers(0, T, size=(B, T))
    means = np.stack([P[:, ix].mean(axis=1) for ix in idx])  # B x M
    ci = np.percentile(means, [2.5, 97.5], axis=0)
    order = np.argsort(-P.mean(axis=1))
    pairs = []
    for a, b in zip(order[:-1], order[1:]):
        keep = float((means[:, a] > means[:, b]).mean())
        pairs.append({"upper": models[a], "lower": models[b], "gap": round(float(P[a].mean() - P[b].mean()), 1),
                      "p_order_kept": round(keep, 3)})
    return {"task_ci": {models[i]: [round(float(ci[0, i]), 1), round(float(ci[1, i]), 1)] for i in range(M)}, "adjacent": pairs}


v10 = {"single": {}, "bimanual": {}}
n_adj_unsure = {"single_ood": 0, "single_total": 0, "bi_ood": 0, "bi_total": 0}
for s in SUITES:
    v10["single"][s] = boot_suite(SA[s]["p"], SM, None)
    v10["bimanual"][s] = boot_suite(BA[s]["p"][:7], BM[:7], None)  # GPT(10회)는 제외
    for a in v10["single"][s]["adjacent"]:
        n_adj_unsure["single_total"] += 1
        if a["p_order_kept"] < 0.975:
            n_adj_unsure["single_ood"] += 1
    for a in v10["bimanual"][s]["adjacent"]:
        n_adj_unsure["bi_total"] += 1
        if a["p_order_kept"] < 0.975:
            n_adj_unsure["bi_ood"] += 1
v10["adjacent_pairs_not_kept_at_97.5pct"] = n_adj_unsure


def boot_average(A, models, ntask, pairs):
    """묶음별로 과제를 복원추출한 뒤 5개 묶음 단순 평균. 평균 행의 쌍 비교가 과제 표집에 얼마나 안정적인가."""
    res = {}
    avgs = np.zeros((B, len(models)))
    for s in SUITES:
        P = A[s]["p"][: len(models)]
        T = P.shape[1]
        idx = rng.integers(0, T, size=(B, T))
        avgs += np.stack([P[:, ix].mean(axis=1) for ix in idx]) / 5.0
    for a, b in pairs:
        ia, ib = models.index(a), models.index(b)
        res[f"{a}>{b}"] = {"obs_gap": round(float(np.mean([A[s]["p"][ia].mean() - A[s]["p"][ib].mean() for s in SUITES])), 2),
                           "p_order_kept": round(float((avgs[:, ia] > avgs[:, ib]).mean()), 3)}
    return res


v10["average_row_pairs"] = {
    "single": boot_average(SA, SM, S_NTASK, [("pi0.5", "FAST"), ("FAST", "pi0"), ("pi0", "PG-FM"), ("PG-FM", "GR00T"), ("pi0", "GR00T")]),
    "bimanual": boot_average(BA, BM[:7], B_NTASK, [("pi0.5", "FAST"), ("FAST", "GR00T"), ("GR00T", "PG-FM"), ("PG-FM", "pi0"), ("GR00T", "pi0")]),
}
OUT["V10_task_bootstrap"] = v10

# ---------------------------------------------------------------- V11 인접 순위 쌍: 비짝지음 z 검정 근사
v11 = {"single": {}, "bimanual": {}}
cnt = {"single_ood_ns": 0, "bi_ood_ns": 0, "single_id_ns": 0, "bi_id_ns": 0, "near_tie_single": 0, "near_tie_bi": 0}
for s in SUITES:
    for tag, A, ms, nmap, rows in (("single", SA, SM, S_NTASK, 50), ("bimanual", BA, BM[:7], B_NTASK, 50)):
        P = A[s]["p"][: len(ms)]
        n = 50 * nmap[s]
        means = P.mean(axis=1)
        order = np.argsort(-means)
        res = []
        for a, b in zip(order[:-1], order[1:]):
            ka = int(round(P[a].sum() / 2))
            kb = int(round(P[b].sum() / 2))
            p = two_prop_p(ka, n, kb, n)
            res.append({"pair": f"{ms[a]}>{ms[b]}", "gap": round(float(means[a] - means[b]), 1), "unpaired_p": round(p, 3)})
            if p >= 0.05:
                if s == "ID":
                    cnt[f"{'single' if tag=='single' else 'bi'}_id_ns"] += 1
                else:
                    cnt[f"{'single' if tag=='single' else 'bi'}_ood_ns"] += 1
            if s != "ID" and means[a] - means[b] <= 1.6:
                cnt["near_tie_single" if tag == "single" else "near_tie_bi"] += 1
        v11[tag][s] = res
v11["counts"] = cnt
OUT["V11_adjacent_pairs"] = v11

# ---------------------------------------------------------------- V12 그림 5 (BiMESA-Spatial 대 ID 대응)
FIG5 = {  # (ID, OOD) 논문 그림 5 라벨, 방법 순서 DP, PG-Bin, PG-FM, GR00T, pi0, FAST, pi0.5
    "arms": [(19.9, 21.9), (45.0, 36.7), (67.9, 61.9), (70.1, 54.5), (62.1, 57.9), (82.5, 70.0), (87.7, 81.6)],
    "handoff": [(10.8, 15.5), (16.0, 19.8), (42.2, 32.2), (37.2, 34.5), (45.8, 37.2), (22.0, 36.2), (61.3, 55.8)],
    "articulated": [(0.0, 0.0), (0.0, 0.7), (0.0, 0.0), (15.3, 2.0), (17.3, 3.3), (35.3, 9.3), (40.0, 10.0)],
}


def stem(n):
    return re.sub(r"\s*\([^)]*\)\s*$", "", n).strip()


id_by_stem = {}
for j, n in enumerate(BA["ID"]["names"]):
    id_by_stem.setdefault(stem(n), []).append(j)
sp_names = BA["Spatial"]["names"]
groups = {"arms": [], "handoff": [], "articulated": []}
unmatched12 = []
for j, n in enumerate(sp_names):
    if "handoff" in n:
        g = "handoff"
    elif "fixture" in n:
        g = "articulated"
    else:
        g = "arms"
    st = stem(n)
    cands = id_by_stem.get(st, [])
    if g == "handoff":
        cands = [c for c in cands if "handoff" in BA["ID"]["names"][c]]
    elif g == "arms":
        cands = [c for c in cands if "handoff" not in BA["ID"]["names"][c]]
    else:
        # 표 12 articulated 행의 이름은 줄바꿈 때문에 괄호가 중간에 있을 수 있어 stem을 다시 만든다
        st2 = re.sub(r"\s*\([^)]*fixture\)", "", n).strip()
        cands = [c for c, nm in enumerate(BA["ID"]["names"]) if nm == st2]
    if len(cands) == 1:
        groups[g].append((j, cands[0]))
    else:
        unmatched12.append((n, len(cands)))
v12 = {"unmatched": unmatched12, "group_sizes": {g: len(v) for g, v in groups.items()}, "per_group": {}}
for g, pairs in groups.items():
    if not pairs:
        continue
    oj = [p[0] for p in pairs]
    ij = [p[1] for p in pairs]
    arr = []
    for i, m in enumerate(BM[:7]):
        ood = float(BA["Spatial"]["p"][i][oj].mean())
        idv = float(BA["ID"]["p"][i][ij].mean())
        arr.append({"method": m, "id": round(idv, 1), "ood": round(ood, 1), "fig5": FIG5[g][i],
                    "err_id": round(idv - FIG5[g][i][0], 1), "err_ood": round(ood - FIG5[g][i][1], 1)})
    v12["per_group"][g] = arr
OUT["V12_fig5"] = v12

# 그림 5 개수: 조 크기와 n
OUT["V12_fig5_n"] = {"arms": 27 * 50, "handoff": 8 * 50, "articulated": 3 * 50}

# ---------------------------------------------------------------- V14 그림 4, 7, 8
FIG4 = {  # (10, 50, 100) 시연/과제
    "ID": {"pi0.5": (53.5, 64.0, 62.8), "pi0": (8.8, 38.3, 43.1)},
    "Spatial": {"pi0.5": (61.6, 70.6, 68.6), "pi0": (4.6, 36.4, 42.2)},
    "Instance": {"pi0.5": (47.3, 51.3, 55.4), "pi0": (6.4, 27.1, 34.0)},
    "Composite": {"pi0.5": (40.0, 44.7, 46.7), "pi0": (5.3, 23.0, 27.5)},
    "Category": {"pi0.5": (32.1, 28.5, 26.4), "pi0": (8.7, 11.9, 11.6)},
}
v14 = {"fig4_pi05_10_vs_pi0_100": {}, "fig4_pi05_category_trend": {}}
for s in SUITES:
    n = 50 * S_NTASK[s]
    a = FIG4[s]["pi0.5"][0]
    b = FIG4[s]["pi0"][2]
    v14["fig4_pi05_10_vs_pi0_100"][s] = {"pi05_10": a, "pi0_100": b, "gap": round(a - b, 1),
                                         "unpaired_p": float(f"{two_prop_p(round(a*n/100), n, round(b*n/100), n):.2g}")}
n = 1000
v14["fig4_pi05_category_trend"] = {"10": 32.1, "50": 28.5, "100": 26.4,
                                   "p_10_vs_100_unpaired": round(two_prop_p(round(32.1 * n / 100), n, round(26.4 * n / 100), n), 4),
                                   "p_10_vs_50_unpaired": round(two_prop_p(round(32.1 * n / 100), n, round(28.5 * n / 100), n), 4)}
# 단조 증가가 아닌 (묶음, 모델) 개수
nonmono = []
for s in SUITES:
    for m, v in FIG4[s].items():
        if not (v[0] <= v[1] <= v[2]):
            nonmono.append(f"{s}/{m}: {v}")
v14["fig4_non_monotone"] = nonmono
# 그림 8b: 50회 롤아웃 가정 (46.0 = 23/50, 40.0 = 20/50, 28.0 = 14/50)
v14["fig8b_fisher"] = {"orig_vs_noRPY(23/50 vs 20/50)": round(stats.fisher_exact([[23, 27], [20, 30]])[1], 3),
                       "orig_vs_noOpt(23/50 vs 14/50)": round(stats.fisher_exact([[23, 27], [14, 36]])[1], 3)}
# 그림 6(a): DP 단일 과제 100/500/1000 시연 = 12/40/46 %
v14["fig6a_fisher_12_vs_46"] = round(stats.fisher_exact([[6, 44], [23, 27]])[1], 5)
# 그림 7: 무작위 지시문 pi0.5
FIG7 = {"ID": 1.6, "Spatial": 3.5, "Instance": 2.8, "Composite": 0.0, "Category": 1.0}
v14["fig7_random_instruction"] = {s: {"orig": T2["MESA-" + s]["pi0.5"], "random": FIG7[s]} for s in SUITES}
# 무작위 지시문이 우연히 맞을 확률(과제 140개 중 1개, 지시문 중복은 무시)
v14["fig7_chance_match"] = round(100 / 140, 2)
OUT["V14_figs"] = v14

# ---------------------------------------------------------------- 추가: 인접 순위 이외 계산 (표 1/F.3 비용)
f3 = {"openpi_runs_H100_hours": 21 * 2, "openpi_runs_L40S_hours": 48 * 4, "gr00t_bs128_H100h": 12, "gr00t_bs1024_H100h": 30 * 4}
single_runs = 4  # PG-FM, pi0, FAST, pi0.5 (openpi)
bi_runs = 5      # PG-Bin, PG-FM, pi0, FAST, pi0.5
f3["train_H100h_openpi_main"] = (single_runs + bi_runs) * 42
f3["train_H100h_gr00t_main"] = 120 + 12
f3["train_H100h_total_without_DP_aux_scaling"] = f3["train_H100h_openpi_main"] + f3["train_H100h_gr00t_main"]
f3["eval_rollouts_main"] = {"single_6_models": 6 * 7000, "bimanual_7_models": 7 * 7250, "gpt": 1450}
f3["eval_rollouts_main"]["total"] = sum(f3["eval_rollouts_main"].values())
OUT["F3_compute_estimates"] = f3

# ---------------------------------------------------------------- V15 MESA-Instance: 실제 자산 교체(복숭아, 호박, 와인) 대 AI 생성 자산
real_kw = ("peach", "squash", "wine")
inst_names = SA["Instance"]["names"]
real_ix = [j for j, n in enumerate(inst_names) if any(k in n for k in real_kw)]
ai_ix = [j for j in range(len(inst_names)) if j not in real_ix]
id_ix_real = [find_id(inst_names[j])[0] for j in real_ix]
id_ix_ai = [find_id(inst_names[j])[0] for j in ai_ix]
v15 = {"n_real_asset_tasks": len(real_ix), "n_ai_asset_tasks": len(ai_ix), "per_method": {}}
for i, m in enumerate(SM):
    g_real = SA["Instance"]["p"][i][real_ix].mean() - id_p[i][id_ix_real].mean()
    g_ai = SA["Instance"]["p"][i][ai_ix].mean() - id_p[i][id_ix_ai].mean()
    v15["per_method"][m] = {"id_real": round(float(id_p[i][id_ix_real].mean()), 1), "ood_real": round(float(SA["Instance"]["p"][i][real_ix].mean()), 1),
                            "gap_real": round(float(g_real), 1),
                            "id_ai": round(float(id_p[i][id_ix_ai].mean()), 1), "ood_ai": round(float(SA["Instance"]["p"][i][ai_ix].mean()), 1),
                            "gap_ai": round(float(g_ai), 1)}
OUT["V15_instance_real_vs_ai_assets"] = v15

# ---------------------------------------------------------------- 요약 출력
Path(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
print("tables parsed rows:", N_ROWS)
print("V1 max abs diff:", v1["max_abs_diff"], "cells>0.06:", v1["n_cells_over_0.06"])
print("V3 worst Wilson err:", v3["worst_abs_err"], "cells>0.15:", v3["n_over_0.15"])
print("V4 cells w/o consistent integer denominator:", v4["n_cells_without_consistent_integer_denominator"])
print("V6", v6)
print("V7 n_sig:", v7["n_significant_0.05"], "bonf:", v7["n_significant_bonferroni"])
print("V8", v8)
print("V10 not kept:", n_adj_unsure)
print("V11 counts:", cnt)
print("V13 bimanual Category:", {m: v13["bimanual"]["Category"]["per_method"][m] for m in ("pi0.5", "GPT", "GR00T")})
print("V14", v14["fig4_pi05_category_trend"], v14["fig8b_fisher"])
