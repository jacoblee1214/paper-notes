"""DexJoCo 논문 표 수치 재계산 (CPU, 표준 라이브러리만 사용).

입력: paper/paper.txt (v2, pdftotext -layout), paper/paper_v1.txt (v1)
출력: analysis/recompute_results.json

검사 항목
- 표 값 파싱 후 success 수 k = mean * 150 / 100 가 정수에 가까운지 (3 seed x 50 episode)
- Avg 행이 11개 task 평균과 맞는지
- 셀의 ± 값이 seed 3개의 표준편차로 가능한지 (ddof 0/1)
- Wilson 95% CI, 두 비율 Fisher 정확검정, task 단위 sign test
- v1 -> v2 수치 변경 목록
"""
import itertools
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_V2 = ROOT / "paper" / "paper.txt"
PAPER_V1 = ROOT / "paper" / "paper_v1.txt"
OUT = ROOT / "analysis" / "recompute_results.json"

TASKS = [
    "Hammer Nail", "Click Mouse", "Pick Bucket", "Pinch Tongs", "Fold Glasses",
    "Water Plant", "Unlock iPad /B", "Hanoi /B", "Assembly /B", "Microwave /B",
    "Photograph /B",
]
TOK = re.compile(r"(\d+\.\d)±(\d+\.\d)")
N_SEEDS, N_EP = 3, 50
TRIALS = N_SEEDS * N_EP  # 150 per task per cell


def parse_rows(text, n_tokens, name_list, occurrence):
    """이름으로 시작하는 줄에서 ± 토큰 n_tokens개를 뽑는다. occurrence번째 블록 사용."""
    rows = {}
    seen = {}
    for line in text.splitlines():
        s = line.strip()
        for name in name_list:
            if s.startswith(name + " ") or s.startswith(name + "\t"):
                toks = TOK.findall(s)
                if len(toks) == n_tokens:
                    seen.setdefault(name, []).append(toks)
    for name in name_list:
        lst = seen.get(name, [])
        if len(lst) > occurrence:
            rows[name] = [(float(m), float(s)) for m, s in lst[occurrence]]
    return rows


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def fisher_two_sided(k1, n1, k2, n2):
    """두 이항 비율의 Fisher 정확검정 (양측). 초기하분포 기반."""
    K = k1 + k2
    N = n1 + n2

    def pmf(x):
        return math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)

    p_obs = pmf(k1)
    lo, hi = max(0, K - n2), min(K, n1)
    total = 0.0
    for x in range(lo, hi + 1):
        px = pmf(x)
        if px <= p_obs * (1 + 1e-9):
            total += px
    return min(1.0, total)


def sign_test(pairs):
    """task 단위 쌍 비교의 양측 부호검정 (동점 제외)."""
    pos = sum(1 for a, b in pairs if a > b)
    neg = sum(1 for a, b in pairs if a < b)
    n = pos + neg
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return {"pos": pos, "neg": neg, "ties": len(pairs) - n, "p_two_sided": min(1.0, p)}


def seed_std_feasible(k_total, mean_pct, std_pct):
    """seed 3개(각 50 episode) 성공 수 (c1,c2,c3), sum=k_total 중 std가 보고값과 0.05 이내인 조합이 있는지."""
    res = {"ddof0": False, "ddof1": False}
    for c in itertools.product(range(N_EP + 1), repeat=N_SEEDS):
        if sum(c) != k_total:
            continue
        r = [100 * x / N_EP for x in c]
        m = sum(r) / 3
        ss = sum((x - m) ** 2 for x in r)
        s0 = math.sqrt(ss / 3)
        s1 = math.sqrt(ss / 2)
        if abs(s0 - std_pct) <= 0.05:
            res["ddof0"] = True
        if abs(s1 - std_pct) <= 0.05:
            res["ddof1"] = True
    return res


def language_probe():
    """부록 Fig. 7 행렬(입력 지시문 x 출력 가능 행동, 논문 텍스트에서 옮김)으로 JS, NMI, 카이제곱을 재계산."""
    rows = {
        "1": [0.100, 0.353, 0.307, 0.060, 0.053, 0.127],
        "2": [0.020, 0.300, 0.360, 0.067, 0.067, 0.187],
        "4": [0.053, 0.353, 0.193, 0.113, 0.100, 0.187],
        "1+1": [0.067, 0.360, 0.253, 0.027, 0.073, 0.220],
        "2+2": [0.053, 0.353, 0.320, 0.060, 0.020, 0.193],
        "two": [0.060, 0.367, 0.293, 0.053, 0.107, 0.120],
        "one plus one": [0.087, 0.380, 0.293, 0.020, 0.067, 0.153],
    }
    names = list(rows)
    K = len(names)

    def kl(p, q, base):
        return sum(a * math.log(a / b, base) for a, b in zip(p, q) if a > 0)

    def js(p, q):
        m = [(a + b) / 2 for a, b in zip(p, q)]
        return 0.5 * kl(p, m, 2) + 0.5 * kl(q, m, 2)

    pairs = [(js(rows[a], rows[b]), a, b) for a, b in itertools.combinations(names, 2)]
    mean_js = sum(v for v, _, _ in pairs) / len(pairs)
    max_js = max(pairs)
    marg = [sum(rows[n][j] for n in names) / K for j in range(6)]
    hx = math.log2(K)
    hy = -sum(p * math.log2(p) for p in marg if p > 0)
    mi = 0.0
    for n in names:
        for j in range(6):
            pxy = rows[n][j] / K
            if pxy > 0:
                mi += pxy * math.log2(pxy / ((1 / K) * marg[j]))
    chi = sum((150 * rows[n][j] - 150 * marg[j]) ** 2 / (150 * marg[j]) for n in names for j in range(6))
    return {
        "mean_js_log2": round(mean_js, 4),
        "max_js_log2": [round(max_js[0], 4), max_js[1], max_js[2]],
        "nmi_sqrt": round(mi / math.sqrt(hx * hy), 4),
        "chi2_n150_per_row": round(chi, 1),
        "chi2_df": (K - 1) * 5,
        "note": "행 n=150 가정. 논문의 p=6.37e-3과 비교용",
    }


def main():
    v2 = PAPER_V2.read_text(encoding="utf-8")
    v1 = PAPER_V1.read_text(encoding="utf-8")
    models2 = ["DP-T rand-obj", "DP-T rand-full", "DP-C rand-obj", "DP-C rand-full",
               "ACT rand-obj", "ACT rand-full", "pi05 rand-obj", "pi05 rand-full",
               "GR00T rand-obj", "GR00T rand-full"]
    t2 = parse_rows(v2, 10, TASKS + ["Avg."], 0)
    assert len(t2) == 12, f"Table II parse: {len(t2)} rows"
    t4 = parse_rows(v2, 5, ["Hammer Nail", "Click Mouse", "Pick Bucket", "Pinch Tongs",
                            "Fold Glasses", "Water Plant", "Unlock iPad /B", "Hanoi /B",
                            "Assembly /B", "Microwave /B", "Photograph /B", "Avg."], 0)
    assert len(t4) == 12, f"Table IV parse: {len(t4)} rows"
    t2v1 = parse_rows(v1, 10, TASKS + ["Avg."], 0)
    t4v1 = parse_rows(v1, 5, ["Hammer Nail", "Click Mouse", "Pick Bucket", "Pinch Tongs",
                              "Fold Glasses", "Water Plant", "Unlock iPad /B", "Hanoi /B",
                              "Assembly /B", "Microwave /B", "Photograph /B", "Avg."], 0)

    out = {"trials_per_cell": TRIALS, "checks": {}}

    # 1) 셀별 k 정수성, Avg 재계산, seed-std 가능성
    k_bad, feas = [], {"ddof0": 0, "ddof1": 0, "n": 0}
    cell_rows = []
    for ci, name in enumerate(models2):
        for ti, task in enumerate(TASKS):
            m, s = t2[task][ci]
            # 보고값은 소수 1자리 반올림이므로 k/150 이 m 과 0.05 %p 이내인 정수 k 를 찾는다
            cands = [kk for kk in range(TRIALS + 1) if abs(100 * kk / TRIALS - m) <= 0.05 + 1e-9]
            if not cands:
                k_bad.append((name, task, m))
                continue
            kr = cands[0]
            f = seed_std_feasible(kr, m, s)
            feas["n"] += 1
            feas["ddof0"] += f["ddof0"]
            feas["ddof1"] += f["ddof1"]
            cell_rows.append({"model": name, "task": task, "mean": m, "std": s, "k": kr,
                              "feasible_ddof0": f["ddof0"], "feasible_ddof1": f["ddof1"]})
    out["checks"]["non_integer_k"] = k_bad
    out["checks"]["seed_std_feasibility"] = feas

    avg_check = []
    for ci, name in enumerate(models2):
        vals = [t2[t][ci][0] for t in TASKS]
        mu = sum(vals) / len(vals)
        rep_m, rep_s = t2["Avg."][ci]
        avg_check.append({"model": name, "mean_of_task_means": round(mu, 2), "reported": rep_m,
                          "diff": round(mu - rep_m, 2),
                          "std_across_tasks": round(math.sqrt(sum((x - mu) ** 2 for x in vals) / 10), 1),
                          "reported_avg_std": rep_s})
    out["checks"]["avg_row"] = avg_check

    # 2) Wilson CI (셀 단위 150 trial) 및 평균행 pooled CI (1650 trial)
    ci_rows = []
    for r in cell_rows:
        lo, hi = wilson(r["k"], TRIALS)
        ci_rows.append({"model": r["model"], "task": r["task"], "mean": r["mean"],
                        "wilson95": [round(lo, 1), round(hi, 1)]})
    out["wilson_cells_sample"] = [x for x in ci_rows if x["task"] == "Unlock iPad /B"]

    # 3) 주요 주장 검정
    def k_of(table, task, ci):
        return round(table[task][ci][0] * TRIALS / 100)

    idx = {n: i for i, n in enumerate(models2)}
    tests = {}
    # Unlock iPad: DP-C rand-obj vs pi05 rand-obj (Fisher, 150 vs 150)
    a = k_of(t2, "Unlock iPad /B", idx["DP-C rand-obj"])
    b = k_of(t2, "Unlock iPad /B", idx["pi05 rand-obj"])
    tests["unlock_ipad_DPC_vs_pi05_rand_obj"] = {"k": [a, b], "p_fisher": round(fisher_two_sided(a, 150, b, 150), 4)}
    # Pinch Tongs: DP-C vs pi05 rand-obj
    a = k_of(t2, "Pinch Tongs", idx["DP-C rand-obj"])
    b = k_of(t2, "Pinch Tongs", idx["pi05 rand-obj"])
    tests["pinch_tongs_DPC_vs_pi05_rand_obj"] = {"k": [a, b], "p_fisher": round(fisher_two_sided(a, 150, b, 150), 4)}
    # pooled over tasks (주의: task 간 독립 가정 위반, 참고용)
    def pooled(ci):
        return sum(k_of(t2, t, ci) for t in TASKS)
    for x, y in [("pi05 rand-obj", "DP-T rand-obj"), ("pi05 rand-full", "DP-T rand-full"),
                 ("pi05 rand-obj", "DP-C rand-obj"), ("GR00T rand-obj", "pi05 rand-obj")]:
        kx, ky = pooled(idx[x]), pooled(idx[y])
        n = TRIALS * len(TASKS)
        tests[f"pooled_{x}_vs_{y}".replace(" ", "_")] = {
            "k": [kx, ky], "rate": [round(100 * kx / n, 2), round(100 * ky / n, 2)],
            "p_fisher_pooled_caveat": round(fisher_two_sided(kx, n, ky, n), 6)}
    # task 단위 sign test (task 11개)
    for x, y in [("pi05 rand-obj", "DP-T rand-obj"), ("pi05 rand-obj", "DP-C rand-obj"),
                 ("pi05 rand-obj", "ACT rand-obj"), ("pi05 rand-full", "DP-T rand-full")]:
        pairs = [(t2[t][idx[x]][0], t2[t][idx[y]][0]) for t in TASKS]
        tests[f"sign_task_{x}_vs_{y}".replace(" ", "_")] = sign_test(pairs)
    # 표 IV: rand-AH vs pi05 rand-obj (Avg), 차이 & 근사 SE
    ah = t4["Avg."][4][0]
    base = t2["Avg."][idx["pi05 rand-obj"]][0]
    tests["rand_AH_vs_partial_AH_avg"] = {"rand_AH": ah, "partial_AH": base, "diff": round(ah - base, 2)}
    # 시드 단위 Welch t (± 를 seed 표준편차(ddof=1, n=3)로 가정; 주장 검정의 보조 근거)
    def welch(m1, s1, m2, s2, n=3):
        se2 = s1 * s1 / n + s2 * s2 / n
        t = (m1 - m2) / math.sqrt(se2)
        df = se2 * se2 / ((s1 * s1 / n) ** 2 / (n - 1) + (s2 * s2 / n) ** 2 / (n - 1))
        return {"diff": round(m1 - m2, 2), "t": round(t, 2), "df": round(df, 2)}
    seed_t = {}
    for t_ in ["Unlock iPad /B", "Pinch Tongs"]:
        a_m, a_s = t2[t_][idx["DP-C rand-obj"]]
        b_m, b_s = t2[t_][idx["pi05 rand-obj"]]
        seed_t[f"{t_}_DPC_vs_pi05_rand_obj"] = welch(a_m, a_s, b_m, b_s)
    a_m, a_s = t2["Avg."][idx["pi05 rand-obj"]]
    b_m, b_s = t2["Avg."][idx["DP-T rand-obj"]]
    seed_t["Avg_pi05_vs_DPT_rand_obj"] = welch(a_m, a_s, b_m, b_s)
    a_m, a_s = t2["Avg."][idx["pi05 rand-full"]]
    b_m, b_s = t2["Avg."][idx["DP-T rand-full"]]
    seed_t["Avg_pi05_vs_DPT_rand_full"] = welch(a_m, a_s, b_m, b_s)
    tests["seed_level_welch"] = seed_t
    out["tests"] = tests
    out["language_probe_fig7"] = language_probe()

    # 4) Table III 지연시간 vs 한 action chunk 실행 시간 (30 Hz, chunk 길이는 본문)
    chunk = {"DP-T": 45, "DP-C": 43, "ACT": 100, "pi05": 30, "GR00T": 50}
    lat = {"ACT": (7.0, 7.4), "DP-C": (37.7, 42.3), "DP-T": (33.6, 36.1), "pi05": (85.1, 83.9), "GR00T": (80.0, 90.0)}
    out["latency_vs_chunk_ms"] = {m: {"chunk_exec_ms": round(1000 * chunk[m] / 30, 0),
                                      "latency_1arm_2arm_ms": lat[m]} for m in chunk}

    # 5) v1 -> v2 수치 변경
    changes = []
    for name, rowsv1, rowsv2, cols in [("Table II", t2v1, t2, models2), ("Table IV", t4v1, t4, None)]:
        for task in rowsv2:
            if task not in rowsv1:
                continue
            v1r, v2r = rowsv1[task], rowsv2[task]
            for j, (a1, a2) in enumerate(zip(v1r, v2r)):
                if abs(a1[0] - a2[0]) > 1e-9 or abs(a1[1] - a2[1]) > 1e-9:
                    changes.append({"table": name, "row": task, "col_index": j,
                                    "v1": list(a1), "v2": list(a2)})
    out["v1_to_v2_changed_cells"] = {"n": len(changes), "cells": changes[:200]}

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("non_integer_k:", len(k_bad))
    print("seed-std feasibility:", feas)
    for row in avg_check:
        print(row)
    print("tests:", json.dumps(tests, ensure_ascii=False)[:2000])
    print("v1->v2 changed cells:", len(changes))


if __name__ == "__main__":
    main()
