"""Chronos Policy 수치 검증 (CPU, 수 초).

1. 논문 Table 3·4·7–9를 paper/paper.txt에서 읽고, 프로젝트 페이지 data.js(반올림 전 값)와 대조한다.
2. 본문 수치 주장(§4.2–4.4, B.3)을 표에서 다시 계산한다.
3. 시뮬레이션 표의 mean ± SD에서 시드별 성공 횟수를 역산한다(GRIMMER식).
4. 시드 단위 Welch 검정, 에피소드 단위 검정, 시드 간 과분산, 실제 로봇 permutation 검정.

실행: python3 analysis/table_check.py  → analysis/table_check_results.json
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "paper" / "paper.txt").read_text()
CONDS = ["nominal", "small_obs", "wide_sensors", "delay_100", "delay_200", "combined"]
PM = r"(\d+\.\d)\s*±\s*(\d+\.\d)"


def load_js(name):
    t = (ROOT / "paper" / "site" / name).read_text()
    return json.loads(t[t.index("=") + 1:].strip().rstrip(";"))


# ---------------------------------------------------------------- 1. 표 읽기
def parse_table3():
    block = TXT[TXT.index("Table 3: Full Conveyor"):TXT.index("Table 4: Full results")]
    out, task = {}, None
    for line in block.splitlines():
        s = line.strip()
        if s in ("Conveyor", "Pivoting"):
            task = s.lower()
            out[task] = {}
            continue
        m = re.match(r"(DP \+ rand\.|DP|RoPE w/o rand\.|AddTime|RoPE \(ours\))\s+(.*)", s)
        if task and m:
            vals = re.findall(PM, m.group(2))
            assert len(vals) == 6, line
            out[task][m.group(1)] = [(float(a), float(b)) for a, b in vals]
    return out


def parse_table4():
    block = TXT[TXT.index("Table 4: Full results"):TXT.index("C     A DDITIONAL")]
    out, task = {}, None
    for line in block.splitlines():
        s = line.strip()
        if s in ("Threading", "Assembly", "Transport", "Coffee", "Drawer Cleanup", "Pouring"):
            task = s
            out[task] = {}
            continue
        m = re.match(r"(DP|Chronos Policy)\s+(.*)", s)
        if task and m:
            vals = re.findall(PM, m.group(2))
            assert len(vals) == 6, line
            out[task][m.group(1)] = [(float(a), float(b)) for a, b in vals]
    return out


def parse_real():
    """Table 7–9: {task: {policy: {scenario: [(score, F, P, X) x 4]}}}"""
    out = {}
    for task, start, end in [("dynamic_cup", "Table 7:", "Table 8:"), ("peg", "Table 8:", "Table 9:"),
                             ("pivoting", "Table 9:", "\x0c" * 0 + "                                                         18")]:
        block = TXT[TXT.index(start):TXT.index(end, TXT.index(start) + 10)]
        out[task], pol = {}, None
        for line in block.splitlines():
            cells = re.findall(r"(\d+\.\d) \((\d+)/(\d+)/(\d+)\)", line)
            if len(cells) != 4:
                continue
            head = line[:line.index(cells[0][0])].strip()
            for p in ("DP + RTC", "Chronos Policy", "DP"):
                if head.startswith(p):
                    pol, head = p, head[len(p):].strip()
                    break
            out[task].setdefault(pol, {})[head] = [(float(s), int(f), int(p_), int(x)) for s, f, p_, x in cells]
    return out


t3, t4, real = parse_table3(), parse_table4(), parse_real()
data = load_js("data.js")
res = {"checks": {}, "claims": [], "seeds": {}, "stats": {}}

# ---------------------------------------------------------------- 1b. 웹페이지 대조
T3_SITE = {"DP": "Diffusion Policy", "DP + rand.": "DP + temporal randomization",
           "RoPE w/o rand.": "RoPE without randomization", "AddTime": "Additive time encoding",
           "RoPE (ours)": "Chronos Policy (RoPE)"}
SITE_TASK = {"conveyor": "conveyor", "pivoting": "flipup"}
mism = []
exact = {}  # (task, method) -> [(mean, sd) x 6] 반올림 전 값
for task, rows in t3.items():
    for m, cells in rows.items():
        mean = data["ablation"][SITE_TASK[task]][T3_SITE[m]]
        sd = data["ablationStd"][SITE_TASK[task]][T3_SITE[m]]
        exact[(task, m)] = list(zip(mean, sd))
        for c, (a, b), mu, s in zip(CONDS, cells, mean, sd):
            if abs(round(mu, 1) - a) > 0.051 or abs(round(s, 1) - b) > 0.051:
                mism.append([task, m, c, a, b, mu, s])
SIM_COND = ["nominal", "sensor_id", "sensor_ood", "inference_0_100ms", "inference_0_200ms",
            "combined_sensor_ood_inference_0_200ms"]
site_std = {}
for sec in data["sections"]:
    for task in sec["tasks"]:
        for r in task.get("results", []):
            site_std[(r["task"], r["method"], r["condition"])] = (r["mean"], r["sd"])
T4_KEY = {"Threading": "threading", "Assembly": "assembly", "Transport": "transport", "Coffee": "coffee",
          "Drawer Cleanup": "drawer_cleanup", "Pouring": "pouring"}
T4_M = {"DP": "Diffusion Policy", "Chronos Policy": "Chronos Policy"}
for task, rows in t4.items():
    for m, cells in rows.items():
        ex = []
        for c, (a, b), sc in zip(CONDS, cells, SIM_COND):
            mu, s = site_std[(T4_KEY[task], T4_M[m], sc)]
            ex.append((mu, s))
            if abs(round(mu, 1) - a) > 0.051 or abs(round(s, 1) - b) > 0.051:
                mism.append([task, m, c, a, b, mu, s])
        exact[(task, m)] = ex
res["checks"]["sim_table_vs_site_mismatches"] = mism

# 실제 로봇: 점수 공식, 시나리오 합계, 웹페이지 대조
REAL_SITE = {"dynamic_cup": "real-dynamic-cup", "peg": "real-peg", "pivoting": "real-flip"}
REAL_COND = ["nominal", "obs_change", "inference_latency", "combined"]
POL_KEY = {"DP": "dp", "DP + RTC": "rtc", "Chronos Policy": "chronos"}
real_issues = []
site_real = {}
for sec in data["sections"]:
    for task in sec["tasks"]:
        for c in task["comparisons"]:
            for clip in c["clips"]:
                if "partialRate" in clip and clip.get("partialRate") is not None:
                    site_real[(task["id"], c["condition"], clip["methodKey"])] = clip
for task, pols in real.items():
    for pol, scen in pols.items():
        subs = [v for k, v in scen.items() if k != "Overall"]
        for k, cells in scen.items():
            for ci, (s, f, p, x) in enumerate(cells):
                n = f + p + x
                if abs(100 * (f + 0.5 * p) / n - s) > 0.05:
                    real_issues.append(["score_formula", task, pol, k, ci])
        for ci in range(4):
            tot = [sum(c[ci][j] for c in subs) for j in (1, 2, 3)]
            if tot != list(scen["Overall"][ci][1:]):
                real_issues.append(["overall_sum", task, pol, ci, tot, scen["Overall"][ci]])
            s, f, p, x = scen["Overall"][ci]
            clip = site_real[(REAL_SITE[task], REAL_COND[ci], POL_KEY[pol])]
            n = f + p + x
            if (clip["trials"], clip["score"], round(clip["successRate"], 3), round(clip["partialRate"], 3)) != \
               (n, s, round(100 * f / n, 3), round(100 * p / n, 3)):
                real_issues.append(["site", task, pol, REAL_COND[ci], clip["score"], s])
res["checks"]["real_issues"] = real_issues


# ---------------------------------------------------------------- 2. 본문 주장 재계산
def avg(xs):
    return sum(xs) / len(xs)


def m3(task, m, c):  # Table 3 반올림 값
    return t3[task][m][CONDS.index(c)][0]


def claim(cid, text, stated, computed, tol=0.051):
    ok = all(abs(a - b) <= tol for a, b in zip(np.atleast_1d(stated), np.atleast_1d(computed)))
    res["claims"].append({"id": cid, "text": text, "stated": stated, "computed": computed, "ok": bool(ok)})


pert = CONDS[1:]
for task, cid, pairs in [("conveyor", "B1", [("DP", 6.3), ("DP + rand.", 26.9), ("RoPE w/o rand.", 48.8),
                                             ("RoPE (ours)", 63.1), ("AddTime", 51.5)]),
                         ("pivoting", "B2", [("DP", 48.4), ("DP + rand.", 21.6), ("RoPE w/o rand.", 33.3),
                                             ("RoPE (ours)", 55.3), ("AddTime", 33.5)])]:
    for m, st in pairs:
        comp_r = avg([m3(task, m, c) for c in pert])
        comp_x = avg([mu for mu, _ in exact[(task, m)][1:]])
        claim(f"{cid}-{m}", f"{task} {m} 교란 5조건 평균 (§4.3)", st, round(comp_x, 2))
        res["claims"][-1]["computed_from_rounded"] = round(comp_r, 2)
claim("A1", "Conveyor nominal DP→Chronos 73.1→82.0 (§4.2)", [73.1, 82.0], [m3("conveyor", "DP", "nominal"), m3("conveyor", "RoPE (ours)", "nominal")])
claim("A2", "Conveyor small obs 15.6 vs 77.6", [15.6, 77.6], [m3("conveyor", "DP", "small_obs"), m3("conveyor", "RoPE (ours)", "small_obs")])
claim("A3", "Conveyor delay 0–200 ms 1.8 vs 76.4", [1.8, 76.4], [m3("conveyor", "DP", "delay_200"), m3("conveyor", "RoPE (ours)", "delay_200")])
claim("A4", "Conveyor combined 0 vs 32.7", [0.0, 32.7], [m3("conveyor", "DP", "combined"), m3("conveyor", "RoPE (ours)", "combined")])
claim("A5", "Pivoting delay 0–200 51.3→60.2, combined 19.6→38.4", [51.3, 60.2, 19.6, 38.4],
      [m3("pivoting", "DP", "delay_200"), m3("pivoting", "RoPE (ours)", "delay_200"), m3("pivoting", "DP", "combined"), m3("pivoting", "RoPE (ours)", "combined")])
# AddTime vs RoPE: Conveyor는 nominal만 AddTime 우세, Pivoting은 모든 조건 RoPE 우세
conv_add_better = [c for c in CONDS if m3("conveyor", "AddTime", c) > m3("conveyor", "RoPE (ours)", c)]
piv_add_better = [c for c in CONDS if m3("pivoting", "AddTime", c) >= m3("pivoting", "RoPE (ours)", c)]
claim("A6", "Conveyor에서 AddTime이 RoPE보다 나은 조건은 nominal뿐 (§4.3)", 1, len(conv_add_better) if conv_add_better == ["nominal"] else -1)
claim("A7", "Pivoting에서 RoPE가 AddTime보다 모든 조건에서 나음", 0, len(piv_add_better))
# 표준 6과제
tasks6 = list(t4)
dp_nom = avg([exact[(t, "DP")][0][0] for t in tasks6])
ch_nom = avg([exact[(t, "Chronos Policy")][0][0] for t in tasks6])
dp_p = avg([mu for t in tasks6 for mu, _ in exact[(t, "DP")][1:]])
ch_p = avg([mu for t in tasks6 for mu, _ in exact[(t, "Chronos Policy")][1:]])
claim("A8", "표준 6과제 교란 평균 59.7→73.4 (§4.4)", [59.7, 73.4], [round(dp_p, 2), round(ch_p, 2)])
fig4c = {"DP": [76.4, 71.8, 43.9, 75.3, 68.1, 39.2], "Chronos Policy": [77.1, 75.7, 70.9, 76.0, 76.4, 67.8]}  # Fig. 4c 막대 라벨
for m, vals in fig4c.items():
    comp = [round(avg([exact[(t, m)][i][0] for t in tasks6]), 2) for i in range(6)]
    claim(f"F4c-{m}", f"Fig. 4c 표준 6과제 조건별 평균 ({m})", vals, comp)
rest = [t for t in tasks6 if t not in ("Assembly", "Coffee", "Pouring")]
res["standard"] = {
    "nominal_avg": {"DP": dp_nom, "Chronos": ch_nom},
    "perturbed_avg": {"DP": dp_p, "Chronos": ch_p},
    "perturbed_avg_without_assembly_coffee_pouring": {
        "DP": avg([mu for t in rest for mu, _ in exact[(t, "DP")][1:]]),
        "Chronos": avg([mu for t in rest for mu, _ in exact[(t, "Chronos Policy")][1:]])},
    "cells_chronos_lower": [[t, c, exact[(t, "DP")][i][0], exact[(t, "Chronos Policy")][i][0]]
                            for t in tasks6 for i, c in enumerate(CONDS)
                            if exact[(t, "Chronos Policy")][i][0] < exact[(t, "DP")][i][0] - 1e-9],
}
# 실제 로봇: Chronos가 모든 조건에서 최고 점수인가
best = []
for task, pols in real.items():
    for ci in range(4):
        sc = {p: pols[p]["Overall"][ci][0] for p in pols}
        best.append(sc["Chronos Policy"] > max(v for p, v in sc.items() if p != "Chronos Policy"))
claim("A9", "실제 로봇 12개 조건 모두 Chronos 최고 점수 (§4.5)", 12, sum(best))
rtc_inf = [real[t]["DP + RTC"]["Overall"][2][0] > real[t]["DP"]["Overall"][2][0] for t in real]
claim("A10", "RTC가 추론 지연에서 DP보다 나음 (3과제 중)", 3, sum(rtc_inf))
claim("A11", "B.3: Conveyor DP+rand nominal 73.1→53.6, Pivoting 61.6→29.8",
      [73.1, 53.6, 61.6, 29.8], [m3("conveyor", "DP", "nominal"), m3("conveyor", "DP + rand.", "nominal"),
                                 m3("pivoting", "DP", "nominal"), m3("pivoting", "DP + rand.", "nominal")])


# ---------------------------------------------------------------- 3. 시드별 성공 횟수 역산
def reconstruct(mean, sd, n_ep):
    """시드 3개, 시드당 n_ep 에피소드. mean·sd(%)를 정확히 맞추는 성공 횟수 조합."""
    total = mean / 100 * 3 * n_ep
    if abs(total - round(total)) > 1e-6:
        return None
    total = round(total)
    sols = []
    for a in range(0, n_ep + 1):
        for b in range(a, n_ep + 1):
            c = total - a - b
            if c < b or c > n_ep:
                continue
            v = np.array([a, b, c]) / n_ep * 100
            if abs(v.std(ddof=1) - sd) < 1e-6:
                sols.append([a, b, c])
    return sols


seeds = {}
n_unique = n_multi = n_none = 0
for (task, m), cells in exact.items():
    n_ep = 150 if task in ("conveyor", "pivoting") else 50
    for c, (mu, s) in zip(CONDS, cells):
        sols = reconstruct(mu, s, n_ep)
        seeds[f"{task}|{m}|{c}"] = sols
        if not sols:
            n_none += 1
        elif len(sols) == 1:
            n_unique += 1
        else:
            n_multi += 1
res["seeds"] = {"n_ep_per_seed": {"conveyor/pivoting": 150, "standard": 50},
                "unique": n_unique, "multiple": n_multi, "none": n_none, "solutions": seeds}

# ---------------------------------------------------------------- 4. 통계
st = {}
# 4a. 시드 간 과분산: 관측 SD vs 이항 SD(시드당 n_ep)
od = []
for (task, m), cells in exact.items():
    n_ep = 150 if task in ("conveyor", "pivoting") else 50
    for c, (mu, s) in zip(CONDS, cells):
        p = mu / 100
        if 0.05 < p < 0.95:
            od.append((s / 100) / np.sqrt(p * (1 - p) / n_ep))
st["overdispersion_ratio_median"] = float(np.median(od))
st["overdispersion_ratio_iqr"] = [float(np.percentile(od, 25)), float(np.percentile(od, 75))]


# 4b. Chronos vs DP / AddTime: 시드 단위 Welch(n=3) vs 에피소드 단위 Fisher(450 vs 450)
def welch(m1, s1, m2, s2, n=3):
    return float(stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False).pvalue)


cmp = []
for task in ("conveyor", "pivoting"):
    for other in ("DP", "AddTime", "RoPE w/o rand."):
        for i, c in enumerate(CONDS):
            (m1, s1), (m2, s2) = exact[(task, "RoPE (ours)")][i], exact[(task, other)][i]
            k1, k2 = round(m1 / 100 * 450), round(m2 / 100 * 450)
            cmp.append({"task": task, "vs": other, "cond": c, "chronos": round(m1, 1), "other": round(m2, 1),
                        "welch_p_seed": round(welch(m1, s1, m2, s2), 4),
                        "fisher_p_episode": float(stats.fisher_exact([[k1, 450 - k1], [k2, 450 - k2]])[1])})
st["sim_tests"] = cmp
st["standard_tests"] = [{"task": t, "cond": c,
                         "dp": round(exact[(t, "DP")][i][0], 1), "chronos": round(exact[(t, "Chronos Policy")][i][0], 1),
                         "welch_p_seed": round(welch(*exact[(t, "Chronos Policy")][i], *exact[(t, "DP")][i]), 4)}
                        for t in tasks6 for i, c in enumerate(CONDS)]

# 4c. 실제 로봇: 시행 점수(1, 0.5, 0) permutation 검정
rng = np.random.default_rng(0)


def trials(f, p, x):
    return np.array([1.0] * f + [0.5] * p + [0.0] * x)


def perm_p(groups_a, groups_b, n_perm=20000):
    """조건별 층화 permutation. groups_*: 조건마다 시행 점수 배열. 양측, 평균 차이."""
    obs = np.mean(np.concatenate(groups_a)) - np.mean(np.concatenate(groups_b))
    cnt = 0
    for _ in range(n_perm):
        da, db = [], []
        for a, b in zip(groups_a, groups_b):
            pool = rng.permutation(np.concatenate([a, b]))
            da.append(pool[:len(a)])
            db.append(pool[len(a):])
        if abs(np.mean(np.concatenate(da)) - np.mean(np.concatenate(db))) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1), float(obs * 100)


rt = []
for task, pols in real.items():
    for other in ("DP", "DP + RTC"):
        ch = [trials(*pols["Chronos Policy"]["Overall"][ci][1:]) for ci in range(4)]
        ot = [trials(*pols[other]["Overall"][ci][1:]) for ci in range(4)]
        cells = []
        for ci in range(4):
            p, d = perm_p([ch[ci]], [ot[ci]], 5000)
            cells.append({"cond": REAL_COND[ci], "diff": round(d, 1), "p": round(p, 3)})
        p_all, d_all = perm_p(ch, ot)
        rt.append({"task": task, "vs": other, "pooled_diff": round(d_all, 1), "pooled_p": round(p_all, 4),
                   "n_per_policy": int(sum(len(a) for a in ch)), "cells": cells})
st["real_tests"] = rt
res["stats"] = st

out = ROOT / "analysis" / "table_check_results.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))

# ---------------------------------------------------------------- 요약 출력
print("Table 3/4 vs 웹페이지 불일치:", len(mism))
print("실제 로봇 표 문제:", real_issues or "없음")
bad = [c for c in res["claims"] if not c["ok"]]
print(f"본문 주장 {len(res['claims'])}개 중 불일치 {len(bad)}개", *[(c['id'], c['stated'], c['computed']) for c in bad], sep="\n  ")
print(f"시드 역산: 유일해 {n_unique}, 복수해 {n_multi}, 해 없음 {n_none}")
print("과분산(관측 SD / 이항 SD) 중앙값 %.2f, IQR %s" % (st["overdispersion_ratio_median"], st["overdispersion_ratio_iqr"]))
print("표준 6과제:", json.dumps(res["standard"], ensure_ascii=False, default=lambda x: round(x, 2))[:600])
for r in cmp:
    if r["vs"] == "DP":
        print(f"  {r['task']:9s} {r['cond']:12s} {r['chronos']:5.1f} vs {r['other']:5.1f}  Welch p={r['welch_p_seed']:.3f}  Fisher(ep) p={r['fisher_p_episode']:.1e}")
for r in rt:
    print(f"  real {r['task']:11s} vs {r['vs']:8s} 차이 {r['pooled_diff']:+5.1f}점 p={r['pooled_p']:.4f}", [(c['cond'][:4], c['diff'], c['p']) for c in r['cells']])
