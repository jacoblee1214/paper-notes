"""논문 표 수치를 다시 계산하고 일관성을 검사한다 (numpy, scipy, 수 초).

입력: paper/paper.txt (pdftotext -layout 결과, 원문 v2 기준)
출력: analysis/recompute_results.json (표 재계산, 검정 결과, 일관성 검사)
실행: python3 analysis/recompute.py
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
OUT = ROOT / "analysis" / "recompute_results.json"

METHODS = ["Fast-WAM", "Fast-WAM-Joint", "Fast-WAM-IDM", "w.o.video-co-train",
           "LingBot-VA", "pi0.5", "Motus"]

# ---------- 1. 표 3 (RoboTwin 과제별) 파싱 ----------
lines = TXT.splitlines()
start = next(i for i, l in enumerate(lines) if l.strip().startswith("Adjust Bottle"))
end = next(i for i, l in enumerate(lines) if l.strip().startswith("Turn Switch"))
rows = {}
for l in lines[start:end + 1]:
    m = re.match(r"^\s*([A-Z][A-Za-z0-9 ]+?)\s{2,}((?:\d+(?:\.\d+)?\s+){13}\d+(?:\.\d+)?)\s*$", l)
    if m:
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        vals = [float(v) for v in m.group(2).split()]
        rows[name] = vals
avg_line = next(l for l in lines[end:end + 6] if l.strip().startswith("Average"))
stated_avg = [float(v) for v in re.findall(r"\d+\.\d+|\d+", avg_line)[-14:]]
assert len(rows) == 50, f"expected 50 tasks, parsed {len(rows)}"
assert all(len(v) == 14 for v in rows.values())

M = np.array(list(rows.values()))  # [50, 14]: 7 methods x (clean, rand)
names = list(rows.keys())
cols = {m: (M[:, 2 * i], M[:, 2 * i + 1]) for i, m in enumerate(METHODS)}
res = {"n_tasks": len(names), "table3": {}, "checks": {}}

# 표 3 평균 재계산과 본문 평균 비교 (Average 행)
for i, m in enumerate(METHODS):
    c, r = cols[m]
    res["table3"][m] = {
        "clean_mean": round(float(c.mean()), 2), "rand_mean": round(float(r.mean()), 2),
        "clean_stated": stated_avg[2 * i], "rand_stated": stated_avg[2 * i + 1],
        "avg_of_both": round(float(((c + r) / 2).mean()), 2),
    }

# 표 1 평균: (clean + rand) / 2 와 표기 값 비교
table1_stated = {"Fast-WAM": 91.8, "Fast-WAM-Joint": 90.6, "Fast-WAM-IDM": 91.3,
                 "w.o.video-co-train": 83.8, "LingBot-VA": 92.2, "pi0.5": 79.8, "Motus": 87.8}
table1_pairs = {"Fast-WAM": (91.88, 91.78), "Fast-WAM-Joint": (90.84, 90.32),
                "Fast-WAM-IDM": (91.16, 91.34), "w.o.video-co-train": (82.76, 84.80),
                "LingBot-VA": (92.90, 91.50), "pi0.5": (82.74, 76.76), "Motus": (88.66, 87.02)}
res["table1_check"] = {m: {"recomputed": round(sum(p) / 2, 3), "stated": table1_stated[m],
                           "ok_at_0.05": abs(sum(p) / 2 - table1_stated[m]) <= 0.05 + 1e-9}
                       for m, p in table1_pairs.items()}

# 과제별 평균 (clean+rand)/2 기준 쌍체 비교: Fast-WAM 대 다른 방법
def task_avg(m):
    c, r = cols[m]
    return (c + r) / 2

def paired(a_name, b_name):
    d = task_avg(a_name) - task_avg(b_name)
    n = len(d)
    t = stats.ttest_1samp(d, 0.0)
    se = d.std(ddof=1) / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    try:
        w = stats.wilcoxon(d)
        wp = float(w.pvalue)
    except ValueError:
        wp = None
    wins = int((d > 0).sum()); losses = int((d < 0).sum()); ties = int((d == 0).sum())
    sign_p = stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else None
    return {
        "mean_diff": round(float(d.mean()), 2),
        "ci95": [round(float(d.mean() - tc * se), 2), round(float(d.mean() + tc * se), 2)],
        "paired_t_p": round(float(t.pvalue), 4),
        "wilcoxon_p": None if wp is None else round(wp, 4),
        "wins_losses_ties": [wins, losses, ties],
        "sign_test_p": None if sign_p is None else round(float(sign_p), 4),
    }

res["robotwin_paired"] = {
    f"Fast-WAM minus {b}": paired("Fast-WAM", b)
    for b in ["Fast-WAM-Joint", "Fast-WAM-IDM", "w.o.video-co-train", "LingBot-VA", "Motus", "pi0.5"]
}
# 이상치 민감도: 과제별 차이의 중앙값, 절댓값이 가장 큰 과제 하나를 뺀 평균
def robust(a_name, b_name):
    d = task_avg(a_name) - task_avg(b_name)
    top = int(np.argmax(np.abs(d)))
    keep = np.delete(d, top)
    return {"median_diff": round(float(np.median(d)), 2),
            "largest_task": names[top], "largest_diff": round(float(d[top]), 2),
            "mean_without_largest": round(float(keep.mean()), 2)}
res["robotwin_robustness"] = {f"Fast-WAM minus {b}": robust("Fast-WAM", b)
                              for b in ["Fast-WAM-Joint", "Fast-WAM-IDM", "LingBot-VA"]}
# 세부 설정(clean, rand) 별 검정은 간단히 평균 차이만 남긴다
res["robotwin_between_task_se"] = {
    m: round(float(task_avg(m).std(ddof=1) / math.sqrt(50)), 2) for m in METHODS
}

# ---------- 2. LIBERO (표 2): 스위트당 500회 가정, 정수 성공 횟수 복원 ----------
LIB = {  # Spatial, Object, Goal, Long
    "Fast-WAM": [98.2, 100.0, 97.0, 95.2],
    "Fast-WAM-Joint": [99.6, 99.4, 98.2, 96.8],
    "Fast-WAM-IDM": [98.8, 97.8, 97.8, 97.6],
    "w.o.video-co-train": [89.2, 99.2, 95.4, 90.0],
    "LingBot-VA": [98.5, 99.6, 97.2, 98.5],
    "Motus": [96.8, 99.8, 96.6, 97.6],
    "pi0.5": [98.8, 98.2, 98.0, 92.4],
    "pi0": [96.8, 98.8, 95.8, 85.2],
}
N_SUITE = 500

def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half

lib = {}
for m, pct in LIB.items():
    counts = [p * N_SUITE / 100 for p in pct]
    integer_ok = all(abs(c - round(c)) < 1e-6 for c in counts)
    counts = [int(round(c)) for c in counts]
    total = sum(counts)
    avg_recomputed = np.mean(pct)
    lib[m] = {
        "suite_counts_of_500": counts, "integer_ok": integer_ok,
        "overall_k_of_2000": total, "overall_pct": round(total / 20, 2),
        "avg_of_suites_recomputed": round(float(avg_recomputed), 2),
        "wilson95_overall": [round(100 * x, 2) for x in wilson(total, 2000)],
    }
res["libero"] = lib

def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    se_u = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {"diff_pts": round(100 * (p1 - p2), 2),
            "ci95_pts": [round(100 * (p1 - p2 - 1.96 * se_u), 2), round(100 * (p1 - p2 + 1.96 * se_u), 2)],
            "z": round(float(z), 3), "p_unpaired": round(float(pval), 5)}

fw_k = lib["Fast-WAM"]["overall_k_of_2000"]
res["libero_two_prop_vs_fastwam"] = {
    m: two_prop(fw_k, 2000, lib[m]["overall_k_of_2000"], 2000)
    for m in ["w.o.video-co-train", "Fast-WAM-Joint", "Fast-WAM-IDM", "LingBot-VA", "Motus", "pi0.5"]
}
# 검출 가능한 최소 차이 (단측 아님, 양측 5%, 검정력 80%, 비쌍체 근사)
p0 = fw_k / 2000
mde_overall = (1.96 + 0.8416) * math.sqrt(2 * p0 * (1 - p0) / 2000)
mde_suite = (1.96 + 0.8416) * math.sqrt(2 * p0 * (1 - p0) / 500)
res["libero_mde_pts"] = {"overall_n2000": round(100 * mde_overall, 2), "per_suite_n500": round(100 * mde_suite, 2)}

# ---------- 3. 실로봇 (그림 4): 값은 그림에서 읽음, 평가 횟수 미보고 ----------
real_read = {"pi0.5": (1.00, 119), "Fast-WAM-IDM": (0.90, 177), "Fast-WAM": (0.75, 152),
             "Fast-WAM-Joint": (0.70, 227), "pi0.5-wo-pretrain": (0.40, 206),
             "w.o.video-co-train": (0.10, 241)}
res["real_world_read_from_fig4"] = {k: {"success": v[0], "completion_s": v[1]} for k, v in real_read.items()}
# 0.05 단위로 읽히므로 n=20 이면 정수 횟수가 된다 (가정, 논문 미기재)
res["real_world_if_n20_counts"] = {k: round(v[0] * 20, 2) for k, v in real_read.items()}

# ---------- 4. 지연 시간 ----------
res["latency_ratio"] = {
    "IDM_over_FastWAM": round(810 / 190, 2),
    "Joint_over_FastWAM": round(580 / 190, 2),
    "note": "초록의 '4배 이상'은 IDM 대비에서만 성립. Joint 대비는 약 3배 (그림 4 우측, 읽은 값)",
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 요약 출력 ----------
print("tasks parsed:", len(names))
for m in METHODS:
    t = res["table3"][m]
    print(f"{m:20s} clean {t['clean_mean']:7.2f} (stated {t['clean_stated']})  rand {t['rand_mean']:7.2f} (stated {t['rand_stated']})")
print("table1 check:", {m: v["ok_at_0.05"] for m, v in res["table1_check"].items()})
for k, v in res["robotwin_paired"].items():
    print(k, v)
for m, v in lib.items():
    print(f"LIBERO {m:20s} counts {v['suite_counts_of_500']} integer_ok={v['integer_ok']} overall {v['overall_pct']} avg_suites {v['avg_of_suites_recomputed']}")
for k, v in res["libero_two_prop_vs_fastwam"].items():
    print("FastWAM vs", k, v)
print("MDE", res["libero_mde_pts"])
print("robust", res["robotwin_robustness"])
print("latency", res["latency_ratio"])
