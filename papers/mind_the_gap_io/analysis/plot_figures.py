"""우리가 만든 그래프: 주요 비교의 Wilson 95% 구간 (논문 그림 아님).

입력: analysis/recompute_results.json (recompute.py 출력)
출력: report/figures/own_forest.png
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트: Linux Noto CJK -> macOS Apple SD Gothic Neo 순서로 찾는다
for _name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    if any(f.name == _name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
OUT = ROOT / "report" / "figures" / "own_forest.png"

KEEP = [
    "Microwave Open, π0: E2E vs E2C (Q2)",
    "Plug Clear, π0: E2E vs E2C (Q2 force gen.)",
    "Blind plug, π0: E2C vs EC2C (Q3)",
    "Wiping strict, π0: E2C vs EC2C (Q3)",
    "Microwave Open, π0: E2C vs EC2C (Q3)",
    "Plug Clear, π0: E2C vs EC2C",
]
rows = [c for c in R["table2_comparisons"] if c["label"] in KEEP]
rows += [c for c in R["table3_comparisons"] if c["label"].startswith("Sort, DP")]
rows += [c for c in R["table5_comparisons"] if c["label"].startswith("A1→B2")]

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
for i, c in enumerate(rows):
    y = len(rows) - i
    for side, col, dx in [("a", "#8a8a85", -0.012), ("b", "#1f5fae", 0.012)]:
        d = c[side]
        p = d["p"]
        lo, hi = d["ci95"]
        ax.plot([lo, hi], [y + dx * 0, y + dx * 0], color=col, lw=2.2, solid_capstyle="round")
        ax.plot([p], [y], "o", color=col, ms=5)
    ax.text(1.03, y, f"p = {c['fisher_p']:.3f}" if c["fisher_p"] >= 0.001 else "p < 0.001",
            va="center", fontsize=7.5, color="#333")
ax.set_yticks(range(len(rows), 0, -1))
ax.set_yticklabels([c["label"] for c in rows], fontsize=7.5)
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("성공률 (Wilson 95% 구간). 회색: 비교 대상 A, 파랑: 비교 대상 B")
ax.set_title("논문 표의 주요 비교 (단일 실행, 시도 수 10–50)", fontsize=10)
ax.grid(axis="x", color="#e5e5e5", lw=0.6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
