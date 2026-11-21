"""Table 2 성공률과 95% Wilson CI를 그린다 (우리가 만든 그래프).

입력: analysis/recompute_results.json (recompute.py 결과)
출력: report/figures/own_ci_forest.png
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
OUT = ROOT / "report" / "figures" / "own_ci_forest.png"

font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
if Path(font_path).exists():
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

tasks = ["Rotation", "Scooping", "Sweeping"]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
for i, t in enumerate(tasks):
    r = RES[t]
    y = len(tasks) - 1 - i
    for dy, key, color, label in ((0.16, "ours", "#1f5fae", "Ours"), (-0.16, "dp", "#9a9a94", "Diffusion Policy")):
        pct = r[f"{key}_pct"]
        lo, hi = r[f"{key}_wilson95"]
        ax.errorbar(pct, y + dy, xerr=[[pct - lo], [hi - pct]], fmt="o", color=color, capsize=3, lw=1.6,
                    label=label if i == 0 else None)
        ax.text(hi + 1.5, y + dy, f"{r[key]}  p={r['fisher_p_two_sided']:.2g}" if key == "ours" else r[key],
                va="center", fontsize=7.5, color="#55544f")
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(list(reversed(tasks)))
ax.set_xlim(0, 115)
ax.set_xlabel("성공률 (%), 점: 보고값, 막대: 95% Wilson 구간 (n = 40)")
ax.axvline(50, color="#dcdad3", lw=0.8, zorder=0)
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Table 2 재계산: 성공률과 신뢰구간 (논문 보고값 기준)", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
