"""실세계 최종 성공률과 95% CI 그래프 (논문 표 I·II의 수치로 직접 그림, 논문 그림 미사용).

입력: analysis/recompute_results.json (recompute.py가 만든 파일)
출력: report/figures/act_real_ci.png
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())
OUT = ROOT / "report" / "figures" / "act_real_ci.png"

FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                         "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})

TEAL = "#1b6f6a"
GREY = "#7a7a75"
ORANGE = "#d9822b"

rows = RES["real_final_vs_best_baseline"]
labels = [r["task"] for r in rows]
pct = [int(r["ACT_final"].split("/")[0]) * 100 / 25 for r in rows]
lo = [r["ACT_CI95"][0] for r in rows]
hi = [r["ACT_CI95"][1] for r in rows]
base = [int(r["baseline_final"].split("/")[0]) * 100 / 25 for r in rows]
bnames = [r["best_baseline"] for r in rows]

fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
x = range(len(rows))
ax.bar(x, pct, color=TEAL, width=0.55, label="ACT (논문 최종 단계)")
ax.errorbar(x, pct, yerr=[[p - l for p, l in zip(pct, lo)], [h - p for p, h in zip(pct, hi)]],
            fmt="none", ecolor="#143b38", capsize=4, lw=1)
ax.scatter(x, base, marker="D", color=ORANGE, zorder=3, s=26, label="최고 기준선 (최종 단계)")
for i, p in enumerate(pct):
    ax.text(i, hi[i] + 3, f"{p:.0f}%", ha="center", fontsize=8, color="#141b1a")
    ax.text(i, -11, "기준선 0%" if base[i] == 0 else f"{bnames[i]} {base[i]:.0f}%", ha="center", fontsize=6.8, color=GREY)
ax.set_xticks(list(x), labels, fontsize=8)
ax.set_ylim(-14, 110)
ax.set_ylabel("최종 단계 성공률 (%)")
ax.axhline(0, color="#999", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.set_title("실세계 6과제, 25회 평가 · 1 시드 (막대 오차: Wilson 95% CI)", fontsize=9.5, pad=24)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
