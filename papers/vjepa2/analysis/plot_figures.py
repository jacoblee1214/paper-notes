"""우리가 만든 그래프 한 장: Table 2(두 랩 합산 n = 20)의 성공률과 Wilson 95% 구간.

실행: python3 analysis/plot_figures.py  → report/figures/own_ci_table2.png
입력 수치는 analysis/recompute_results.json에서 읽는다(재계산 결과와 같은 값).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures", "own_ci_table2.png")

# 한글 폰트: macOS 기본 글꼴 → Linux Noto CJK 순서로 찾는다(경로는 후보 목록만 쓴다)
for path in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

with open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8") as f:
    res = json.load(f)["table2_avg_check"]

labels = {"Grasp_Cup": "Grasp (컵)", "Grasp_Box": "Grasp (상자)", "ReachObj_Cup": "Reach w/ obj (컵)",
          "ReachObj_Box": "Reach w/ obj (상자)", "PnP_Cup": "Pick-and-place (컵)", "PnP_Box": "Pick-and-place (상자)"}
tasks = list(labels)
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
for i, t in enumerate(tasks):
    for m, color, dy in [("Octo", "#9a9891", -0.16), ("VJEPA2AC", "#1f5fae", 0.16)]:
        r = res[m][t]
        p = r["successes_of_20"] / 20
        lo, hi = r["wilson95"]
        y = len(tasks) - 1 - i + dy
        ax.plot([lo * 100, hi * 100], [y, y], color=color, lw=2)
        ax.plot(p * 100, y, "o", color=color, ms=6,
                label=("Octo (BC 기준선)" if m == "Octo" else "V-JEPA 2-AC") if i == 0 else None)
        ax.text(102, y, f"{r['successes_of_20']}/20", va="center", fontsize=8, color=color)
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([labels[t] for t in reversed(tasks)], fontsize=9)
ax.set_xlim(-2, 115)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("성공률 (%) · 점 = 관측값, 선 = Wilson 95% 구간 (두 랩 합산 n = 20)", fontsize=8.5)
ax.set_title("Table 2 재계산: V-JEPA 2-AC 대 Octo", fontsize=10.5, loc="left")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8, frameon=False)
ax.grid(axis="x", color="#dcdad3", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
