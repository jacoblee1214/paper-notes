"""우리가 그린 그래프 한 장: LIBERO 스위트별 Wilson 95% 구간과 RoboCasa 기준선 비교.

실행: python3 report/plot_figures.py  → report/figures/own_ci.png
논문 그림은 쓰지 않는다 (공개본 규칙, README 참조).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=170)

ax = axes[0]
suites = ["Spatial", "Object", "Goal", "Long"]
vals = [98.1, 100.0, 98.2, 97.6]
lo = [RES["libero_wilson95"][s][0] for s in suites]
hi = [RES["libero_wilson95"][s][1] for s in suites]
ax.errorbar(range(4), vals, yerr=[[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]],
            fmt="o", color="#1f5fae", capsize=4, ms=6)
ax.axhline(98.5, color="#888", lw=0.8, ls="--")
ax.text(3.45, 98.55, "평균 98.5", fontsize=7.5, color="#555", ha="right", va="bottom")
ax.set_xticks(range(4), suites, fontsize=8)
ax.set_ylim(94, 101.5)
ax.set_ylabel("성공률 (%)", fontsize=8.5)
ax.set_title("LIBERO 스위트별 (Wilson 95%, n=1500)", fontsize=9)
ax.tick_params(labelsize=8)

ax = axes[1]
names = ["Cosmos\nPolicy", "FLARE", "GR00T-N1.5\n+HAMLET", "Video\nPolicy", "GR00T-N1.5"]
robo = [67.1, 66.4, 66.4, 66.0, 64.1]
colors = ["#1f5fae", "#9aa7b8", "#9aa7b8", "#9aa7b8", "#c9d1dc"]
ax.bar(range(len(robo)), robo, color=colors)
for i, v in enumerate(robo):
    ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5)
lo_c, hi_c = RES["robocasa_cosmos_wilson95"]
ax.errorbar([0], [67.1], yerr=[[67.1 - lo_c], [hi_c - 67.1]], fmt="none", ecolor="#333", capsize=4)
ax.set_xticks(range(len(robo)), names, fontsize=7.2)
ax.set_ylim(60, 70)
ax.set_ylabel("평균 성공률 (%)", fontsize=8.5)
ax.set_title("RoboCasa 24 과제 평균 (논문 표 2)", fontsize=9)
ax.tick_params(labelsize=8)

fig.text(0.5, 0.005, "그래프: 우리가 그림. 값은 논문 표 1, 2에서 옮김. 기준선 시도 수는 논문에 없어 구간은 Cosmos만 표시.",
         ha="center", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = FIG / "own_ci.png"
fig.savefig(out)
print("wrote", out)
