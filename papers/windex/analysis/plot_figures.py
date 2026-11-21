"""WinDEX 분석 그래프 두 장을 report/figures/에 만든다. 우리가 그린 그래프이며 논문 그림을 쓰지 않는다.

실행: python3 analysis/plot_figures.py   (matplotlib 필요, 수 초)
입력: analysis/recompute_results.json 과 논문 표에서 옮긴 값.
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

KO_FONTS = ["Apple SD Gothic Neo", "AppleSDGothicNeo", "Noto Sans CJK KR", "NanumGothic"]
installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in KO_FONTS if f in installed), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False
INK, ACCENT, GREY = "#1b1b1a", "#1f5fae", "#9a978d"

# 그림 1: 정책 학습 시뮬 성공률 (Wilson 95%)
ps = RES["policy_success"]
names = ["WinDEX", "HaWoR", "WiLoR"]
keys = ["WinDEX references", "HaWoR references", "WiLoR references"]
vals = [ps[k]["k"] for k in keys]
lo = [100 * (ps[k]["k"] / 100 - ps[k]["wilson95"][0]) for k in keys]
hi = [100 * (ps[k]["wilson95"][1] - ps[k]["k"] / 100) for k in keys]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
bars = ax.bar(names, vals, color=[ACCENT, GREY, GREY], width=0.55)
ax.errorbar(names, vals, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=5, lw=1)
for b, v, h in zip(bars, vals, hi):
    ax.text(b.get_x() + b.get_width() / 2, v + h + 2, f"{v}%", ha="center", color=INK, fontsize=10)
ax.set_ylim(0, 100)
ax.set_ylabel("시뮬 성공률 (%)")
ax.set_title("Hand2Policy 시뮬 성공률 (방법당 100회, 95% Wilson 구간)", fontsize=10.5, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "논문 그림 12와 초록의 평균값을 옮김. 물체별 값은 텍스트에 없음.", fontsize=7.2, color="#55544f")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_policy_wilson.png")
plt.close(fig)

# 그림 2: 같은 구성의 손목 오차가 표마다 다르게 보고된 값 (표 II, III, IV)
rows = [
    ("표 II: 교사 WinDEX-Net, 층 11", 15.70, ACCENT),
    ("표 IV: WinDEX-Window (P), 우리 행", 15.70, ACCENT),
    ("표 III: WinDEX-Net, 2회", 25.9, GREY),
    ("표 III: WinDEX-Window, 2회", 27.1, GREY),
    ("표 III: WinDEX-Stream (인과)", 42.7, GREY),
]
fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
labels = [r[0] for r in rows][::-1]
values = [r[1] for r in rows][::-1]
colors = [r[2] for r in rows][::-1]
ax.barh(labels, values, color=colors, height=0.55)
for i, v in enumerate(values):
    ax.text(v + 0.6, i, f"{v:.1f} mm", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 52)
ax.set_xlabel("손목 위치 오차 (mm, 논문 보고값)")
ax.set_title("같은 구성인데 표마다 다른 손목 오차", fontsize=10.5, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", labelsize=8.2)
fig.text(0.01, 0.01, "표 II와 IV는 같은 값, 표 III은 다른 값. 평가 집합은 논문에 적혀 있지 않음.", fontsize=7.2, color="#55544f")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_wrist_by_table.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
