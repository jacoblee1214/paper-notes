"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_table4_ci.png     표 4 세 과제의 성공률과 Wilson 95% 구간 (과제당 50회 가정)
  own_ablation_ci.png   표 5, 6, 2의 Bottle Handover 성공률과 Wilson 95% 구간 (50회 가정)
  own_probe_vs_success.png  인코더별 프로브 r2(부록 E 그림)와 Handover 성공률(표 2)
  own_toy_aliasing.png  토이: 상태 별칭 과제에서 이력 길이별 성공률

입력: analysis/recompute_results.json, analysis/toy_aliasing_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TOY = json.loads((ROOT / "analysis" / "toy_aliasing_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

INK, MUTED, ACC, ACC2, WARN, GREY = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a", "#9a9890"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color="#e6e4dd", lw=0.8)
    ax.set_axisbelow(True)


# 1. 표 4
T4 = {"Bottle Handover": [44, 38, 74, 88], "Drop Catch": [80, 38, 66, 88], "Flick Catch": [48, 20, 66, 76]}
methods = ["RTC", "VLASH", "TEMPO", "TEMPO+UVT"]
colors = [GREY, MUTED, ACC, ACC2]
fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=160)
w = 0.2
for j, (m, c) in enumerate(zip(methods, colors)):
    xs, ys, lo, hi = [], [], [], []
    for i, (t, v) in enumerate(T4.items()):
        k = round(v[j] / 2)
        l, h = wilson(k, 50)
        xs.append(i + (j - 1.5) * w)
        ys.append(v[j])
        lo.append(v[j] - 100 * l)
        hi.append(100 * h - v[j])
    ax.bar(xs, ys, w * 0.92, color=c, label=m)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    for x, y in zip(xs, ys):
        ax.text(x, 2, f"{y}", ha="center", va="bottom", color="white", fontsize=7.5)
ax.set_xticks(range(3))
ax.set_xticklabels(list(T4))
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 105)
ax.set_title("표 4 성공률과 Wilson 95% 구간 (과제당 50회 가정)", color=INK, fontsize=10.5, loc="left")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_table4_ci.png")
plt.close(fig)

# 2. 절제와 백본 (Handover)
rows = [("VLASH (기준)", 38, MUTED), ("RTC (기준)", 44, GREY), ("π0.5 (기준)", 26, GREY),
        ("ACT만", 52, ACC2), ("MOT만", 56, ACC2), ("TEMPO 전체 (VLASH 기반)", 74, ACC),
        ("π0.5 + TEMPO", 56, ACC), ("RTC + TEMPO", 80, ACC),
        ("인코더 VideoLaVIT", 52, "#7aa3d6"), ("인코더 VidTwin", 60, "#7aa3d6"), ("인코더 Reducio", 76, "#7aa3d6")]
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
for i, (name, v, c) in enumerate(rows):
    k = round(v / 2)
    l, h = wilson(k, 50)
    ax.barh(i, v, color=c, height=0.66)
    ax.errorbar(v, i, xerr=[[v - 100 * l], [100 * h - v]], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    ax.text(1.5, i, f"{v}%", va="center", color="white", fontsize=8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 105)
ax.set_xlabel("Bottle Handover 성공률 (%)")
ax.axvline(38, color=MUTED, lw=0.8, ls="--")
ax.set_title("표 2, 5, 6의 Handover 성공률과 Wilson 95% 구간 (50회 가정)", color=INK, fontsize=10.5, loc="left")
style(ax)
ax.yaxis.grid(False)
ax.xaxis.grid(True, color="#e6e4dd", lw=0.8)
fig.tight_layout()
fig.savefig(FIG / "own_ablation_ci.png")
plt.close(fig)

# 3. 프로브 r2 대 성공률
enc = {"VideoLaVIT": (0.45, 52), "Reducio": (0.50, 76), "SAM 2.1-tiny": (0.58, 74), "VidTwin": (0.59, 60)}
fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=160)
for name, (r2, s) in enc.items():
    ax.scatter(r2, s, s=60, color=ACC, zorder=3)
    ax.annotate(name, (r2, s), textcoords="offset points", xytext=(6, 5), fontsize=8.5, color=INK)
ax.axhline(38, color=MUTED, lw=0.8, ls="--")
ax.text(0.452, 39.5, "VLASH 기준 38%", color=MUTED, fontsize=8)
ax.set_xlabel("선형 프로브 R² (부록 E 그림, Bottle Handover)")
ax.set_ylabel("Handover 성공률 % (표 2)")
ax.set_xlim(0.43, 0.63)
ax.set_ylim(30, 85)
ax.set_title("인코더 4개: 프로브 R²와 과제 성공률", color=INK, fontsize=10.5, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_probe_vs_success.png")
plt.close(fig)

# 4. 토이
res = TOY["결과"]
labels = [r["변형"].replace(" (", "\n(") for r in res]
vals = [100 * r["성공률_평균"] for r in res]
fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=160)
cols = [WARN if r["K"] == 0 else ACC for r in res]
ax.bar(range(len(res)), vals, color=cols, width=0.68)
for i, r in enumerate(res):
    lo, hi = r["wilson95_(시드 합산 900회, 독립 근사)"]
    ax.errorbar(i, vals[i], yerr=[[vals[i] - 100 * lo], [100 * hi - vals[i]]], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    ax.text(i, vals[i] + 3, f"{vals[i]:.0f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(len(res)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 112)
ax.set_title("토이: 별칭이 있는 1차원 왕복 과제 (시드 3개, 롤아웃 900개)", color=INK, fontsize=10.5, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_toy_aliasing.png")
plt.close(fig)
print("saved 4 figures to", FIG)
