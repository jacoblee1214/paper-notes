"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 입력은 recompute.py와 같은 논문 표 값이다.

실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
한글 폰트: Linux Noto CJK -> macOS Apple SD Gothic Neo 순서로 찾는다. 없으면 기본 폰트.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

import glob
_KR_FONT_CANDIDATES = (sorted(glob.glob("/usr/share/fonts/**/NotoSansCJK*-Regular.ttc", recursive=True))
                       + ["/System/Library/Fonts/AppleSDGothicNeo.ttc"])
for _p in _KR_FONT_CANDIDATES:
    if Path(_p).exists():
        font_manager.fontManager.addfont(_p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def dot_ci(ax, rows, title):
    """rows: (라벨, 성공 수, 시도 수, 색)"""
    for i, (label, k, n, color) in enumerate(rows):
        p = k / n
        lo, hi = wilson(k, n)
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=color, capsize=3, ms=6)
        ax.text(1.02, i, f"{k}/{n}", va="center", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.axvline(0, color="#cccccc", lw=0.5)
    ax.set_xlabel("성공률 (Wilson 95% 구간)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color="#eeeeee")


# 그림 1: 표 1 장기 과제 ALL (과제 3개 x 10회 가정, V1)
rows = [
    ("GC-BC (BC-RNN) 20", 0, 30, "#999999"),
    ("GC-BC (BC-trans) 20", 0, 30, "#999999"),
    ("C-BeT 20", 0, 30, "#999999"),
    ("LMP 20", 2, 30, "#999999"),
    ("R3M-BC 20", 0, 30, "#999999"),
    ("Ours (0% human) 20", 7, 30, "#1f5fae"),
    ("Ours 20", 14, 30, "#d1495b"),
    ("Ours (0% human) 40", 12, 30, "#1f5fae"),
    ("Ours 40", 21, 30, "#d1495b"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
dot_ci(ax, rows, "표 1 장기 과제 (3과제 ALL, 과제당 10회 가정)")
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png", dpi=170)
plt.close(fig)

# 그림 2: 표 2 스터디 데스크 ALL (학습 4과제 x10, 미학습 3과제 x10)
rows2 = [
    ("학습 과제: GC-BC (BC-trans)", 0, 40, "#999999"),
    ("학습 과제: LMP", 0, 40, "#999999"),
    ("학습 과제: Ours (0% human)", 8, 40, "#1f5fae"),
    ("학습 과제: Ours (w/o GMM)", 11, 40, "#7a7a7a"),
    ("학습 과제: Ours (w/o KL)", 15, 40, "#7a7a7a"),
    ("학습 과제: Ours (50% human)", 12, 40, "#7a7a7a"),
    ("학습 과제: Ours", 22, 40, "#d1495b"),
    ("미학습: Ours (0% human)", 3, 30, "#1f5fae"),
    ("미학습: Ours (w/o GMM)", 2, 30, "#7a7a7a"),
    ("미학습: Ours (w/o KL)", 6, 30, "#7a7a7a"),
    ("미학습: Ours (50% human)", 8, 30, "#7a7a7a"),
    ("미학습: Ours", 14, 30, "#d1495b"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.8))
dot_ci(ax, rows2, "표 2 스터디 데스크 ALL (20 demos, 시도 수 가정 V1)")
fig.tight_layout()
fig.savefig(FIG / "own_study_ci.png", dpi=170)
plt.close(fig)

# 그림 3: 표 4 시뮬 (5 시드 평균 ± SD, 과제별)
import json
sim = {
    "GC-BC (BC-RNN)": [(0.96, 0.02), (0, 0), (0, 0), (0, 0), (0, 0)],
    "GC-BC (BC-trans)": [(0.95, 0.03), (0.01, 0.02), (0, 0), (0.01, 0.02), (0, 0)],
    "Ours (0% human)": [(0.92, 0.06), (0.70, 0.05), (0.80, 0.07), (0.74, 0.11), (0.78, 0.04)],
}
sim_long = {
    "GC-BC (BC-RNN)": [(0, 0)] * 5,
    "GC-BC (BC-trans)": [(0, 0)] * 5,
    "Ours (0% human)": [(0.58, 0.04), (0.29, 0.10), (0.62, 0.11), (0.59, 0.05), (0.67, 0.09)],
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
colors = {"GC-BC (BC-RNN)": "#999999", "GC-BC (BC-trans)": "#555555", "Ours (0% human)": "#1f5fae"}
for ax, data, title in ((axes[0], sim, "부분 과제 (첫 subgoal)"), (axes[1], sim_long, "장기 과제 (3개 이상 subgoal)")):
    for m, pts in data.items():
        xs = list(range(1, 6))
        ys = [p[0] for p in pts]
        es = [p[1] for p in pts]
        ax.errorbar([x + (0.12 if m.startswith("Ours") else (-0.12 if "trans" in m else 0)) for x in xs], ys,
                    yerr=es, fmt="o", capsize=3, color=colors[m], label=m, ms=5)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"과제 {i}" for i in range(1, 6)], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", color="#eeeeee")
axes[0].set_ylabel("성공률 (5 시드 평균 ± SD)")
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "own_sim_ci.png", dpi=170)
plt.close(fig)
print("wrote figures to", FIG)
