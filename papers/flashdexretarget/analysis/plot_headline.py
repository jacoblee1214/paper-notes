"""리포트용 그래프 두 장. 실행: python3 analysis/plot_headline.py

1) headline_table1.png: Table I의 성공률 대 GPU-h (로그축), XHand·Sharpa
2) reward_axis.png: 회전축에 따른 점군 보상 오차 (analysis/reward_toy.py와 같은 식)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import reward_toy as rt
import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) Table I
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200, sharey=True)
for ax, (hand, title) in zip(axes, (("xhand", "(a) XHand"), ("sharpa", "(b) Sharpa Wave Hand"))):
    for name, v in tc.TABLE[hand].items():
        ours = name == tc.FDR
        c = BLUE if ours else MUTED
        gh, sp, mto = v[5], v[0], v[1]
        ax.plot([gh, gh], [mto, sp], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([gh], [sp], s=46, facecolor="white", edgecolor=c, linewidth=2, zorder=3)
        ax.scatter([gh], [mto], s=46, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{name}\n{gh:,} GPU-h", (gh, sp), xytext=(8, 2), textcoords="offset points",
                    fontsize=7.5, color=INK if ours else INK2, fontweight="bold" if ours else "normal")
    ax.set_xscale("log")
    ax.set_xlim(15, 9000)
    ax.set_ylim(-0.03, 1.0)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("학습 GPU-h (로그축, RTX 3090)", color=INK2)
    style(ax)
axes[0].set_ylabel("50개 모션 중 성공 비율", color=INK2)
h1 = axes[0].scatter([], [], s=46, facecolor="white", edgecolor=INK2, linewidth=2, label="SR_SPIDER (10 cm / 0.5 rad, 평균 중심화)")
h2 = axes[0].scatter([], [], s=46, color=INK2, label="SR_MT-Obj (3 cm / 30°, 손별 최악)")
fig.legend(handles=[h1, h2], loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.01, "출처: Table I. 기준선은 모션마다 따로 학습(성공하면 중단), FlashDexRetarget은 정책 하나를 300M(XHand)·400M(Sharpa) step 학습.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig(OUT / "headline_table1.png")
print("wrote", OUT / "headline_table1.png")

# 2) 회전축 의존성
deg = np.arange(0, 181, 5)
stick = rt.fps(rt.box_surface((0.20, 0.03, 0.03)), 128)
sphere = rt.fps(rt.sphere_surface(0.04), 128)
I3, z3 = np.eye(3), np.zeros(3)
series = [
    ("막대 20×3×3 cm, 장축(x) 회전", stick, np.array([1.0, 0, 0]), ORANGE),
    ("막대 20×3×3 cm, 수직축(z) 회전", stick, np.array([0, 0, 1.0]), BLUE),
    ("구 r = 4 cm, 임의 축 (파랑과 겹침)", sphere, np.array([0.3, 0.5, 0.8]), AQUA),
]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
for label, sp, axis, c in series:
    e = [100 * rt.pcd_error(sp, rt.rotmat(axis, math.radians(d)), z3, I3, z3) for d in deg]
    ax.plot(deg, e, color=c, lw=2, label=label)
ax.axhline(3.0, color=INK2, lw=1, ls="--")
ax.axvline(30, color=INK2, lw=1, ls=":")
ax.text(181, 3.15, "3 cm", va="bottom", fontsize=7.5, color=INK2)
ax.text(32, 11.3, "30°: SR_MT-Obj 회전 기준", fontsize=7.5, color=INK2)
ax.set_xlabel("회전각 (도)", color=INK2)
ax.set_ylabel("worst-3 점 오차 e_obj (cm)", color=INK2)
ax.set_xlim(0, 195)
ax.set_ylim(0, 12)
ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
style(ax)
fig.text(0.01, 0.01, "식 6을 numpy로 옮겨 계산(ρ = 0.058 m로 정규화, FPS 128점, worst-3 평균). analysis/reward_toy.py",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "reward_axis.png")
print("wrote", OUT / "reward_axis.png")
