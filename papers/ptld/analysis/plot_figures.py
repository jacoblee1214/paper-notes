"""자체 그래프 두 장 (논문 그림이 아님). 실행: python3 analysis/plot_figures.py

  report/figures/own_fig6_readout.png : 그림 6 총 회전 막대(그림 판독 값)와 "182%"에 해당하는 비
  report/figures/own_cube_and_goals.png : 표 3(정육면체) PTLD 학생 대 교사 비, 표 4 평균과 근사 95% 구간
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
font = next((f for f in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "AppleGothic"]
             if any(f == x.name for x in font_manager.fontManager.ttflist)), None)
if font:
    plt.rcParams["font.family"] = font
plt.rcParams["axes.unicode_minus"] = False

fig6 = json.loads((HERE / "fig6_readout.json").read_text())
names = ["RMA", "AAC", "PD", "TA", "PTLD"]
labels = ["RMA\n(고유수용)", "AAC\n(고유수용)", "고유수용\n증류", "촉각 적응\n(Sparsh-skin)", "PTLD"]
colors = ["#b8b6ae", "#8fb3e8", "#b9b4e6", "#e3b0e6", "#1f5fae"]

# ---- 그림 1
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=170, sharey=True)
for ax, key, title in zip(axes, ["rotation_cw", "rotation_ccw"], ["시계 방향", "반시계 방향"]):
    vals = [fig6[key][n] for n in names]
    errs = [fig6["errorbar_halfwidth"][key].get(n, 0) for n in names]
    ax.bar(range(5), vals, color=colors, yerr=errs, capsize=2, width=0.7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(f"총 회전 (rad), {title}", fontsize=9)
    ax.grid(axis="y", color="#dcdad3", lw=0.6)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(i + 0.38, v, f"{v:.0f}", ha="left", va="center", fontsize=7)
r = fig6["rotation_cw"]["PTLD"] / fig6["rotation_cw"]["AAC"] - 1
axes[0].annotate(f"PTLD / AAC = +{100 * r:.0f}%", xy=(4, fig6["rotation_cw"]["PTLD"]), xytext=(1.3, 62),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
r2 = fig6["rotation_ccw"]["PTLD"] / fig6["rotation_ccw"]["AAC"] - 1
axes[1].annotate(f"PTLD / AAC = +{100 * r2:.0f}%", xy=(4, fig6["rotation_ccw"]["PTLD"]), xytext=(1.3, 82),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
axes[0].set_ylabel("rad", fontsize=8)
axes[0].set_ylim(0, 100)
fig.suptitle("그림 6 막대의 판독 값(오차 약 ±1 눈금). 오차 막대 정의는 논문에 없음", fontsize=8, y=0.99)
fig.tight_layout()
fig.savefig(OUT / "own_fig6_readout.png")
plt.close(fig)

# ---- 그림 2
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=170, gridspec_kw={"width_ratios": [1.1, 1]})
metrics = ["RotR", "TTF", "RotP(↓)"]
ratios = [4.5 / 17.3, 0.34 / 0.33, 50.8 / 5.5]
a.bar(range(3), ratios, color=["#c0504d", "#8a8a86", "#c0504d"], width=0.55)
a.axhline(1, color="#333", lw=0.8, ls="--")
a.set_xticks(range(3))
a.set_xticklabels(metrics, fontsize=8)
a.set_ylabel("촉각 학생 / 교사 AAC (표 3)", fontsize=8)
a.set_title("정육면체: 학생 대 교사 비 (1 = 같음)", fontsize=9)
for i, v in enumerate(ratios):
    a.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=8)
a.set_ylim(0, 10.5)

n = 10
data = [("고유수용", 2.1, 2.14), ("촉각+고유수용", 3.3, 1.55)]
for i, (lab, m, s) in enumerate(data):
    half = 2.262 * s / math.sqrt(n)  # t(0.975, 9), ± 를 SD로 가정
    b.errorbar(i, m, yerr=half, fmt="o", color="#1f5fae", capsize=4)
    b.text(i + 0.08, m, f"{m} ± {half:.2f}", fontsize=8, va="center")
b.set_xticks([0, 1])
b.set_xticklabels([d[0] for d in data], fontsize=8)
b.set_xlim(-0.5, 1.9)
b.set_ylim(0, 6)
b.set_ylabel("N_goals", fontsize=8)
b.set_title("표 4: 평균과 근사 95% 구간 (n=10, ±=SD 가정)", fontsize=9)
b.grid(axis="y", color="#dcdad3", lw=0.6)
fig.tight_layout()
fig.savefig(OUT / "own_cube_and_goals.png")
plt.close(fig)
print("wrote", OUT / "own_fig6_readout.png", OUT / "own_cube_and_goals.png")
