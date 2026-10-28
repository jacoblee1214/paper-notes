"""리포트용 그래프 세 장 (논문 수치와 toy 결과로 직접 그림). 실행: python3 analysis/plot_figures.py

1) table_consistency.png: Table IV의 f_max = λ·F*_wall(여유 = 1/λ)과 실험 1·2의 플라스틱 컵 "Ours" 비교
2) ranking_invariance.png: 재료 계수만 바꿨을 때 선택이 유지되는 비율 vs α (force_map_toy.py T1)
3) friction_need.png: 플라스틱 컵(1.9 g)을 마찰로 들기 위한 손가락당 최소 수직력 vs 보고된 f_max
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

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


# 1) 표 정합성
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=200)
ax = axes[0]
lam = np.array([0.3, 0.7, 1.0])
f = np.array([0.09, 0.21, 0.29])
ls = np.linspace(0, 1.05, 50)
ax.plot(ls, 0.3 * ls, color=MUTED, lw=1.5, ls="--", label="f = λ · F*_wall")
ax.scatter(lam, f, s=50, color=BLUE, zorder=3)
for l_, f_, name in zip(lam, f, ("gently", "default", "firmly")):
    ax.annotate(f"{name}\nf = {f_:.2f} N, 여유 = {0.3 / f_:.2f}", (l_, f_), xytext=(8, -16 if l_ > 0.9 else -4),
                textcoords="offset points", fontsize=7.5, color=INK)
ax.axhline(0.3, color=ORANGE, lw=1, label="벽 허용 하중 F*_wall = 0.3 N")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_xlim(0, 1.25)
ax.set_ylim(0, 0.36)
ax.set_xlabel("상호작용 계수 λ", color=INK2)
ax.set_ylabel("f_max (N)", color=INK2)
ax.set_title("(a) Table IV: 여유는 정확히 1/λ", fontsize=10, color=INK)
style(ax)

ax = axes[1]
labels = ["실험 1 (Table II)\n기준: 테두리", "실험 2 (Table III)\n기준: 벽"]
ref = [3.7, 0.3]
fm = [3.7 / 5.9, 0.21]
x = np.arange(2)
ax.bar(x - 0.18, ref, width=0.34, color=MUTED, label="기준 허용 하중 F*")
ax.bar(x + 0.18, fm, width=0.34, color=BLUE, label="f_max (여유에서 역산)")
for xi, r, v in zip(x, ref, fm):
    ax.text(xi - 0.18, r + 0.08, f"{r:g} N", ha="center", fontsize=7.5, color=INK2)
    ax.text(xi + 0.18, v + 0.08, f"{v:.2f} N", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylim(0, 4.3)
ax.set_ylabel("N", color=INK2)
ax.set_title("(b) 플라스틱 컵 \"Ours\": 두 실험이 다른 기준", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
style(ax)
fig.text(0.01, 0.01, "출처: 논문 Table II–IV. 실험 2는 '힘 지도로 고른 파지를 고정'한다고 하지만 기준은 벽 0.3 N이고, 실험 1의 선택은 테두리 3.7 N.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "table_consistency.png")
print("wrote", OUT / "table_consistency.png")

# 2) 재순위화의 재료 불변성
res = json.loads((ROOT / "analysis" / "force_map_toy_results.json").read_text())["T1_ranking_invariance"]
alphas = [0, 0.01, 0.1, 1]
xs = np.arange(len(alphas))
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
for key, color, label in (("no_bonus_n2", BLUE, "가산점 없음, F_base ∝ t²"), ("with_bonus_n2", ORANGE, "층 가산점 있음, F_base ∝ t²"),
                          ("with_bonus_n3", AQUA, "층 가산점 있음, F_base ∝ t³")):
    ys = [res[key]["same_choice_across_k_m"][f"alpha{a:g}"] for a in alphas]
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, label=label)
ax.set_xticks(xs, ["0", "0.01", "0.1", "1"])
ax.set_ylim(0, 1.05)
ax.set_xlabel("Eq. 5의 분산 가중치 α (1/N)", color=INK2)
ax.set_ylabel("재료 계수 k_m만 바꿔도 같은 후보를 고른 비율", color=INK2, fontsize=8)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
style(ax)
fig.text(0.01, 0.01, "toy: 종이컵 두께장, 후보 50개 중 5개, 400회. k_m ∈ {0.35, 1, 10}. analysis/force_map_toy.py",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "ranking_invariance.png")
print("wrote", OUT / "ranking_invariance.png")

# 3) 마찰로 들기 위한 최소 수직력 (플라스틱 컵 1.9 g)
mu = np.linspace(0.1, 1.0, 100)
w = 1.9e-3 * 9.81
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
ax.plot(mu, w / (2 * mu), color=BLUE, lw=2, label="손가락 2개 접촉")
ax.plot(mu, w / (5 * mu), color=AQUA, lw=2, label="손가락 5개 접촉")
for val, txt, c, dy, va in ((0.08, "under-stiffness 0.08 N: 실패", ORANGE, -0.004, "top"),
                            (0.09, "gently 0.09 N: 성공", INK2, 0.004, "bottom"),
                            (0.21, "default 0.21 N: 성공", INK2, 0.004, "bottom"),
                            (0.30, "벽 허용 하중 0.30 N", MUTED, 0.004, "bottom")):
    ax.axhline(val, color=c, lw=1, ls="--" if c != ORANGE else "-")
    ax.text(1.0, val + dy, txt, ha="right", va=va, fontsize=7.5, color=c if c != MUTED else INK2)
ax.axvspan(0.4, 0.6, color=GRID, alpha=0.6, lw=0)
ax.text(0.5, 0.115, "코드 기본\nμ_d 0.4 – μ_s 0.6", ha="center", fontsize=7, color=INK2)
ax.set_xlim(0.1, 1.0)
ax.set_ylim(0, 0.34)
ax.set_xlabel("마찰계수 μ", color=INK2)
ax.set_ylabel("손가락당 최소 수직력 (N)", color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper center")
style(ax)
fig.text(0.01, 0.01, "정적 근사: μ · n · f ≥ m g. 출처: 질량 Sec. IV-A, f_max Table III–IV, 마찰계수 공식 코드 기본값.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "friction_need.png")
print("wrote", OUT / "friction_need.png")
