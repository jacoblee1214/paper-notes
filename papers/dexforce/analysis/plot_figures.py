"""리포트 그래프 세 장. 실행: cd analysis && python3 plot_figures.py

1) kf_gain.png: Fig. 9의 k_f 허용 범위를 힘 배율 G = k_p·k_f(k_p = 310)로 바꿔 그림. G = 1(시연 힘 그대로)과 논문 값 G = 1.40
2) obs_ablation_ci.png: Fig. 8 성공률과 Wilson 95% 구간, Fisher p
3) toy_impedance.png: (a) 접촉 강성별 힘 배율 G(k_f)  (b) slide cube toy에서 k_f에 따른 블록 이동 거리
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import impedance_toy as it  # noqa: E402
import table_check as tc  # noqa: E402

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


# 1) k_f → 힘 배율
dig = json.loads((ROOT / "analysis" / "fig9_digitize_results.json").read_text())["bars"]
order = ["unscrew nut", "open AirPods", "slide cube", "reorient smiley", "grasp battery", "flip box"]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
for i, t in enumerate(order):
    lo, hi = tc.KP * dig[t]["min_mean"], tc.KP * dig[t]["max_mean"]
    ax.plot([i, i], [lo, hi], color=BLUE, lw=7, solid_capstyle="butt", alpha=0.85)
    cens = dig[t]["max_mean"] + dig[t].get("max_sd", 0) > tc.KF_CAP
    ax.annotate(f"{lo:.2f}–{hi:.2f}×" + (" (상한 절단)" if cens else ""), (i, hi), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=7, color=INK2)
ax.axhline(1.0, color=AQUA, lw=1.5)
ax.axhline(tc.KP * tc.KF_USED, color=ORANGE, lw=1.5, ls="--")
ax.axhline(tc.KP * tc.KF_CAP, color=MUTED, lw=1, ls=":")
ax.text(5.45, 1.0, "G = 1\n(시연 힘 그대로)", fontsize=7, color=INK2, va="center")
ax.text(5.45, tc.KP * tc.KF_USED + 0.15, "논문 k_f = 0.0045\n→ G = 1.40", fontsize=7, color=INK2, va="bottom")
ax.text(5.45, tc.KP * tc.KF_CAP, "sweep 상한\nk_f = 0.025", fontsize=7, color=INK2, va="center")
ax.set_xticks(range(6), [t.replace(" ", "\n") for t in order], fontsize=7.5)
ax.set_ylabel("힘 배율 G = k_p · k_f  (강체 접촉 근사)", color=INK2)
ax.set_xlim(-0.6, 6.4)
ax.set_ylim(0, 8.4)
style(ax)
fig.text(0.01, 0.01, "Fig. 9 막대(시연 5개 평균 최솟값–최댓값)를 픽셀 판독(±0.00012)하고 k_p = 310을 곱함. analysis/fig_digitize.py", fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "kf_gain.png")

# 2) 관측 절제 성공률과 구간
tasks = list(tc.FIG8.keys())
fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=200)
cols = [("ft", "RGB, F/T", BLUE), ("rgb", "RGB only", ORANGE), ("bin", "RGB, 0/1", AQUA)]
for j, (key, lab, c) in enumerate(cols):
    for i, t in enumerate(tasks):
        k = tc.FIG8[t][key][0]
        lo, hi = tc.wilson(k)
        x = i + (j - 1) * 0.22
        ax.plot([x, x], [lo, hi], color=c, lw=1.2)
        ax.scatter([x], [k / tc.N], color=c, s=22, zorder=3, label=lab if i == 0 else None)
for i, t in enumerate(tasks):
    p = tc.fisher(tc.FIG8[t]["ft"][0], tc.FIG8[t]["rgb"][0])
    ax.text(i, 1.04, f"p={p:.3f}" if p >= 0.001 else f"p={p:.0e}", ha="center", fontsize=6.5, color=INK2)
ax.axvline(5.5, color=GRID, lw=1, ls="--")
ax.set_xticks(range(len(tasks)), [t.replace(" - ", "\n").replace(" ", "\n", 1) for t in tasks], fontsize=7)
ax.set_ylabel("성공률 (30회)", color=INK2)
ax.set_ylim(-0.03, 1.1)
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
style(ax)
fig.text(0.01, 0.01, "Fig. 8의 성공 수와 Wilson 95% 구간. p는 RGB, F/T vs RGB only의 양측 Fisher 정확검정. 오른쪽 두 칸은 OOD.", fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(OUT / "obs_ablation_ci.png")

# 3) toy
res = json.loads((ROOT / "analysis" / "impedance_toy_results.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), dpi=200)
kfg = np.linspace(0, 0.012, 121)
for ke, c in ((1e3, AQUA), (1e4, ORANGE), (1e5, BLUE)):
    axes[0].plot(kfg, it.gain_formula(kfg, ke), color=c, lw=2, label=f"k_e = {int(ke):,} N/m")
axes[0].axvline(1 / tc.KP, color=INK2, lw=1, ls=":")
axes[0].axvline(tc.KF_USED, color=INK2, lw=1, ls="--")
axes[0].text(1 / tc.KP, 3.55, " 1/k_p", fontsize=7, color=INK2)
axes[0].text(tc.KF_USED, 3.25, " 0.0045", fontsize=7, color=INK2)
axes[0].set_xlabel("k_f (m/N)", color=INK2)
axes[0].set_ylabel("재생 힘 / 시연 힘", color=INK2)
axes[0].set_title("(a) 직렬 스프링 힘 배율 (시뮬 = 해석식)", fontsize=9)
axes[0].legend(frameon=False, fontsize=7)
style(axes[0])
cur = res["T2_curve_ke3000"]
axes[1].plot(cur["kf"], cur["block_disp_mm"], color=BLUE, lw=2)
axes[1].axhspan(30, 50, color=AQUA, alpha=0.15)
axes[1].axhline(40, color=AQUA, lw=1)
axes[1].text(0.0105, 41, "목표 40 mm (±10 mm 띠)", fontsize=7, color=INK2)
axes[1].axvline(tc.KF_USED, color=INK2, lw=1, ls="--")
axes[1].set_xlabel("k_f (m/N)", color=INK2)
axes[1].set_ylabel("블록 이동 (mm)", color=INK2)
axes[1].set_title("(b) slide cube toy (k_e = 3,000 N/m)", fontsize=9)
axes[1].set_ylim(-2, 48)
axes[1].set_xticks([0, 0.005, 0.01, 0.015, 0.02])
style(axes[1])
fig.text(0.01, 0.01, "k_p = 310, k_v = 2(논문 값). k_f = 0이면 수직력이 모자라 손끝이 미끄러지고, k_f가 크면 레일 마찰이 커져 덜 간다. analysis/impedance_toy.py", fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "toy_impedance.png")
print("wrote 3 figures to", OUT)
