"""리포트용 그래프 두 장. 실행: python3 analysis/plot_headline.py (먼저 grounding_toy.py 실행)

1) headline_success.png: Table I 성공률과 Wilson 95% 구간 (n = 10 또는 20)
2) grounding_toy.png: toy에서 장면별 반복성 σ 분포와, 이상치 시점 수에 따른 오차
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def bars(ax, groups, methods, values, ns, colors):
    w = 0.8 / len(methods)
    for j, (m, c) in enumerate(zip(methods, colors)):
        for i, g in enumerate(groups):
            v, n = values[m][i], ns[i]
            k = round(v * n)
            lo, hi = tc.wilson(k, n)
            x = i + (j - (len(methods) - 1) / 2) * w
            ax.bar(x, v, width=w * 0.92, color=c, label=m if i == 0 else None)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
            ax.text(x, hi + 0.02, f"{k}/{n}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1.15)
    style(ax)


fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]})
bars(axes[0], ["Standard", "Cluttered & Precise", "All"], ["MOKA", "ReKep", "Ours"],
     {m: v for m, v in tc.T1.items()}, [10, 10, 20], [ORANGE, AQUA, BLUE])
axes[0].set_title("(a) zero-shot 기준선 (Franka)", fontsize=10, color=INK)
axes[0].set_ylabel("성공률", color=INK2)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)
bars(axes[1], ["Approach", "Grasp", "Success"], ["GR00T", "Being-H0", "Ours"],
     {m: v for m, v in tc.T1_VLA.items()}, [20, 20, 20], [ORANGE, AQUA, BLUE])
axes[1].set_title("(b) 100개 시연 미세조정 VLA (xArm, Throw Away Trash)", fontsize=10, color=INK)
axes[1].legend(frameon=False, fontsize=7.5, loc="upper right", ncol=3)
fig.text(0.01, 0.01, "출처: Table I. 막대 위 숫자는 성공/시행, 오차막대는 Wilson 95% 구간. 모두 한 번씩의 실제 로봇 시행.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "headline_success.png")
print("wrote", OUT / "headline_success.png")

# 2) grounding toy
res = json.loads((ROOT / "analysis" / "grounding_toy_results.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200)
ax = axes[0]
for key, label, c in (("consensus_refit", "다시점 합의 + inlier 재적합", BLUE),
                      ("consensus_literal", "다시점 합의 (Eq. 6 그대로, 동률은 첫 쌍)", AQUA),
                      ("single_rgbd", "단일 RGB-D 역투영 (MOKA식)", ORANGE)):
    s = np.sort(np.array(res["G1_sigma_samples_mm"][key]))
    ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=c, lw=2, label=label)
ax.set_xscale("log")
ax.set_xlim(2, 1000)
lit_max = max(res["G1_sigma_samples_mm"]["consensus_literal"])
if lit_max > 1000:
    ax.annotate(f"Eq. 6 그대로: 최대 σ {lit_max / 1000:.1f} m (범위 밖)", xy=(1000, 0.995), xytext=(60, 0.7),
                fontsize=7, color=INK2, arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.8})
ax.set_xlabel("장면별 반복성 σ (mm, 로그축)", color=INK2)
ax.set_ylabel("누적 비율 (장면)", color=INK2)
ax.set_title("(a) 같은 입력 8회 반복의 σ 분포 (200 장면)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.grid(color=GRID, lw=0.8)
style(ax)
ax = axes[1]
ks = [0, 1, 2, 3]
for key, label, c in (("consensus_refit", "합의 + 재적합", BLUE), ("all_view_lsq", "4개 시점 최소제곱(평균)", ORANGE)):
    med = [res["G2_outlier_views"][f"{k}_bad_views"][key]["median_mm"] for k in ks]
    ax.plot(ks, med, marker="o", ms=6, color=c, lw=2, label=label)
ax.set_yscale("log")
ax.set_xticks(ks, ["0", "1", "2", "3"])
ax.set_xlabel("4개 시점 중 VLM이 엉뚱한 곳을 가리킨 시점 수", color=INK2)
ax.set_ylabel("3D 오차 중앙값 (mm, 로그축)", color=INK2)
ax.set_title("(b) 이상치 시점 수에 따른 오차", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.grid(color=GRID, lw=0.8)
style(ax)
fig.text(0.01, 0.01, "toy 가정: VLM 2D 오차 = 고정 편향 4 px + 실행별 3 px + 질의당 5% 이상치(40–100 px), ε_tri = 10 px, 카메라 배치 가정. analysis/grounding_toy.py",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "grounding_toy.png")
print("wrote", OUT / "grounding_toy.png")
