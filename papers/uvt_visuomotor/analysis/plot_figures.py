"""UVT 직접 그린 그래프 세 장 (CPU). 실행: python3 analysis/plot_figures.py  (table_check.py, uvt_toy.py 결과 필요)

1) success_ci.png: 실물 과제별 성공률(Wilson 95%)과 LIBERO suite별 차이(가정 n=500, 10k 스텝)
2) toy_dyn_gain.png: 선형 toy에서 동역학 고유 이득(잡음 블록 대비)과 목표 구성 효과(행동 전용 대비), 압축 차원 r별
3) power_mde.png: 과제당 50회 두 팔 비교의 최소 검출 차이와 논문의 실물 차이
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
TC = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
TOY = json.loads((ROOT / "analysis" / "uvt_toy_results.json").read_text())

FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, GRID = "#0b0b0b", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---- 1. 성공률 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
tasks = TC["real_tasks"]
names = list(tasks)
for i, t in enumerate(names):
    v = tasks[t]
    n = 50
    for j, (k, col, lab) in enumerate(((v["base_full"], MUTED, "기준"), (v["uvt_full"], BLUE, "UVT"))):
        lo, hi = [x / 100 for x in (v["base_full_wilson"] if j == 0 else v["uvt_full_wilson"])]
        ax1.bar(i + (j - 0.5) * 0.36, k / n, width=0.34, color=col, label=lab if i == 0 else None)
        ax1.errorbar(i + (j - 0.5) * 0.36, k / n, yerr=[[k / n - lo], [hi - k / n]], color=INK, capsize=3, lw=0.8)
        ax1.text(i + (j - 0.5) * 0.36, hi + 0.03, f"{k}/50", ha="center", fontsize=8)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names)
ax1.set_ylim(0, 1.0)
ax1.set_ylabel("전체 성공률 (50회)")
ax1.set_title("실물: Wilson 95% 구간", fontsize=10)
ax1.legend(frameon=False, fontsize=8, loc="upper left")
style(ax1)

# LIBERO suite별 10k 차이, 가정 n=500, Wald 근사 95% (시드 분산 미포함)
rows = [r for r in TC["libero_n500_assumed"]["suites"] if r["step"] == "10k"]
xs = np.arange(len(rows))
diffs = np.array([r["diff_pp"] for r in rows])
ses = np.array([math.sqrt((r["base"] / 100) * (1 - r["base"] / 100) / 500 + (r["uvt"] / 100) * (1 - r["uvt"] / 100) / 500) * 100
                for r in rows])
ax2.bar(xs, diffs, color=[AQUA if r["holm_p"] < 0.05 else MUTED for r in rows], width=0.55)
ax2.errorbar(xs, diffs, yerr=1.96 * ses, fmt="none", ecolor=INK, capsize=3, lw=0.8)
ax2.axhline(0, color=INK, lw=0.6)
ax2.set_xticks(xs)
ax2.set_xticklabels([r["suite"] for r in rows])
ax2.set_ylabel("UVT - 기준 (pp), 10k 스텝")
ax2.set_title("LIBERO: suite별 차이 (가정 n=500)", fontsize=10)
for x, d, r in zip(xs, diffs, rows):
    ax2.text(x, d + 1.96 * ses[x] + 0.8, f"Holm p={r['holm_p']:.2g}", ha="center", fontsize=7)
style(ax2)
fig.tight_layout()
fig.savefig(FIG / "success_ci.png", dpi=200)
plt.close(fig)

# ---- 2. 선형 toy: 동역학 고유 이득과 목표 구성 효과 ----
res = TOY["results"]
rhos = TOY["settings"]["rhos"]
rs = TOY["settings"]["rs"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
for ax, n in zip(axes, (60, 200)):
    for r, col in zip([2, 4, 6], (MUTED, BLUE, ORANGE)):
        ys = [res[f"n={n},rho={rho}"][f"dyn_gain_r{r}"] for rho in rhos]
        es = [res[f"n={n},rho={rho}"][f"dyn_gain_se_r{r}"] for rho in rhos]
        ax.errorbar(rhos, ys, yerr=[1.96 * e for e in es], marker="o", ms=4, color=col, capsize=2, lw=1.2, label=f"r={r}")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_title(f"선형 toy, n={n} (시드 {TOY['settings']['seeds']}개)", fontsize=10)
    ax.set_xlabel("동역학 예측도 ρ = R²(d | x)")
    style(ax)
axes[0].set_ylabel("동역학 고유 이득 (MSE 감소)")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "toy_dyn_gain.png", dpi=200)
plt.close(fig)

# ---- 3. 검출력 ----
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ps = np.linspace(0.05, 0.95, 91)
mde = [100 * (1.96 + 0.8416) * math.sqrt(2 * p * (1 - p) / 50) for p in ps]
ax.plot(ps * 100, mde, color=BLUE, lw=1.6, label="과제당 50회, 두 팔 (80% 검출력)")
ax.axhline(16, color=ORANGE, lw=0.9, ls="--")
ax.text(3, 16.8, "Lift Pot 차이 16pp (논문)", color=ORANGE, fontsize=8)
ax.axhline(10, color=AQUA, lw=0.9, ls="--")
ax.text(3, 10.8, "Close/Plate 전체 성공 차이 10pp", color=AQUA, fontsize=8)
ax.set_xlabel("기준 성공률 (%)")
ax.set_ylabel("최소 검출 차이 (pp)")
ax.set_ylim(0, 45)
ax.legend(frameon=False, fontsize=8, loc="upper right")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "power_mde.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("*.png") if p.name in ("success_ci.png", "toy_dyn_gain.png", "power_mde.png")))
