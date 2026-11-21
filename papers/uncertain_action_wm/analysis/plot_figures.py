"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 입력은 recompute_results.json과 auroc_mc_check_results.json.

1) own_table1_ci.png: 표 1의 ∆ACTUS 점과 95% 구간 (PushT, Cube, 대조 방법)
2) own_auroc_law.png: 검출 법칙의 정확 AUROC와 식 (8) 근사, 토이 몬테카를로 점
3) own_thm2_decomp.png: Thm. 2 분해 (Lbox, Llat, Lest), 표 1 값에서 계산
"""
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REC = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))
MC = json.loads((HERE / "auroc_mc_check_results.json").read_text(encoding="utf-8"))

if platform.system() == "Darwin":
    FONT = "Apple SD Gothic Neo"
else:
    FONT = "Noto Sans CJK KR"
plt.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, GRID = "#1b1b1a", "#6b6a64", "#e4e2dc"
BLUE, GREY, ORANGE = "#1f5fae", "#8c8a84", "#b5651d"

# 1) 표 1: ∆ACTUS (CI는 표 값, 점은 ACTUS 대비 차이)
ci = {"Frozen": (-21.2, -28.0, -14.9, -5.4, -9.1, -1.9),
      "AdaJEPA": (-23.5, -29.1, -18.2, -8.4, -12.8, -4.2),
      "OpTTA": (-26.2, -31.7, -21.0, -8.6, -12.3, -4.6),
      "JEPA-TTT": (-18.9, -23.8, -14.2, -4.1, -7.9, -0.2),
      "Sandwich": (-22.8, -28.7, -17.2, -6.3, -10.4, -2.5),
      "ALRA": (-22.8, -28.5, -17.4, -9.4, -13.7, -4.5),
      "Robust": (-33.6, -40.1, -27.3, -14.0, -16.1, -12.0),
      "ActLin": (-8.4, -11.6, -5.4, -6.7, -10.5, -3.0),
      "Lie-SGD": (-3.3, -5.3, -1.3, -4.4, -7.6, -1.2),
      "Oracle@1": (0.2, -0.7, 1.2, 0.9, -0.7, 2.3),
      "Oracle@0": (-0.7, -1.7, 0.3, 3.5, 1.6, 5.5),
      "JEPA-TTT+A": (1.5, 0.6, 2.5, -0.1, -1.5, 1.1),
      "AdaJEPA+A": (1.6, 0.7, 2.4, 0.3, -1.2, 1.6)}
names = list(ci)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), sharey=True)
for ax, (idx, title) in zip(axes, [(0, "PushT (∆ACTUS, 점수)"), (3, "OGBench-Cube (∆ACTUS, 점수)")]):
    y = np.arange(len(names))[::-1]
    for yi, nm in zip(y, names):
        v = ci[nm]
        c, lo, hi = v[idx], v[idx + 1], v[idx + 2]
        col = ORANGE if nm.startswith("Oracle") else (BLUE if nm.endswith("+A") else GREY)
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="butt")
        ax.plot(c, yi, "o", color=col, ms=4.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("ACTUS 대비 차이 (포인트, 95% 구간)", color=MUTED)
axes[0].set_yticks(list(range(len(names)))[::-1])
axes[0].set_yticklabels(names, color=INK)
fig.text(0.01, 0.01, "표 1의 값을 옮겼습니다. 구간은 논문 부트스트랩 값입니다(재계산 불가). 주황은 오라클, 파랑은 ACTUS 조합.",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_table1_ci.png", dpi=170)
plt.close(fig)

# 2) 검출 법칙
lam = np.linspace(0.2, 12, 200)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for K, col in ((4, BLUE), (6, ORANGE)):
    exact = []
    for l in lam:
        f = lambda x, K=K, l=l: stats.chi2.pdf(x, K) * stats.ncx2.sf(x, K, l)
        exact.append(quad(f, 0, np.inf, limit=200)[0])
    approx = stats.norm.cdf(lam / (2 * np.sqrt(K + lam)))
    ax.plot(lam, exact, color=col, lw=1.8, label=f"정확 (K={K})")
    ax.plot(lam, approx, color=col, lw=1.1, ls="--", label=f"식 (8) 근사 (K={K})")
ax.axhline(0.75, color=GREY, lw=0.8, ls=":")
ax.text(11.8, 0.76, "AUROC 0.75", ha="right", color=MUTED, fontsize=8)
for K, col, paper in ((4, BLUE, 3.6), (6, ORANGE, 4.2)):
    ax.plot(paper, 0.75, "o", color=col, ms=5, mfc="white")
ax.set_xlabel("비중심 λ = n·||δ||²_F / κ", color=MUTED)
ax.set_ylabel("윈도 AUROC (정상 대 이동)", color=MUTED)
ax.set_title("검출 법칙: 정확 곡선과 식 (8) 근사 (λ0.75: 논문 3.6, 4.2 표시)", loc="left", fontsize=9.5, color=INK)
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.text(0.01, 0.01, f"토이 확인(AR(1) ρ=0.5, n=20, κ 추정 {MC['kappa_estimated_from_nominal']}): 몬테카를로 AUROC는 정확 곡선과 0.02 안쪽, 식 (8)보다 가깝습니다(analysis/auroc_mc_check.py).",
         fontsize=6.8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_auroc_law.png", dpi=170)
plt.close(fig)

# 3) Thm. 2 분해
d = REC["thm2_decomposition"]
keys = [k for k in d]
labels = ["Cube" if k.startswith("cube") else "PushT" for k in keys]
comp = {"Lbox (명령 상자)": [d[k]["Lbox"] for k in keys],
        "Llat (지연)": [d[k]["Llat"] for k in keys],
        "Lest (추정)": [d[k]["Lest"] for k in keys]}
fig, ax = plt.subplots(figsize=(5.6, 3.6))
x = np.arange(2)
bottom_pos = np.zeros(2)
cols = [BLUE, GREY, ORANGE]
for (nm, vals), col in zip(comp.items(), cols):
    v = np.array(vals, dtype=float)
    ax.bar(x, np.where(v > 0, v, 0), bottom=bottom_pos, width=0.5, color=col, label=nm)
    bottom_pos += np.where(v > 0, v, 0)
    for xi, vi, b in zip(x, v, bottom_pos - np.where(v > 0, v, 0)):
        if abs(vi) > 0.05:
            ax.text(xi + 0.27, b + vi / 2, f"{vi:+.1f}", va="center", fontsize=8, color=INK)
for xi, t in zip(x, [REC["thm2_decomposition"][k]["total"] for k in keys]):
    ax.text(xi, bottom_pos[xi] + 0.3, f"합 {t:.1f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x, labels)
ax.set_ylabel("성공률 손실 (포인트)", color=MUTED)
ax.set_title("Thm. 2 분해: 표 1 값에서 계산 (음수 Llat은 0 아래로 그리지 않음)", loc="left", fontsize=9, color=INK)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.text(0.01, 0.01, "PushT의 Llat = -1.0은 오라클@1이 오라클@0보다 높은 값이라 생긴 음수입니다. 잡음 범위 안입니다(표 1 CI).",
         fontsize=6.8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_thm2_decomp.png", dpi=170)
plt.close(fig)
print("ok", sorted(p.name for p in FIG.glob("own_*.png")))
