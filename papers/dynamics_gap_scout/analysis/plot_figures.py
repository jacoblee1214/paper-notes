"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_ise_welch.png   : 표 1 SCOUT 대 이력 조건 GMM (평균 ± 표준오차)과 보고 p 대 정규근사 p
own_door_wilson.png : 표 2 문 성공률과 Wilson 95% 구간 (시도 170회 가정)
own_realworld.png   : 그림 3 위치별 시도 수 (95% CI 반폭에서 역산한 SD), 표 5 p 값
실행: python3 analysis/plot_figures.py   (numpy, matplotlib, scipy; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, WARN = "#1b1b1a", "#1f5fae", "#9a9890", "#b5552b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ── 1. 표 1: 평균 ± SE, 보고 padj 대 정규근사 p ─────────────────────────
tasks = ["Slide Brick", "Push Bar", "Pick Bar", "Open Box", "Turn Faucet"]
hist_m, hist_se = [3.92, 4.65, 3.36, 1.87, 1.61], [0.23, 0.22, 0.11, 0.05, 0.03]
scout_m, scout_se = [3.37, 4.14, 2.83, 1.83, 1.54], [0.19, 0.18, 0.17, 0.05, 0.03]
padj = [4.93e-24, 7.29e-13, 3.75e-26, 7.40e-5, 1.0]
p_norm = []
for i in range(5):
    z = (hist_m[i] - scout_m[i]) / math.sqrt(hist_se[i] ** 2 + scout_se[i] ** 2)
    p_norm.append(2 * (1 - stats.norm.cdf(abs(z))))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
x = np.arange(5)
ax.errorbar(x - 0.13, hist_m, yerr=hist_se, fmt="o", color=GREY, capsize=3, label="이력 조건 GMM")
ax.errorbar(x + 0.13, scout_m, yerr=scout_se, fmt="s", color=ACC, capsize=3, label="SCOUT")
ax.set_xticks(x, tasks, fontsize=8.5)
ax.set_ylabel("평균 시도 수 (± 표준오차)")
ax.set_title("표 1: 평균과 표준오차 (원문 보고값)", fontsize=10)
ax.legend(frameon=False, fontsize=8.5)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
ax.bar(x - 0.18, [-math.log10(p) for p in padj], width=0.36, color=GREY, label="보고 padj (MW, Holm)")
ax.bar(x + 0.18, [-math.log10(p) for p in p_norm], width=0.36, color=WARN, label="정규근사 p (SE만 사용)")
ax.axhline(-math.log10(0.05), color=INK, lw=0.8, ls="--")
ax.text(4.45, -math.log10(0.05) + 0.6, "p = 0.05", fontsize=7.5, ha="right")
ax.set_xticks(x, tasks, fontsize=8.5, rotation=20)
ax.set_ylabel("-log10 p")
ax.set_title("부록 C.1: 보고 p 대 SE 근사 p", fontsize=10)
ax.legend(frameon=False, fontsize=7.8, loc="upper right")
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "근사 p는 평균과 표준오차에서 얻은 정규 Welch 근사입니다. 원자료가 없어 순위 검정과 같지 않습니다.",
         fontsize=7.5, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_ise_welch.png", dpi=170)
plt.close(fig)

# ── 2. 표 2: 문 성공률과 Wilson 구간 (n = 170 가정) ─────────────────────
N = 170
methods = ["원시 GMM", "이력 조건\nGMM", "PNDiT\n(생성기)", "HAVE\n5 샘플", "HAVE\n20 샘플", "SCOUT\n1 스텝", "SCOUT\n5 스텝"]
srs = [0.61, 0.68, 0.68, 0.82, 0.84, 0.83, 0.97]
fig, ax = plt.subplots(figsize=(9, 4.2))
xs = np.arange(len(methods))
lo, hi = zip(*[wilson(round(s * N), N) for s in srs])
colors = [GREY if "SCOUT" not in m else ACC for m in methods]
ax.bar(xs, srs, color=colors, width=0.6)
ax.errorbar(xs, srs, yerr=[np.array(srs) - np.array(lo), np.array(hi) - np.array(srs)],
            fmt="none", ecolor=INK, capsize=3, lw=0.9)
for xi, s, h in zip(xs, srs, hi):
    ax.text(xi, h + 0.015, f"{s:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(xs, methods, fontsize=8)
ax.set_ylim(0, 1.08)
ax.set_ylabel("성공률")
ax.set_title("표 2: 문 벤치마크 성공률과 Wilson 95% 구간 (시도 170회 가정)", fontsize=10)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "시도 수는 논문에 없습니다. 저장소의 34개 테스트 물체 x 5회 = 170을 가정했습니다.", fontsize=7.5, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_door_wilson.png", dpi=170)
plt.close(fig)

# ── 3. 그림 3: 위치별 시도 수 (CI 반폭 -> SD -> 평균의 95% 구간) ─────────
t975 = stats.t.ppf(0.975, 9)
cells = {
    "Pick Bar": {"비-TTT": [(3.2, 1.9), (3.6, 1.8), (3.0, 1.6)], "SCOUT": [(2.0, 0.5), (2.7, 2.0), (1.8, 0.7)]},
    "Push Bar": {"비-TTT": [(6.1, 3.0), (6.1, 2.5), (4.6, 2.0)], "SCOUT": [(2.9, 0.8), (3.7, 0.5), (2.1, 2.0)]},
}
pvals = {"Pick Bar": 0.035, "Push Bar": 0.003}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=False)
positions = ["왼쪽", "가운데", "오른쪽"]
for ax, (task, arms) in zip(axes, cells.items()):
    for j, (arm, col) in enumerate([("비-TTT", GREY), ("SCOUT", ACC)]):
        for i, (m, h) in enumerate(arms[arm]):
            ax.errorbar(i + (j - 0.5) * 0.22, m, yerr=h, fmt="o", color=col, capsize=3,
                        label=arm if i == 0 else None)
    ax.set_xticks(range(3), positions, fontsize=8.5)
    ax.set_ylabel("성공까지 시도 수 (95% CI)")
    ax.set_title(f"{task}  (풀링 Mann-Whitney p = {pvals[task]})", fontsize=10)
    ax.set_ylim(0, 10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "값은 원문 그림 3 표의 텍스트에서 옮겼습니다. 시행은 위치별 10회이고 최대 10 시도입니다.",
         fontsize=7.5, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_realworld.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
