"""우리가 만든 그래프 두 장. 논문 그림이 아니라 표 값(recompute_results.json)에서 그린다.

실행: python3 analysis/plot_figures.py → report/figures/own_rounds.png, own_forest.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
if Path(FONT).exists():
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
INK, ACCENT, GREY = "#1b1b1a", "#1f5fae", "#9a978e"

# 1) 피드백 라운드에 따른 누적 성공률 (Table II 합계, Wilson 95% 구간)
rounds = list(range(6))
with_sr = [26, 34, 36, 37, 38, 39]
without_sr = [20, 31, 36, 37, 38, 38]
from math import sqrt


def wilson(k, n=40, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
for series, color, label, dx in ((with_sr, ACCENT, "자기 검토 있음", 0.06), (without_sr, GREY, "자기 검토 없음", -0.06)):
    ys = [100 * k / 40 for k in series]
    lo = [100 * wilson(k)[0] for k in series]
    hi = [100 * wilson(k)[1] for k in series]
    x = [r + dx for r in rounds]
    ax.errorbar(x, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                fmt="o-", color=color, ecolor=color, elinewidth=0.8, capsize=2, label=label, markersize=4)
ax.set_xlabel("피드백 라운드 (시뮬레이션, 시도 40회)", color=INK)
ax.set_ylabel("누적 성공률 (%)", color=INK)
ax.set_xticks(rounds)
ax.set_ylim(40, 105)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_rounds.png")
plt.close(fig)

# 2) 방법별 전체 성공률과 Clopper-Pearson 95% 구간 (Table III, Table IV)
rows = [
    ("OaP, 2라운드 (시뮬)", 36, 40),
    ("OaP, 1라운드 (시뮬)", 34, 40),
    ("OaP, 0라운드 자기 검토 (시뮬)", 26, 40),
    ("CoRAL (adapted, 시뮬)", 31, 40),
    ("ReKep (시뮬)", 17, 40),
    ("L2R (시뮬)", 11, 40),
    ("pi0.5 (시뮬)", 10, 40),
    ("OpenVLA-OFT (시뮬)", 0, 40),
    ("OaP, 실로봇 (선정된 프로그램)", 32, 40),
]
from scipy import stats


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
for i, (name, k, n) in enumerate(rows):
    p = 100 * k / n
    lo, hi = cp(k, n)
    y = len(rows) - 1 - i
    color = ACCENT if name.startswith("OaP") else GREY
    ax.plot([100 * lo, 100 * hi], [y, y], color=color, lw=1.4)
    ax.plot([p], [y], "o", color=color, ms=4)
    ax.text(102, y, f"{k}/{n}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
ax.set_xlim(0, 118)
ax.set_xlabel("성공률 (%), Clopper-Pearson 95% 구간", color=INK)
ax.grid(axis="x", alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_forest.png")
plt.close(fig)
print("wrote", FIG / "own_rounds.png", FIG / "own_forest.png", "intervals ok:", len(RES["intervals"]) > 0)
