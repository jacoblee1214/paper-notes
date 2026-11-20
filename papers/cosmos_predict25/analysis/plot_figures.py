"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 여기서 만들지 않는다.

  own_real_ci.png          표 13 실로봇 성공률과 Wilson 95% 구간 (재계산)
  own_pref_sensitivity.png 그림 6–7 사람 선호: 가상 N에 따른 승률 차이 검정 p값 (N 미보고)
  own_transfer_consistency.png 표 14 다시점 일관성: 실제 영상 대비 Transfer1과 Transfer2.5 (낮을수록 좋음)

입력: analysis/recompute_results.json (표 수치는 recompute.py가 같이 넣음)
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
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
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


# 1. 실로봇 성공률 (표 13)
t13 = R["table13_real_robot"]
names = ["Base", "Baseline", "Proposed"]
labels = ["기본 정책", "표준 증강", "Transfer2.5 증강"]
rates = [100 * t13[n]["rate"] for n in names]
lo = [100 * t13[n]["rate"] - 100 * t13[n]["wilson95"][0] for n in names]
hi = [100 * t13[n]["wilson95"][1] - 100 * t13[n]["rate"] for n in names]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
x = np.arange(len(names))
ax.bar(x, rates, color=[MUTED, ACC2, ACC], width=0.55)
ax.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=5, lw=1)
for xi, n, r in zip(x, names, rates):
    ax.text(xi, r + 6, f"{t13[n]['successes']}/{t13[n]['trials']}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, labels)
ax.set_ylim(0, 105)
ax.set_ylabel("성공률 (%)", color=INK)
ax.set_title("실로봇 성공률 (표 13 재계산, 95% Wilson 구간)", fontsize=10, color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png")
plt.close(fig)

# 2. 사람 선호 검정 민감도 (그림 6–7, N 미보고)
pref = R["human_preference_fig6_7"]
Ns = np.array([50, 100, 200, 400, 800, 1600, 3200])
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
styles = [(ACC2, "--"), (MUTED, ":"), (ACC, "-"), (WARN, "-.")]
for (name, rec), (color, ls) in zip(pref.items(), styles):
    share = rec["win_share_of_decisive"]
    frac = rec["decisive_fraction_of_all_votes"]  # 무승부를 뺀 비율 (보고서 수치 그대로)
    pv = []
    for N in Ns:
        nd = int(round(N * frac))
        kw = int(round(nd * share))
        pv.append(stats.binomtest(kw, nd, 0.5).pvalue)
    ax.plot(Ns, pv, color=color, ls=ls, lw=1.6, label=name)
ax.axhline(0.05, color=INK, lw=0.8, ls="--")
ax.text(Ns[-1], 0.055, "p = 0.05", ha="right", fontsize=8, color=INK)
ax.set_xscale("log")
ax.set_xlabel("가상 비교 수 N (무승부를 뺀 비율은 보고서 값 그대로)", color=INK, fontsize=9)
ax.set_ylabel("양측 p값", color=INK)
ax.set_title("사람 선호: 논문 승률이 유지될 때 N에 따른 p값 (그림 6–7)", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=7, frameon=False)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_pref_sensitivity.png")
plt.close(fig)

# 3. 다시점 일관성 (표 14): 값이 낮을수록 좋은 지표, 실제 영상 기준 대비 비율
real_tse, real_cse = 1.193, 1.832
rows = {
    "TSE": {"Transfer1-7B": 1.017, "Transfer2.5-2B": 1.246, "실제 영상": real_tse},
    "CSE": {"Transfer1-7B": 1.835, "Transfer2.5-2B": 2.310, "실제 영상": real_cse},
}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), dpi=200)
for ax, (metric, vals) in zip(axes, rows.items()):
    keys = list(vals.keys())
    colors = [MUTED, ACC, INK]
    ax.bar(range(3), [vals[k] for k in keys], color=colors, width=0.6)
    for i, k in enumerate(keys):
        ax.text(i, vals[k] + 0.05, f"{vals[k]:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(3), ["Transfer1", "Transfer2.5", "실제"], fontsize=8)
    ax.set_title(f"{metric} (낮을수록 좋음)", fontsize=9, color=INK, loc="left")
    ax.set_ylim(0, max(vals.values()) * 1.25)
    style(ax)
fig.suptitle("다시점 일관성 (표 14, 다중 뷰 생성)", fontsize=10, color=INK, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(FIG / "own_transfer_consistency.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
