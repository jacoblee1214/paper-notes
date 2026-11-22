"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 만들거나 자르지 않는다.

  own_scaling.png     원자 24과제의 과제별 성공률(점)과 설정별 평균 ± 평가 잡음 95% 구간 (Fig. 13 값)
  own_paired.png      과제별 차이 G3000 - H50 (대응 비교, 정렬한 막대)
  own_realworld.png   실로봇 과제 평균 ± 3시드 SD (Fig. 10 값), 보이는 객체와 미본 객체

입력: analysis/recompute_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

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
INK, MUTED, ACC, ACC2 = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

SETTINGS = ["H50", "G100", "G300", "G3000"]
LABELS = {"H50": "사람 50/과제", "G100": "생성 100/과제", "G300": "생성 300/과제", "G3000": "생성 3000/과제"}
tasks = R["fig13_tasks"]
rng = np.random.default_rng(1)

# ------------------------------------------------ 그래프 1: 설정별 과제 점과 평균 구간
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
for i, s in enumerate(SETTINGS):
    vals = np.array([t[s] for t in tasks]) * 100
    x = i + rng.uniform(-0.18, 0.18, len(vals))
    ax.scatter(x, vals, s=12, color=MUTED, alpha=0.55, linewidths=0, zorder=2)
    m = R["fig13_column_means"][s] * 100
    lo, hi = (np.array(R["avg_ci_binomial"][s]) * 100)
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="D", color=ACC, ms=6, capsize=4, lw=1.6, zorder=3)
    ax.text(i + 0.22, m, f"{R['fig13_paper_average'][s] * 100:.1f}%", va="center", color=ACC, fontsize=8.5)
ax.set_xticks(range(len(SETTINGS)))
ax.set_xticklabels([LABELS[s] for s in SETTINGS], fontsize=8)
ax.set_ylabel("과제별 성공률 (%)")
ax.set_ylim(-3, 104)
ax.set_title("원자 24과제: 과제별 점과 평균 ± 평가 잡음 95% 구간", fontsize=10, loc="left", color=INK)
fig.text(0.01, 0.01, "과제당 50회 평가, 학습 시드 수는 미확인이라 구간은 하한. 논문 Fig. 13 값에서 계산.",
         fontsize=6.8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_scaling.png")
plt.close(fig)

# ------------------------------------------------ 그래프 2: 과제별 차이 G3000 - H50
diff = sorted(((t["G3000"] - t["H50"]) * 100, t["task"]) for t in tasks)
vals = [d for d, _ in diff]
labels = [n for _, n in diff]
colors = [ACC if v > 0 else ACC2 for v in vals]
fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
ax.barh(range(len(vals)), vals, color=colors, height=0.7)
ax.axvline(0, color=INK, lw=0.8)
ax.axvline(R["paired_task_test"]["G3000_vs_H50"]["mean_diff"] * 100, color=MUTED, lw=1, ls="--")
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=6.5)
ax.set_xlabel("성공률 차이 G3000 - H50 (pp)")
t_ = R["paired_task_test"]["G3000_vs_H50"]
ax.set_title(f"과제별 차이: 양 {t_['n_positive']}개, 음 {t_['n_negative']}개, 평균 {t_['mean_diff']*100:.1f} pp (점선)",
             fontsize=9.5, loc="left", color=INK)
fig.text(0.01, 0.01, "과제 대응 t 검정 p < 1e-6 (과제를 표본으로 봄). 논문 Fig. 13 값에서 계산.",
         fontsize=6.8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_paired.png")
plt.close(fig)

# ------------------------------------------------ 그래프 3: 실로봇 과제 평균 ± 근사 SD
RW = R["real_world"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), dpi=200, sharey=True)
for ax, cat, title in zip(axes, ["seen", "unseen"], ["보이는 객체", "미본 객체"]):
    d = RW[cat]
    means = [d["real"]["avg_from_tasks"], d["sim"]["avg_from_tasks"]]
    sds = [d["real"]["sd_of_avg_approx"], d["sim"]["sd_of_avg_approx"]]
    xs = [0, 1]
    ax.bar(xs, means, yerr=sds, capsize=5, color=[MUTED, ACC], width=0.55,
           error_kw={"lw": 1.2, "ecolor": INK})
    for x, m, sd in zip(xs, means, sds):
        ax.text(x, m + sd + 1.0, f"{m:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(["실로봇만", "실로봇 + 시뮬"], fontsize=8)
    ax.set_title(f"{title} (p ≈ {d['welch_p_approx']:.3f})", fontsize=9, loc="left", color=INK)
axes[0].set_ylabel("과제 평균 성공률 (%)")
axes[0].set_ylim(0, 40)
fig.text(0.01, 0.01, "막대: 3시드 과제 평균. 오차 막대: 과제 독립 가정 근사 SD (논문 Fig. 10 값에서 계산).",
         fontsize=6.4, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_realworld.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
