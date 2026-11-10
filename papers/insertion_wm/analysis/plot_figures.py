"""InsertionWM 직접 그린 그래프 네 장 (CPU). 실행: python3 analysis/plot_figures.py → report/figures/

1) fig_headline_ci.png: 헤드라인 수치(미지 56%, 학습 55%, AutoMate 7%)와 Wilson 95% 구간, AutoMate 시도 수 가정 표시
2) fig_scaling_read.png: 논문 Fig. 1에서 읽은 학습 객체 수 대 성공률(눈으로 읽음, ±2pp). 학습 규모가 다른 점을 표시
3) fig_toy_model_error.png: toy 계획자 성공률 대 모델 오차 δ(c 별), 그리고 대 간극 c(δ=0.05)
4) fig_toy_compare.png: toy 미지 기하에서 계획자와 모델 없는 정책(kNN) 성공률 비교
논문 그림은 사용하지 않는다. 수치는 table_check_results.json과 insertion_toy_results.json에서 읽는다.
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
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, GRID = "#0b0b0b", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
T = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
Y = json.loads((ROOT / "analysis" / "insertion_toy_results.json").read_text())


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 1) 헤드라인과 Wilson 구간
fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
rows = [
    ("미지 객체 10개\n(우리, 1000회)", 0.56, wilson(560, 1000), BLUE),
    ("학습 객체 90개\n(우리, 9000회, 본문 55.0%)", 0.545, wilson(round(0.545 * 9000), 9000), BLUE),
    ("AutoMate 7%\n(시도 100회 가정)", 0.07, wilson(7, 100), MUTED),
    ("AutoMate 7%\n(시도 1000회 가정)", 0.07, wilson(70, 1000), MUTED),
]
for i, (lab, p, (lo, hi), col) in enumerate(rows):
    ax.errorbar(p * 100, i, xerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]], fmt="o", color=col, ms=6, capsize=4)
    ax.text(p * 100 + 2.5, i + 0.18, f"{p*100:.1f}%", color=INK, fontsize=9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 75)
ax.set_xlabel("성공률 (%), Wilson 95% 구간")
style(ax)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.grid(axis="y", visible=False)
ax.set_title("헤드라인 수치와 불확실성 (시도 수는 논문 기재값 또는 가정)", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig_headline_ci.png")
plt.close(fig)

# 2) 논문 Fig. 1에서 읽은 규모 곡선 (값은 눈으로 읽음)
ours_x = [5, 15, 45, 75, 90]
ours_unseen = [27, 40, 50, 53, 56]
ours_seen = [59, 64, 62, 57, 55]
auto_seen_x = [5, 15, 25, 45, 50, 60, 75, 80]
auto_seen = [85, 81, 55, 27, 24, 22, 21, 16]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=200, sharey=True)
ax = axes[0]
ax.plot(ours_x, ours_seen, "-o", color=BLUE, label="우리 (학습 객체)")
ax.plot(auto_seen_x, auto_seen, "-s", color=MUTED, label="AutoMate (학습 객체, 본문 80개 15.6%)")
ax.set_title("학습 객체 안(seen)", fontsize=9, loc="left")
ax.set_xlabel("학습 객체 수")
ax.set_ylabel("성공률 (%)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
style(ax)
ax = axes[1]
ax.plot(ours_x, ours_unseen, "-o", color=ORANGE, label="우리 (미지 객체 10개)")
ax.scatter([20], [7], s=36, color=MUTED, zorder=3, label="AutoMate 미지 7% (x 약 15–25)")
ax.axvline(15, color=GRID, lw=0.8)
ax.annotate("같은 x에서 우리 약 40%", xy=(15, 40), xytext=(28, 18), fontsize=7,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
ax.set_title("학습 객체 밖(unseen), 헤드라인 비교", fontsize=9, loc="left")
ax.set_xlabel("학습 객체 수")
ax.legend(fontsize=7, frameon=False, loc="lower right")
style(ax)
ax.set_ylim(0, 100)
fig.suptitle("논문 Fig. 1을 눈으로 읽어 다시 그림 (x 위치와 값 모두 대략, 학습 규모가 다른 점에 주의)", fontsize=8.5, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig_scaling_read.png")
plt.close(fig)

# 3) 계획자: 모델 오차 δ, 간극 c
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=200, sharey=True)
ax = axes[0]
dd = Y["planner_vs_model_error_c0.1_N128"]
xs = [float(k) for k in dd]
ys = [dd[k]["rate"] * 100 for k in dd]
lo = [dd[k]["wilson95"][0] * 100 for k in dd]
hi = [dd[k]["wilson95"][1] * 100 for k in dd]
ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]], fmt="-o", color=BLUE, capsize=3)
ax.axvline(0.05, color=GRID, lw=1)
ax.text(0.052, 8, "δ = c/2 (c = 0.1)", fontsize=7, color=MUTED)
ax.text(0.0, 8, "", fontsize=7)
ax.set_title("간극 c = 0.1에서 모델 오차 δ (N=128)", fontsize=9, loc="left")
ax.set_xlabel("벽 반폭 오차 δ (단위: peg 반지름 0.5)")
ax.set_ylabel("성공률 (%)")
style(ax)
ax = axes[1]
cc = Y["planner_vs_clearance_d0.05_N128"]
xs = [float(k) for k in cc]
ys = [cc[k]["rate"] * 100 for k in cc]
lo = [cc[k]["wilson95"][0] * 100 for k in cc]
hi = [cc[k]["wilson95"][1] * 100 for k in cc]
ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]], fmt="-o", color=AQUA, capsize=3)
ax.axvline(0.1, color=GRID, lw=1)
ax.text(0.105, 8, "c = 2δ = 0.1", fontsize=7, color=MUTED)
ax.set_xticks([0.05, 0.1, 0.2, 0.4])
ax.set_title("모델 오차 δ = 0.05에서 간극 c (N=128)", fontsize=9, loc="left")
ax.set_xlabel("간극 c")
style(ax)
ax.set_ylim(0, 105)
fig.suptitle("toy: δ ≥ c/2이면 모델이 간극을 좁게 본 에피소드가 전부 실패한다 (각 셀 150회, Wilson 95%)", fontsize=8.5, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig_toy_model_error.png")
plt.close(fig)

# 4) toy: 미지 기하에서 계획자 대 모델 없는 정책
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=200)
cmp_ = Y["planner_vs_bc_unseen"]
names = {"clearance_0.05": "간극 0.05\n(학습 0.2–0.6)", "clearance_0.1": "간극 0.1\n(학습 0.2–0.6)",
         "offset_0.1_hr0.6": "오프셋 |h|≤0.6\n(학습 |h|≤0.3)"}
x = range(len(cmp_))
w = 0.36
for i, (k, v) in enumerate(cmp_.items()):
    for j, (lab, col) in enumerate((("계획자 (δ=0.05, 모델 오차)", BLUE), ("모델 없는 정책 (kNN)", ORANGE))):
        d = v["planner" if j == 0 else "bc"]
        ax.bar(i + (j - 0.5) * w, d["rate"] * 100, w, color=col, label=lab if i == 0 else None)
        ax.errorbar(i + (j - 0.5) * w, d["rate"] * 100,
                    yerr=[[d["rate"] * 100 - d["wilson95"][0] * 100], [d["wilson95"][1] * 100 - d["rate"] * 100]],
                    color=INK, capsize=3, lw=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels([names[k] for k in cmp_], fontsize=8)
ax.set_ylim(0, 115)
ax.set_ylabel("성공률 (%)")
ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
style(ax)
ax.set_title("toy: 미지 기하에서 두 방식 비교 (150회, Wilson 95%). 이 toy에서는 모델 없는 정책이 더 좋음", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig_toy_compare.png")
plt.close(fig)
print("wrote figures to", OUT)
