"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_real_wilson.png   실로봇 성공 횟수(20회)와 Wilson 95% 구간, 모달리티별
  own_sd_check.png      시뮬 표의 ±를 이항 표본 잡음과 비교 (n = 64, 192)
  own_sensitivity.png   (a) 보정 RMSE 대 최소 검출 힘 (b) 접촉 임계값 대 최소 검출 변위와 지터

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

INK, MUTED, ACC, ACC2, WARN, GREY = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a", "#9a9890"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ---------------------------------------------------------------- 그림 1
counts = R["V2_real_counts"]
tasks = ["Coin", "Chip", "Letter", "Syringe"]
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(4)
w = 0.26
for j, (name, col) in enumerate([("W", GREY), ("W+M", ACC2), ("W+FE", ACC)]):
    ks = counts[name]
    ys = [100 * k / 20 for k in ks]
    lo = [y - wilson(k, 20)[0] for y, k in zip(ys, ks)]
    hi = [wilson(k, 20)[1] - y for y, k in zip(ys, ks)]
    ax.bar(x + (j - 1) * w, ys, w, color=col, label=name, yerr=[lo, hi], capsize=3, error_kw={"lw": 1, "ecolor": INK})
    for xi, k, y in zip(x, ks, ys):
        ax.text(xi + (j - 1) * w, 3, f"{k}/20", ha="center", fontsize=8, color="white" if col != GREY else INK)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 105)
ax.set_title("실로봇 표 1: 20회 시행의 성공 횟수와 Wilson 95% 구간", fontsize=11, color=INK)
ax.legend(frameon=False, ncol=3, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "성공 횟수 = 표의 % x 20 (표는 20회를 명시). 우리가 계산.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_real_wilson.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 그림 2
rows = []
for name, cells in R["V5_sim_sd_vs_binomial"].items():
    for c in cells:
        rows.append((f"{name}/{c['task']}", c["sd_reported"], c["binom_sd_n64"], c["binom_sd_n192"]))
fig, ax = plt.subplots(figsize=(9, 4.6))
xs = np.arange(len(rows))
ax.scatter(xs, [r[1] for r in rows], color=INK, zorder=3, label="표의 ± (보고)")
ax.scatter(xs, [r[2] for r in rows], color=ACC2, marker="_", s=160, zorder=2, label="이항 잡음 sd (n = 64)")
ax.scatter(xs, [r[3] for r in rows], color=ACC, marker="_", s=160, zorder=2, label="이항 잡음 sd (n = 192)")
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 20])
ax.set_yticklabels(["1", "2", "5", "10", "20"])
ax.minorticks_off()
ax.set_xticks(xs)
ax.set_xticklabels([r[0] for r in rows], rotation=90, fontsize=6.5)
ax.set_ylabel("표준편차 (%p, 로그)")
ax.set_title("시뮬 표 1–3의 ±와 평가 횟수만으로 생기는 이항 잡음", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "n = 64: 평가 시드 하나의 롤아웃 수. n = 192: 학습 시드 하나의 평가 3개 합. 성공률은 표의 평균으로 대체. 우리가 계산.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_sd_check.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 그림 3
S = R["V7_sensitivity"]
T = R["V8_thresholds"]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.5]})
labels = ["Fx", "Fy", "Fz"]
fmin = [4.30, 4.22, 9.93]
rmse = [13.0, 9.0, 47.0]
xx = np.arange(3)
a.bar(xx - 0.18, fmin, 0.36, color=ACC, label="최소 검출 힘 (부록 A.2.2)")
a.bar(xx + 0.18, rmse, 0.36, color=WARN, label="보류 표본 RMSE (부록 A.2.1)")
a.set_xticks(xx)
a.set_xticklabels(labels)
a.set_ylabel("mN")
a.set_title("(a) 힘 성분별 비교", fontsize=10, color=INK)
a.legend(frameon=False, fontsize=8)
names = list(T.keys())
yy = np.arange(len(names))
b.barh(yy - 0.18, [T[n]["paper_mm"] for n in names], 0.36, color=ACC, label="논문 표 6")
b.barh(yy + 0.18, [T[n]["code_mm"] for n in names], 0.36, color=ACC2, label="공식 설정 파일")
b.axvline(S["dl_min_mm"], color=WARN, lw=1.5, ls="--")
b.axvline(0.012, color=GREY, lw=1.2, ls=":")
b.text(0.0135 * 1.06, 1.0, "최소 검출 변위 0.0135 mm (점선: 지터 0.012)", color=WARN, fontsize=8, transform=b.get_xaxis_transform(), va="bottom")
b.set_yticks(yy)
b.set_yticklabels(names, fontsize=9)
b.set_xscale("log")
b.set_xlim(0.0035, 0.16)
b.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1])
b.set_xticklabels(["0.005", "0.01", "0.02", "0.05", "0.1"])
b.minorticks_off()
b.set_xlabel("접촉 임계값 Δz (mm, 로그)")
b.set_title("(b) 접촉 임계값과 검출 한계", fontsize=10, color=INK, pad=22)
b.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.0))
b.invert_yaxis()
for ax in (a, b):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "RMSE와 최소 검출 힘은 정의가 다른 양이라 크기 비교만 한다. 설정 파일 값은 공식 저장소 정적 읽기. 우리가 계산.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(FIG / "own_sensitivity.png", dpi=170)
plt.close(fig)
print("wrote 3 figures")
