"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_il_ci.png      : 모방학습 성공률(그림 8 막대 라벨, 시행 수 25 가정)과 증강 없음 Wilson 95% 구간
own_force_gap.png  : 힘 추정 MAE (그림 6(b) 판독 값, 표 6)와 생성 데이터 단독 값의 출처별 차이
own_refcopy.png    : 표 1 MSE(로그 축)와 참조 복사 기준선(부록 E.1), SSIM
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c25b1a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e3dc", lw=.6)
    ax.set_axisbelow(True)


# ---- 1. 모방학습
demos = [20, 35, 50, 70, 100]
no = [6, 7, 12, 14, 15]
a9 = [5, 11, 18, 17, 18]
a4 = [6, 9, 17, 17, 18]
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=170)
x = np.arange(len(demos))
for xs, ks, col, lab, off in [(x, no, GREY, "증강 없음", -0.12), (x, a9, ACC, "증강 사본 9개", 0.12)]:
    ci = np.array([wilson(k, 25) for k in ks])
    ax.errorbar(xs + off, [100 * k / 25 for k in ks], yerr=[[100 * k / 25 - c[0] for k, c in zip(ks, ci)], [c[1] - 100 * k / 25 for k, c in zip(ks, ci)]],
                fmt="o", color=col, capsize=3, label=lab, ms=5)
ax.plot(x + 0.0, [100 * k / 25 for k in a4], "s", color=ORG, ms=4, alpha=.8, label="증강 사본 4개 (점만)")
ax.set_xticks(x, [str(d) for d in demos])
ax.set_xlabel("실제 시연 수")
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 100)
ax.set_title("모방학습 성공률 (그림 8 라벨, 칸당 25회 가정, Wilson 95%)", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_il_ci.png")
plt.close(fig)

# ---- 2. 힘 추정 MAE
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=170, gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
cats = ["1/20", "1/4", "1/3", "1/2", "전체"]
real = [0.53, 0.37, 0.29, 0.24, 0.06]
g15 = [0.24, 0.18, 0.12, 0.10, 0.065]
g30 = [0.21, 0.16, 0.10, 0.08, 0.065]
xx = np.arange(len(cats))
w = .26
ax.bar(xx - w, real, w, color=GREY, label="실제만")
ax.bar(xx, g15, w, color=ACC, label="+ 생성 15k")
ax.bar(xx + w, g30, w, color=ORG, label="+ 생성 30k")
ax.axhline(0.06, color=INK, lw=.8, ls="--")
ax.text(4.45, 0.075, "전체 실제 0.06", fontsize=7.5, ha="right")
ax.set_xticks(xx, cats)
ax.set_xlabel("실제 데이터 비율")
ax.set_ylabel("MAE (N)")
ax.set_title("그림 6(b) 판독 값 (눈금 ±0.01)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8)
style(ax)
ax = axs[1]
srcs = ["표 6\n(4–10 N?)", "그림 6(b)\n판독", "v1 표 5\n(1–10 N)"]
v15 = [0.21, 0.27, 0.35]
v30 = [0.17, 0.23, 0.33]
xx = np.arange(3)
ax.bar(xx - .18, v15, .36, color=ACC, label="생성 15k")
ax.bar(xx + .18, v30, .36, color=ORG, label="생성 30k")
ax.axhline(0.053, color=INK, lw=.8, ls="--")
ax.text(2.4, 0.065, "실제 15k 0.053", fontsize=7.5, ha="right")
ax.set_xticks(xx, srcs, fontsize=8)
ax.set_ylabel("MAE (N)")
ax.set_title("생성 데이터 단독 MAE: 출처별", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_force_gap.png")
plt.close(fig)

# ---- 3. 표 1 + 참조 복사
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=170)
meth = ["Simulator", "Sim2Real", "Separate", "Hybrid", "ControlTac"]
mse = [1065, 253, 199, 37, 26]
ssim = [.69, .73, .72, .75, .79]
col = [GREY, GREY, GREY, GREY, ACC]
ax = axs[0]
ax.barh(meth, mse, color=col)
ax.set_xscale("log")
ax.axvline(233, color=ORG, lw=1.3, ls="--")
ax.text(240, 4.35, "참조 복사 233", color=ORG, fontsize=8)
ax.set_xlabel("MSE (낮을수록 좋음, 로그 축)")
ax.set_title("FeelAnyForce 미관측 객체", fontsize=10, color=INK)
style(ax)
ax = axs[1]
ax.barh(meth, ssim, color=col)
ax.set_xlim(.6, .85)
ax.axvline(.78, color=ORG, lw=1.3, ls="--")
ax.text(.781, 4.35, "참조 복사 0.78", color=ORG, fontsize=8)
ax.set_xlabel("SSIM (높을수록 좋음)")
ax.set_title("같은 표 1 값 (표 1, 부록 E.1)", fontsize=10, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_refcopy.png")
plt.close(fig)
print("saved 3 figures to", FIG)
