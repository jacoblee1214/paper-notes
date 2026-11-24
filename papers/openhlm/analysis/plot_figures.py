"""OpenHLM 분석용 그래프 세 장을 그린다 (우리가 그린 그래프, 논문 그림 아님).

실행: python3 analysis/plot_figures.py → report/figures/own_ablation.png, own_cotrain.png, own_longhorizon.png
값은 논문 그림의 막대 위 숫자를 읽은 것(그림 판독 값)과 표 3의 수치이며, analysis/recompute.py 와 같은 값을 쓴다.
"""
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"):
    if any(f.name == cand for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = cand
        break
matplotlib.rcParams["axes.unicode_minus"] = False

C1, C2, C3, C4, GREY = "#1f5fae", "#c9792b", "#2c8c6b", "#8a4fa0", "#8a8983"

# ---- 1. 절제: 4과제 평균과 "같은 구성 재측정" 폭
labels = ["기본값", "투영\n무작위 초기화", "휴머노이드\n순서", "상대\n행동", "고유감각\n제거", "PaliGemma\n초기화", "처음부터\n학습", "1단계\nflow", "1단계\ndrifting"]
avg = [91.25, 82.08, 87.08, 86.25, 81.25, 59.58, 41.67, 72.92, 70.83]
cols = [C1] + [C2] * 4 + [C4] * 2 + [C3] * 2
fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=170)
ax.axhspan(91.25 - 9.75, 91.25, color=GREY, alpha=0.18, lw=0)
ax.bar(range(9), avg, color=cols, width=0.68)
for i, v in enumerate(avg):
    ax.text(i, v + 1.3, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(range(9))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 105)
ax.set_ylabel("4과제 평균 task progress (%)")
ax.text(8.45, 91.25 - 9.75 + 1.5, "같은 기본 구성의 두 측정(그림 3 대 그림 4)에서\n과제별 차이 평균 9.8점", ha="right", va="bottom", fontsize=8, color="#555")
ax.set_title("그림 4 절제의 평균과 재측정 폭 (그림 판독 값, 과제당 5회)", fontsize=10.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "own_ablation.png")
plt.close(fig)

# ---- 2. 공동 학습: 과제별, 루브릭 점수 단위 눈금
tasks = ["9 Pig", "10 Gum", "11 Shelf\nCube", "12 Pouring"]
base = [20, 47, 40, 25]
stat = [100, 93, 73, 80]
humi = [80, 100, 73, 15]
orac = [100, 100, 87, 90]
x = np.arange(4)
w = 0.2
fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=170)
for off, v, c, lab in [(-1.5, base, GREY, "8과제 기준"), (-0.5, stat, C1, "정지 자세 공동 학습"), (0.5, humi, C4, "HuMI 공동 학습"), (1.5, orac, C3, "12과제 오라클")]:
    ax.bar(x + off * w, v, w, color=c, label=lab)
    for xi, vi in zip(x, v):
        ax.text(xi + off * w, vi + 1.5, f"{vi}", ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=9)
ax.set_ylim(0, 128)
ax.set_ylabel("task progress (%)")
ax.legend(fontsize=8, ncol=4, frameon=False, loc="upper center")
ax.set_title("보류 과제 4개 (그림 12 판독 값, 과제당 5회). Pouring에서 HuMI 15% 대 기준 25%는 점수 3점 대 5점(20점 만점)", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "own_cotrain.png")
plt.close(fig)

# ---- 3. 장기 과제: 표 3 평균과 Wilson 구간(루브릭 점수를 이항 시행으로 본 근사)
names = ["Ψ0", "GR00T N1.6", "OpenHLM\n(HuMI 공동 학습)", "OpenHLM\n(오라클)"]
mean = [48.8, 57.5, 87.5, 97.5]
se = [4.4, 4.6, 3.7, 1.7]
wil = [(38.1, 59.5), (46.6, 67.7), (78.5, 93.1), (91.3, 99.3)]
fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=170)
cols = [GREY, GREY, C1, C3]
ax.bar(range(4), mean, color=cols, width=0.6)
ax.errorbar(range(4), mean, yerr=se, fmt="none", ecolor="#222", capsize=4, lw=1.3, label="논문 표 3의 ±SE")
for i, (lo, hi) in enumerate(wil):
    ax.plot([i + 0.2, i + 0.2], [lo, hi], color=C2, lw=2.2, label="Wilson 95% (점수를 이항 시행으로 본 근사)" if i == 0 else None)
    ax.text(i, mean[i] + se[i] + 2.3, f"{mean[i]:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(4))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 112)
ax.set_ylabel("task progress (%)  (10 롤아웃, 8점 만점)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("장기 과제 비교 (표 3). 데모 시간: 2.70 h, 2.70 h, 1.14 h, 2.73 h", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "own_longhorizon.png")
plt.close(fig)
print("wrote 3 figures")
