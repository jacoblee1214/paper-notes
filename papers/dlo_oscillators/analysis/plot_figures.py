"""논문 수치를 우리가 다시 그린 그래프 세 장을 만든다 (CPU, 수 초).

출력: report/figures/own_sim_ci.png, own_real_ci.png, own_timing.png
한글 폰트는 macOS(Apple SD Gothic Neo) → Linux(Noto CJK) 순서로 찾는다.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, SOFT = "#1b1b1a", "#1f5fae", "#8a8880", "#c9d7ee"

# ---- 그림 1: Table 1 위치 RMSE (2초), 평균 ± SD, 5 시드 ----
datasets = ["Baseline", "Heavy", "Low damping", "Compliant"]
t1 = {
    "NODE": [(14.6, 0.6), (12.5, 1.1), (15.4, 1.6), (11.0, 1.3)],
    "LNN": [(17.6, 1.3), (13.6, 0.7), (24.8, 1.6), (12.9, 0.6)],
    "Φ-CON": [(14.0, 0.6), (6.6, 0.9), (14.8, 1.0), (4.5, 0.5)],
}
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
w = 0.26
colors = {"NODE": GREY, "LNN": SOFT, "Φ-CON": ACC}
for i, name in enumerate(t1):
    xs = [j + (i - 1) * w for j in range(len(datasets))]
    means = [m for m, _ in t1[name]]
    sds = [s for _, s in t1[name]]
    ax.bar(xs, means, w, yerr=sds, capsize=3, color=colors[name], edgecolor=INK, linewidth=0.5, label=name)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets)
ax.set_ylabel("위치 RMSE at 2 s [mm]")
ax.set_title("표 1 재작도: 2초 예측 위치 오차 (평균 ± SD, 5 시드)", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "출처: arXiv:2611.23563 v1 표 1. 막대와 오차선은 논문 값, 그림은 저자 작성.", fontsize=6.5, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_sim_ci.png")
plt.close(fig)

# ---- 그림 2: 실로봇 끝점 오차 (PD 대 PD+FF) 와 per-target 개선율 95% 구간 ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=200, gridspec_kw={"width_ratios": [1.1, 1]})
ends = ["왼쪽", "오른쪽"]
pd_m, pd_s = [63.7, 48.3], [8.0, 8.5]
ff_m, ff_s = [48.4, 32.3], [5.6, 5.2]
xs = [0, 1]
a1.bar([x - 0.18 for x in xs], pd_m, 0.36, yerr=pd_s, capsize=3, color=GREY, edgecolor=INK, linewidth=0.5, label="PD (임피던스)")
a1.bar([x + 0.18 for x in xs], ff_m, 0.36, yerr=ff_s, capsize=3, color=ACC, edgecolor=INK, linewidth=0.5, label="PD + Φ-CON FF")
a1.set_xticks(xs)
a1.set_xticklabels(ends)
a1.set_ylabel("최종 끝점 오차 [mm]")
a1.set_title("끝점별 최종 오차 (76개 정상상태 목표)", fontsize=9)
a1.legend(frameon=False, fontsize=7)
a1.spines[["top", "right"]].set_visible(False)

# 개선율 95% CI (정규·t 근사, n = 76, SD는 목표 간 분산으로 가정)
rel = [(23.5, 8.8), (32.2, 10.5)]
n_ = 76
cis = []
for m, s in rel:
    h = stats.t.ppf(0.975, n_ - 1) * s / math.sqrt(n_)
    cis.append((m - h, m + h))
for x, (m, s), (lo, hi) in zip(xs, rel, cis):
    a2.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=ACC, capsize=4, markersize=6)
a2.axhline(27.9, color=GREY, linestyle="--", linewidth=0.8)
a2.text(1.45, 28.6, "평균 27.9%", fontsize=7, color=GREY, ha="right")
a2.set_xticks(xs)
a2.set_xticklabels(ends)
a2.set_xlim(-0.5, 1.5)
a2.set_ylim(0, 45)
a2.set_ylabel("목표별 상대 개선율 [%]")
a2.set_title("개선율 평균과 95% 구간 (n = 76)", fontsize=9)
a2.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "출처: arXiv:2611.23563 v1 §6.3 본문 수치. 구간은 정규·t 근사, 저자 계산.", fontsize=6.5, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_real_ci.png")
plt.close(fig)

# ---- 그림 3: 계산 시간 Table 2 대 Table 8 (같은 측정이라고 본문이 적음) ----
labels = ["Φ-CON CPU", "Φ-CON GPU", "LNN CPU", "LNN GPU"]
t2 = [(12.6, 3.4), (72.1, 8.1), (25.5, 2.9), (114.6, 4.1)]
t8 = [(13.9, 3.0), (64.8, 15.8), (25.4, 3.9), (105.2, 5.0)]
fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=200)
xs = list(range(len(labels)))
ax.bar([x - 0.18 for x in xs], [m for m, _ in t2], 0.36, yerr=[s for _, s in t2], capsize=3,
       color=GREY, edgecolor=INK, linewidth=0.5, label="표 2 (제어 한 걸음)")
ax.bar([x + 0.18 for x in xs], [m for m, _ in t8], 0.36, yerr=[s for _, s in t8], capsize=3,
       color=ACC, edgecolor=INK, linewidth=0.5, label="표 8 (예측 한 번)")
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylabel("시간 [us] (평균 ± SD)")
ax.set_title("같은 측정으로 적힌 두 표의 계산 시간이 다릅니다", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "출처: arXiv:2611.23563 v1 표 2, 표 8과 부록 C 문장. 그림은 저자 작성.", fontsize=6.5, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_timing.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
