"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 여기서 그리지 않는다).

own_task_ci.png   : 표 4·5 과제별 성공률과 Wilson 95% 구간, 실데이터 기준선 (n = 100 가정)
own_ewm_check.png : 표 2·3 평균 x100과 그림 1(a) 막대 비교, EWMScore와 데이터 엔진 평균 산점도
own_gate_toy.png  : 논문 식(min(1, Sdyn/γ))과 코드 규칙(dyn <= 0.1213이면 곱함)의 점수 차이
실행: python3 analysis/plot_figures.py   (numpy, scipy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
R = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def wilson(p_pct, n=100, z=1.96):
    p = p_pct / 100
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half) * 100, (center + half) * 100


# ---------------------------------------------------------------- 1. 과제별 성공률과 Wilson 구간
t4 = R["table4_stats"]["rows"]
names = list(t4.keys())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, key, real, title in [
    (axes[0], "task1", 77, "과제 1 (adjust bottle)"),
    (axes[1], "task2", 66, "과제 2 (click bell)"),
]:
    ys = np.arange(len(names))
    vals = [t4[n][key] for n in names]
    lo = [wilson(v)[0] for v in vals]
    hi = [wilson(v)[1] for v in vals]
    ax.errorbar(vals, ys, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                fmt="o", color="#1f5fae", ecolor="#8aa9d6", capsize=3)
    ax.axvspan(wilson(real)[0], wilson(real)[1], color="#e8a33d", alpha=0.25)
    ax.axvline(real, color="#b5651d", lw=1.2, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("성공률 (%)")
    ax.set_xlim(-2, 100)
    ax.grid(alpha=0.25)
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels(names, fontsize=8.5)
axes[0].invert_yaxis()
fig.suptitle("데이터 엔진 성공률 (표 4) 과 Wilson 95% 구간, 주황 띠 = 실데이터 학습 기준선 Wilson 구간", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "own_task_ci.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. 재계산 대 그림 1, EWMScore와 데이터 엔진
ew = R["ewmscore_recompute"]["per_model"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
xs = [ew[m]["fig1"] for m in ew]
ys = [ew[m]["mean16_x100"] for m in ew]
ax.scatter(xs, ys, color="#1f5fae", s=28)
lim = [40, 65]
ax.plot(lim, lim, color="#888", lw=1, ls="--")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("그림 1(a) EWMScore (논문)")
ax.set_ylabel("표 2·3 평균 x100 (재계산)")
ax.set_title(f"재계산 일치 (최대 차이 {R['ewmscore_recompute']['max_abs_diff']})", fontsize=10)
ax.grid(alpha=0.25)
ax = axes[1]
v = R["ewm_vs_tasks_n6"]
models = v["models"]
ewm6 = [ew[m]["fig1"] for m in models]
avg4 = [np.mean([t4[m]["task1"], t4[m]["task2"]]) for m in models]
ax.scatter(ewm6, avg4, color="#1f5fae", s=36)
for m, x, y in zip(models, ewm6, avg4):
    ax.annotate(m, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7.5)
ax.set_xlabel("EWMScore (그림 1(a))")
ax.set_ylabel("데이터 엔진 평균 성공률 (%, 표 4)")
ax.set_title(f"n = 6, 논문 r = {v['data_engine_avg_r_paper']:.3f}, 재계산 r = {v['data_engine_avg_r'][0]:.3f} (p = {v['data_engine_avg_r'][1]:.2f})", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_ewm_check.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. 동적 게이트 규칙 차이 (toy 식과 같은 식)
GAMMA = 0.1213
dyn = np.linspace(0.0, 0.5, 501)
raw = 0.90
paper = raw * np.minimum(1.0, dyn / GAMMA)
code = np.where(dyn <= GAMMA, raw * dyn, raw)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(dyn, paper, color="#1f5fae", lw=2, label="논문 식: raw · min(1, Sdyn/γ)")
ax.plot(dyn, code, color="#b5651d", lw=2, ls="--", label="코드: dyn ≤ 0.1213이면 raw · dyn")
ax.axvline(GAMMA, color="#888", lw=1, ls=":")
ax.text(GAMMA + 0.005, 0.12, "임계 0.1213", fontsize=8, color="#555")
ax.set_xlabel("Sdyn (동적 정도)")
ax.set_ylabel("보정 후 점수 (raw = 0.90 가정)")
ax.set_title("정적 영상 페널티 규칙 비교 (toy, 가상 raw 값)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_gate_toy.png", dpi=200)
plt.close(fig)
print("wrote figures to", FIG)
