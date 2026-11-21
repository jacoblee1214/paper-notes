"""우리가 그린 그래프 두 장 (논문 그림은 쓰지 않는다). 실행: python3 analysis/plot_figures.py
입력: recompute_results.json (Wilson 95% 구간). 출력: report/figures/own_task_ci.png, own_ablation.png
한글 폰트: macOS AppleSDGothicNeo, Linux Noto CJK 순서로 찾는다.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

for name in ["Apple SD Gothic Neo", "AppleSDGothicNeo", "Noto Sans CJK KR", "Noto Sans KR"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

from scipy.stats import binomtest  # noqa: F401  (환경 확인용)
import sys
sys.path.insert(0, HERE)
from recompute import wilson, T1, T3, TASKS, k_of

INK = "#1b1b1a"; ACC = "#1f5fae"; GREY = "#9a988f"

# 그림 1: 과제별 성공률 (20회당 Wilson 95% 구간), 3DTouch 대 세 기준선
sel = ["pi0", "pi0.5", "pi0.5+tactile", "Tactile-VLA", "3DTouch"]
labels = {"pi0": "π0", "pi0.5": "π0.5", "pi0.5+tactile": "π0.5+촉각", "Tactile-VLA": "Tactile-VLA", "3DTouch": "3DTouch"}
fig, axes = plt.subplots(1, 5, figsize=(11, 3.6), sharey=True)
for ax, (ti, task) in zip(axes, enumerate(TASKS)):
    for xi, m in enumerate(sel):
        k = k_of(T1[m][0][ti])
        lo, hi = wilson(k, 20)
        ax.errorbar(xi, 100 * k / 20, yerr=[[100 * (k / 20 - lo)], [100 * (hi - k / 20)]],
                    fmt="o", color=ACC if m == "3DTouch" else GREY, ecolor=ACC if m == "3DTouch" else GREY, capsize=3)
    ax.set_title(task, fontsize=10, color=INK)
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([labels[m] for m in sel], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
axes[0].set_ylabel("성공률 (%), 20회", fontsize=9)
fig.suptitle("과제별 성공률과 Wilson 95% 구간 (논문 표 1 값 재구성)", fontsize=10.5, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_task_ci.png"), dpi=170)
plt.close(fig)

# 그림 2: 절제 평균 (100회, Wilson 95% 구간)
order = ["3DTouch (Full)", "w/o geom, Global tactile", "w/o geom, Local tactile", "Point cloud only",
         "Geometric relations only", "Tactile only (Global)", "RGB only (=pi0)"]
nice = ["Full", "기하 관계 제거\n(전역 촉각)", "기하 관계 제거\n(국소 촉각)", "점군만", "기하 관계만", "촉각만", "RGB만 (π0)"]
fig, ax = plt.subplots(figsize=(8.2, 3.6))
for i, key in enumerate(order):
    k = sum(k_of(v) for v in T3[key][0])
    lo, hi = wilson(k, 100)
    ax.errorbar(100 * k / 100, i, xerr=[[100 * (k / 100 - lo)], [100 * (hi - k / 100)]],
                fmt="o", color=ACC if i == 0 else GREY, ecolor=ACC if i == 0 else GREY, capsize=3)
    ax.text(100 * hi + 1.5, i, f"{k}/100", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(nice, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("성공률 (%), 5과제 합산 100회, Wilson 95% 구간", fontsize=9)
ax.grid(axis="x", color="#e5e3dc", lw=0.6)
ax.set_title("표 3 절제: 합산 성공률과 구간 (독립 시행 가정)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_ablation.png"), dpi=170)
plt.close(fig)
print("wrote", sorted(f for f in os.listdir(FIG) if f.startswith("own_")))
