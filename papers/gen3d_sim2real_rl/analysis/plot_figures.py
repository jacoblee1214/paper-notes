"""논문 수치로 그래프 세 장을 그린다 (우리가 그림). 출력: report/figures/own_*.png

실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

BLUE, ORANGE, GREY, GREEN = "#1f5fae", "#d9822b", "#7a7a75", "#2e8b57"

# 표 1 (N = 훈련 장면 수). N=100은 3 시드, 나머지는 3 실행
N = np.array([1, 3, 10, 25, 50, 100])
ID = np.array([94.3, 88.7, 87.0, 84.3, 80.5, np.nan])
ID_sd = np.array([2.6, 4.7, 3.3, 2.4, 0.4, np.nan])
OOD = np.array([53.2, 61.3, 72.4, 77.6, 77.9, np.nan])
OOD_sd = np.array([5.5, 1.8, 0.4, 0.8, 0.9, np.nan])
W = np.array([51.6, 60.0, 72.1, 78.3, 79.2, 78.9])
W_sd = np.array([5.3, 4.6, 0.9, 0.5, 0.3, 0.9])
dG = np.array([41.1, 27.4, 14.6, 6.7, 2.6, np.nan])

# 그림 1: 훈련 장면 수에 따른 ID, OOD 성공률과 ID-OOD 차이
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
m = ~np.isnan(ID)
ax[0].errorbar(N[m], ID[m], yerr=ID_sd[m], marker="o", color=BLUE, capsize=3, label="ID (훈련 장면)")
ax[0].errorbar(N[m], OOD[m], yerr=OOD_sd[m], marker="s", color=ORANGE, capsize=3, label="OOD (나머지 50장면)")
ax[0].errorbar(N, W, yerr=W_sd, marker="^", color=GREY, capsize=3, ls="--", label="전체 W (100장면)")
ax[0].set_xscale("log"); ax[0].set_xticks(N); ax[0].set_xticklabels([str(x) for x in N])
ax[0].set_xlabel("훈련 장면 수 N"); ax[0].set_ylabel("성공률 (%)"); ax[0].set_ylim(0, 105)
ax[0].set_title("표 1 재구성: EmbodiedGen 장면", fontsize=9)
ax[0].legend(fontsize=7, loc="lower right"); ax[0].grid(alpha=0.25)
ax[1].plot(N[m], dG[m], marker="o", color=GREEN)
for x, y in zip(N[m], dG[m]):
    ax[1].annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xticks(N[m]); ax[1].set_xticklabels([str(x) for x in N[m]])
ax[1].set_xlabel("훈련 장면 수 N"); ax[1].set_ylabel("dG = SR(ID) - SR(OOD) (pp)")
ax[1].set_title("일반화 격차는 N에 따라 줄어듭니다", fontsize=9); ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIG / "own_scaling.png", dpi=170); plt.close(fig)

# 그림 2: 실로봇 장면별 성공률 (대응 비교)과 전체 성공률의 Wilson 95% 구간
il = [0.6, 0.2, 0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.1, 0.3, 0.0, 0.2]
rl = [0.9, 0.7, 0.9, 0.7, 1.0, 1.0, 0.7, 0.8, 0.7, 0.6, 0.5, 0.5]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
for a, b in zip(il, rl):
    ax[0].plot([0, 1], [a * 100, b * 100], color=GREY, lw=0.8, alpha=0.7)
ax[0].scatter([0] * 12, np.array(il) * 100, color=GREY, zorder=3, label="IL πpre")
ax[0].scatter([1] * 12, np.array(rl) * 100, color=BLUE, zorder=3, label="RL N=100")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["IL πpre", "RL N=100"])
ax[0].set_ylabel("장면별 성공률 (%, 10회)"); ax[0].set_ylim(-5, 105)
ax[0].set_title("12장면 대응 비교 (12/12 장면에서 RL이 높음)", fontsize=9); ax[0].grid(alpha=0.25)
for xi, k, col in ((0, 26, GREY), (1, 90, BLUE)):
    lo, hi = wilson(k, 120)
    ax[1].errorbar([xi], [k / 120 * 100], yerr=[[k / 120 * 100 - lo * 100], [hi * 100 - k / 120 * 100]],
                   fmt="o", color=col, capsize=4)
    ax[1].annotate(f"{k}/120", (xi, k / 120 * 100), textcoords="offset points", xytext=(8, -2), fontsize=8)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["IL", "RL"]); ax[1].set_xlim(-0.5, 1.5)
ax[1].set_ylim(0, 100); ax[1].set_title("전체 성공률, Wilson 95%", fontsize=9); ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIG / "own_real.png", dpi=170); plt.close(fig)

# 그림 3: 외부 평가 세트 (훈련에 쓰이지 않은 자산 출처)의 성공률 변화
ev_N = ["0", "1", "3", "10", "25", "50", "100"]
sim = {"EmbodiedGen OOD (50장면)": [9.6, 53.2, 61.3, 72.4, 77.6, 77.9, np.nan],
       "SimplerEnv (3장면)": [23.7, 36.1, 47.4, 54.3, 70.1, 68.4, 74.3],
       "Google Scanned (18장면)": [12.1, 42.3, 48.8, 59.3, 62.7, 62.5, 60.7],
       "Hunyuan3D-2.1 (25장면)": [10.2, 42.7, 50.0, 58.2, 61.8, 63.3, 62.5]}
fig, ax = plt.subplots(figsize=(6.2, 3.1))
for (k, v), col in zip(sim.items(), [ORANGE, BLUE, GREEN, GREY]):
    xs = [i for i, y in enumerate(v) if not np.isnan(y)]
    ax.plot(xs, [v[i] for i in xs], marker="o", color=col, label=k)
ax.set_xticks(range(len(ev_N))); ax.set_xticklabels(ev_N)
ax.set_xlabel("훈련 장면 수 N (0 = 사전학습 πpre)"); ax.set_ylabel("성공률 (%)"); ax.set_ylim(0, 100)
ax.set_title("외부 평가 세트의 성공률 (표 1, 각 3 실행 평균)", fontsize=9)
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIG / "own_bench.png", dpi=170); plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
