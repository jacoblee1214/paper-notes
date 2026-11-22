"""논문 표 값으로 그린 우리 그래프 세 장 (report/figures/own_*.png).

논문 그림은 쓰지 않습니다(사람 영상 프레임과 실로봇 사진이 섞여 있음). 모든 값은 논문 표를 옮긴 것입니다.
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(FIG, exist_ok=True)

# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다
for name in ["Noto Sans CJK KR", "Noto Sans KR", "AppleSDGothicNeo", "Apple SD Gothic Neo"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

BLUE = "#1f5fae"
GRAY = "#8a8a85"
ORANGE = "#c96a18"

# ---------------------------------------------------------------
# 그래프 1: 표 19 ARCTIC (SPIDER 대 DexMachina), 작업별 평균 ± SD
# ---------------------------------------------------------------
tasks = ["box", "ketchup", "ketchup-\ns01-u02", "laptop", "mixer", "notebook", "waffle-\niron"]
t19 = {
    "dpos (m)": [(0.0172, 0.0069, 0.0906, 0.0373), (0.8917, 0.1311, 1.1865, 0.3021),
                 (0.7573, 0.3397, 1.1569, 0.2231), (0.0427, 0.0171, 0.0573, 0.0840),
                 (0.0176, 0.0098, 0.1211, 0.1563), (0.9968, 0.0245, 0.4969, 0.0682),
                 (0.2224, 0.1585, 0.5113, 0.0682)],
    "drot (rad)": [(0.0988, 0.0198, 0.3709, 0.1155), (1.7862, 0.1723, 2.0616, 0.1768),
                   (1.5418, 0.3500, 2.0379, 0.2021), (0.2155, 0.0726, 0.1980, 0.2344),
                   (0.1203, 0.0360, 0.7360, 0.2814), (1.9491, 0.2248, 1.1062, 0.2850),
                   (1.0033, 0.2985, 1.1791, 0.0903)],
    "darti (rad)": [(0.0929, 0.0801, 0.2897, 0.0522), (0.7836, 0.0850, 0.8057, 0.0806),
                    (1.1717, 0.0227, 1.1863, 0.0851), (1.0548, 0.5011, 0.4094, 0.3296),
                    (0.1440, 0.0118, 0.3654, 0.1230), (0.7024, 0.0007, 0.7468, 0.1131),
                    (0.5528, 0.3142, 0.5480, 0.0753)],
}
fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
x = np.arange(len(tasks))
for ax, (metric, rows) in zip(axes, t19.items()):
    s = np.array(rows)
    w = 0.38
    ax.bar(x - w / 2, s[:, 0], w, yerr=s[:, 1], color=BLUE, capsize=3, label="SPIDER (5 시드)")
    ax.bar(x + w / 2, s[:, 2], w, yerr=s[:, 3], color=GRAY, capsize=3, label="DexMachina (논문 표 19)")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(loc="upper right", frameon=False)
axes[0].set_title("표 19 재그림: ARCTIC 7개 작업, 낮을수록 좋음 (막대 = 평균, 선 = 표준편차)")
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(tasks)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_t19_arctic.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# 그래프 2: 표 17 Oakink 16개 궤적, 궤적별 et와 er (SPIDER 대 ManipTrans), 로그 축
# ---------------------------------------------------------------
sp_et = [34.81, 40.74, 44.27, 25.96, 31.09, 26.88, 5.99, 20.56, 45.45, 22.06, 14.00, 11.36, 12.64, 21.04, 6.25, 6.77]
mt_et = [34.50, 40.17, 44.31, 13.40, 34.43, 25.64, 21.33, 11.66, 45.41, 17.04, 14.03, 16.97, 18.61, 23.99, 3.93, 3.13]
sp_er = [101.76, 47.17, 80.72, 27.95, 29.26, 40.46, 56.57, 52.75, 51.92, 59.61, 85.78, 23.57, 33.06, 49.52, 85.93, 44.97]
mt_er = [112.67, 35.20, 80.54, 137.87, 57.47, 73.22, 57.51, 58.05, 51.81, 70.61, 85.77, 15.78, 99.13, 122.98, 50.81, 38.57]
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.9))
for ax, (lab, a, b, unit) in zip(axes, [("et: 물체 이동 오차", sp_et, mt_et, "cm"),
                                        ("er: 물체 회전 오차", sp_er, mt_er, "deg")]):
    lo = min(min(a), min(b)) * 0.8
    hi = max(max(a), max(b)) * 1.2
    ax.plot([lo, hi], [lo, hi], color="#999", lw=0.8, ls="--")
    ax.scatter(b, a, color=ORANGE, s=24, label="궤적 16개")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"ManipTrans ({unit})")
    ax.set_ylabel(f"SPIDER ({unit})")
    ax.set_title(lab, fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].text(0.03, 0.92, "점선 위: SPIDER가 더 큼\n점선 아래: SPIDER가 더 작음", transform=axes[0].transAxes, fontsize=7.5)
fig.suptitle("표 17 재그림: 궤적별 평균 (5 시드 평균, 로그 축)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_t17_pertraj.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# 그래프 3: 표 1 처리량 대 ADD-AUC (Oakink 와 GigaHands, 로그 축)
# ---------------------------------------------------------------
pts = [  # (방법, 데이터셋, FPS, ADD-AUC, 오차막대)
    ("Kinematic (MuJoCo)", "Oakink", 248.9, 11.0, 11.1),
    ("ManipTrans (IsaacGym, RL)", "Oakink", 0.1, 19.4, 27.5),
    ("SPIDER (IsaacGym)", "Oakink", 2.5, 22.3, 28.1),
    ("DexMachina (Genesis, RL)", "ARCTIC", 0.05, 41.0, 25.3),
    ("SPIDER (Genesis)", "ARCTIC", 1.5, 43.6, 29.8),
    ("Kinematic (MuJoCo)", "GigaHands", 244.2, 24.1, 22.3),
    ("SPIDER (MuJoCo Warp)", "Oakink", 6.1, 48.8, 24.9),
    ("SPIDER+C (MuJoCo Warp)", "Oakink", 0.4, 56.9, 24.7),
    ("SPIDER (MuJoCo Warp)", "GigaHands", 5.9, 56.9, 20.6),
    ("SPIDER+C (MuJoCo Warp)", "GigaHands", 0.5, 57.7, 23.9),
]
styles = {"Oakink": "o", "GigaHands": "s", "ARCTIC": "^"}
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for name, ds, fps, auc, sd in pts:
    col = BLUE if "SPIDER" in name else GRAY
    ax.errorbar(fps, auc, yerr=sd, fmt=styles[ds], color=col, ms=6, capsize=2, alpha=0.9)
    ax.annotate(f"{name.split(' (')[0]} / {ds}", (fps, auc), textcoords="offset points", xytext=(5, 4), fontsize=6.8)
ax.set_xscale("log")
ax.set_xlabel("처리량 FPS (로그 축, 높을수록 빠름)")
ax.set_ylabel("ADD-AUC (%) ± SD, 높을수록 좋음")
ax.set_title("표 1 재그림: 방법별 처리량과 추적 정확도 (보고된 값, 서로 다른 장비와 시뮬레이터)", fontsize=8.5)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_t1_fps_auc.png"), dpi=200)
plt.close(fig)

print("wrote", sorted(f for f in os.listdir(FIG) if f.startswith("own_")))
