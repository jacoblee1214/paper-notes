"""우리가 논문 표 숫자로 그린 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 복사하지 않는다.

  own_robocasa_scaling.png  Table 4 (RoboCasa 24과제 평균): 실제 궤적 수별 기준 대 +240k 신경 궤적
  own_real_paired.png       Table 5 (실로봇): 정책·과제별 기준 대 +신경 궤적 짝지음
  own_table1_tasks.png      Table 1 (새 동사·새 환경): 과제별 기준 대 DreamGen 점수

입력: 숫자는 analysis/recompute.py 와 같은 값을 직접 적었다 (표 번호 주석 참고). 네트워크 불필요.
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

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

# ---------- Table 4 (p.17): 평균 행 ----------
sizes = ["30 traj", "100 traj", "300 traj"]
base = [17.44, 32.07, 49.59]
with_nt = [23.32, 39.94, 57.61]
only_nt = 20.55
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
x = np.arange(3)
w = 0.36
ax.bar(x - w / 2, base, w, color="#b7c5d8", label="실제 궤적만 (GR00T N1)")
ax.bar(x + w / 2, with_nt, w, color=ACC, label="실제 궤적 + 신경 궤적 240k")
for i, (b, n) in enumerate(zip(base, with_nt)):
    ax.text(i - w / 2, b + 1, f"{b:.1f}", ha="center", fontsize=8, color=INK)
    ax.text(i + w / 2, n + 1, f"{n:.1f}", ha="center", fontsize=8, color=INK)
ax.axhline(only_nt, color=ACC2, lw=1.2, ls="--")
ax.text(2.45, only_nt + 1, f"신경 궤적만 {only_nt:.2f}", color=ACC2, ha="right", fontsize=8)
ax.set_xticks(x, sizes)
ax.set_ylabel("평균 성공률 (%), 24과제")
ax.set_ylim(0, 70)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("RoboCasa 평균 성공률 (논문 Table 4 값)", fontsize=10, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_robocasa_scaling.png")
plt.close(fig)

# ---------- Table 5 (p.18): 실로봇 짝지음 ----------
cells = [  # (라벨, 기준, +신경 궤적), 값이 없는 칸은 제외
    ("GR1 Hammering", 60.0, 65.0), ("GR1 Wiping", 36.6, 49.0), ("GR1 Folding", 27.0, 37.0), ("GR1 Stacking", 25.0, 35.0),
    ("Franka Pick&Place", 40.0, 60.0), ("Franka Cube", 10.0, 20.0), ("Franka Tool", 20.0, 30.0),
    ("SO-100 Pick&Place", 17.0, 26.0), ("SO-100 Tic-Tac-Toe", 25.0, 65.0),
]
dp = [("GR1 Hammering", 35.0, 15.0), ("GR1 Wiping", 23.3, 33.3), ("GR1 Folding", 6.6, 26.4), ("GR1 Stacking", 25.0, 35.0),
      ("Franka Pick&Place", 20.0, 30.0), ("Franka Cube", 0.0, 20.0), ("Franka Tool", 10.0, 10.0)]
pi0 = [("Franka Pick&Place", 30.0, 40.0), ("Franka Cube", 10.0, 20.0), ("Franka Tool", 20.0, 20.0)]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.0), dpi=200, gridspec_kw={"width_ratios": [1.6, 1.2, 0.9]})
for ax, (title, data) in zip(axes, [("GR00T N1", cells), ("DP (Diffusion Policy)", dp), ("π0", pi0)]):
    for i, (lab, b, n) in enumerate(data):
        col = ACC if n > b else (ACC2 if n < b else MUTED)
        ax.plot([b, n], [i, i], color=col, lw=2)
        ax.scatter([b], [i], color="#b7c5d8", zorder=3, s=22)
        ax.scatter([n], [i], color=col, zorder=3, s=22)
    ax.set_yticks(range(len(data)), [d[0] for d in data], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-2, 75)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel("성공률 (%)", fontsize=8)
fig.text(0.02, 0.01, "회색 점: 기준, 파란 점: +신경 궤적, 주황: 하락, 회색 짧은 점: 변화 없음", fontsize=7, color=MUTED)
fig.suptitle("실로봇 과제별 짝지음 비교 (논문 Table 5 값; 파란 선 = 상승)", fontsize=9.5, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(FIG / "own_real_paired.png")
plt.close(fig)

# ---------- Table 1 (p.7): 과제별 점수 ----------
beh = [("Open Microwave", 0, 23), ("Open Macbook", 0, 45), ("Close Lunchbox", 0, 10), ("Hit Tambourine", 5, 15),
       ("Hit Keyboard", 0, 90), ("Grab button", 45, 75), ("Pour Water", 40, 55), ("Water flowers", 50, 95),
       ("Light Candle", 10, 15), ("Use Vacuum", 0, 55), ("Iron shirt", 0, 20), ("Take Spoon Out", 7, 17),
       ("Unroll mat", 0, 55), ("Move Mouse", 0, 35)]
env = [("Pick up Tangerine", 0, 30), ("Box sandwich", 0, 10), ("Weigh the Orange", 0, 20), ("Put cup in trash", 0, 45),
       ("Put pear in basket", 0, 35), ("Put sauce on tray", 0, 45), ("Water Flowers", 0, 15), ("Lift Basket", 0, 55),
       ("Swirl Around Spoon", 0, 15), ("Use Whisk", 0, 25), ("Close soup container", 0, 55), ("Uncover Pot", 0, 30),
       ("Cover Pot", 0, 35)]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), dpi=200, gridspec_kw={"width_ratios": [14, 13]})
for ax, (title, data, avg_txt) in zip(axes, [("새 동사 (seen 환경, 14과제)", beh, "평균 11.2 → 43.2"),
                                           ("새 환경 (13과제)", env, "평균 0.0 → 28.5 (표); 단순평균 31.9")]):
    y = np.arange(len(data))
    ax.barh(y - 0.2, [d[1] for d in data], 0.4, color="#b7c5d8", label="GR00T N1 (pick-and-place만)")
    ax.barh(y + 0.2, [d[2] for d in data], 0.4, color=ACC, label="+ DreamGen 신경 궤적")
    ax.set_yticks(y, [d[0] for d in data], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("성공률 점수 (%)", fontsize=8)
    ax.set_title(f"{title}\n{avg_txt}", fontsize=8.5, loc="left", color=INK)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle("새 동사·새 환경 과제별 점수 (논문 Table 1 값; 부분 점수 포함)", fontsize=9.5, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(FIG / "own_table1_tasks.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
