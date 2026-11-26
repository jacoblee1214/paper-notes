"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_sim_ci.png     표 1의 시뮬레이션 성공률(과제당 100회 가정)과 Wilson 95% 구간: Dex4D 대 NovaFlow-CL, LVP
  own_real_ci.png    표 3의 실로봇 성공 횟수(과제당 20회)와 Wilson 95% 구간, 합계 Fisher p
  own_pair_toy.png   토이 1: 세 점 인코더의 상대 회전 오차(지도학습 대리 과제)와 표 2의 학생 SR
  own_kabsch_toy.png 토이 2: Kabsch 회전 오차 대 점 잡음·가시 점 수, 이상치 비율(RANSAC 비교)

입력: analysis/recompute.py 의 표 값(이 파일에 복사), analysis/toy_pair_encoding_results.json, analysis/toy_kabsch_results.json
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
PT = json.loads((ROOT / "analysis" / "toy_pair_encoding_results.json").read_text(encoding="utf-8"))
KB = json.loads((ROOT / "analysis" / "toy_kabsch_results.json").read_text(encoding="utf-8"))

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
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

INK, MUTED, ACC, ACC2, WARN, GREEN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a", "#3b7d4f"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


# ---------------------------------------------------------------- 1. 시뮬레이션 성공률 + 구간
TASKS = ["Apple2Plate", "Pour", "Hammer", "StackCup", "RotateBox", "Sponge2Bowl"]
SIM = {"Dex4D": [.86, .86, .28, .60, .44, .56], "NovaFlow-CL": [.58, .73, .27, .44, .33, .27], "LVP": [.26, .39, .00, .10, .00, .11]}
cols = {"Dex4D": ACC, "NovaFlow-CL": ACC2, "LVP": MUTED}
fig, ax = plt.subplots(figsize=(9, 3.9))
x = np.arange(len(TASKS))
for j, (m, v) in enumerate(SIM.items()):
    k = np.round(np.array(v) * 100)
    lo = np.array([wilson(a, 100)[0] for a in k])
    hi = np.array([wilson(a, 100)[1] for a in k])
    ax.errorbar(x + (j - 1) * 0.2, v, yerr=[np.array(v) - lo, hi - np.array(v)], fmt="o", color=cols[m], capsize=3, label=m, ms=5)
ax.set_xticks(x)
ax.set_xticklabels(TASKS, fontsize=9)
ax.set_ylabel("성공률 (표 1)")
ax.set_ylim(-0.02, 1.0)
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
ax.set_title("시뮬레이션 성공률과 Wilson 95% 구간 (과제당 100회 가정)", fontsize=10.5, color=INK, loc="left")
ax.text(2, 0.62, "Hammer: 28 대 27\n(구분 안 됨)", fontsize=8.5, color=WARN, ha="center")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_sim_ci.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 2. 실로봇
REAL_T = ["LiftToy", "Broccoli2Plate", "Meat2Bowl", "Pour", "합계(80)"]
REAL = {"Dex4D": [14, 10, 13, 8, 45], "NovaFlow-CL": [10, 7, 7, 0, 24], "LVP": [9, 6, 7, 0, 22], "π0.5-DROID": [5, 6, 9, 0, 20]}
cols = {"Dex4D": ACC, "NovaFlow-CL": ACC2, "LVP": MUTED, "π0.5-DROID": GREEN}
fig, ax = plt.subplots(figsize=(9, 3.9))
x = np.arange(len(REAL_T))
for j, (m, v) in enumerate(REAL.items()):
    n = np.array([20, 20, 20, 20, 80])
    p = np.array(v) / n
    lo = np.array([wilson(a, b)[0] for a, b in zip(v, n)])
    hi = np.array([wilson(a, b)[1] for a, b in zip(v, n)])
    ax.errorbar(x + (j - 1.5) * 0.15, p, yerr=[p - lo, hi - p], fmt="o", color=cols[m], capsize=3, label=m, ms=5)
ax.set_xticks(x)
ax.set_xticklabels(REAL_T, fontsize=9)
ax.set_ylabel("성공률 (표 3)")
ax.set_ylim(-0.02, 1.0)
ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper right")
ax.set_title("실로봇 성공률과 Wilson 95% 구간 (과제당 20회, 합계 80회)", fontsize=10.5, color=INK, loc="left")
ax.text(4.0, 0.12, "합계 Fisher p:\n0.0001–0.0013", fontsize=8.5, color=WARN, ha="center")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 3. 토이 1: 인코더 비교
S = PT["summary"]
names = ["paired", "decoupled", "mlp"]
labels = ["Paired\n(6D 점 쌍)", "Decoupled\n(두 PointNet)", "MLP\n(점 펼침)"]
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))
ax = axs[0]
rot = [S[k]["rot_err_mean_deg"] for k in names]
sd = [S[k]["rot_err_mean_deg_sd"] for k in names]
ax.bar(range(3), rot, yerr=sd, color=[ACC, ACC2, MUTED], capsize=4, width=0.6)
ax.axhline(90, color=WARN, lw=1, ls="--")
ax.text(2.45, 91.5, "항등 변환만 내놓을 때 약 90도", fontsize=8, color=WARN, ha="right")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("상대 회전 오차 평균 (도)")
ax.set_title("토이 1: 지도학습 대리 과제 (시드 3개)", fontsize=10, color=INK, loc="left")
for i, v in enumerate(rot):
    ax.text(i, v + 3 + sd[i], f"{v:.0f}", ha="center", fontsize=9, color=INK)
ax.set_ylim(0, 110)
style(ax)
ax = axs[1]
t2 = [0.600, 0.203, 0.057]
ax.bar(range(3), t2, color=[ACC, ACC2, MUTED], width=0.6)
for i, v in enumerate(t2):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("학생 정책 SR (논문 표 2)")
ax.set_ylim(0, 0.72)
ax.set_title("논문 표 2 (6과제 평균)", fontsize=10, color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_pair_toy.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 4. 토이 2: Kabsch
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))
ax = axs[0]
ns = [64, 32, 16, 8]
for sg, c in zip([2, 5, 10], [GREEN, ACC, WARN]):
    med = [KB["gauss"][f"sigma{sg}mm_n{n}"]["median_deg"] for n in ns]
    p90 = [KB["gauss"][f"sigma{sg}mm_n{n}"]["p90_deg"] for n in ns]
    ax.plot(range(4), med, "-o", color=c, label=f"점 잡음 {sg} mm (중앙값)")
    ax.plot(range(4), p90, ":", color=c)
ax.set_xticks(range(4))
ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("가시 점 수 n")
ax.set_ylabel("Kabsch 회전 오차 (도)")
ax.set_title("가우시안 잡음만 (점선: 90분위)", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8)
style(ax)
ax = axs[1]
qs = [0.0, 0.1, 0.2, 0.3]
for n, ls in [(32, "-"), (16, "--")]:
    kb = [KB["outlier"][f"q{q}_n{n}"]["kabsch_median_deg"] for q in qs]
    rs = [KB["outlier"][f"q{q}_n{n}"]["ransac_median_deg"] for q in qs]
    ax.plot(qs, kb, ls, marker="o", color=WARN, label=f"Kabsch n={n}")
    ax.plot(qs, rs, ls, marker="s", color=ACC, label=f"RANSAC n={n}")
ax.set_xlabel("이상치 점 비율 (5 cm 변위)")
ax.set_ylabel("회전 오차 중앙값 (도)")
ax.set_title("점 잡음 5 mm + 이상치", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_kabsch_toy.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
