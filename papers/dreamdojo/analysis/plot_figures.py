"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 여기서 만들지 않는다.

  own_mde.png       표 2·3·5·7의 PSNR 차이와 탐지 한계(샘플 SD 가정 1 dB, 2 dB)
  own_winrate.png   표 4 인간 선호도 승률과 Wilson 95% 구간 (50쌍 가정)
  own_fig5b.png     그림 5b의 막대 값(그림 라벨을 읽음)과 본문 '2배' 주장의 대조

입력: analysis/recompute_results.json (재계산 결과), 그림 라벨 값은 아래 상수에 적음
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
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
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

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

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

# ------------------------------------------------ 그래프 1: PSNR 차이와 탐지 한계
diffs = {  # 표에서 옮긴 차이 (dB). 라벨은 어느 표의 어떤 비교인지
    "표 2 In-lab: 잠재 행동 - 행동 없음": 0.116,
    "표 2 EgoDex: 잠재 행동 - 행동 없음": 0.420,
    "표 3 In-lab 평가: +HV 추가": 0.044,
    "표 3 HV 평가: +HV 추가": 0.018,
    "표 5 GR-1: 시간 일관성 손실 추가": 0.004,
    "표 5 반사실: 시간 일관성 손실 추가": 0.197,
    "표 7 In-lab: 사전학습 교사 - 없음 (증류)": 0.429,
    "표 7 반사실: 사전학습 교사 - 없음 (증류)": 0.109,
}
labels = list(diffs.keys())
vals = [diffs[k] for k in labels]
mde1 = RES["mde_psnr_db_n100"]["sd_diff_1.0dB"]
mde2 = RES["mde_psnr_db_n100"]["sd_diff_2.0dB"]

fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
y = np.arange(len(labels))
ax.barh(y, vals, color=[ACC if v >= mde1 else "#b7c5d8" for v in vals], edgecolor="white")
ax.axvline(mde1, color=ACC2, lw=1.2, ls="--", label=f"탐지 한계, SD 1 dB ({mde1:.2f} dB)")
ax.axvline(mde2, color=WARN, lw=1.2, ls=":", label=f"탐지 한계, SD 2 dB ({mde2:.2f} dB)")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("PSNR 차이 (dB), 샘플 100개, 양측 0.05, 검정력 0.8 가정")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "own_mde.png")
plt.close(fig)

# ------------------------------------------------ 그래프 2: 인간 선호도 승률
t4 = RES["table4_ci_assumed_n"]
names = list(t4.keys())
pct = [t4[n]["pct"] for n in names]
lo = [t4[n]["n=50"]["wilson95"][0] for n in names]
hi = [t4[n]["n=50"]["wilson95"][1] for n in names]
err = np.array([[p - l for p, l in zip(pct, lo)], [h - p for p, h in zip(pct, hi)]])

fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200)
x = np.arange(len(names))
ax.errorbar(x, pct, yerr=err, fmt="o", color=ACC, ecolor="#8fa9c9", capsize=3)
ax.axhline(50, color=MUTED, lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("승률 (%)")
ax.set_ylim(30, 95)
ax.set_title("표 4 승률과 Wilson 95% 구간 (판정 50쌍 가정, 정수 분모 없음)", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "own_winrate.png")
plt.close(fig)

# ------------------------------------------------ 그래프 3: 그림 5b 값과 '2배' 주장
groups = ["그룹 1 (분산 큼)", "그룹 2 (대부분 수렴)"]
best = [53, 63]          # 최고 체크포인트 E
uniform = [37, 55]       # 균등 샘플링
dream = [70, 68]         # DreamDojo 계획
fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=200)
x = np.arange(len(groups))
w = 0.25
ax.bar(x - w, best, w, color="#b7c5d8", label="최고 체크포인트")
ax.bar(x, uniform, w, color="#8fa9c9", label="균등 샘플링")
ax.bar(x + w, dream, w, color=ACC, label="DreamDojo 계획")
for i in range(len(groups)):
    ratio = dream[i] / uniform[i]
    ax.text(x[i] + w, dream[i] + 2, f"{ratio:.2f}배\n(균등 대비)", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 85)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("그림 5b 막대 값(라벨 판독, ±1%p). 본문은 두 그룹 모두 '거의 2배'라고 씀", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "own_fig5b.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
