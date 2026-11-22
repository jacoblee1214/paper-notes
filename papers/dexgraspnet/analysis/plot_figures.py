"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_table1_density.png  Table I의 grasp 수 / 물체 수 재계산과 보고된 하한(DDG 행 불일치)
  own_speedup.png         계산 비용 주장의 분해: 유효율 × 스텝 수 × 처리량 = 58.6 대 보고 57.1
  own_entropy_bins.png    관절 엔트로피를 유효 빈 수(2^H)로 바꾼 값과 표본 수에 따른 편향

입력: analysis/recompute_results.json
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
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

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


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


# 1. Table I: grasp 수 / 물체 수
t1 = R["C1_table1_grasps_per_object"]
names = list(t1.keys())
vals = [t1[k]["per_object"] for k in names]
floors = [t1[k]["reported_per_object_floor"] for k in names]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
y = np.arange(len(names))
colors = [WARN if not t1[k]["consistent"] else ACC for k in names]
ax.barh(y, vals, color=colors, alpha=0.85, height=0.6)
ax.scatter(floors, y, marker="|", s=260, color=INK, zorder=3, label="표 I 보고값(하한)")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel("물체당 grasp 수 (로그 축, 원자료 = 표 I의 grasp 수 / 물체 수)")
for i, v in enumerate(vals):
    ax.text(v * 1.08, i, f"{v:,.1f}", va="center", fontsize=8, color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("표 I 재계산: DDGdata 행만 보고값(>100)과 어긋납니다 (빨강)", fontsize=10, color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_table1_density.png")
plt.close(fig)

# 2. 계산 비용 분해
sp = R["C5_speedup_decomposition"]
labels = ["유효율\n18% / 3%", "스텝 수\n10000 / 6000", "grasp·스텝당\n처리량", "곱(재계산)", "보고값\n400 / 7"]
factors = [sp["yield_ratio"], sp["step_ratio"], sp["per_grasp_step_throughput_ratio"]]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
cum = 1.0
bottoms = []
for f in factors:
    bottoms.append(cum)
    cum *= f
xs = np.arange(3)
ax.bar(xs, [f - 1 for f in factors], bottom=[1] * 3, color=ACC, alpha=0.85, width=0.55)
for i, f in enumerate(factors):
    ax.text(i, bottoms[i] + (f - 1) + 0.3, f"×{f:.2f}", ha="center", fontsize=9, color=INK)
ax.bar([3], [sp["speedup_recomputed"]], color=ACC2, alpha=0.85, width=0.55)
ax.text(3, sp["speedup_recomputed"] + 1.5, f"×{sp['speedup_recomputed']}", ha="center", fontsize=9, color=INK)
ax.bar([4], [sp["speedup_reported_400_over_7"]], color=MUTED, alpha=0.85, width=0.55)
ax.text(4, sp["speedup_reported_400_over_7"] + 1.5, f"×{sp['speedup_reported_400_over_7']}", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("10000 유효 grasp당 GPU 시간 비율", fontsize=8)
ax.set_title("계산 비용 주장: 세 요인의 곱이 보고된 57배와 맞습니다 (처리량은 하드웨어로 설명 안 됨)", fontsize=9, color=INK, loc="left")
ax.set_ylim(0, 75)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_speedup.png")
plt.close(fig)

# 3. 엔트로피: 유효 빈 수와 표본 수 편향
ent = R["C9_entropy"]["effective_bins_2^H"]
keys = {"DDGdata_dataset": "DDGdata\n데이터", "DexGraspNet_dataset": "DexGN\n데이터",
        "T4_DDG_on_DexGraspNet": "DDG\n(DexGN 학습)", "T4_GraspTTA_DDGdata_on_DexGraspNet": "GraspTTA\n(DDG 학습)",
        "T4_GraspTTA_on_DexGraspNet": "GraspTTA\n(DexGN 학습)"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.6), dpi=200, gridspec_kw={"width_ratios": [1.35, 1]})
xs = np.arange(len(keys))
v = [ent[k] for k in keys]
a1.bar(xs, v, color=[MUTED, ACC, ACC2, ACC2, ACC2], alpha=0.85, width=0.6)
a1.axhline(100, color=WARN, lw=1, ls="--")
a1.text(len(keys) - 0.5, 96, "100 bin 상한", ha="right", fontsize=8, color=WARN)
for i, x in enumerate(v):
    a1.text(i, x + 2, f"{x:.0f}", ha="center", fontsize=8, color=INK)
a1.set_xticks(xs)
a1.set_xticklabels(list(keys.values()), fontsize=6.5)
a1.set_ylabel("유효 빈 수 2^H (H는 비트)", fontsize=8)
a1.set_ylim(0, 115)
a1.set_title("보고된 H를 2^H로 바꾸면", fontsize=9, color=INK, loc="left")
style(a1)
sens = R["C9_sample_size_sensitivity"]
Ns = [int(n) for n in sens.keys() if int(n) <= 100000]
bias = [sens[str(n)]["MM_bias_bits_if_K100"] for n in Ns]
a2.plot(Ns, bias, marker="o", color=ACC, lw=1.5)
a2.set_xscale("log")
a2.set_xlabel("표본 수 N (로그)", fontsize=8)
a2.set_ylabel("Miller-Madow 보정량 (bit, K=100)", fontsize=8)
a2.set_title("N이 작으면 H가 0.7 bit까지 낮게 나옵니다", fontsize=9, color=INK, loc="left")
a2.axhline(0.428, color=WARN, lw=0.8, ls=":")
a2.text(Ns[0], 0.45, "DDG 대 GraspTTA 격차 0.43 bit", fontsize=7, color=WARN)
style(a2)
fig.tight_layout()
fig.savefig(FIG / "own_entropy_bins.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
