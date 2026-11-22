"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 입력은 논문 표 2, 3, 4 값이다.

실행: python3 analysis/plot_figures.py
논문 그림은 쓰지 않는다(arXiv non-exclusive). 값은 표에서 옮긴 것이고, 검정 값은 analysis/recompute_results.json과 같다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
font = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if font:
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, ACC, GREY = "#1b1b1a", "#55544f", "#1f5fae", "#9a978f"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "text.color": INK, "font.size": 9.5})


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------- 그림 1: 실로봇 객체별 성공 (표 4) ----------
objects = ["Rice Paddle", "Dustpan", "Bouquet", "Pitcher", "Small Bucket",
           "Bottle", "Umbrella", "Shampoo", "Hammer", "Brush"]
sr_dgn = [40, 60, 40, 50, 50, 70, 80, 60, 80, 60]
sr_our = [60, 70, 50, 80, 50, 100, 90, 60, 90, 70]
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
y = list(range(len(objects)))[::-1]
for yi, a, b in zip(y, sr_dgn, sr_our):
    ax.plot([a, b], [yi, yi], color=GREY, lw=2, zorder=1)
ax.scatter(sr_dgn, y, color=GREY, s=46, label="기준선 DexGraspNet 2.0*", zorder=2)
ax.scatter(sr_our, y, color=ACC, s=46, label="OpenDexGrasp", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(objects)
ax.set_xlabel("실로봇 성공률 (%, 객체당 10회)")
ax.set_xlim(25, 110)
ax.axvline(59, color=GREY, ls=":", lw=1)
ax.axvline(72, color=ACC, ls=":", lw=1)
ax.set_ylim(-1.3, 9.6)
ax.text(58, -0.9, "기준선 평균 59", color=MUTED, fontsize=8, ha="right")
ax.text(73, -0.9, "OpenDexGrasp 평균 72", color=ACC, fontsize=8, ha="left")
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
clean(ax)
ax.set_title("객체별 실로봇 성공률 (논문 표 4 값, 우리가 그림)", fontsize=10.5, loc="left")
fig.text(0.01, 0.01, "객체 단위 우위 8, 동률 2, 역전 0. 시도 단위 Fisher p = 0.074, 객체 단위 sign test p = 0.008.",
         fontsize=7.8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "own_realworld.png")
plt.close(fig)

# ---------- 그림 2: 시뮬레이션 네 분할 SR과 인스턴스 수 (표 2, 표 6) ----------
splits = ["기능-seen\n(75)", "기능-unseen\n(15)", "비기능-seen\n(138)", "비기능-unseen\n(15)"]
sr_b = [50.88, 43.22, 63.27, 54.49]
sr_o = [68.07, 62.96, 75.46, 66.73]
z = [2.18, 1.11, 2.22, 0.69]
fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
x = list(range(4))
w = 0.36
ax.bar([i - w / 2 for i in x], sr_b, width=w, color=GREY, label="기준선 DexGraspNet 2.0*")
ax.bar([i + w / 2 for i in x], sr_o, width=w, color=ACC, label="OpenDexGrasp")
for i in x:
    ax.text(i - w / 2, sr_b[i] + 1, f"{sr_b[i]:.1f}", ha="center", fontsize=7.8, color=MUTED)
    ax.text(i + w / 2, sr_o[i] + 1, f"{sr_o[i]:.1f}", ha="center", fontsize=7.8, color=ACC)
    ax.text(i, 93, f"z = {z[i]:.2f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_ylim(0, 100)
ax.set_ylabel("SR (%)")
ax.legend(loc="lower right", frameon=False, fontsize=8.5, ncol=2, bbox_to_anchor=(1.0, -0.36))
clean(ax)
ax.set_title("시뮬레이션 성공률과 인스턴스 기준 z (논문 표 2, 우리가 계산)", fontsize=10.5, loc="left")
fig.text(0.01, 0.01, "z는 두 비율 검정, 표본 크기는 인스턴스 수로 둠(괄호). 시도를 독립으로 보면 모든 z가 커짐.",
         fontsize=7.8, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "own_sim_splits.png")
plt.close(fig)

# ---------- 그림 3: 절제 실험의 SR 차이와 보수적 z (표 3, 243 군집) ----------
names = ["affordance\n제거", "Align\n제거", "Scale\n축소", "VLM\n제거"]
diff = [4.16, 2.27, 11.13, 14.39]
zz = [1.00, 0.55, 2.61, 3.35]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=200, gridspec_kw={"width_ratios": [1, 1]})
a1.bar(names, diff, color=ACC)
for i, v in enumerate(diff):
    a1.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
a1.set_ylabel("SR 차이 (pp, 전체 대비)")
a1.set_title("절제별 성공률 감소", fontsize=9.5, loc="left")
clean(a1)
a2.bar(names, zz, color=GREY)
a2.axhline(1.96, color=INK, ls="--", lw=1)
a2.text(3.45, 2.05, "z = 1.96", fontsize=7.8, ha="right")
for i, v in enumerate(zz):
    a2.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
a2.set_ylabel("z (243 군집 근사)")
a2.set_title("보수적 z", fontsize=9.5, loc="left")
clean(a2)
fig.suptitle("표 3 절제 실험: 전체 243 인스턴스를 군집으로 쓴 근사 (우리가 계산)", fontsize=10, x=0.01, ha="left")
fig.text(0.01, 0.01, "표 3에는 분모와 신뢰구간이 없어 근사임. VLM 제거와 Scale 축소만 z > 1.96.", fontsize=7.8, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(OUT / "own_ablation.png")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
