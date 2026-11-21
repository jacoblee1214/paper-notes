"""우리가 그린 그래프 두 장을 만든다 (논문 그림은 쓰지 않는다).

  report/figures/own_table1.png : 표 I 열별 개선률 (최강 기준선 대비, 양수 = TacBridge가 좋음)
  report/figures/own_ood_ci.png  : 시험 직물 5개일 때 쌍체 격차 2.8 pp의 95% 반폭 대 직물 간 SD
실행: python3 analysis/plot_figures.py  (matplotlib만 사용)
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(FIG, exist_ok=True)

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 표 I: (라벨, 방향, TacBridge, 최강 기준선)
ROWS = [
    ("TAG GS (정확도)", "higher", 73.13, 80.82),
    ("Cloth GS (정확도)", "higher", 35.65, 26.76),
    ("Pose DG (정확도)", "higher", 63.79, 56.22),
    ("Slip F1 DG", "higher", 85.77, 83.33),
    ("Force Change DG (RMSE)", "lower", 91.26, 98.33),
    ("Slip F1 Mini", "higher", 97.90, 97.50),
    ("Force Change Mini (RMSE)", "lower", 104.56, 102.64),
    ("Force DG (RMSE)", "lower", 680.83, 821.26),
    ("Force Mini (RMSE)", "lower", 185.94, 297.96),
    ("ToucHD DG (RMSE)", "lower", 1011.79, 1953.82),
    ("ToucHD Mini (RMSE)", "lower", 1009.31, 3208.10),
]


def improvement(direction, tb, base):
    return 100 * ((tb - base) / base if direction == "higher" else (base - tb) / base)


def plot_table1():
    labels = [r[0] for r in ROWS]
    vals = [improvement(*r[1:]) for r in ROWS]
    colors = ["#1f5fae" if v >= 0 else "#b5472f" for v in vals]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    y = range(len(ROWS))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color="#333", lw=0.8)
    for yi, v in zip(y, vals):
        ax.text(v + (1 if v >= 0 else -1), yi, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.set_xlabel("최강 기준선 대비 개선률 (%, 양수 = TacBridge가 좋음)", fontsize=9)
    ax.set_title("표 I 열별 개선률 (논문 값에서 계산)", fontsize=10.5, loc="left")
    ax.set_xlim(-60, 85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "own_table1.png"))
    plt.close(fig)


def plot_ood_ci():
    t_crit = 2.776  # t(4), 97.5%
    sds = [i / 10 for i in range(5, 101)]
    half = [t_crit * s / math.sqrt(5) for s in sds]
    gap = 2.8
    s_thr = gap / (t_crit / math.sqrt(5))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    ax.plot(sds, half, color="#1f5fae", lw=1.8, label="95% 반폭 = 2.776 × s / √5")
    ax.axhline(gap, color="#b5472f", lw=1.2, ls="--", label=f"관측 격차 {gap} pp")
    ax.axvline(s_thr, color="#555", lw=0.9, ls=":")
    ax.text(s_thr + 0.1, 9.2, f"p = 0.05 경계 s ≈ {s_thr:.2f} pp", fontsize=8)
    ax.set_xlabel("직물 간 쌍체 차이의 표준편차 s (pp), 시험 직물 5개", fontsize=9)
    ax.set_ylabel("95% 반폭 (pp)", fontsize=9)
    ax.set_title("OOD 5 직물에서 2.8 pp 격차의 검정력 (추정)", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "own_ood_ci.png"))
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_ood_ci()
    print("wrote", FIG)
