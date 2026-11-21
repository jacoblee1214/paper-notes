"""논문 표 값을 그래프 두 장으로 그린다(우리가 그린 그래프, 논문 그림은 쓰지 않는다).

실행: python3 analysis/plot_figures.py  →  report/figures/own_table2_ci.png, own_table1_planning.png
값은 recompute.py와 같은 표에서 가져온다(표 1 = Table 1, 표 2 = Table 2, v4).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
]
for f in FONT_CANDIDATES:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

INK, ACCENT, GREY, SOFT = "#1b1b1a", "#1f5fae", "#9a988f", "#c9d7ea"

# ---- 표 2 (CRPSsum, 낮을수록 좋음). 값과 ±는 Table 2 (v4)에서 옮김.
DS = ["Exchange", "Solar", "Electricity", "Traffic", "Taxi", "Wikipedia"]
T2 = {
    "VES": [0.005, 0.900, 0.880, 0.350, None, None],
    "VAR": [0.005, 0.830, 0.039, 0.290, None, None],
    "VAR-Lasso": [0.012, 0.510, 0.025, 0.150, None, 3.100],
    "GARCH": [0.023, 0.880, 0.190, 0.370, None, None],
    "DeepAR": [None, 0.336, 0.023, 0.055, None, 0.127],
    "LSTM-Copula": [0.007, 0.319, 0.064, 0.103, 0.326, 0.241],
    "GP-Copula": [0.007, 0.337, 0.025, 0.078, 0.208, 0.086],
    "KVAE": [0.014, 0.340, 0.051, 0.100, None, 0.095],
    "NKF": [None, 0.320, 0.016, 0.100, None, 0.071],
    "Transformer-MAF": [0.005, 0.301, 0.021, 0.056, 0.179, 0.063],
    "TimeGrad": [0.006, 0.287, 0.021, 0.044, 0.114, 0.049],
    "ScoreGrad": [0.006, 0.256, 0.019, 0.041, 0.101, 0.043],
    "Ours": [0.003, 0.289, 0.023, 0.040, 0.075, 0.085],
}
T2_SD_OURS = [0.001, 0.002, 0.001, 0.004, 0.002, 0.007]
T_CRIT = stats.t.ppf(0.975, df=4)

fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
for ax, j in zip(axes.ravel(), range(len(DS))):
    items = [(k, v[j]) for k, v in T2.items() if v[j] is not None and k != "Ours"]
    items.sort(key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    ypos = np.arange(len(items))
    ax.barh(ypos, vals, color=SOFT, edgecolor="none", height=0.62)
    ours = T2["Ours"][j]
    half = T_CRIT * T2_SD_OURS[j] / np.sqrt(5)
    ax.axvspan(ours - half, ours + half, color=ACCENT, alpha=0.15, lw=0)
    ax.axvline(ours, color=ACCENT, lw=1.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7, color=INK)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(f"{DS[j]}", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("CRPSsum (낮을수록 좋음)", fontsize=7, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.98, 0.02, f"Ours {ours:.3f} ± CI95 {half:.4f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=ACCENT)
fig.suptitle("표 2 재구성: 시계열 CRPSsum 순위와 우리 방법의 95% 구간 (n = 5 시드 가정)", fontsize=10, color=INK)
fig.text(0.01, 0.005, "값: Table 2 (arXiv v4). 구간: 논문 ±를 표준편차로 가정한 t 분포 95% 구간. 기준선의 ±는 표시하지 않음.",
         fontsize=6.5, color=GREY)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(FIG / "own_table2_ci.png", dpi=170)
plt.close(fig)

# ---- 표 1 (planning 평균 보상). 시행 수가 적혀 있지 않아 ±는 논문 값 그대로 보인다.
ENV = ["U-Maze", "Medium", "Large"]
T1 = {
    "Maze2D": {
        "Ours": [(116.7, 2.0), (149.4, 7.5), (159.0, 2.7)],
        "Ours wo MCG": [(110.1, 3.9), (136.1, 10.2), (142.8, 5.6)],
        "Diffuser*": [(113.9, 3.1), (121.5, 2.7), (123.0, 6.4)],
    },
    "Multi2D": {
        "Ours": [(119.1, 4.0), (152.3, 9.9), (167.1, 2.7)],
        "Ours wo MCG": [(107.7, 4.9), (145.6, 6.5), (129.8, 1.5)],
        "Diffuser*": [(128.9, 1.8), (127.2, 3.4), (132.1, 5.8)],
    },
}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
colors = {"Ours": ACCENT, "Ours wo MCG": "#7fa6d6", "Diffuser*": GREY}
width = 0.26
for ax, (task, data) in zip(axes, T1.items()):
    x = np.arange(len(ENV))
    for i, (name, rows) in enumerate(data.items()):
        means = [r[0] for r in rows]
        sds = [r[1] for r in rows]
        ax.bar(x + (i - 1) * width, means, width, yerr=sds, capsize=2.5, color=colors[name],
               label=name, error_kw={"lw": 0.8, "ecolor": INK})
    ax.set_xticks(x)
    ax.set_xticklabels(ENV, fontsize=8, color=INK)
    ax.set_title(task, fontsize=9, color=INK, loc="left")
    ax.set_ylim(0, 185)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
axes[0].set_ylabel("평균 에피소드 보상 (높을수록 좋음)", fontsize=7.5, color=INK)
axes[1].legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle("표 1 재구성: 2D 미로 계획의 평균 보상 (오차 막대는 논문의 ±)", fontsize=10, color=INK)
fig.text(0.01, 0.01, "값: Table 1 (arXiv v4). 시행 수 N이 논문에 없어 신뢰구간은 그리지 않음. Diffuser*는 손으로 짠 PD 제어기 사용.",
         fontsize=6.5, color=GREY)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(FIG / "own_table1_planning.png", dpi=170)
plt.close(fig)
print("wrote", FIG / "own_table2_ci.png", "and", FIG / "own_table1_planning.png")
