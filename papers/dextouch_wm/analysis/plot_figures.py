"""리포트용 그래프 세 장 (직접 계산한 결과만 사용). 실행: python3 analysis/plot_figures.py

1) table3_pearson_n3.png: Table III Pearson의 n=3 불확실성 (p-value, 임계 0.997)
2) transfer_toy.png: 도메인 격차 delta와 리타게팅 오차 s_r에 따른 소스 데이터 이득 (토이 A)
3) sparse_metric.png: 희소 접촉 지표 대 MSE (토이 B)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
T3 = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())["table3"]
TT = json.loads((ROOT / "analysis" / "transfer_toy_results.json").read_text())
SM = json.loads((ROOT / "analysis" / "sparse_metric_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# 1) Table III: Pearson과 p-value
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
tasks = ["Place Shoes", "Place Phone", "Stack Bowls", "Stand Bottle"]
x = np.arange(4)
w = 0.38
for i, (name, col) in enumerate([("WM-Robot", BLUE), ("WM-Mix", ORANGE)]):
    ax[0].bar(x + (i - 0.5) * w, T3[name]["pearson_recomputed"], w, color=col, label=name)
ax[0].axhline(0.997, color=INK2, ls="--", lw=0.8)
ax[0].text(3.6, 0.99, "p<0.05 임계 |r|≥0.997", ha="right", va="bottom", fontsize=7, color=INK2)
ax[0].set_xticks(x, tasks, rotation=15, fontsize=7.5)
ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel("Pearson r (n=3 정책)")
ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")
style(ax[0])
for i, (name, col) in enumerate([("WM-Robot", BLUE), ("WM-Mix", ORANGE)]):
    ax[1].bar(x + (i - 0.5) * w, T3[name]["p_two_sided_df1"], w, color=col, label=name)
ax[1].axhline(0.05, color=INK2, ls="--", lw=0.8)
ax[1].set_xticks(x, tasks, rotation=15, fontsize=7.5)
ax[1].set_ylabel("양측 p (df=1)")
ax[1].set_ylim(0, 1)
style(ax[1])
fig.suptitle("Table III 상관은 정책 3개로 계산됨: 유의하지 않음", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "table3_pearson_n3.png", dpi=200)
plt.close(fig)

# 2) 토이 A: 격차와 리타게팅 오차에 따른 소스 이득
fig, ax = plt.subplots(figsize=(6.4, 3.0))
cols = {"delta=0.0": MUTED, "delta=0.25": AQUA, "delta=0.5": ORANGE, "delta=1.0": BLUE}
for d in ["0.0", "0.25", "0.5", "1.0"]:
    ys = [TT["summary"][f"delta={d},s_r={s}"]["gain_at_400"] for s in ["0.0", "0.25", "0.5", "1.0"]]
    ax.plot([0, 0.25, 0.5, 1.0], ys, marker="o", color=cols[f"delta={d}"], label=f"격차 δ={d}")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("리타게팅 입력 오차 s_r")
ax.set_ylabel("로봇 MSE 이득 (소스 400 추가)")
ax.legend(frameon=False, fontsize=7.5)
style(ax)
ax.set_title("토이 A: 사람 데이터는 격차와 리타게팅 오차가 작을 때만 도움", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "transfer_toy.png", dpi=200)
plt.close(fig)

# 3) 토이 B: MSE 대 접촉 지표 (접촉 3%)
r = SM["results"]["contact_frac=0.03"]
names = ["zero", "blur", "shift", "noisy"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].bar(names, [r[n]["MSE"] for n in names], color=[MUTED, BLUE, ORANGE, AQUA])
ax[0].set_ylabel("MSE (낮을수록 좋음)")
ax[0].set_title("MSE", fontsize=9)
style(ax[0])
ax[1].bar(names, [r[n]["ContactF1"] for n in names], color=[MUTED, BLUE, ORANGE, AQUA])
ax[1].set_ylabel("Contact-F1 (높을수록 좋음)")
ax[1].set_ylim(0, 1)
ax[1].set_title("Contact-F1", fontsize=9)
style(ax[1])
fig.suptitle("토이 B (접촉 3%): 접촉을 전혀 예측하지 않은 zero가 위치만 1칸 틀린 shift보다 MSE가 낮다", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "sparse_metric.png", dpi=200)
plt.close(fig)
print("wrote figures to", OUT)
