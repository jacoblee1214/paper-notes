"""리포트용 그래프 세 장. 실행: (cd analysis && python3 plot_figures.py)

1) real_tradeoff.png: 실물 6개 과제의 성공률 대 속도 향상 (과제별 점 + 6개 평균, Fig. 1(b) 재구성)
2) es_vs_tmax.png: t_max에 따른 ES (Table 1 GR00T N1.5, RoboCasa PnP·Others)
3) toy_window.png: 추적 toy — 사건 전 세밀 구간 길이 × 판정 지연별 성공률과 속도 (kp 150)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, YELLOW, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a9f"
COLOR = {"base": MUTED, "Fixed Quantization": ORANGE, "DemoSpeedUp": AQUA, "SAIL": YELLOW, "PaceVLA": BLUE}
LABEL = {"base": "GR00T N1.5 (기준)", "Fixed Quantization": "+ 고정 양자화", "DemoSpeedUp": "+ DemoSpeedUp",
         "SAIL": "+ SAIL", "PaceVLA": "+ PaceVLA"}
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) 실물 trade-off
fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=200)
for m in tc.METHODS:
    srs = [tc.T2[m][i][0] for i in range(5)] + [tc.T4[m][0]]
    sps = [tc.T2["base"][i][1] / tc.T2[m][i][1] for i in range(5)] + [tc.T4[m][1] / tc.T4["base"][1]]
    ax.scatter(sps, srs, s=14, color=COLOR[m], alpha=0.45, linewidths=0, zorder=2)
    ax.scatter([np.mean(sps)], [np.mean(srs)], s=170 if m == "PaceVLA" else 70, color=COLOR[m], edgecolor="white", linewidth=1.5, zorder=3,
               marker="*" if m == "PaceVLA" else "o", label=f"{LABEL[m]}  ({np.mean(sps):.2f}×, {np.mean(srs):.1f}%)")
ax.set_xlabel("기준 대비 속도 (성공 시행 시간비, 컨베이어는 처리량비)", color=INK2)
ax.set_ylabel("성공률 (%, 부분 점수 포함)", color=INK2)
ax.set_xlim(0.9, 2.05)
ax.set_ylim(38, 100)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
style(ax)
fig.text(0.01, 0.01, "작은 점: 과제 6개(Table 2·4), 큰 점: 단순 평균 = Fig. 1(b)의 점. 시간은 완전 성공 시행만의 평균이라 실패 비용이 빠져 있음.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "real_tradeoff.png")
print("wrote", OUT / "real_tradeoff.png")

# 2) ES vs t_max
res = json.loads((ROOT / "analysis" / "quant_toy_results.json").read_text())["T3_es_vs_tmax"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=200, sharey=True)
for ax, col in zip(axes, ("RoboCasa PnP", "RoboCasa Others")):
    for m in tc.METHODS[1:]:
        xs = [int(k) for k in res[col][m]]
        ys = list(res[col][m].values())
        ax.plot(xs, ys, color=COLOR[m], lw=2, marker="o", ms=4, label=LABEL[m])
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.axvline(1500, color=INK2, lw=1, ls=":")
    ax.text(1530, 1.47, "논문 t_max", fontsize=7.5, color=INK2)
    ax.axvline(720, color=INK2, lw=0.8, ls=":")
    ax.text(745, 1.47, "GR00T 기본 720", fontsize=7.5, color=INK2)
    ax.set_title(f"{col} (GR00T N1.5)", fontsize=10, color=INK)
    ax.set_xlabel("실패 1회에 매기는 step 수 t_max", color=INK2)
    style(ax)
axes[0].set_ylabel("ES (기준 대비 성공 처리량)", color=INK2)
axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.text(0.01, 0.01, "Table 1의 SR·TPR에서 성공 궤적 길이를 복원(조화평균)하고 t_max만 바꿔 식 8을 다시 계산. t_max = 0은 성공 시행의 step 비.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "es_vs_tmax.png")
print("wrote", OUT / "es_vs_tmax.png")

# 3) toy
rows = json.loads((ROOT / "analysis" / "quant_toy_results.json").read_text())["T2_tracking_toy"]["rows"]
Ws, Ds = (6, 12, 24, 48), (0, 6, 12)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), dpi=200)
for ax, key, title, fmt in ((axes[0], "SR", "성공률", "{:.2f}"), (axes[1], "speedup", "기준 대비 속도", "{:.2f}×")):
    M = np.array([[rows[f"kp150/adaptive W{w} delay{d}"][key] for w in Ws] for d in Ds])
    im = ax.imshow(M, cmap="Blues", vmin=0 if key == "SR" else 1.0, vmax=1.0 if key == "SR" else 2.0, aspect="auto")
    for i in range(len(Ds)):
        for j in range(len(Ws)):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if (v > 0.6 if key == "SR" else v > 1.6) else INK)
    ax.set_xticks(range(len(Ws)), [f"{w}" for w in Ws])
    ax.set_yticks(range(len(Ds)), [f"{d}" for d in Ds])
    ax.set_xlabel("사건(쥐기·놓기) 전 세밀 구간 길이 W (step)", color=INK2)
    ax.set_ylabel("판정 지연 (step)", color=INK2)
    ax.set_title(title, fontsize=10, color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
f2 = rows["kp150/fixed q=2"]
fig.text(0.01, 0.01, f"kp 150, q=2 · 고정 q=2는 성공률 {f2['SR']:.2f}·속도 {f2['speedup']:.2f}×, kp 300 이상에서는 고정 q=2도 {rows['kp300/fixed q=2']['SR']:.2f}. analysis/quant_toy.py",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "toy_window.png")
print("wrote", OUT / "toy_window.png")
