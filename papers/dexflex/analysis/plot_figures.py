"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다(논문 그림 조각은 만들지 않는다).

  own_teleop_wilson.png   표 2 풀링 성공률(방법별 120시행)과 Wilson 95% 구간, 과제별 점
  own_policy_wilson.png   그림 5 정책 학습 성공률(방법별 60롤아웃)과 Wilson 구간, 과제별 점
  own_fig3_readout.png    그림 3 판독값(진행도·의도 오차)과 표 1 평균의 대조
  own_selection_toy.png   후보 선택 toy: 예측기 정확도와 후보 수에 따른 안전 선택 확률

입력: analysis/recompute_results.json, analysis/fig3_readout.json
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
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
G = json.loads((ROOT / "analysis" / "fig3_readout.json").read_text(encoding="utf-8"))

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
COLORS = {"Raw": "#8a8a85", "Filter": "#4f86c6", "DexGen": "#d9822b", "DexFLEX w/o Tactile": "#a98fbb", "DexFLEX": "#2e2466"}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


# 1. 표 2 풀링 + Wilson
T2 = {"Raw": [50, 30, 40, 30, 15, 10], "Filter": [60, 40, 50, 40, 25, 20], "DexGen": [70, 55, 60, 55, 40, 35],
      "DexFLEX w/o Tactile": [75, 60, 65, 60, 50, 45], "DexFLEX": [90, 80, 85, 80, 70, 65]}
pooled = RES["table2"]["V3_pooled"]
fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=170)
for i, m in enumerate(T2):
    lo, hi = pooled[m]["wilson95"]
    r = pooled[m]["rate"]
    ax.barh(i, 100 * r, color=COLORS[m], alpha=0.85, height=0.55)
    ax.plot([100 * lo, 100 * hi], [i, i], color=INK, lw=1.6)
    ax.scatter(T2[m], np.full(6, i) + np.linspace(-0.2, 0.2, 6), s=14, color="white", edgecolor=INK, lw=0.7, zorder=3)
    ax.text(101, i, f"{pooled[m]['k']}/120 = {100 * r:.1f}%", va="center", fontsize=8.5, color=INK)
ax.set_yticks(range(5))
ax.set_yticklabels(list(T2))
ax.invert_yaxis()
ax.set_xlim(0, 118)
ax.set_xlabel("성공률 (%)  막대: 6과제 풀링(120회), 선: Wilson 95%, 점: 과제별(20회)", color=MUTED, fontsize=8.5)
clean(ax)
ax.set_title("표 2 실로봇 텔레옵: 풀링 성공률과 구간", fontsize=10.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_teleop_wilson.png")
plt.close(fig)

# 2. 그림 5 정책 학습
F5 = {"Direct": [5, 3, 0], "DexGen": [10, 6, 2], "DexFLEX": [12, 9, 7]}
cols5 = {"Direct": "#8a8a85", "DexGen": "#d9822b", "DexFLEX": "#2e2466"}
tot5 = RES["fig5_policy"]["totals"]
fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=170)
for i, m in enumerate(F5):
    lo, hi = tot5[m]["wilson95"]
    ax.barh(i, 100 * tot5[m]["rate"], color=cols5[m], alpha=0.85, height=0.55)
    ax.plot([100 * lo, 100 * hi], [i, i], color=INK, lw=1.6)
    ax.scatter([5 * k for k in F5[m]], np.full(3, i) + np.linspace(-0.15, 0.15, 3), s=16, color="white", edgecolor=INK, lw=0.7, zorder=3)
    ax.text(101, i, f"{tot5[m]['k']}/60 = {100 * tot5[m]['rate']:.1f}%", va="center", fontsize=8.5, color=INK)
ax.set_yticks(range(3))
ax.set_yticklabels(["직접 관절 행동", "DexGen", "DexFLEX"])
ax.invert_yaxis()
ax.set_xlim(0, 118)
ax.set_xlabel("성공률 (%)  막대: 3과제 풀링(60회), 선: Wilson 95%, 점: 과제별(20회)", color=MUTED, fontsize=8.5)
clean(ax)
ax.set_title("그림 5 정책 학습: DexFLEX 대 DexGen은 구간이 겹침(Fisher p = 0.09)", fontsize=10.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_policy_wilson.png")
plt.close(fig)

# 3. 그림 3 판독과 표 1
deg = ["additive_noise", "command_scaling", "delayed_command", "support_finger_removal"]
lab = ["가산 잡음", "명령 배율", "지연", "지지 손가락 제거"]
meth = {"raw": "Raw", "filter": "Filter", "dexgen": "DexGen", "notactile": "DexFLEX w/o Tactile", "dexflex": "DexFLEX"}
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=170)
for ax, metric, title, t1 in [(axs[0], "progress", "과제 진행도 (높을수록 좋음)", 0.78), (axs[1], "intent_error", "의도 오차 (낮을수록 좋음)", 0.18)]:
    for k, name in meth.items():
        ys = [G["values"][metric][k][d] for d in deg]
        ax.plot(range(4), ys, marker="o", ms=4, lw=1.4, color=COLORS[name], label=name)
    mean4 = float(np.mean([G["values"][metric]["dexflex"][d] for d in deg]))
    ax.axhline(t1, color=WARN, lw=1.2, ls="--")
    ax.axhline(mean4, color=COLORS["DexFLEX"], lw=0.9, ls=":")
    box = dict(facecolor="white", edgecolor="none", pad=0.6)
    ax.text(3.2, t1, f"표 1: {t1}", color=WARN, fontsize=8, va="bottom", bbox=box)
    ax.text(3.2, mean4, f"판독 4조건 평균: {mean4:.2f}", color=COLORS["DexFLEX"], fontsize=8, va="top", bbox=box)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_xlim(-0.3, 4.6)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    clean(ax)
axs[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.suptitle("그림 3 판독값(PDF 벡터 좌표)과 표 1 DexFLEX 평균의 대조", fontsize=10.5, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "own_fig3_readout.png")
plt.close(fig)

# 4. 후보 선택 toy
K2 = RES["K2_selection_toy"]
fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=170)
for N, ls in [(4, "-"), (16, "--"), (64, ":")]:
    ys = [K2[f"N={N},acc={a}"]["consequence_safe"] for a in (0.5, 0.7, 0.9)]
    ax.plot([0.5, 0.7, 0.9], ys, ls=ls, marker="o", color=ACC, label=f"결과 예측 점수, 후보 {N}개")
    ax.plot([0.5, 0.7, 0.9], [K2[f"N={N},acc=0.5"]["nearest_safe"]] * 3, ls=ls, color=WARN, lw=1, label=f"초안 최근접, 후보 {N}개")
ax.axhline(0.5, color=MUTED, lw=0.8)
ax.text(0.505, 0.51, "무작위 선택 약 0.50", fontsize=7.5, color=MUTED)
ax.set_xlabel("결과 예측기의 안전 여부 정확도", color=MUTED, fontsize=8.5)
ax.set_ylabel("안전한 후보를 고를 확률", color=MUTED, fontsize=8.5)
ax.set_ylim(0, 0.9)
ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper left")
clean(ax)
ax.set_title("toy: 초안이 위험 쪽을 가리킬 때의 선택 (가정은 03 K2)", fontsize=10, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_selection_toy.png")
plt.close(fig)
print("wrote 4 figures")
