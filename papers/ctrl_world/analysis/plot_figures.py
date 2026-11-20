"""우리가 만든 그래프 세 장을 report/figures/에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_rank_scatter.png   표 3의 실로봇 대 세계모델 (지시 준수, 성공), 적합선과 y = x (논문 수치를 우리가 다시 그림)
  own_toy_ceiling.png    토이 A·B의 쌍별 순서 맞힘 확률 (toy_rank_noise_results.json)
  own_improvement.png    표 4–7의 범주별 기저 대 개선 (논문 수치를 우리가 다시 그림)

입력: analysis/recompute_results.json, analysis/toy_rank_noise_results.json, 표 값 (recompute.py에서 가져옴)
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
TOY = json.loads((ROOT / "analysis" / "toy_rank_noise_results.json").read_text(encoding="utf-8"))

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

# 표 3 (recompute.py와 같은 값): (실로봇 지시, 세계모델 지시, 실로봇 성공, 세계모델 성공)
T3 = [
    (0.75, 0.60, 0.70, 0.55), (0.90, 0.75, 0.85, 0.70), (0.90, 0.80, 0.85, 0.70),
    (0.40, 0.20, 0.40, 0.20), (0.55, 0.45, 0.55, 0.40), (0.80, 0.75, 0.75, 0.65),
    (0.05, 0.00, 0.05, 0.00), (0.20, 0.20, 0.15, 0.05), (0.80, 0.65, 0.80, 0.30),
    (0.20, 0.10, 0.20, 0.00), (0.50, 0.35, 0.50, 0.35), (0.80, 0.70, 0.80, 0.65),
    (0.15, 0.20, 0.15, 0.00), (0.60, 0.50, 0.50, 0.00), (0.80, 0.70, 0.70, 0.05),
    (0.30, 0.20, 0.30, 0.20), (0.10, 0.00, 0.10, 0.00), (0.85, 0.60, 0.75, 0.50),
    (0.75, 0.45, 0.60, 0.40), (0.80, 0.70, 0.70, 0.55), (0.80, 0.65, 0.65, 0.55),
]
arr = np.array(T3)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


# 1) 산점도
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, (xi, yi, title, fit) in zip(axes, [
    (0, 1, "지시 준수 (21쌍)", "0.87x - 0.04"),
    (2, 3, "성공 (21쌍)", "0.81x - 0.11"),
]):
    x, y = arr[:, xi], arr[:, yi]
    ax.scatter(x, y, s=36, color=ACC, zorder=3)
    lim = [-0.05, 1.0]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="y = x")
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(0, 0.95, 50)
    ax.plot(xs, m * xs + b, color=ACC2, lw=1.6, label=f"적합 y = {m:.2f}x {b:+.2f}")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("실로봇", color=INK); ax.set_ylabel("세계모델", color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
fig.suptitle("표 3: 세계모델 대 실로봇 (논문 수치, 우리가 다시 그림)", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "own_rank_scatter.png", dpi=170)
plt.close(fig)

# 2) 토이 쌍별 확률
A = TOY["A_noise_only_pair_correct_prob"]["per_pair"]
B = TOY["B_wm_model_pair_correct_prob"]["per_pair"]
idx = np.arange(len(A))
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(idx - 0.2, A, width=0.4, color=ACC, label="A: 실로봇 20회 측정 잡음만")
ax.bar(idx + 0.2, B, width=0.4, color=ACC2, label="B: 세계모델 편차(-0.20, sd 0.154) + 20회 측정")
ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(idx); ax.set_xticklabels([str(i + 1) for i in idx], fontsize=8)
ax.set_xlabel("비동점 쌍 번호 (과제 안 정책 쌍 18개)", color=INK)
ax.set_ylabel("순서를 맞힐 확률", color=INK)
ax.set_ylim(0, 1.08)
ax.set_title("토이: 20회 측정으로 쌍 순서를 맞힐 확률 (관측 18/18, 우리가 계산)", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_toy_ceiling.png", dpi=170)
plt.close(fig)

# 3) 범주별 기저 대 개선
cats = ["공간 이해\n(8)", "모양 이해\n(4)", "수건 방향\n(4)", "새 물체\n(2)"]
base = [0.2875, 0.4374, 0.575, 0.25]
after = [0.875, 0.9125, 0.80, 0.75]
x = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - 0.2, base, width=0.4, color=MUTED, label="기저 (π0.5)")
ax.bar(x + 0.2, after, width=0.4, color=ACC, label="합성 데이터 미세조정 후")
for xi, b_, a_ in zip(x, base, after):
    ax.text(xi - 0.2, b_ + 0.015, f"{b_:.2f}", ha="center", fontsize=8, color=INK)
    ax.text(xi + 0.2, a_ + 0.015, f"{a_:.2f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylim(0, 1.05); ax.set_ylabel("성공률 (범주 평균)", color=INK)
ax.set_title("표 4–7: 범주별 기저와 개선 (논문 수치, 우리가 다시 그림). 평균 38.7% → 83.4%", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_improvement.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
