"""우리가 그린 그래프 두 장. 논문 그림은 쓰지 않는다.

  report/figures/own_k_gain.png  : 표 2 값을 K에 따라 그린 것 (평균·최종 거리, 확률 모델 대 BC 기준선)
  report/figures/own_ood.png     : 표 5 OOD 과제와 ID 평균의 비교 (Dec+BC)

실행: python3 analysis/plot_figures.py   (matplotlib 필요, CPU 수 초)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "report", "figures")
os.makedirs(OUT, exist_ok=True)

FONT_CANDIDATES = [
    "Noto Sans CJK KR", "Noto Sans KR", "AppleSDGothicNeo", "Apple SD Gothic Neo",
]
for name in FONT_CANDIDATES:
    try:
        from matplotlib import font_manager
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

K = [1, 5, 10]
T2 = {
    "Dec+BC": ([0.045, 0.045, 0.045], [0.062, 0.062, 0.062]),
    "Dec+DDPM": ([0.053, 0.044, 0.041], [0.071, 0.050, 0.044]),
    "Dec+FM": ([0.052, 0.042, 0.040], [0.071, 0.049, 0.043]),
    "EncDec+DDPM": ([0.052, 0.042, 0.039], [0.071, 0.048, 0.043]),
    "EncDec+FM": ([0.051, 0.041, 0.038], [0.070, 0.047, 0.041]),
}
STY = {
    "Dec+BC": ("#55544f", "--"),
    "Dec+DDPM": ("#1f5fae", "-"),
    "Dec+FM": ("#2a9d8f", "-"),
    "EncDec+DDPM": ("#8e6bbf", "-"),
    "EncDec+FM": ("#c0392b", "-"),
}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=False)
for ax, idx, title in [(axes[0], 0, "평균 거리 (Avg)"), (axes[1], 1, "최종 거리 (Final)")]:
    for name, vals in T2.items():
        color, ls = STY[name]
        ax.plot(K, vals[idx], marker="o", color=color, ls=ls, lw=1.6, label=name)
    ax.set_xscale("log")
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("샘플 수 K (log)")
    ax.set_ylabel("거리 (m)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
fig.suptitle("표 2 재그림: K가 늘면 확률 모델은 좋아지고, BC는 그대로", fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_k_gain.png"), dpi=160)
plt.close(fig)

OOD = [
    ("ID 평균", 0.045, 0.062),
    ("Jigsaw Puzzle", 0.047, 0.065),
    ("Tetra Board", 0.060, 0.082),
    ("Knit Scarf", 0.064, 0.093),
    ("Play Reversi", 0.065, 0.096),
    ("Blowdry Hair", 0.083, 0.118),
    ("Stamp Paper", 0.099, 0.162),
]
fig, ax = plt.subplots(figsize=(7.4, 3.6))
names = [o[0] for o in OOD]
avg = [o[1] for o in OOD]
fin = [o[2] for o in OOD]
y = range(len(OOD))
ax.barh([i + 0.2 for i in y], avg, height=0.38, color="#1f5fae", label="평균 거리")
ax.barh([i - 0.2 for i in y], fin, height=0.38, color="#c0392b", label="최종 거리")
ax.set_yticks(list(y))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("거리 (m), Dec+BC")
ax.set_title("표 5 재그림: 분포 밖 과제는 ID보다 먼 거리 (단일 시드)", fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_ood.png"), dpi=160)
plt.close(fig)

print("wrote", os.path.join(OUT, "own_k_gain.png"), os.path.join(OUT, "own_ood.png"))
