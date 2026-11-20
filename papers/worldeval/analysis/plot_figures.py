"""우리가 만든 그래프 두 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_pearson_null.png  정책 4개일 때 무관한 프록시가 Pearson r을 얼마나 높게 만드는지(toy_pearson_null, 같은 난수 시드)
  own_checkpoint_ci.png 논문 표 3의 체크포인트별 성공률(k/40)과 Wilson 95% 구간

입력: 논문 표 3 값(코드에 하드코딩, 출처 표시), analysis/recompute.py 로직 재사용
실행: python3 analysis/plot_figures.py   (matplotlib, numpy, CPU, 수 초)
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

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

INK, MUTED, ACC, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#a33a3a"


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


# 1) 귀무 분포: 프록시가 무작위일 때 정책 4개의 Pearson r
TRUE = np.array([0.9, 0.6, 0.4, 0.1])
rng = np.random.default_rng(0)
M = 200_000
proxy = rng.random((M, 4))
a = TRUE - TRUE.mean(); b = proxy - proxy.mean(1, keepdims=True)
r0 = (b @ a) / np.sqrt((b * b).sum(1) * (a * a).sum())
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
ax.hist(r0, bins=80, color=MUTED, alpha=0.7, density=True)
ax.axvline(0.942, color=ACC, lw=1.6)
ax.axvline(0.958, color=WARN, lw=1.0, ls="--")
ax.text(0.93, ax.get_ylim()[1] * 0.9, "논문 표 1 평균 r = 0.942", color=ACC, fontsize=8, ha="right")
ax.set_xlabel("무작위 프록시의 Pearson r (정책 4개)", color=INK, fontsize=9)
ax.set_ylabel("밀도", color=INK, fontsize=9)
ax.set_title("무작위 프록시도 r ≥ 0.942를 약 2.6%에서 만든다 (우리 토이)", color=INK, fontsize=10, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_pearson_null.png")
plt.close(fig)

# 2) 체크포인트별 성공률과 Wilson 95% 구간 (논문 표 3 값, k/40)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

rows = [
    ("π0 · 과제 A · 5k", 29), ("π0 · 과제 A · 30k", 31), ("π0 · 과제 A · 최종", 37),
    ("π0 · 과제 B · 5k", 3), ("π0 · 과제 B · 30k", 6), ("π0 · 과제 B · 최종", 24),
    ("π0 · 과제 C · 5k", 12), ("π0 · 과제 C · 30k", 18), ("π0 · 과제 C · 최종", 30),
    ("DexVLA · 과제 B · 최종", 10),
]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
for i, (lab, k) in enumerate(rows[::-1]):
    lo, hi = wilson(k, 40)
    ax.plot([lo, hi], [i, i], color=ACC, lw=2)
    ax.plot(k / 40, i, "o", color=ACC, ms=4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
ax.set_xlim(0, 1)
ax.set_xlabel("실물 성공률 (40 롤아웃, Wilson 95% 구간)", color=INK, fontsize=9)
ax.set_title("논문 표 3 체크포인트 성공률의 불확실성 (우리 계산)", color=INK, fontsize=10, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_checkpoint_ci.png")
plt.close(fig)
print("saved", FIG)
