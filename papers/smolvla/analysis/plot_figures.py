"""우리가 그린 그래프 한 장: 표 6·9·10·12·13의 LIBERO 평균과 이항 95% 신뢰구간(400 에피소드 가정).
실행: python3 analysis/plot_figures.py -> report/figures/own_ablation_ci.png
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

KOREAN_FONTS = ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc"]
for f in KOREAN_FONTS:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        break

OUT = Path(__file__).resolve().parents[1] / "report" / "figures" / "own_ablation_ci.png"
plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rows = [
    ("표 6 CA", 79.0), ("표 6 SA", 74.5), ("표 6 CA+SA (논문 '제안')", 85.5),
    ("표 9 폭 1.00", 82.3), ("표 9 폭 0.75 (본문 선택)", 77.5), ("표 9 폭 0.50", 80.3),
    ("표 10 흐름 매칭", 80.25), ("표 10 회귀 L1", 75.25),
    ("표 12 청크 10", 84.0), ("표 12 청크 50 (본 모델)", 80.3),
]
N = 400


def ci(p):
    q = p / 100
    h = 1.96 * math.sqrt(q * (1 - q) / N) * 100
    return p - h, p + h


fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
for i, (label, p) in enumerate(rows):
    lo, hi = ci(p)
    ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color="#1f5fae", ecolor="#8aa9d6", capsize=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("LIBERO 평균 성공률 (%), 이항 95% 신뢰구간 (4스위트 x 100회 가정)", fontsize=8)
ax.set_xlim(65, 92)
ax.grid(axis="x", color="#e5e5e5")
ax.set_title("SmolVLA 어블레이션: 점 추정과 시행 수 기반 구간 (우리가 계산한 그래프)", fontsize=9)
fig.text(0.01, 0.01, "구간은 독립 시행 가정이라 과제 군집을 반영하지 않습니다. 논문에 시드 반복과 분산이 없습니다.",
         fontsize=6.5, color="#55544f")
fig.tight_layout(rect=(0, 0.03, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
