"""논문 표 값으로 그린 그래프 두 장. 실행: python3 analysis/plot_figures.py
출력: report/figures/own_ablation.png, report/figures/own_sensitivity.png
한글 폰트: macOS(AppleSDGothicNeo) → Linux(Noto CJK) 순서로 찾는다.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

KO_FONT_FILES = ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
for path in KO_FONT_FILES:
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 1. 표 6 절제: 데이터셋별 성능 하락 (논문 값). 막대 색: 학습에 쓰인 센서(TAG, Feel) 대 쓰이지 않은 센서(OF1, OF2)
abl = {
    "텍스트 제거": [4.91, 1.60, 0.75, 0.50],
    "시각 제거": [6.27, 3.23, 1.50, 0.80],
    "TacQuad 텍스트 제거": [0.12, 0.34, 0.41, 0.11],
    "크로스센서 매칭 제거": [0.28, 1.10, 0.37, 0.22],
    "범용 센서 토큰 제거": [0.03, 1.50, 1.22, 0.62],
    "동적 인지 제거": [2.89, 1.25, 1.00, 0.32],
}
cols = ["TAG (seen)", "Feel (seen)", "OF 1.0 (unseen)", "OF 2.0 (unseen)"]
colors = ["#1f5fae", "#1f5fae", "#c0622b", "#c0622b"]
fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=160)
names = list(abl.keys())
width = 0.12
for j, col in enumerate(cols):
    xs = [i + (j - 1.5) * width * 1.0 for i in range(len(names))]
    ys = [abl[n][j] for n in names]
    ax.bar(xs, ys, width=width, color=colors[j], alpha=0.95 if j % 2 == 0 else 0.55, label=col)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8.5, rotation=12)
ax.set_ylabel("성능 하락 (pp, 논문 표 6)", fontsize=9)
ax.set_title("표 6 절제: 제거 시 하락량 (단일 실행, 구간 없음)", fontsize=10)
ax.axhline(0, color="#666", lw=0.6)
ax.legend(fontsize=7.5, ncol=4, loc="upper right", frameon=False)
ax.grid(axis="y", alpha=0.25)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "그래프: 우리가 논문 표 6의 수치로 그림. 파란색은 학습에 쓰인 센서, 주황색은 쓰이지 않은 센서.", fontsize=6.5, color="#55544f")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_ablation.png")
plt.close(fig)

# 2. 표본 수 n에 따른 95% 구간 반폭과 검출 가능한 차이 (이항, 정확도 p=0.8 가정)
ns = [i * 50 for i in range(2, 401)]  # 100 ~ 20,000
p = 0.8
half = [100 * 1.96 * math.sqrt(p * (1 - p) / n) for n in ns]
mdd = [100 * (1.96 + 0.84) * math.sqrt(2 * p * (1 - p) / n) for n in ns]
fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=160)
ax.plot(ns, half, color="#1f5fae", lw=1.6, label="95% 구간 반폭 (정확도 80% 가정)")
ax.plot(ns, mdd, color="#c0622b", lw=1.6, ls="--", label="검출 가능한 차이 (검정력 0.8, 비짝지음)")
ax.axhspan(0.03, 0.45, color="#999", alpha=0.18)
ax.set_ylim(0.02, 12)
ax.text(19000, 0.12, "표 1·6의 많은 차이 (0.03–0.45 pp)", fontsize=7.5, va="center", ha="right", color="#333")
ax.axhline(3.0, color="#666", lw=0.6, ls=":")
ax.text(19000, 3.4, "표 3 OF1 차이 약 3 pp", fontsize=7.5, ha="right", color="#333")
ax.set_xlabel("평가 표본 수 n (논문은 n을 보고하지 않음)", fontsize=9)
ax.set_ylabel("pp (percentage point)", fontsize=9)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_title("표본 수에 따른 불확실성: 논문 차이와 비교", fontsize=10)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.grid(alpha=0.25, which="both")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "그래프: 우리가 이항 근사로 그림. 실제 테스트 표본 수는 논문에 없어 n 범위로 보여 줌.", fontsize=6.5, color="#55544f")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_sensitivity.png")
plt.close(fig)
print("wrote", FIG / "own_ablation.png", FIG / "own_sensitivity.png")
