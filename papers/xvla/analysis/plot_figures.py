"""X-VLA 분석의 그래프 두 장 (우리가 그린 것). 실행: python3 analysis/plot_figures.py
결과: report/figures/own_libero_pm.png, report/figures/own_sota_margin.png

한글 폰트는 Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다. 논문 그림은 넣지 않는다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if font_path:
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

INK = "#1b1b1a"
GREY = "#8a8880"
BLUE = "#1f5fae"
GREEN = "#2e7d4f"
RED = "#b23b3b"


def wilson(p_pct, n, z=1.96):
    """Wilson 95% 구간 (퍼센트 단위)."""
    p = p_pct / 100
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h) * 100, (c + h) * 100


# ---------------------------------------------------------------- 그림 1: 표 3 대 표 6 (LIBERO LoRA)
suites = ["Spatial", "Object", "Goal", "Long"]
t3 = [95.4, 96.6, 96.0, 84.2]   # 표 3, LoRA, 전체 데모
t6 = [96.6, 95.4, 95.0, 84.2]   # 표 6, LoRA, 50 데모 (같은 설정이라고 적힘)
N = 500                         # 스위트당 시도 수 가정 (과제 10 x 50)

fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
xs = range(len(suites))
for i, (a, b) in enumerate(zip(t3, t6)):
    for val, dx, color, label in ((a, -0.12, BLUE, "표 3 (전체 데모)"), (b, 0.12, GREY, "표 6 (50 데모)")):
        lo, hi = wilson(val, N)
        ax.errorbar(i + dx, val, yerr=[[val - lo], [hi - val]], fmt="o", color=color, ms=5, capsize=3,
                    label=label if i == 0 else None)
ax.set_xticks(list(xs))
ax.set_xticklabels([f"LIBERO-{s}" for s in suites], color=INK)
ax.set_ylabel("성공률 (%)", color=INK)
ax.set_ylim(80, 100)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("LIBERO LoRA: 두 표의 스위트별 값 (Wilson 95%, 500회 가정)", fontsize=9.5, color=INK)
fig.tight_layout()
fig.savefig(OUT / "own_libero_pm.png")
plt.close(fig)

# ---------------------------------------------------------------- 그림 2: X-VLA 대 최고 기존 SOTA (표 2)
rows = [
    ("Simpler VM", 80.4, 78.0),
    ("Simpler VA", 75.7, 72.7),
    ("Simpler WidowX", 95.8, 71.9),
    ("LIBERO Spatial", 98.2, 98.4),
    ("LIBERO Object", 98.6, 98.8),
    ("LIBERO Goal", 97.8, 97.9),
    ("LIBERO Long", 97.6, 94.5),
    ("LIBERO 평균", 98.1, 97.1),
    ("CALVIN ABC→D", 4.43, 4.53),
    ("RoboTwin Easy", 70.0, 46.4),
    ("RoboTwin Hard", 39.0, 16.4),
    ("VLABench", 51.1, 39.7),
    ("NAVSIM PDMS", 87.3, 81.7),
]
labels = [r[0] for r in rows]
diffs = [round(x - m, 2) for _, x, m in rows]
colors = [GREEN if d > 0 else RED for d in diffs]

fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
y = list(range(len(rows)))[::-1]
ax.barh(y, diffs, color=colors, height=0.62)
ax.axvline(0, color=INK, lw=0.8)
for yi, d, (lab, x, m) in zip(y, diffs, rows):
    ax.text(d + (0.4 if d >= 0 else -0.4), yi, f"{d:+.2f}", va="center",
            ha="left" if d >= 0 else "right", fontsize=7.4, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8, color=INK)
ax.set_xlabel("X-VLA - 표 2 최고 기존 SOTA (점수 단위, 과제별 척도가 다름)", fontsize=8, color=INK)
ax.set_xlim(-6, 28)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("표 2에서 X-VLA가 최고 기존 값을 넘는 열 (초록) 과 못 넘는 열 (빨강)", fontsize=9.2, color=INK)
fig.tight_layout()
fig.savefig(OUT / "own_sota_margin.png")
plt.close(fig)

print("wrote", OUT / "own_libero_pm.png", OUT / "own_sota_margin.png", "font:", font_path)
