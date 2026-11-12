"""분석 리포트용 그래프 네 장 (백서 수치와 toy 결과로 직접 그림). 실행: cd analysis && python3 plot_figures.py

1) yam_sr_ci.png     : 표 1a 과제별 성공률과 Wilson 95% 구간 (과제당 시도 20으로 가정)
2) elo_pairs.png     : 표 13 쌍별 Elo 차이와 95% 구간
3) latency_speedup.png : 표 27 지연 시간 기준과 최적화 값, 배율
4) power_curve.png   : 시도 수별 두 비율 검정 검출력 곡선 (toy_power_fit.py 결과)
"""
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"


def style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=6.8, color=INK2)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1) 표 1a: 과제별 성공률 (SR, %)과 Wilson 95% (n=20 가정)
TASKS = ["Hot Wheels", "Toy Tractor", "Toy Plane", "iPhone→Box", "Fixed-Pipe", "Wire→Breaker"]
SR = {
    "GR00T 2": [85, 45, 70, 65, 70, 95],
    "GR00T 2 Turbo": [50, 40, 55, 55, 90, 100],
    "GR00T 1.7": [0, 0, 0, 15, 35, 0],
    "π0.5": [0, 0, 10, 5, 30, 30],
}
cols = {"GR00T 2": BLUE, "GR00T 2 Turbo": AQUA, "GR00T 1.7": ORANGE, "π0.5": MUTED}
fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200)
w = 0.2
x = np.arange(len(TASKS))
for i, (nm, vals) in enumerate(SR.items()):
    xs = x + (i - 1.5) * w
    lo = [wilson(round(v / 100 * 20), 20)[0] * 100 for v in vals]
    hi = [wilson(round(v / 100 * 20), 20)[1] * 100 for v in vals]
    ax.bar(xs, vals, w * 0.92, color=cols[nm], label=nm)
    ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5)
ax.set_xticks(x, TASKS, fontsize=7.6)
ax.set_ylabel("성공률 (%)", color=INK2)
ax.set_ylim(0, 115)
style(ax, "y")
ax.legend(ncol=4, frameon=False, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
note(fig, "백서 표 1a(YAM 장기 과제, 값은 표 그대로). 막대 오차는 과제당 시도 20을 가정한 Wilson 95% 구간.")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(OUT / "yam_sr_ci.png")
plt.close(fig)

# 2) 표 13 Elo 쌍 비교
PAIRS = [
    ("GR00T 2 - π0.5", 236, 171, 325),
    ("GR00T 2 - GR00T 1.7", 95, 50, 148),
    ("GR00T 2 Turbo - GR00T 1.7", 109, 54, 168),
    ("GR00T 1.7 - π0.5", 141, 78, 227),
    ("GR00T 2 - Turbo (초기)", 12, -20, 48),
    ("GR00T 2 - Turbo (기준)", -14, -58, 28),
]
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
ys = np.arange(len(PAIRS))[::-1]
for y, (lab, d, lo, hi) in zip(ys, PAIRS):
    c = BLUE if lo > 0 else (ORANGE if hi < 0 else MUTED)
    ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="butt")
    ax.plot([d], [y], "o", color=c, ms=5)
    ax.text(hi + 7, y, f"{d:+d} [{lo:+d}, {hi:+d}]", va="center", fontsize=7, color=INK2)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(ys, [p[0] for p in PAIRS], fontsize=7.6)
ax.set_xlabel("Elo 차이 (A - B), 95% 구간 (쌍 부트스트랩)", color=INK2)
ax.set_xlim(-90, 380)
style(ax, "x")
note(fig, "백서 표 13. 파란색은 구간이 0 위, 회색은 0을 포함. 기대 승률은 동점을 뺀 승부 기준이다.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "elo_pairs.png")
plt.close(fig)

# 3) 표 27 지연 시간
LAT = [
    ("Dexter · GR00T 2 · RTX 5090", 449.20, 152.78),
    ("Dexter · GR00T 2 · DGX Station GB300", 219.02, 48.47),
    ("G1 · Turbo · RTX 5090", 224.73, 38.40),
    ("G1 · Turbo · Jetson Thor (120 W)", 720.41, 123.97),
    ("YAM · GR00T 2 · RTX 5090", 550.96, 179.79),
    ("YAM · GR00T 2 · DGX Station GB300", 386.06, 54.00),
]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
ys = np.arange(len(LAT))[::-1]
for y, (lab, b, o) in zip(ys, LAT):
    ax.barh(y + 0.19, b, 0.36, color=MUTED, label="기준(BF16 eager)" if y == ys[0] else None)
    ax.barh(y - 0.19, o, 0.36, color=BLUE, label="최적화" if y == ys[0] else None)
    ax.text(b + 8, y + 0.19, f"{b:.0f} ms", va="center", fontsize=6.8, color=INK2)
    ax.text(o + 8, y - 0.19, f"{o:.0f} ms  ({b / o:.1f}×)", va="center", fontsize=6.8, color=INK)
ax.set_yticks(ys, [l[0] for l in LAT], fontsize=7.4)
ax.set_xlabel("모델 예측 지연, p50 (ms)", color=INK2)
ax.set_xlim(0, 880)
style(ax, "x")
ax.legend(frameon=False, fontsize=7.4, loc="lower right")
note(fig, "백서 표 27의 중앙값. 최적화는 TensorRT·CUDA graph·FP8 등 CUDA 경로이므로 맥에서는 같은 값이 나오지 않는다.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "latency_speedup.png")
plt.close(fig)

# 4) 검출력 곡선 (toy 결과)
T = json.loads((ROOT / "analysis" / "toy_power_fit_results.json").read_text())
curve = T["power_curve_p1_0.5"]
fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
styles = {"n=20": (ORANGE, "과제당 20 (YAM)"), "n=120": (MUTED, "6과제 × 20 합"),
          "n=213": (BLUE, "Dexter 71과제 × 3")}
for key, (c, lab) in styles.items():
    d = curve[key]
    ax.plot([float(k) for k in d], [float(v) for v in d.values()], "-o", color=c, ms=3, lw=1.4, label=lab)
ax.axhline(0.8, color=INK2, lw=0.6, ls="--")
ax.text(0.405, 0.82, "검출력 0.8", fontsize=6.8, color=INK2, ha="right")
ax.set_xlabel("두 성공률 차이 (기준 0.5 대비)", color=INK2)
ax.set_ylabel("검출력 (α=0.05, 양측)", color=INK2)
ax.set_ylim(0, 1.02)
style(ax, "both")
ax.legend(frameon=False, fontsize=7.4, loc="lower right")
note(fig, "몬테카를로 20,000회, 과제 군집 효과는 넣지 않음. 과제 간 차이를 고려하면 실제 검출력은 더 낮다.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "power_curve.png")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
