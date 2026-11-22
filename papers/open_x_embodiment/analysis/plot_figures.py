"""보고 수치와 메타데이터 재집계로 우리가 그린 그래프 세 장 (matplotlib). 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
입력: 보고 값(하드코딩, 출처는 recompute.py 주석과 같음), analysis/meta/ 의 스프레드시트 스냅샷.
구간: 시행 수가 논문에 없으므로, 반올림으로 설명되는 가장 작은 n을 '가정'으로 두고 Wilson 95%를 그린다.
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SHEET = ROOT / "analysis" / "meta" / "oxe_dataset_sheet_2026-11-22.csv"

for cand in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

INK = "#1b1b1a"
ACC = "#1f5fae"
GREY = "#8c8a83"
ORANGE = "#c8702a"
LIGHT = "#b9c9e4"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ------------------------------------------------ 그림 1: 그림 4 라벨 값 (5도메인 + 평균), 우리가 다시 그림
DOM = ["Kitchen\nManip.", "Cable\nRouting", "NYU Door\nOpening", "Autolab\nUR5", "Task-Agn.\nPlay", "평균"]
ORIG = [43, 24, 53, 53, 33, 41]
RT1 = [48, 18, 65, 25, 68, 44]
RT1X = [63, 56, 80, 45, 72, 63]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
x = list(range(len(DOM)))
w = 0.27
b1 = ax.bar([i - w for i in x], ORIG, w, color=LIGHT, label="Original Method (도메인별 기준)")
b2 = ax.bar(x, RT1, w, color=GREY, label="RT-1")
b3 = ax.bar([i + w for i in x], RT1X, w, color=ACC, label="RT-1-X")
for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{int(b.get_height())}",
                ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(DOM, fontsize=7.5)
ax.set_ylabel("성공률 (%)", fontsize=8)
ax.set_ylim(0, 95)
ax.axvspan(4.5, 5.5, color="#f1f1ee", zorder=0)
ax.legend(fontsize=6.8, frameon=False, loc="upper left", ncol=3)
ax.set_title("논문 그림 4 라벨 값을 다시 그림 (n 미보고, 도메인 간 기준 방법이 다름)", fontsize=8.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_fig4_replot.png")
plt.close(fig)

# ------------------------------------------------ 그림 2: 표 I 구간 (Bridge 열 n=30 가정)
labels = ["Original", "RT-1", "RT-1-X", "RT-2-X 55B"]
iris = [13, 40, 27, 50]
rail = [13, 30, 27, 30]
N = 30
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=200, sharey=True)
for ax, vals, title in zip(axes, [iris, rail], ["Bridge IRIS (WidowX)", "Bridge RAIL (WidowX)"]):
    ks = [round(v * N / 100) for v in vals]
    for i, (lab, k) in enumerate(zip(labels, ks)):
        lo, hi = wilson(k, N)
        col = ACC if lab == "RT-1-X" else (ORANGE if lab.startswith("RT-2-X") else GREY)
        ax.errorbar(i, 100 * k / N, yerr=[[100 * k / N - lo], [hi - 100 * k / N]], fmt="o", color=col,
                    capsize=4, ms=6)
        ax.text(i + 0.12, 100 * k / N, f"{vals[i]}%", fontsize=7, va="center", color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title + "\n(가정 n=30)", fontsize=8, color=INK)
    ax.set_ylim(0, 85)
    ax.axhline(13, color=GREY, lw=0.6, ls=":")
axes[0].set_ylabel("성공률 (%) · Wilson 95%", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "own_table1_ci_assumed_n30.png")
plt.close(fig)

# ------------------------------------------------ 그림 3: 표 II 창발 평가, 가장 작은 반올림 호환 n 가정
ROWS = [  # (라벨, 창발 %, 가정 n (반올림 기준 가장 작은 후보))
    ("(1) RT-2 55B", 27.3, 11),
    ("(2) RT-2-X 55B", 75.8, 33),
    ("(3) RT-2-X 55B, Bridge 제외", 42.8, 138),
    ("(4) RT-2-X 5B, 이력 2", 44.4, 9),
    ("(5) RT-2-X 5B, 이력 없음", 14.5, 55),
    ("(7) RT-2-X 5B, 미세조정만", 48.7, 39),
]
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
for i, (lab, pct, n) in enumerate(ROWS):
    k = round(pct * n / 100)
    lo, hi = wilson(k, n)
    ax.errorbar(pct, i, xerr=[[pct - lo], [hi - pct]], fmt="o", color=ACC if "(2)" in lab else GREY,
                capsize=3, ms=5)
    ax.text(101, i, f"n={n}", fontsize=6.8, va="center", color=INK)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.2)
ax.invert_yaxis()
ax.set_xlim(0, 115)
ax.set_xlabel("창발 평가 성공률 (%) · Wilson 95% (가정 n)", fontsize=8)
ax.set_title("표 II 창발 평가: n은 논문에 없고, 반올림 기준 가장 작은 후보를 가정", fontsize=8.2, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_table2_emergent_ci.png")
plt.close(fig)

# ------------------------------------------------ 그림 4: 스프레드시트 기준 에피소드 점유 (v1.0) 와 혼합 9개 임베디먼트
rows = list(csv.reader(open(SHEET, encoding="utf-8")))
data = [r for r in rows[15:] if any(c.strip() for c in r)]
data = [r + [""] * (22 - len(r)) for r in data]
MIX = {"RT-1 Robot Action", "QT-Opt", "Berkeley Bridge", "Freiburg Franka Play", "USC Jaco Play",
       "Berkeley Cable Routing", "Roboturk", "NYU VINN", "Austin VIOLA", "Berkeley Autolab UR5",
       "TOTO Benchmark", "Language Table"}
recs = []
ver = "v1.0"
for r in data:
    if r[0].strip():
        ver = r[0].strip()
    if ver != "v1.0":
        continue
    try:
        ep = float(r[3].replace(",", ""))
    except ValueError:
        ep = 0.0
    recs.append((r[1].strip(), ep, r[1].strip() in MIX))
recs.sort(key=lambda t: -t[1])
top = recs[:14]
tot = sum(t[1] for t in recs)
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
names = [t[0] for t in top][::-1]
vals = [t[1] for t in top][::-1]
cols = [ACC if t[2] else LIGHT for t in top][::-1]
ax.barh(range(len(names)), vals, color=cols)
for i, v in enumerate(vals):
    ax.text(v * 1.02, i, f"{int(v):,} (v1.0 중 {100 * v / tot:.1f}%)", va="center", fontsize=6.5, color=INK)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=6.8)
ax.set_xscale("log")
ax.set_xlim(1e3, 2e6)
ax.set_xlabel("에피소드 수 (로그 축, 스프레드시트 v1.0 60개 중 상위 14개)", fontsize=7.5)
ax.set_title("진한 막대 = 논문 IV-C 혼합 12개 출처 (혼합 안에서는 QT-Opt와 Language Table이 90%)", fontsize=8.2, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_mixture_episode_share.png")
plt.close(fig)
print("saved", sorted(p.name for p in FIG.glob("own_*.png")))
