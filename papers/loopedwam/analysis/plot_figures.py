"""분석 그래프 두 장 (우리가 그림). 입력은 논문 표 5 수치와 recompute.py 결과.

실행: python3 analysis/plot_figures.py   -> report/figures/own_latency_fit.png, own_pareto_ci.png
한글 폰트: macOS Apple SD Gothic Neo, 없으면 Linux Noto CJK 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 표 5: (Ka, N, 호출 수 c, 지연 ms, LIBERO 평균)
T5 = [(1, 2, 24, 24.7, 95.18), (2, 2, 36, 27.6, 95.98), (4, 2, 60, 33.2, 95.63),
      (1, 5, 60, 33.3, 95.23), (2, 5, 90, 39.1, 95.55), (1, 10, 120, 47.9, 95.33),
      (4, 5, 150, 54.5, 96.10), (2, 10, 180, 59.9, 95.67), (4, 10, 300, 86.2, 96.22)]
RES = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))
fit = RES["V3_table5_elastic"]["latency_fit"]

# 그림 1: 호출 수 대 지연 (선형 적합)
fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=160)
xs = [t[2] for t in T5]
ys = [t[3] for t in T5]
colors = {1: "#9a9a94", 2: "#1f5fae", 4: "#c0522a"}
for ka in (1, 2, 4):
    pts = [t for t in T5 if t[0] == ka]
    ax.scatter([p[2] for p in pts], [p[3] for p in pts], s=40, color=colors[ka], label=f"Ka={ka}", zorder=3)
xx = [0, 320]
ax.plot(xx, [fit["intercept_ms"] + fit["slope_ms_per_call"] * x for x in xx], color="#333", lw=1, ls="--",
        label=f"적합: {fit['intercept_ms']:.1f} + {fit['slope_ms_per_call']:.3f}c (R²={fit['r2']:.3f})")
ax.set_xlabel("질의당 행동 블록 호출 수 c = N(6+6Ka)")
ax.set_ylabel("지연 (ms, H100, 배치 1)")
ax.set_title("표 5 아홉 설정의 지연은 호출 수에 선형", fontsize=10)
ax.grid(alpha=0.25)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "own_latency_fit.png")
plt.close(fig)

# 그림 2: 지연 대 LIBERO 평균 (표 5 아홉 설정, 6000 에피소드 기준 95% 구간 약 ±0.5점)
half = 0.50
fig, ax = plt.subplots(figsize=(6.2, 3.9), dpi=160)
for ka, n, c, lat, acc in T5:
    ax.errorbar(lat, acc, yerr=half, fmt="o", ms=5, color=colors[ka], capsize=2, lw=0.9)
    ax.annotate(f"({ka},{n})", (lat, acc), textcoords="offset points", xytext=(4, 4), fontsize=6.5)
ax.axhline(96.6, color="#777", lw=0.8, ls=":")
ax.text(88, 96.68, "Fast-WAM* 96.6 (10스텝, 107.5 ms)", fontsize=6.5, ha="right", color="#555")
ax.scatter([107.5], [96.6], marker="s", color="#555", s=28, zorder=4)
ax.set_xlabel("지연 (ms, H100, 배치 1)")
ax.set_ylabel("LIBERO 평균 성공률 (%)")
ax.set_title("엘라스틱 설정 아홉 개: 정확도는 95.2–96.2 안", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_pareto_ci.png")
plt.close(fig)
print("wrote", FIG / "own_latency_fit.png", FIG / "own_pareto_ci.png")
