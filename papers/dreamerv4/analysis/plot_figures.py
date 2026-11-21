"""우리가 그린 그래프 두 장. 논문 표 값만 사용 (Table 7, Table 2). 출력: report/figures/own_*.png
실행: python3 analysis/plot_figures.py  (matplotlib, CPU)"""
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

N = 1000
T7 = {  # Table 7 success rates (%), n = 1000
    "Dreamer 4": [90.1, 29.0, 0.7],
    "WM+BC": [89.4, 16.9, 0.0],
    "VLA (Gemma 3)": [76.7, 11.2, 0.0],
    "BC": [53.8, 0.6, 0.0],
}
ITEMS = ["돌 곡괭이", "철 곡괭이", "다이아몬드"]

def wilson(p_pct, n=N, z=1.96):
    k = round(p_pct * n / 100); p = k / n
    d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*(c-h), 100*(c+h)

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
width = 0.2
colors = ["#1f5fae", "#8fa9c9", "#c9b48f", "#b9b7b0"]
for i, (name, vals) in enumerate(T7.items()):
    xs = [j + (i-1.5)*width for j in range(len(ITEMS))]
    lo_err, hi_err = [], []
    for v in vals:
        lo, hi = wilson(v)
        lo_err.append(max(0, v-lo)); hi_err.append(max(0, hi-v))
    ax.bar(xs, vals, width, label=name, color=colors[i], yerr=[lo_err, hi_err], capsize=2, error_kw={"lw": 0.8})
ax.set_xticks(range(len(ITEMS))); ax.set_xticklabels(ITEMS)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 105)
ax.set_title("Minecraft 오프라인 과제 성공률 (Table 7, n = 1000, Wilson 95% 구간)", fontsize=10)
ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("7/1000 (0.34–1.44%)", xy=(2-0.3, 0.7), xytext=(1.6, 22), fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.6))
fig.tight_layout()
fig.savefig(OUT / "own_milestone_ci.png")
plt.close(fig)

# Table 2: FVD cascade (log scale). Row labels in paper order.
rows = [("확산 포싱", 306), ("+4스텝", 875), ("+shortcut", 329), ("+x-예측", 326), ("+x-손실", 151),
        ("+ramp 가중치", 102), ("+교차 배치", 80), ("+긴 문맥 4층마다", 70), ("+GQA", 71),
        ("+시간 전용 긴 문맥", 91), ("+등록 토큰", 91), ("+공간 토큰 128", 66), ("+공간 토큰 256", 57)]
fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
bars = ax.bar(range(len(vals)), vals, color="#1f5fae")
bars[1].set_color("#c44e52")  # 4스텝만 (품질 붕괴 행)
ax.set_yscale("log")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("FVD (낮을수록 좋음, 로그)")
ax.set_title("설계 절제의 FVD 흐름 (Table 2, 1024 영상, 단일 실행)", fontsize=10)
for i, v in enumerate(vals):
    ax.text(i, v*1.07, str(v), ha="center", fontsize=6.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "own_fvd_cascade.png")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
