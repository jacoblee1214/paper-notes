"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1.png : 표 1 방법별 12과제 평균과 보수적 95% 구간 (베르누이 분산 상한, 과제당 16회 가정)
own_ablation.png : 표 2 절제 평균과 구간, 최소 검출 차이
own_fig5_gap.png : 그림 5 판독 값의 시연 수별 격차
실행: python3 analysis/plot_figures.py   (numpy, matplotlib)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((Path(__file__).resolve().parent / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for n in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == n for f in font_manager.fontManager.ttflist):
            return n
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
ACC, GREY = "#1f5fae", "#9a9890"

# 1
u = R["V6_table1_mean_uncertainty"]
names = [k for k in u if not k.startswith("worst")]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(names, [u[k]["mean_pct"] for k in names], xerr=[u[k]["ci95_half_pts"] for k in names],
        color=[ACC if k == "Ours" else GREY for k in names], capsize=3)
ax.set_xlabel("12과제 평균 (%) ± 95% (베르누이 분산 상한)")
ax.invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "own_table1.png", dpi=170); plt.close(fig)

# 2
t2 = R["V3_table2"]
ab = R["V10_table2_ablation_tests"]
fig, ax = plt.subplots(figsize=(7, 3.2))
ks = list(t2)
import math
vals = [t2[k]["mean"] for k in ks]
err = [1.96 * 100 * math.sqrt(v / 100 * (1 - v / 100) / 96) for v in vals]
ax.barh(ks, vals, xerr=err, color=[ACC] + [GREY] * 5, capsize=3)
ax.set_xlabel("6과제 평균 (%) ± 95% (96회 가정, 베르누이 분산 상한)")
ax.invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "own_ablation.png", dpi=170); plt.close(fig)

# 3
g = R["V13_figure5_data_efficiency"]["gap_blue_minus_green"]
N = [10, 20, 50, 100, 200]
fig, ax = plt.subplots(figsize=(6, 3.2))
for k, v in g.items():
    ax.plot(N, v, marker="o", label=k)
ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(N)
ax.set_xlabel("사후학습 시연 수"); ax.set_ylabel("중간학습 있음 빼기 없음 (%p, 그림 판독 값)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "own_fig5_gap.png", dpi=170); plt.close(fig)
print("ok")
