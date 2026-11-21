"""우리가 그린 그래프 두 장(report/figures/own_*.png). 논문 그림이 아니므로 공개 가능.

1) own_factor_real.png: 표 IV 배율(최선/평균)과 표 III 실로봇 과제 성공률의 Wilson 95% 구간
2) own_selection_toy.png: 토이 1(선택 편향)의 K별 배율
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
REC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_selection_results.json").read_text(encoding="utf-8"))

for cand in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 1) 배율과 실로봇 구간
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.2]})
fac = REC["table4_improvement_factor"]
names = list(fac)
vals = [fac[k]["factor"] for k in names]
a.barh(names, vals, color="#1f5fae")
a.axvline(1.5, color="#888", ls="--", lw=1)
a.axvline(3.8, color="#c0392b", ls="--", lw=1)
a.set_xlabel("최선 / 평균 (Scrape는 평균 / 최선)")
a.set_title("표 IV: 도메인 파라미터 최선의 배율", fontsize=10)
for i, v in enumerate(vals):
    a.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
a.set_xlim(0, 4.4)

real = REC["table3_real_world"]
tasks = list(real)
for i, t in enumerate(tasks):
    r = real[t]
    lo, hi = r["task_wilson95"]
    k = int(r["task"].split("/")[0])
    p = k / 20 * 100
    b.plot([lo, hi], [i, i], color="#1f5fae", lw=2)
    b.plot(p, i, "o", color="#1f5fae")
    b.text(hi + 1.5, i, f"{k}/20", va="center", fontsize=8)
b.set_yticks(range(len(tasks)))
b.set_yticklabels(tasks)
b.invert_yaxis()
b.set_xlim(0, 110)
b.set_xlabel("과제 성공률 (%), Wilson 95% 구간, n = 20")
b.set_title("표 III: 실로봇 과제 성공률", fontsize=10)
fig.text(0.01, 0.01, "우리가 그린 그래프. 값은 논문 표 IV, 표 III에서 옮겨 recompute.py로 계산.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_factor_real.png", dpi=160)
plt.close(fig)

# 2) 토이 1
fig, ax = plt.subplots(figsize=(6, 4))
for K in sorted({r["K"] for r in TOY["rows"]}):
    rows = [r for r in TOY["rows"] if r["K"] == K]
    ax.plot([r["noise_cv"] * 100 for r in rows], [r["mean_max_over_mean"] for r in rows], "o-", label=f"K = {K}")
ax.axhspan(1.5, 3.8, color="#c0392b", alpha=0.08)
ax.text(5.2, 3.6, "논문 보고 범위 1.5–3.8", fontsize=8, color="#c0392b")
ax.set_xlabel("측정 잡음 상대 SD (%)")
ax.set_ylabel("E[max / 평균] (진짜 차이 0)")
ax.set_title("토이 1: 잡음만으로 생기는 선택 편향", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0.95, 4.0)
fig.text(0.01, 0.01, "우리가 그린 그래프. analysis/toy_selection.py 결과.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_selection_toy.png", dpi=160)
plt.close(fig)
print("wrote", FIG / "own_factor_real.png", FIG / "own_selection_toy.png")
