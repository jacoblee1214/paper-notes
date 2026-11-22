"""우리가 그린 그래프 두 장 (논문 그림을 쓰지 않는다). 입력은 recompute.py와 같은 표 값.

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"Fixed-5": "#8a8a85", "Fixed-15": "#1f5fae", "IG": "#b07d2b", "IG+P": "#6c8f3c", "SCR": "#c0392b"}

# 1) 후보 계획 수 대비 성공률 (표 4 값), 종점 혼합선
W = {"Fixed-5": (84000, 53.125), "Fixed-15": (189000, 87.5), "IG": (168000, 81.25),
     "IG+P": (171000, 84.375), "SCR": (144000, 78.125)}
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=160)
ax.plot([84, 189], [53.125, 87.5], color="#444", lw=1, ls="--", label="두 종점 사이 혼합 (표 4 값)")
for k, (w, s) in W.items():
    ax.scatter(w / 1000, s, s=46, color=C[k], zorder=3)
    dy = {"Fixed-5": -4.2, "Fixed-15": 1.8, "IG": -4.2, "IG+P": 2.0, "SCR": -4.6}[k]
    ax.annotate(f"{k} {s:.1f}%", (w / 1000, s), xytext=(6, dy * 1.2), textcoords="offset points", fontsize=8)
q = 4 / 7
ax.scatter(144, (1 - q) * 53.125 + q * 87.5, s=40, facecolors="none", edgecolors="#444", zorder=3)
ax.annotate("SCR 작업량의 혼합 72.8%", (144, 72.77), xytext=(10, -10), textcoords="offset points", fontsize=7.5, color="#444")
ax.set_xlabel("후보 계획 평가 수 (천 개, 32 궤적 합계)")
ax.set_xlim(60, 205)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(45, 95)
ax.grid(alpha=0.25)
ax.set_title("PushT: 작업량 대비 성공률 (표 4 재배치, 우리가 그림)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "own_frontier.png")
plt.close(fig)

# 2) 궤적 단위 부트스트랩 분포와 논문의 장면 군집 구간, 시드별 차이
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=160, gridspec_kw={"width_ratios": [1.4, 1]})
ci = RES["trial_bootstrap_SCR_workload_pp"]["ci95"]
a1.plot([-6.70, 16.07], [0.5, 0.5], color="#888", lw=3, solid_capstyle="butt")
a1.plot([ci[0], ci[1]], [0.25, 0.25], color="#1f5fae", lw=3, solid_capstyle="butt")
a1.set_ylim(0, 1)
a1.axvline(0, color="#444", lw=0.8)
a1.axvline(5.357, color="#c0392b", lw=1.4)
a1.set_yticks([])
a1.set_xlim(-12, 24)
a1.set_xlabel("SCR - 작업량 맞춘 혼합 (pp)")
a1.set_title("차이 5.36pp와 구간", fontsize=9)
a1.text(16.6, 0.5, "장면 군집", fontsize=7, va="center", color="#555")
a1.text(16.6, 0.25, "궤적 iid", fontsize=7, va="center", color="#1f5fae")
a1.grid(alpha=0.25, axis="x")
seeds = ["시드 42", "시드 99", "합동"]
vals = [11.486, -1.136, 5.357]
a2.bar(seeds, vals, color=["#1f5fae", "#8a8a85", "#c0392b"], width=0.55)
a2.axhline(0, color="#444", lw=0.8)
for i, v in enumerate(vals):
    a2.text(i, v + (0.6 if v >= 0 else -1.6), f"{v:+.2f}", ha="center", fontsize=8)
a2.set_ylabel("pp")
a2.set_ylim(-4, 15)
a2.set_title("시드별 차이 (논문 표 8)", fontsize=9)
a2.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "own_seed_bootstrap.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
