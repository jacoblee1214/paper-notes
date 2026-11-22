"""우리가 그린 그래프 세 장을 report/figures/에 만든다(논문 그림은 쓰지 않는다).

own_neoreal_ci.png   : NeoReal 9과제 성공률과 Wilson 95% 구간 (n = 20 가정, 그래프 값 기준)
own_throughput.png   : Figure 12 처리량과 성공률로 역산한 시도 속도 (과제별)
own_sim_paired.png   : 20과제 쌍비교 (N0 - pi0.5, 과제별 점수차)
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

for f in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
C_N0, C_PI, C_ACT = "#1f5fae", "#8a8a85", "#d9a441"

# 1. NeoReal
r = RES["C_neoreal_n20"]["per_task"]
names = [p["task"] for p in r]
n0 = np.array([p["n0_k"] * 5 for p in r], float)
pi = np.array([p["pi_k"] * 5 for p in r], float)
lo = np.array([p["n0_ci95"][0] for p in r]); hi = np.array([p["n0_ci95"][1] for p in r])
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(x - 0.2, pi, 0.4, color=C_PI, label="pi0.5")
ax.bar(x + 0.2, n0, 0.4, color=C_N0, label="N0-VTLA")
ax.errorbar(x + 0.2, n0, yerr=[n0 - lo, hi - n0], fmt="none", ecolor="#222", capsize=3, lw=1)
ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("성공률 (%)"); ax.set_ylim(0, 105)
ax.set_title("NeoReal 9과제: 그래프 값과 Wilson 95% 구간 (시도 20회 가정)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "own_neoreal_ci.png", dpi=160); plt.close(fig)

# 2. 처리량 역산
f12 = RES["K6_throughput_implied_attempts_per_h"]
tasks = list(f12.keys()); variants = ["pi0.5-SFT", "N0-SFT", "pi0.5+chi0", "pi0.5+ALTER", "N0+ALTER"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)
for ax, t in zip(axes, tasks):
    vals = [f12[t][v]["implied_attempts_per_h"] for v in variants]
    cols = [C_PI, "#7fa7d9", C_ACT, "#c07ac0", C_N0]
    ax.bar(range(5), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(5), ["pi0.5\nSFT", "N0\nSFT", "chi0", "pi0.5\nALTER", "N0\nALTER"], fontsize=7)
    ax.set_title(t, fontsize=9); ax.set_ylabel("시도/시간 (역산)", fontsize=8)
fig.suptitle("Figure 12 처리량 / 성공률 = 시도 속도: 4개 변형은 40/h 근처, pi0.5-SFT Cardboard만 80/h", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "own_throughput.png", dpi=160); plt.close(fig)

# 3. 20과제 쌍비교
d = RES["C_paired_20tasks_vs_pi05"]["per_task_diff"]
items = sorted(d.items(), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(8, 5.4))
vals = [v for _, v in items]
ax.barh(range(len(items)), vals, color=[C_N0 if v >= 0 else "#b03a2e" for v in vals])
ax.set_yticks(range(len(items)), [k for k, _ in items], fontsize=7)
ax.axvline(0, color="#333", lw=0.8)
ax.set_xlabel("N0-VTLA - pi0.5 (성공률 점수차, %p)")
m = RES["C_paired_20tasks_vs_pi05"]["mean_diff_pp"]
ax.set_title(f"20과제 쌍비교: 평균 {m:.1f} %p, 17/20 과제에서 우위", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_sim_paired.png", dpi=160); plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
