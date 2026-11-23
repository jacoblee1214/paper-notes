"""재계산 결과로 그래프 세 장을 그린다 (report/figures/own_*.png).

실행: python3 analysis/plot_figures.py   (recompute_results.json이 있어야 함)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
res = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"):
    if Path(path).exists():
        fm.fontManager.addfont(path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
V_C, VT_C = "#7f8c9a", "#1f5fae"
names = ["Test Tube", "Do Puzzle", "Assembly #1", "Assembly #2"]

# 1. 실로봇 RL 후 성공률 (Wilson 95%, 20회 가정)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(4); w = 0.36
for off, key, col, lab in ((-w / 2, "V_rl", V_C, "Vision-only"), (w / 2, "VT_rl", VT_C, "Visuo-tactile")):
    rows = res["real_wilson"][key]
    p = np.array([r["p"] for r in rows]) * 100
    lo = np.array([r["wilson95"][0] for r in rows]) * 100; hi = np.array([r["wilson95"][1] for r in rows]) * 100
    ax.bar(x + off, p, w, color=col, label=lab)
    ax.errorbar(x + off, p, yerr=[p - lo, hi - p], fmt="none", ecolor="black", capsize=3, lw=1)
for i, r in enumerate(res["real_per_task_VT_vs_V"]["rl"]):
    ax.text(i, 109, f"p={r['fisher_p']:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 128); ax.set_ylabel("실로봇 성공률 (%)"); ax.set_yticks(range(0, 101, 20))
ax.set_title("RL 후 실로봇 성공률 (표 2, 셀당 20회 가정, Wilson 95%, Fisher p)", fontsize=10); ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "own_real_ci.png", dpi=170); plt.close(fig)

# 2. 시뮬 대 실로봇 (공동학습 단계와 RL 단계 16셀)
SIM = {"V": {"cotrain": [.42, .12, .39, .39], "rl": [.50, .54, .77, .61]}, "VT": {"cotrain": [.41, .16, .47, .58], "rl": [.72, .71, .92, .79]}}
REAL = {"V": {"cotrain": [.35, .10, .40, .35], "rl": [.35, .25, .80, .60]}, "VT": {"cotrain": [.50, .25, .45, .55], "rl": [.70, .45, .95, .80]}}
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.plot([0, 1], [0, 1], color="#bbb", lw=1)
for st, mk in (("cotrain", "o"), ("rl", "s")):
    for pol, col in (("V", V_C), ("VT", VT_C)):
        ax.scatter(SIM[pol][st], REAL[pol][st], marker=mk, color=col, s=36, label=f"{'Vision' if pol == 'V' else 'Visuo-tactile'}, {'공동학습' if st == 'cotrain' else 'RL 후'}")
ax.annotate("Do Puzzle\n(RL 후)", (0.54, 0.25), (0.62, 0.12), fontsize=7, arrowprops=dict(arrowstyle="-", lw=.6))
ax.set_xlabel("시뮬 성공률 (표 1, 128개 환경)"); ax.set_ylabel("실로봇 성공률 (표 2, 20회 가정)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title(f"시뮬 대 실로봇 16셀 (r={res['sim_vs_real_16cells']['pearson_r']:.2f})", fontsize=10); ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "own_sim_vs_real.png", dpi=170); plt.close(fig)

# 3. 단계별 합산 성공률 (80회 합산, Wilson)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
labs = ["Real-only FT", "Sim-real co-training", "RL post-training"]; keys = ["real_only", "cotrain", "rl"]
for off, pol, col, lab in ((-w / 2, "V", V_C, "Vision-only"), (w / 2, "VT", VT_C, "Visuo-tactile")):
    p = np.array([res["real_pooled_VT_vs_V"][k][f"{pol}_pct"] for k in keys])
    lo = np.array([res["real_pooled_VT_vs_V"][k][f"{pol}_wilson"][0] for k in keys]) * 100
    hi = np.array([res["real_pooled_VT_vs_V"][k][f"{pol}_wilson"][1] for k in keys]) * 100
    ax.bar(np.arange(3) + off, p, w, color=col, label=lab)
    ax.errorbar(np.arange(3) + off, p, yerr=[p - lo, hi - p], fmt="none", ecolor="black", capsize=3, lw=1)
for i, k in enumerate(keys):
    ax.text(i, 99, f"합산 Fisher p={res['real_pooled_VT_vs_V'][k]['fisher_p_pooled']:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_ylim(0, 118); ax.set_ylabel("4과제 합산 성공률 (%)"); ax.set_yticks(range(0, 101, 20))
ax.set_title("단계별 합산 성공률 (80회 가정, Wilson 95%)", fontsize=10); ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "own_stage_pooled.png", dpi=170); plt.close(fig)
print("figures written")
