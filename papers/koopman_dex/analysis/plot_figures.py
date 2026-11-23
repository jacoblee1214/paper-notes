"""그래프 네 장 (우리가 그림): report/figures/own_*.png. 실행: python3 analysis/plot_figures.py"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
RES = json.loads((HERE / "recompute_results.json").read_text())
C = {"K-UBM": "#2a9d3f", "ACT": "#8c1c1c", "Diffusion": "#5f4b8b", "UVA": "#8c564b", "etc": "#777777"}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

# 1. 실로봇 성공률과 jerk
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
ks = {"Diffusion": 29, "ACT": 37, "K-UBM": 34}
for i, (m, k) in enumerate(ks.items()):
    lo, hi = wilson(k, 40)
    ax[0].bar(i, 100 * k / 40, color=C[m]); ax[0].errorbar(i, 100 * k / 40, yerr=[[100 * k / 40 - lo], [hi - 100 * k / 40]], color="k", capsize=4)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(ks); ax[0].set_ylim(0, 105); ax[0].set_ylabel("성공률 (%)")
ax[0].set_title("실로봇 성공률 (과제당 10회 가정, 40회)\nWilson 95% 구간", fontsize=9)
for j, part in enumerate(("arm", "hand")):
    d = RES["V6_table10"]["jerk"][part]
    for i, m in enumerate(("Diffusion", "ACT", "K-UBM")):
        ax[1 + j].bar(i, d[m]["mean"], color=C[m]); ax[1 + j].errorbar(i, d[m]["mean"], yerr=d[m]["sd1"], color="k", capsize=4)
    ax[1 + j].set_xticks(range(3)); ax[1 + j].set_xticklabels(("Diffusion", "ACT", "K-UBM"))
    ax[1 + j].set_title(("팔" if part == "arm" else "손") + " RMS jerk (rad/s³), 평균 ± 과제 간 SD", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "own_real.png", dpi=170); plt.close(fig)

# 2. 반응성 표 9
tasks = ["Bucket", "Laptop", "Toilet", "Door", "Tool", "Reorient", "Reloc"]
SR = {"Diffusion": [23,22,13,0,12,1,16], "ACT": [24,23,11,10,9,0,10], "K-UBM": [15,17,7,15,23,17,27]}
fig, ax = plt.subplots(figsize=(8.8, 3.4))
w = 0.26
for i, (m, v) in enumerate(SR.items()):
    x = np.arange(7) + (i - 1) * w
    p = np.array(v) / 30 * 100
    err = np.array([[pp - wilson(k, 30)[0], wilson(k, 30)[1] - pp] for pp, k in zip(p, v)]).T
    ax.bar(x, p, w, color=C[m], label=m); ax.errorbar(x, p, yerr=err, fmt="none", color="k", capsize=2, lw=0.8)
ax.set_xticks(range(7)); ax.set_xticklabels(tasks); ax.set_ylabel("섭동 후 성공률 (%)"); ax.set_ylim(0, 105); ax.legend(ncol=3, fontsize=8)
ax.set_title("표 9 재그림: 과제당 30회, Wilson 95% 구간 (전체 210회: 87 / 87 / 121)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "own_react.png", dpi=170); plt.close(fig)

# 3. 시뮬 평균 (표 값과 K-UBM 그림 판독)
f3 = RES["V4_fig3_readoff"]
rows = [("KODex(V3,V2)", 26.7, 46.7), ("KOROL", 26.8, 37.6), ("ACT(20)", 73.4, 69.9), ("Diff(2,4,8)", 70.8, 52.7),
        ("UVA(판독)", f3["Flow"]["UVA"], f3["DynaMo"]["UVA"]), ("K-UBM(판독)", f3["Flow"]["KUBM"], f3["DynaMo"]["KUBM"])]
fig, ax = plt.subplots(figsize=(8.2, 3.3))
x = np.arange(len(rows))
ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label="Flow", color="#3b7dd8"); ax.bar(x + 0.2, [r[2] for r in rows], 0.4, label="DynaMo", color="#e0892f")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8); ax.set_ylabel("7과제 평균 성공률 (%)"); ax.set_ylim(0, 100); ax.legend(fontsize=8)
ax.set_title("시뮬 평균: 표 3-6 값, UVA와 K-UBM은 그림 3의 막대 판독 값(오차 약 1%p)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "own_sim_avg.png", dpi=170); plt.close(fig)

# 4. 롤아웃 시간 토이
T = json.loads((HERE / "toy_rollout_timing_results.json").read_text())
fig, ax = plt.subplots(figsize=(6.4, 3.2))
labels = ["본 토이: numpy\n(dz=412)", "본 토이: torch+변환\n(dz=412)", "논문 그림 5\n(실로봇)", "논문 표 1\n(시뮬 bucket)"]
vals = [T["dexart_dz412"]["numpy_matvec_ms"], T["dexart_dz412"]["torch_linear_plus_numpy_ms"], 0.017, 0.42]
ax.bar(range(4), vals, color=["#999", "#999", C["K-UBM"], C["K-UBM"]]); ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("ms / 스텝 (로그)"); ax.set_title("K-UBM 한 스텝 롤아웃(z ← Kz) 시간", fontsize=9)
for i, v in enumerate(vals): ax.text(i, v * 1.15, f"{v:g}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "own_timing.png", dpi=170); plt.close(fig)
print("ok")
