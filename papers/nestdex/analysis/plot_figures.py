"""리포트용 그래프 네 장 (직접 계산). 실행: python3 analysis/plot_figures.py

1) time_decomposition.png: Table II 성공 1건당 시간 vs 시행 1회당 시간
2) outer_policy_ci.png: Table III 성공률과 Wilson 95% 구간
3) effort_public_data.png: 공개 데이터(30회)의 유지 구간 노력값, 조건·결과별
4) ensemble_tradeoff.png: Eq. 2 toy — jerk 감소와 접촉 뒤 overshoot·지연
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
E = json.loads((ROOT / "analysis" / "ensemble_toy_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# 1) 시간 분해
rows = R["C2_time_decomposition"]["rows"]
tasks = ["Bottle Disposal", "Dual-Object Transfer", "Ingredient and Pot Transfer"]
labels = ["Bottle\nDisposal", "Dual-Object\nTransfer", "Ingredient &\nPot Transfer"]
T2 = {"Bottle Disposal": (41.37, 88.88), "Dual-Object Transfer": (36.19, 121.63), "Ingredient and Pot Transfer": (43.26, 55.29)}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200, sharey=True)
x = np.arange(len(tasks))
for ax, (title, key) in zip(axes, [("성공 1건당 시간 (Table II)", "per_success"), ("시행 1회당 시간 (= 성공 1건당 × 성공률)", "per_attempt")]):
    if key == "per_success":
        c = [T2[t][0] for t in tasks]
        a = [T2[t][1] for t in tasks]
    else:
        c = [rows[t]["copilot_per_attempt_s_upper"] for t in tasks]
        a = [rows[t]["anyteleop_per_attempt_s"] for t in tasks]
    ax.bar(x - 0.19, c, 0.36, color=BLUE, label="Copilot (NestDex)")
    ax.bar(x + 0.19, a, 0.36, color=MUTED, label="AnyTeleop")
    for xi, v in zip(x - 0.19, c):
        ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
    for xi, v in zip(x + 0.19, a):
        ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_title(title, fontsize=10, color=INK)
    style(ax)
axes[0].set_ylabel("초", color=INK2)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, 0.01, "Copilot 값에는 내부 정책 시연 수집의 상각분이 들어 있어 시행당 시간의 상한이다. 실패·성공 시행 길이가 같다고 가정.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "time_decomposition.png")

# 2) Table III 신뢰구간
T3 = {"Copilot, H-VAE": [100, 75, 90, 100], "Copilot, no H-VAE": [65, 60, 80, 85], "AnyTeleop, no H-VAE": [None, 40, 20, 75]}
cols = {"Copilot, H-VAE": BLUE, "Copilot, no H-VAE": AQUA, "AnyTeleop, no H-VAE": MUTED}
names4 = ["Tongs\nTransfer", "Bottle\nDisposal", "Dual-Object\nTransfer", "Ingredient &\nPot Transfer"]
fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=200)
for j, (k, vals) in enumerate(T3.items()):
    for i, v in enumerate(vals):
        xi = i + (j - 1) * 0.24
        if v is None:
            ax.text(xi, 0.04, "N/A", ha="center", fontsize=7, color=INK2, rotation=90)
            continue
        kk = round(v / 100 * 20)
        lo, hi = wilson(kk, 20)
        ax.plot([xi, xi], [lo, hi], color=cols[k], lw=2, solid_capstyle="round")
        ax.scatter([xi], [kk / 20], color=cols[k], s=36, zorder=3, label=k if i == 1 else None, edgecolor="white", linewidth=1)
ax.set_xticks(range(4), names4, fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("자율 정책 성공률 (20회, Wilson 95%)", color=INK2)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)
fig.text(0.01, 0.01, "출처: Table III. 과제별로 유의한 H-VAE 효과는 Tongs Transfer뿐(Holm p = 0.033), 4과제 합산 MH p = 0.0018.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "outer_policy_ci.png")

# 3) 공개 데이터 노력값
d = json.loads((ROOT / "paper" / "inner-policy-effort.json").read_text())
fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=200)
ylabels = []
for yi, c in enumerate(d["conditions"]):
    ylabels.append(c["label"].replace("Closed-loop, ", "Closed-loop,\n"))
    rng = np.random.default_rng(yi)
    for tr in c["trials"]:
        col = AQUA if tr["outcome"] == "success" else ORANGE
        ax.scatter(tr["holdMedian"], yi + rng.uniform(-0.12, 0.12), color=col, s=30, edgecolor="white", linewidth=0.8, zorder=3)
thr = R["C5_public_effort_data"]["pooled_effort_predicts_success"]["best_threshold"]
ax.axvline(thr, color=INK2, lw=1, ls="--")
ax.text(thr + 0.01, 2.42, f"문턱 {thr:.2f}: 30회 중 90% 판별 (AUC 0.91)", fontsize=7.5, color=INK2)
ax.scatter([], [], color=AQUA, s=30, label="성공")
ax.scatter([], [], color=ORANGE, s=30, label="실패")
ax.set_yticks(range(3), ylabels, fontsize=8)
ax.set_ylim(-0.5, 2.7)
ax.set_xlabel("닫힘 뒤 0.5–5.5 s 손 전체 노력 RMS 중앙값 (드라이버 단위)", color=INK2)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)
fig.text(0.01, 0.01, "출처: 프로젝트 사이트 공개 데이터 inner-policy-effort.json(시행별 값). 성공 여부는 유지 노력 하나로 대부분 갈린다.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "effort_public_data.png")

# 4) 앙상블 trade-off
e2 = E["E2_jerk_and_response"]
keys = ["no_ensemble", "ensemble_m=0.01", "ensemble_m=0.05", "ensemble_m=0.1", "ensemble_m=0.3"]
lab = ["앙상블 없음", "m = 0.01", "m = 0.05", "m = 0.1", "m = 0.3"]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), dpi=200)
jr = [e2[k]["jerk_ratio_vs_no_ensemble"] for k in keys]
ov = [e2[k]["overshoot_past_stop_noise_free"] for k in keys]
cl = [MUTED, BLUE, BLUE, BLUE, BLUE]
axes[0].bar(range(5), jr, color=cl)
axes[0].axhline(1, color=INK2, lw=1, ls=":")
for i, v in enumerate(jr):
    axes[0].text(i, v + 0.05, f"{v:.2f}×", ha="center", fontsize=7.5, color=INK2)
axes[0].set_xticks(range(5), lab, fontsize=8)
axes[0].set_title("P95 jerk 감소 배수 (잡음 있는 정책)", fontsize=10, color=INK)
axes[1].bar(range(5), ov, color=[MUTED, ORANGE, ORANGE, ORANGE, ORANGE])
for i, v in enumerate(ov):
    axes[1].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
axes[1].set_xticks(range(5), lab, fontsize=8)
axes[1].set_title("접촉 뒤 목표를 넘어서는 양 (잡음 없음, 0–1 닫힘)", fontsize=10, color=INK)
for ax in axes:
    style(ax)
fig.text(0.01, 0.01, "toy: 100 Hz, chunk 30. 앙상블은 접촉 전에 만든 예측을 최대 0.3 s 동안 섞는다(지연은 m과 무관하게 chunk 길이). m 값은 논문에 없다.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "ensemble_tradeoff.png")
print("wrote 4 figures to", OUT)
