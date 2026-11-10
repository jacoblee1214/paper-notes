"""직접 만든 그래프 네 장 (CPU). 실행: python3 analysis/plot_figures.py → report/figures/

1) forest_welch.png: Table II 열별 foresight - residual 차이와 Welch 95% 구간(시드 sd 해석)
2) product_check.png: 개별 단계 성공의 곱과 연속 성공(보고값)의 비교
3) toy_staged.png: 단계별 학습 곡선과 최종 참 성공률(추정기 오차별)
4) toy_joint.png: 끝까지 학습: 희소 보상과 PBRS(정확·편향·악용 Φ)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
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
INK, GRID = "#1b1b1a", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED, RED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f", "#c0392b"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
T = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
S = json.loads((ROOT / "analysis" / "shaping_toy_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("saved", name)


# 1) Welch 구간
rows = T["T2_foresight_vs_residual"]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
labels = [r["column"] for r in rows]
for i, r in enumerate(rows):
    d = r["as_sd"]["diff_pp"]
    lo, hi = r["as_sd"]["ci95"]
    ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=ORANGE if d > 10 else BLUE, capsize=3, ms=5)
    ax.text(hi + 1.0, i, f"{d:+.1f}pp (p={r['as_sd']['p']:.2g})", va="center", fontsize=7.5)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("foresight - residual (pp), 시드 4개 Welch 95% 구간 (sd 해석)")
ax.set_xlim(-12, 60)
style(ax)
save(fig, "forest_welch.png")

# 2) 곱 검사
pc = T["subtask_product_check"]
cats = ["Grasp+Insert", "Insert+Rotate", "All (3단계)"]
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
for ax, name, title in [(axes[0], "residual", "π0 + Residual"), (axes[1], "foresight", "π0 + Residual with Foresight")]:
    d = pc[name]
    prod = [d["G*I_product_pct"], d["I*R_product_pct"], d["G*I*R_product_pct"]]
    rep = [d["G+I_reported"], d["I+R_reported"], d["All_reported"]]
    x = np.arange(3)
    ax.bar(x - 0.2, prod, 0.38, color=MUTED, label="개별 성공의 곱 (oracle 초기화)")
    ax.bar(x + 0.2, rep, 0.38, color=ORANGE if name == "foresight" else BLUE, label="연속 성공 (보고값)")
    for xi, p_, r_ in zip(x, prod, rep):
        ax.text(xi + 0.2, r_ + 1.5, f"{r_:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=7.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 128)
    style(ax)
axes[0].set_ylabel("성공률 (%)")
axes[0].legend(fontsize=7, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.0))
save(fig, "product_check.png")

# 3) toy 단계별 학습
C = S["_curves_every_10_or_20"]
P = S["_meta"]["P_star"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
xs = np.arange(len(C["foresight_exact"])) * 10
for key, col, lab in [("myopic_constant", MUTED, "상수 보상(기준선)"), ("foresight_exact", BLUE, "foresight, V 정확"),
                      ("foresight_noisy", AQUA, "foresight, 표본 잡음"), ("foresight_trap_exploit", RED, "foresight, 악용 오차")]:
    ax.plot(xs, C[key], color=col, lw=1.6, label=lab)
ax.axhline(P, color=INK, ls="--", lw=0.8)
ax.text(xs[-1], P + 0.012, f"최적 {P:.3f}", ha="right", fontsize=7.5)
ax.set_xlabel("단계별 학습 반복 (누적)")
ax.set_ylabel("참 전체 성공률")
ax.legend(fontsize=7, frameon=False, loc="center right")
style(ax)
ax = axes[1]
keys = ["myopic_constant", "sparse_global_staged", "foresight_exact", "foresight_noisy", "foresight_bias+0.15",
        "foresight_bias-0.15", "foresight_overconf_x3", "foresight_scale_matched_const", "foresight_shuffled",
        "foresight_trap_exploit"]
short = ["상수", "희소(전역)", "fore 정확", "fore 잡음", "fore +0.15", "fore -0.15", "fore 과대확신",
         "fore 상수(평균)", "fore 섞임", "fore 악용"]
vals = [S[k]["final_mean"] for k in keys]
sds = [S[k]["final_sd"] for k in keys]
colors = [MUTED if "상수" in s else (RED if "악용" in s else BLUE) for s in short]
y = np.arange(len(keys))
ax.barh(y, vals, xerr=sds, color=colors, height=0.6, error_kw={"lw": 0.8})
ax.axvline(P, color=INK, ls="--", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(short, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("최종 참 성공률 (20 초기값)")
ax.set_xlim(0, 0.6)
style(ax)
save(fig, "toy_staged.png")

# 4) toy joint
J = S["_curves_every_10_or_20"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
xs = np.arange(len(J["joint_sparse"])) * 20
for key, col, lab in [("joint_sparse", MUTED, "희소 보상"), ("joint_pbrs_exact", BLUE, "PBRS, Φ 정확"),
                      ("joint_pbrs_trap_exploit", RED, "PBRS, Φ 악용 오차")]:
    ax.plot(xs, J[key], color=col, lw=1.6, label=lab)
ax.axhline(P, color=INK, ls="--", lw=0.8)
ax.set_xlabel("업데이트 (배치 32 에피소드)")
ax.set_ylabel("참 전체 성공률 (4 초기값 평균)")
ax.legend(fontsize=7, frameon=False, loc="lower right")
style(ax)
ax = axes[1]
keys = ["joint_sparse", "joint_pbrs_exact", "joint_pbrs_bias+0.15", "joint_pbrs_bias-0.15", "joint_pbrs_trap_exploit"]
short = ["희소", "PBRS 정확", "PBRS +0.15", "PBRS -0.15", "PBRS 악용"]
vals = [S[k]["final_mean"] for k in keys]
sds = [S[k]["final_sd"] for k in keys]
y = np.arange(len(keys))
ax.barh(y, vals, xerr=sds, color=[MUTED, BLUE, BLUE, BLUE, BLUE], height=0.6, error_kw={"lw": 0.8})
ax.axvline(P, color=INK, ls="--", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(short, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("최종 참 성공률 (20 초기값)")
ax.set_xlim(0.4, 0.53)
style(ax)
save(fig, "toy_joint.png")
