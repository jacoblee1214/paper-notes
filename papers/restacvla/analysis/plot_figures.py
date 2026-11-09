"""리포트용 그래프 네 장 (직접 계산). 실행: (cd analysis && python3 plot_figures.py) 또는 python3 analysis/plot_figures.py

1) success_ci.png: Table I 과제 성공률(최종 단계)과 Wilson 95% 구간, ResTacVLA vs T-UniT Fisher p
2) phase_decomposition.png: 두 단계 과제의 정렬(A)·완료(I) 성공 횟수와 P(I|A)
3) gate_toy.png: σ 게이트 vs 예측 오차 게이트 toy (접촉 기준 평균, 충돌이 있는 예시 에피소드)
4) residual_vs_raw.png: 잔차 vs 원시 촉각 toy — ridge 시험 MSE 상대 개선(%)과 ρ²
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
G = json.loads((ROOT / "analysis" / "gate_toy_results.json").read_text())


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


def frac(s):
    k, n = s.split("/")
    return int(k), int(n)


# 1) 과제 성공률 + Wilson 구간 ------------------------------------------------
T1 = R["T1_table1"]
cols = {"Lightbulb": "Lightbulb-I", "Plug": "Plug-I", "Peg-in-Hole": "Peg-I", "Transfer": "Transfer", "Wiping": "Wiping"}
labels = ["Lightbulb\n(완료, 25회)", "Plug\n(완료, 25회)", "Peg-in-Hole\n(완료, 15회)", "Peg Transfer\n(15회)", "Plate Wiping\n(15회)"]
methods = [("pi05", "π0.5 (시각만)", MUTED), ("pi05_T-ResNet", "π0.5 + T-ResNet", "#c8c8cc"),
           ("pi05_T-UniT", "π0.5 + T-UniT (최강 기준선)", BLUE), ("ResTacVLA", "ResTacVLA", ORANGE)]
fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=200)
style(ax)
w = 0.19
x = np.arange(len(cols))
for j, (m, lab, c) in enumerate(methods):
    ks = [frac(T1[m]["counts"][cc]) for cc in cols.values()]
    p = np.array([k / n for k, n in ks])
    lo = np.array([wilson(k, n)[0] for k, n in ks])
    hi = np.array([wilson(k, n)[1] for k, n in ks])
    xx = x + (j - 1.5) * w
    ax.bar(xx, 100 * p, w * 0.92, color=c, label=lab, zorder=2)
    ax.errorbar(xx, 100 * p, yerr=[100 * (p - lo), 100 * (hi - p)], fmt="none", ecolor=INK2, lw=0.8, capsize=2, zorder=3)
sig = R["T3_significance"]["ResTacVLA_vs_pi05_T-UniT"]
for i, t in enumerate(cols):
    ax.text(x[i] + 0.5 * w, 101, f"p = {sig[t]['fisher_p']:.2f}", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(x, labels)
ax.set_ylim(0, 108)
ax.set_ylabel("성공률 (%)")
cmh = sig["pooled_CMH(5 tasks, final phase)"]
ax.set_title(f"과제 성공률과 Wilson 95% 구간 · 위 숫자: ResTacVLA vs T-UniT Fisher p (5과제 합산 CMH p = {cmh['p']:.3f})",
             fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.2))
fig.tight_layout()
fig.savefig(OUT / "success_ci.png", bbox_inches="tight")
plt.close(fig)

# 2) 단계 분해 -------------------------------------------------------------
PD = R["T4_phase_decomposition"]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=200, sharey=True)
for ax, task in zip(axes, ["Lightbulb", "Plug", "Peg-in-Hole"]):
    style(ax)
    ms = [("pi05", "π0.5", MUTED), ("pi05_T-UniT", "T-UniT", BLUE), ("ResTacVLA", "ResTacVLA", ORANGE)]
    for j, (m, lab, c) in enumerate(ms):
        a, n = frac(PD[task][m]["A"])
        i, _ = frac(PD[task][m]["I"])
        ax.bar(j - 0.18, 100 * a / n, 0.34, color=c, alpha=0.35, zorder=2)
        ax.bar(j + 0.18, 100 * i / n, 0.34, color=c, zorder=2)
        ax.text(j, 100 * a / n + 3, f"P(I|A)\n{i}/{a}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(3), [lab for _, lab, _ in ms])
    ax.set_title(task, fontsize=9, color=INK)
    ax.set_ylim(0, 112)
axes[0].set_ylabel("성공률 (%)")
axes[0].text(-0.45, 106, "연한 막대 = 정렬 A, 진한 막대 = 완료 I", fontsize=7.5, color=INK2)
dec = PD["log_decomposition_3tasks"]["ResTacVLA_vs_pi05_T-UniT"]
fig.suptitle(f"T-UniT 대비 완료 성공 이득(log 배율)의 {100 * dec['share_of_log_gain_from_A']:.0f}%가 정렬 단계에서 나온다 "
             f"(세 과제 합산 A {dec['A_sum']}, I {dec['I_sum']})", fontsize=9, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "phase_decomposition.png", bbox_inches="tight")
plt.close(fig)

# 3) 게이트 toy ------------------------------------------------------------
A = G["A_gate"]
tl = A["timeline_aligned_to_contact"]
rel = np.array(tl["rel_steps"])
names = [("sigma_gate(paper Eq.3)", "σ 게이트 (논문 Eq. 3)", ORANGE),
         ("raw_error_gate", "예측 오차 ‖z - μ‖² 게이트", BLUE),
         ("precision_weighted_surprise_gate", "정밀도 가중 오차 (고전적 surprise)", MUTED)]
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3), dpi=200)
ax = axes[0]
style(ax)
for k, lab, c in names:
    ax.plot(rel, tl["mean_g"][k], color=c, lw=1.6, label=lab)
ax.axvline(0, color=INK2, lw=0.8, ls="--")
ax.axvspan(0, 175 - 120, color=GRID, alpha=0.6, lw=0)
ax.text(20, 0.03, "접촉 구간", fontsize=8, color=INK2)
ax.set_xlabel("접촉 시작 기준 스텝 (충돌 없는 에피소드 평균)")
ax.set_ylabel("게이트 g")
ax.set_ylim(-0.02, 1.02)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 0.88))
ax.set_title("기대한 접촉: σ 게이트는 접촉 전부터 열린다", fontsize=9, loc="left")
ax = axes[1]
style(ax)
ex = A["example_episode"]
t = np.arange(len(ex["sigma"]))
err = np.array(ex["raw_error"])
ax.fill_between(t, 0, np.clip(err / err.max(), 0, 1), color=AQUA, alpha=0.25, lw=0, label="촉각 예측 오차 (정규화)")
for k, lab, c in names[:2]:
    ax.plot(t, ex["g"][k], color=c, lw=1.4, label=lab)
cs = ex["collide_steps"]
ax.axvspan(cs[0], cs[-1] + 1, color=ORANGE, alpha=0.12, lw=0)
ax.annotate("시각에 보이지 않는\n자유 공간 충돌", xy=(cs[0] + 6, 0.5), xytext=(cs[0] + 22, 0.42), fontsize=7.5, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))
ax.axvline(ex["t_on"], color=INK2, lw=0.8, ls="--")
ax.set_xlabel("스텝 (예시 에피소드 하나)")
ax.set_ylim(-0.02, 1.02)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3)
m = A["main(sigma_n=0.05, collide_mag=1.0)"]
ax.set_title(f"예상 밖 충돌 검출: σ 게이트 {m['sigma_gate(paper Eq.3)']['unexpected_collision_recall']:.0%}, "
             f"오차 게이트 {m['raw_error_gate']['unexpected_collision_recall']:.0%}", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(OUT / "gate_toy.png", bbox_inches="tight")
plt.close(fig)

# 4) 잔차 vs 원시 ---------------------------------------------------------
grid = G["B_residual_vs_raw"]["grid"]
rhos = [0.0, 0.5, 0.9, 0.99]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
style(ax)
series = [("dv=256|N=150|cmp=sep=5000", "시각 256차원, 데이터 150, CMP 별도 5000개", ORANGE, "-"),
          ("dv=256|N=150|cmp=sep=300", "시각 256차원, 데이터 150, CMP 별도 300개", BLUE, "-"),
          ("dv=256|N=150|cmp=in-sample", "시각 256차원, 데이터 150, CMP를 같은 데이터로", MUTED, "-"),
          ("dv=256|N=1500|cmp=in-sample", "시각 256차원, 데이터 1500, CMP를 같은 데이터로", MUTED, "--"),
          ("dv=32|N=150|cmp=sep=5000", "시각 32차원, 데이터 150, CMP 별도 5000개", AQUA, "--")]
xs = np.arange(len(rhos))
for key, lab, c, ls in series:
    ys = [grid[f"{key}|rho2={r}"]["relative_gain_pct"] for r in rhos]
    se = [100 * grid[f"{key}|rho2={r}"]["raw_minus_residual_se"] / grid[f"{key}|rho2={r}"]["mse_raw"] for r in rhos]
    ax.errorbar(xs, ys, yerr=1.96 * np.array(se), color=c, ls=ls, lw=1.5, marker="o", ms=3.5, capsize=2, label=lab)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(xs, [f"ρ² = {r}" for r in rhos])
ax.set_ylabel("원시 대비 잔차의 시험 MSE 개선 (%)")
ax.set_title("잔차가 이기는 조건: 시각 차원 ≫ 데이터, 정확한 CMP, 높은 ρ² (ridge, λ 검증 선택, 시드 20)", fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=7.2, loc="lower left", bbox_to_anchor=(1.0, 0.0))
fig.tight_layout()
fig.savefig(OUT / "residual_vs_raw.png", bbox_inches="tight")
plt.close(fig)
print("wrote", [p.name for p in OUT.glob("*.png") if not p.name.startswith("paper_")])
