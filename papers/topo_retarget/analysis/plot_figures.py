"""논문 수치와 우리 계산으로 그린 그래프 다섯 장 (report/figures/own_*.png). 논문 그림은 쓰지 않는다.

입력: analysis/recompute_results.json, analysis/toy_interaction_mesh_results.json
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_interaction_mesh_results.json").read_text(encoding="utf-8"))

KO_FONT = next((f.name for f in font_manager.fontManager.ttflist
                if any(k in f.name for k in ("Apple SD Gothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun"))), "DejaVu Sans")
plt.rcParams["font.family"] = KO_FONT
plt.rcParams["axes.unicode_minus"] = False
INK, GREY, BLUE, ORANGE = "#1b1b1a", "#8a8880", "#1f5fae", "#c9772a"
METHODS = ["TopoRetarget", "OmniRetarget", "Mink", "DexPilot", "GeoRT"]
COL = [BLUE, "#b9b7ae", "#b9b7ae", "#b9b7ae", "#b9b7ae"]


def style(ax, grid="y"):
    ax.grid(axis=grid, color="#e5e3dc", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# 그림 1: 표 2 성공률과 Wilson 95% 구간
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, ds, title in zip(axes, ("Ho-cap", "Pen-Spin"), ("Ho-cap (클립 32개)", "MoCap Pen-Spin (클립 32개)")):
    for i, m in enumerate(METHODS):
        r = RES["table2"][ds][m]
        lo, hi = r["wilson95"]
        ax.errorbar(i, r["rate_exact"], yerr=[[r["rate_exact"] - lo], [hi - r["rate_exact"]]], fmt="o", color=COL[i], ecolor=COL[i], capsize=4, ms=7)
        ax.text(i + 0.12, r["rate_exact"], f"{r['k']}/32", fontsize=8, va="center", color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels(METHODS, rotation=20, fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 100)
    style(ax)
axes[0].set_ylabel("성공률 (%) 과 Wilson 95% 구간")
fig.tight_layout()
fig.savefig(FIG / "own_success_wilson.png", dpi=170)
plt.close(fig)

# 그림 2: Topo 대 기준선 성공률 차이(%p)와 Newcombe 95% 구간
fig, ax = plt.subplots(figsize=(8.6, 4.2))
tests = RES["table2_tests"]
ylabels = []
for i, t in enumerate(tests):
    lo, hi = t["newcombe95_pp"]
    c = BLUE if t["dataset"] == "Pen-Spin" else ORANGE
    ax.errorbar(t["diff_pp"], i, xerr=[[t["diff_pp"] - lo], [hi - t["diff_pp"]]], fmt="o", color=c, capsize=3)
    ylabels.append(f"{t['dataset']} 대 {t['baseline']}  (p={t['fisher_p']:.3f})" if t["fisher_p"] >= 0.001 else f"{t['dataset']} 대 {t['baseline']}  (p<0.001)")
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(len(tests)))
ax.set_yticklabels(ylabels, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("TopoRetarget 성공률 - 기준선 성공률 (%p), Newcombe 95% 구간, p는 Fisher 정확검정(보정 전)")
style(ax, "x")
fig.tight_layout()
fig.savefig(FIG / "own_success_diff.png", dpi=170)
plt.close(fig)

# 그림 3: 표 1 네 지표
T1 = RES["table1"]
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6))
for ax, (key, title, log) in zip(axes, (("Contact precision (mm)", "접촉 정밀도 (mm, 낮을수록 좋음)", False),
                                        ("Contact alignment (deg)", "접촉 정렬 (도)", False),
                                        ("Max penetration (mm)", "최대 침투 (mm)", True),
                                        ("Solve time (ms/frame)", "프레임당 풀이 시간 (ms)", True))):
    vals = [T1[key]["values"][m] for m in METHODS]
    ax.bar(range(5), vals, color=COL)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Topo", "Omni", "Mink", "DexP", "GeoRT"], fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v * (1.08 if log else 1.0) + (0 if log else max(vals) * 0.01), f"{v:g}", ha="center", fontsize=7.5)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_table1_metrics.png", dpi=170)
plt.close(fig)

# 그림 4: κ와 그래프 가중치
G = TOY["graph_stats_every5th_frame"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
d = np.linspace(0, 0.2, 200)
for kappa, c in ((30, BLUE), (10, ORANGE), (0, GREY)):
    axes[0].plot(100 * d, np.exp(-kappa * d), color=c, label=f"κ = {kappa}" + (" (균일, 간선 길이 무시)" if kappa == 0 else ""))
axes[0].set_xlabel("간선 길이 (cm)")
axes[0].set_ylabel("정규화 전 가중치 exp(-κ·길이)")
axes[0].legend(fontsize=8)
style(axes[0])
labels = ["κ = 30", "κ = 10", "κ = 0"]
keys = ["kappa_30", "kappa_10", "kappa_0"]
x = np.arange(3)
axes[1].bar(x - 0.2, [G[k]["ho_share"] for k in keys], 0.4, color=BLUE, label="손 꼭짓점 가중치 중 물체 꼭짓점 몫")
axes[1].bar(x + 0.2, [G[k]["far_share"] for k in keys], 0.4, color=ORANGE, label="가중치 중 5 cm 넘는 간선의 몫")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylim(0, 0.65)
axes[1].legend(fontsize=8, loc="upper left")
style(axes[1])
fig.tight_layout()
fig.savefig(FIG / "own_graph_weights.png", dpi=170)
plt.close(fig)

# 그림 5: 축소 리타게팅 장난감
S = TOY["retarget_toy"]
fig, ax = plt.subplots(figsize=(9.2, 4.0))
groups = [("scale1_kappa30_nvnorm0", "scale1_kappa30_nvnorm1", "scale1_kappa0_nvnorm0"), ("scale1.25_kappa30_nvnorm0", "scale1.25_kappa30_nvnorm1", "scale1.25_kappa0_nvnorm0")]
names = ["손 중심 IK", "메시 κ=30\n(1/Nv 없음)", "메시 κ=30\n(1/Nv 있음)", "메시 κ=0\n(1/Nv 없음)"]
colors = [GREY, BLUE, "#6b9bd1", ORANGE]
for gi, g in enumerate(groups):
    vals = [S[g[0]]["keypoint_ik"]["prec_mm"], S[g[0]]["im"]["prec_mm"], S[g[1]]["im"]["prec_mm"], S[g[2]]["im"]["prec_mm"]]
    for j, (v, c) in enumerate(zip(vals, colors)):
        ax.bar(gi * 5 + j, v, color=c, label=names[j] if gi == 0 else None)
        ax.text(gi * 5 + j, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks([1.5, 6.5])
ax.set_xticklabels(["로봇 손 크기 = 사람 손 × 1.0", "로봇 손 크기 = 사람 손 × 1.25"])
ax.set_ylabel("접촉 키포인트 변위 (mm, 낮을수록 좋음)")
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_toy_scale.png", dpi=170)
plt.close(fig)
print("wrote 5 figures")
