"""리포트용 그래프 네 장. 실행: cd analysis && python3 plot_figures.py  (pca_check·exploration_toy·fig_digitize 결과 JSON이 있어야 함)

1) pca_spectra.png: 공개 PCA 자산의 손별 누적 분산과 90% 지점 k
2) noise_budget.png: 설정별 초기 탐색 분산(관절당 평균)과 k차원 부분공간 집중도 (Sharpa)
3) toy_cost.png: SPIDER식 표본 최적화 toy에서 IID 대비 최종 비용 비 (목표 종류 × 두 잡음 영역)
4) dexmachina_gain.png: Fig. 5 판독값으로 다시 그린 DexMachina ADD-AUC
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import pca_check as pc

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9a9a9f"


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def foot(fig, text):
    fig.text(0.01, 0.01, text, fontsize=6.8, color=INK2)


# 1) 누적 분산
fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=200)
colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
for h, c in zip(pc.HANDS, colors):
    H = pc.hand(h)
    ev = H["t"]["explained_variance"]
    cum = np.cumsum(ev) / ev.sum()
    x = np.arange(1, len(ev) + 1)
    k = pc.k_at(ev, 0.9)
    ax.plot(x, 100 * cum, color=c, lw=2, marker="o", ms=3, label=f"{h} (d={len(ev)}, k={k})")
    ax.scatter([k], [100 * cum[k - 1]], s=46, color=c, edgecolor="white", zorder=3)
ax.axhline(90, color=INK2, lw=1, ls="--")
ax.text(22.5, 88, "90% (논문 설정)", fontsize=7.5, color=INK2, ha="right", va="top")
ax.legend(frameon=False, fontsize=7.4, loc="lower right", bbox_to_anchor=(1.0, 0.12))
ax.set_xlim(0.5, 22.8)
ax.set_ylim(30, 101)
ax.set_xlabel("주성분 수", color=INK2)
ax.set_ylabel("누적 분산 (%)", color=INK2)
style(ax)
foot(fig, "공개 PCA 자산(repo/src/action_bench/assets/*/pca.safetensors)에서 직접 계산. 점 = 90%에 처음 닿는 k. analysis/pca_check.py")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "pca_spectra.png")

# 2) 탐색 분산 예산
pcr = json.loads((A / "pca_check_results.json").read_text())["P2_initial_covariance"]["sharpa"]
names = [("joint_space baseline", "Joint-Space\n(σ=1)", MUTED),
         ("paper_text (matched, split, √λ)", "논문 서술 = DeXtreme\npreset", BLUE),
         ("release SimToolReal preset", "공개 SimToolReal\npreset", ORANGE)]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), dpi=200)
for ax, key, title, fmt in ((axes[0], "mean_var_per_joint", "관절당 평균 초기 분산 (정규화 좌표)", "{:.1f}"),
                            (axes[1], "share_in_k_subspace", "k=9 사람 부분공간 안의 분산 비율", "{:.0%}")):
    vals = [pcr[n][key] for n, _, _ in names]
    xs = np.arange(len(names))
    ax.bar(xs, vals, color=[c for _, _, c in names], width=0.58)
    for x, v in zip(xs, vals):
        ax.text(x, v * 1.02, fmt.format(v), ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, [l for _, l, _ in names], fontsize=7.5)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_ylim(0, max(vals) * 1.22)
    style(ax)
axes[1].axhline(9 / 22, color=INK2, lw=1, ls=":")
axes[1].text(2.35, 9 / 22 + 0.01, "등방이면 9/22", fontsize=7, color=INK2, ha="right")
foot(fig, "Sharpa(d=22, k=9). Joint-Space 초기 σ는 SimToolReal·DeXtreme 원 설정의 sigma_init 0(σ=1). 공개 preset 값으로 계산. analysis/pca_check.py")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "noise_budget.png")

# 3) toy 비용 비
toy = json.loads((A / "exploration_toy_results.json").read_text())["T1_spider_like_final_cost_ratio_median"]
methods = [("eigen+iid", "EigenDEXplore\n(총 분산 맞춤)", BLUE), ("eigen+iid x2", "EigenDEXplore\n(분산 2배)", ORANGE),
           ("random-corr+iid", "무작위 상관\n(같은 스펙트럼)", AQUA)]
kinds = [("human", "사람 분포 목표"), ("mixed", "30% 부분공간 밖"), ("isotropic", "등방 목표")]
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200, sharey=True)
for ax, (regime, vals) in zip(axes, toy.items()):
    w = 0.25
    for i, (m, lab, c) in enumerate(methods):
        ys = [vals[k][m] for k, _ in kinds]
        xs = np.arange(len(kinds)) + (i - 1) * w
        ax.bar(xs, ys, width=w * 0.92, color=c, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=6.6, color=INK2)
    ax.axhline(1.0, color=INK, lw=1)
    ax.text(-0.42, 1.02, "IID = 1", fontsize=7, color=INK2, ha="left", va="bottom")
    ax.set_xticks(np.arange(len(kinds)), [l for _, l in kinds], fontsize=7.8)
    ax.set_title({"noise_ok (N32,T4,s0=0.3)": "(a) 기준 잡음 적당 (N=32, T=4)",
                  "noise_too_small (N32,T8,s0=0.15)": "(b) 기준 잡음 작음 (N=32, T=8)"}[regime], fontsize=9.5, color=INK)
    style(ax)
axes[0].set_ylabel("최종 비용 / IID 최종 비용 (중앙값)", color=INK2)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, [x.replace("\n", " ") for x in l_], frameon=False, fontsize=7.6, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
foot(fig, "식 3(하위 10% 가중 평균)을 numpy로 단순화. Sharpa PCA 공분산, 300회 쌍 비교. 1보다 작을수록 IID보다 낫다. 'Eigen만'은 사람 목표 0.94–0.95, 나머지 1.8–2.9배라 뺌. analysis/exploration_toy.py")
fig.tight_layout(rect=(0, 0.05, 1, 0.92))
fig.savefig(OUT / "toy_cost.png")

# 4) DexMachina 판독
fd = json.loads((A / "fig_digitize_results.json").read_text())["values"]
hands = ["Allegro", "XHand", "Schunk", "Inspire"]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), dpi=200, sharey=True)
for ax, setting, title in ((axes[0], "curriculum", "(a) 커리큘럼 + 보조 보상"), (axes[1], "task_only", "(b) 과제 보상만")):
    xs = np.arange(len(hands) * 2)
    labels, jv, ov = [], [], []
    for task in ("box", "notebook"):
        for h in hands:
            labels.append(f"{h}\n{'상자' if task == 'box' else '노트'}")
            jv.append(fd[f"{task}/{setting}"][h]["joint"])
            ov.append(fd[f"{task}/{setting}"][h]["ours"])
    ax.bar(xs - 0.2, jv, width=0.38, color=MUTED, label="Joint-Space")
    ax.bar(xs + 0.2, ov, width=0.38, color=BLUE, label="EigenDEXplore")
    ax.set_xticks(xs, labels, fontsize=6.6)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_ylim(0, 1)
    style(ax)
axes[0].set_ylabel("ADD-AUC (판독)", color=INK2)
axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
s = json.loads((A / "fig_digitize_results.json").read_text())["summary"]
axes[1].text(0.02, 0.95, f"평균 {s['task_only']['mean_joint']:.2f} → {s['task_only']['mean_ours']:.2f} (+{s['task_only']['ratio_of_means_pct']:.0f}%)",
             transform=axes[1].transAxes, fontsize=7.5, color=INK)
axes[0].text(0.02, 0.95, f"평균 {s['curriculum']['mean_joint']:.2f} → {s['curriculum']['mean_ours']:.2f} (+{s['curriculum']['ratio_of_means_pct']:.0f}%)",
             transform=axes[0].transAxes, fontsize=7.5, color=INK)
foot(fig, "논문 Fig. 5 막대를 300 dpi에서 픽셀 판독(본문 값 4개와 오차 0.004 이하). 오차막대는 생략. analysis/fig_digitize.py")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "dexmachina_gain.png")
print("wrote 4 figures to", OUT)
