"""우리가 만든 그래프 다섯 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_sr_by_task.png     그림 7의 성공률(과제당 20회 가정)과 Wilson 95% 구간: Robot-only 대 Human Pretrain + Cotrain FT, 두 백본
  own_pooled.png         다섯 과제를 합친 성공률(100회 가정)과 Wilson 95% 구간, 전략 네 개 x 백본 둘
  own_temporal_ablation.png  표 III의 칸별 성공률(20회 가정)과 Wilson 구간, Robot-only(그림 7)와 DexRoam
  own_min_detectable.png 20회 대 20회에서 p < 0.05가 되는 최소 차이(기준 성공 횟수별, Fisher 정확 검정)
  own_metric_toy.png     토이: 56차원 MMD/SWD 보정 곡선과 단계별 값

입력: analysis/recompute_results.json, analysis/toy_alignment_metric_results.json (먼저 두 스크립트를 실행할 것)
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TY = json.loads((ROOT / "analysis" / "toy_alignment_metric_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACC, ACC2, WARN, GREEN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a", "#3b7d4f"
GREYBAR = "#b9b8b0"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


TASKS = ["Pick\nChips Can", "Throw\nTrash", "Deliver\nFruit", "Push Chair &\nClose Laptop", "Pour\nWater"]
SR = {
    "GR00T N1.7": [[.50, .70, .65, .90], [.35, .60, .55, .65], [.40, .55, .60, .75], [.10, .20, .15, .25], [.10, .25, .15, .25]],
    "π0.5": [[.40, .50, .75, .75], [.35, .65, .55, .65], [.60, .70, .70, .75], [.20, .30, .35, .40], [.05, .30, .10, .30]],
}

# ---------------------------------------------------------------- 1. 과제별 성공률 + 구간
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, (bb, rows) in zip(axes, SR.items()):
    x = np.arange(5)
    for off, idx, col, lab in ((-0.18, 0, GREYBAR, "Robot-only"), (0.18, 3, ACC, "Human Pretrain + Cotrain FT")):
        k = np.array([round(r[idx] * 20) for r in rows])
        lo, hi = zip(*[wilson(int(v), 20) for v in k])
        p = k / 20
        ax.bar(x + off, p, 0.34, color=col, label=lab)
        ax.errorbar(x + off, p, yerr=[p - np.array(lo), np.array(hi) - p], fmt="none", ecolor=INK, lw=1, capsize=2.5)
        for xi, v in zip(x + off, k):
            ax.text(xi, 1.02, f"{v}/20", ha="center", va="bottom", fontsize=7, color=MUTED, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=8)
    ax.set_title(bb, fontsize=11, color=INK)
    ax.set_ylim(0, 1.12)
    style(ax)
axes[0].set_ylabel("성공률 (과제당 20회 가정, Wilson 95%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIG / "own_sr_by_task.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 2. 합친 성공률
fig, ax = plt.subplots(figsize=(7.6, 4.0))
labels = ["Robot-only", "Cotrain", "Pretrain\n+ Robot FT", "Pretrain\n+ Cotrain FT"]
cols = [GREYBAR, "#8fb3dc", "#5f8fc7", ACC]
w = 0.18
for gi, (bb, key) in enumerate((("GR00T N1.7", "GR00T"), ("π0.5", "pi05"))):
    P = RC["V6"]["pooled"][key]
    for i in range(4):
        k = P["k"][i]
        lo, hi = P["wilson95"][i]
        p = k / 100
        xpos = gi + (i - 1.5) * w * 1.15
        ax.bar(xpos, p, w, color=cols[i], label=labels[i].replace("\n", " ") if gi == 0 else None)
        ax.errorbar(xpos, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, lw=1, capsize=2.5)
        ax.text(xpos, hi + 0.015, f"{k}", ha="center", fontsize=7.5, color=MUTED)
    c = P["cmh_vs_robot"][2]["p"]
    ax.text(gi, 0.78, f"Pretrain+Cotrain FT 대 Robot-only\n층화 검정 p = {c:.1e}", ha="center", fontsize=7.5, color=ACC)
ax.set_xticks([0, 1])
ax.set_xticklabels(["GR00T N1.7", "π0.5"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("다섯 과제 합산 성공률 (100회 가정, Wilson 95%)")
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=4)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_pooled.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 3. 표 III 칸별
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
cells = RC["V7"]["cells"]
strat_lab = {"Cotrain": "Cotrain", "Pretrain+RobotFT": "Pretrain\n+ Robot FT", "Pretrain+CotrainFT": "Pretrain\n+ Cotrain FT"}
for ax, task in zip(axes, ["Pick Chips Can", "Throw Trash"]):
    cs = [c for c in cells if c["task"] == task]
    x = np.arange(3)
    for off, key, col, lab in ((-0.27, "k_robot_only_fig7", GREYBAR, "Robot-only (그림 7)"), (0.0, "k_wo_temporal", WARN, "w/o Temporal"),
                               (0.27, "k_dexroam", ACC, "DexRoam (표 III)")):
        ks = [c[key] for c in cs]
        p = np.array(ks) / 20
        lo, hi = zip(*[wilson(k, 20) for k in ks])
        ax.bar(x + off, p, 0.25, color=col, label=lab)
        ax.errorbar(x + off, p, yerr=[p - np.array(lo), np.array(hi) - p], fmt="none", ecolor=INK, lw=1, capsize=2)
        for xi, k in zip(x + off, ks):
            ax.text(xi, 1.0, f"{k}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([strat_lab[c["strategy"]] for c in cs], fontsize=8.5)
    ax.set_title(task + " (Absolute Action은 6칸 모두 0/20)", fontsize=10, color=INK)
    ax.set_ylim(0, 1.08)
    style(ax)
axes[0].set_ylabel("GR00T N1.7 성공률 (20회 가정, Wilson 95%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG / "own_temporal_ablation.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 4. 최소 구분 차이
md = {int(k): v for k, v in RC["V13"]["min_detectable_pp_by_base_k"].items()}
ks = sorted(md)
fig, ax = plt.subplots(figsize=(6.6, 3.7))
ax.plot([k / 20 * 100 for k in ks], [md[k] for k in ks], "o-", color=ACC, ms=4)
ax.axhspan(13, 27, color=WARN, alpha=0.12)
ax.text(52, 14.5, "논문 본문의 평균 향상 폭 +13–27 %p", color=WARN, fontsize=8)
ax.set_xlabel("기준(Robot-only) 성공률 (%)")
ax.set_ylabel("p < 0.05가 되는 최소 차이 (%p)")
ax.set_ylim(0, 50)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_min_detectable.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 5. 지표 토이
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
A = TY["A_calibration"]
s = [float(k.split("_")[1]) for k in A]
ax = axes[0]
ax.plot(s, [v["MMD"] for v in A.values()], "o-", color=ACC, label="MMD", ms=4)
ax.plot(s, [v["SWD"] for v in A.values()], "s-", color=ACC2, label="SWD", ms=4)
ax.axhspan(0.274, 0.413, color=ACC, alpha=0.12)
ax.axhspan(0.553, 0.631, color=ACC, alpha=0.06)
ax.text(0.02, 0.33, "논문 정렬 뒤 MMD 범위", fontsize=7.5, color=ACC)
ax.text(0.02, 0.57, "논문 정렬 전 MMD 범위", fontsize=7.5, color=ACC)
ax.set_xlabel("모든 56차원을 평행이동한 크기 (강건 표준편차 단위)")
ax.set_ylabel("지표 값 (도메인당 표본 2,000개)")
ax.set_title("56차원 가우시안에서의 보정", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)

ax = axes[1]
modes = ["raw", "relative_native", "relative_resampled", "floor_robot_split"]
mlab = ["절대 행동", "상대 행동\n(원래 시간축)", "상대 행동\n+ 진행 재표집", "같은 분포\n바닥값"]
x = np.arange(4)
for off, lab, key, col in ((-0.2, "같은 행동 (B)", "B_structure", ACC), (0.2, "다른 행동 (C)", "C_behavior_diff", WARN)):
    v = [TY[key][m]["MMD"] for m in modes]
    ax.bar(x + off, v, 0.38, color=col, label=lab)
    for xi, vi in zip(x + off, v):
        ax.text(xi, vi + 0.003, f"{vi:.3f}", ha="center", fontsize=7.5, color=MUTED)
ax.set_xticks(x)
ax.set_xticklabels(mlab, fontsize=8)
ax.set_ylabel("MMD (토이)")
ax.set_title("사람 = 로봇 행동(B)과 사람이 다른 행동(C)에서의 단계별 MMD", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_metric_toy.png", dpi=170)
plt.close(fig)
print("wrote 5 figures to", FIG)
