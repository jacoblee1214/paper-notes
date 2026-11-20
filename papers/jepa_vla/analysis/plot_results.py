"""재계산 결과를 그래프 3장으로 그린다 (직접 그린 그림만 사용; 논문 그림은 넣지 않음).

실행: python3 analysis/plot_results.py  → report/figures/delta_forest.png, libero_plus_suites.png, toy_probe.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "03_recompute_results.json").read_text())
TOY = json.loads((ROOT / "analysis" / "03_toy_encoder_probe_results.json").read_text())

FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#1b1b1a", "#55555a", "#e4e4e7"
TEAL, GRAY, ORANGE = "#1b6f6a", "#9a9aa0", "#d9822b"


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) 스위트별 성능 차이와 Newcombe 95% 구간 (LIBERO, 에피소드 500 가정)
rows = []
for tag, key in (("기본 VLA (Table 1, 1/10 데이터)", "T1_ci"),
                 ("OpenVLA-OFT 재현 (Table 3)", "T3_ci_vs_repro")):
    for r in RES[key]:
        rows.append((f"{tag.split(' ')[0]} · {r['suite']}", r["delta_pts"], r["ci95_pts"]))
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
ys = list(range(len(rows)))[::-1]
for y, (lab, d, (lo, hi)) in zip(ys, rows):
    ax.plot([lo, hi], [y, y], color=TEAL, lw=2)
    ax.plot(d, y, "o", color=TEAL, ms=5)
    ax.text(hi + 0.6, y, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=7.5, color=INK2)
ax.axvline(0, color=GRAY, lw=1)
ax.set_yticks(ys, [r[0] for r in rows], fontsize=8)
ax.set_xlabel("성공률 차이 (%p, 우리 방법 - 기준선)")
ax.set_title("LIBERO 스위트별 차이와 95% 구간 (에피소드 500개 가정, 학습 시드 변동 미포함)", fontsize=9)
ax.set_xlim(-6, 40)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "delta_forest.png")
plt.close(fig)

# 2) LIBERO-plus 스위트별 Total: WorldVLA, 기준선, 우리 방법
suites = ["Spatial", "Object", "Goal", "Long"]
raw = RES["T9_12_raw_total_order"]["values"]
src = {"WorldVLA": [raw[s]["WorldVLA"][7] for s in suites],
       "기준선": [raw[s]["base"][7] for s in suites],
       "우리 방법": [raw[s]["ours"][7] for s in suites]}
import numpy as np
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
xs = np.arange(len(suites))
wdt = 0.26
for i, (name, colr) in enumerate((("WorldVLA", GRAY), ("기준선", "#c9c9cf"), ("우리 방법", TEAL))):
    vals = src[name]
    ax.bar(xs + (i - 1) * wdt, vals, width=wdt, color=colr, label=name)
    for x, v in zip(xs + (i - 1) * wdt, vals):
        ax.text(x, v + 0.6, f"{v:.1f}", ha="center", fontsize=6.8, color=INK2)
ax.set_xticks(xs, [f"LIBERO-{s}" for s in suites])
ax.set_ylabel("LIBERO-plus Total (%)")
ax.set_title("LIBERO-plus 스위트별 Total: 우리 방법이 WorldVLA를 넘는 곳은 2/4", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")
ax.set_ylim(0, 40)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "libero_plus_suites.png")
plt.close(fig)

# 3) 장난감 실험: 교란 종류별 인코더가 담는 정보 (낮을수록 많이 담음, 로그 축)
encs = ["PCA", "SFA", "RANDOM"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=200, sharey=True)
for ax, (key, title) in zip(axes, (("nuisance_const_scale2", "상수 교란 (조명형)"),
                                   ("nuisance_fluct_scale2", "변동 교란 (예측 불가)"))):
    m = TOY["settings"][key]["mean_over_seeds"]
    tgts = [("relevant_state", "과제 상태", TEAL), ("nuisance", "교란 n", ORANGE)]
    xs = np.arange(len(encs))
    for j, (t, lab, colr) in enumerate(tgts):
        vals = [max(m[e][t], 1e-4) for e in encs]
        ax.bar(xs + (j - 0.5) * 0.36, vals, width=0.36, color=colr, label=lab)
        for x, v in zip(xs + (j - 0.5) * 0.36, vals):
            ax.text(x, v * 1.25, f"{v:.3f}", ha="center", fontsize=6.3, color=INK2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(xs, ["PCA\n(분산)", "SFA\n(느린 특징)", "RANDOM\n(대조)"], fontsize=7.5)
    ax.set_title(title, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("테스트 MSE / 타깃 분산 (로그)")
axes[0].set_ylim(1e-4, 3)
axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.suptitle("장난감 실험 (선형 인코더, 합성 데이터): 시간 구조 목적은 변동 교란만 버린다", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "toy_probe.png")
plt.close(fig)
print("wrote figures to", FIG)
