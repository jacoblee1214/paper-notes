"""PhysTwin 논문 표 값으로 우리가 그린 그래프 세 장 (matplotlib). 실행: python3 analysis/plot_figures.py
결과: report/figures/own_*.png. 논문 그림은 쓰지 않는다 (라이선스: arXiv non-exclusive).
한글 폰트: macOS AppleSDGothicNeo 우선, 없으면 Linux Noto CJK 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "report" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
RES = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))

for cand in [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

C_GS = "#7f8c9a"
C_SP = "#c9ced6"
C_OURS = "#1f6f8b"

# ---- 그림 1: 오차 지표 감소율 (PhysTwin 대비)
labels = ["CD", "추적 오차", "LPIPS"]
keys = ["cd", "track", "lpips"]
imp = RES["improvement_table1"]
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
width = 0.2
xs = range(len(labels))
series = [
    ("재구성 · GS-Dynamics", imp["recon_vs_gs"], C_GS, -1.5),
    ("미래 예측 · GS-Dynamics", imp["future_vs_gs"], C_OURS, -0.5),
    ("재구성 · Spring-Gaus", imp["recon_vs_spring"], C_SP, 0.5),
    ("미래 예측 · Spring-Gaus", imp["future_vs_spring"], "#8fb8c8", 1.5),
]
for name, d, color, off in series:
    vals = [d[k] for k in keys]
    bars = ax.bar([x + off * width for x in xs], vals, width, label=name, color=color)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
ax.set_xticks(list(xs))
ax.set_xticklabels(labels)
ax.set_ylabel("감소율 (%)")
ax.set_ylim(0, 100)
ax.set_title("표 1: PhysTwin이 기준선보다 줄인 오차 (보고된 평균으로 계산)", fontsize=9)
ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTDIR / "own_reduction.png")
plt.close(fig)

# ---- 그림 2: 검정 가능 SD 상한 / 기준선 평균 (n = 22 가정)
sd = RES["sd_max_for_p005_vs_gs_dynamics_n22"]
names = {"cd": "CD", "track": "추적 오차", "iou": "IoU", "psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS"}
rows = []
for task, tname in [("recon", "재구성"), ("future", "미래 예측")]:
    for m in ["cd", "track", "iou", "psnr", "ssim", "lpips"]:
        v = sd[f"{task}_{m}"]
        rows.append((f"{tname} {names[m]}", v["sd_max_over_baseline_mean_pct"]))
rows = rows[::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
colors = [C_OURS if p >= 50 else "#d07a5a" for _, p in rows]
ax.barh([r[0] for r in rows], [r[1] for r in rows], color=colors)
for i, (_, p) in enumerate(rows):
    ax.text(p + 2, i, f"{p:.1f}%", va="center", fontsize=7)
ax.axvline(50, color="#444", lw=0.8, ls="--")
ax.text(51, len(rows) - 0.6, "50% 기준", fontsize=7, color="#444")
ax.set_xlabel("유의하려면 허용되는 시나리오별 SD (기준선 평균 대비 %, n = 22, 양측 p < 0.05)")
ax.set_title("표 1 vs GS-Dynamics: 논문이 SD를 주지 않아 검정 여부는 판단 불가", fontsize=9)
ax.tick_params(axis="y", labelsize=7)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTDIR / "own_sd_threshold.png")
plt.close(fig)

# ---- 그림 3: 희소 최적화 ablation (full 대비 오차 증가율)
abl = RES["ablation_error_increase_pct_vs_full"]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
groups = [
    ("재구성 CD", abl["recon"]["zero"]["cd"], abl["recon"]["first"]["cd"]),
    ("재구성 추적", abl["recon"]["zero"]["track"], abl["recon"]["first"]["track"]),
    ("미래 CD", abl["future"]["zero"]["cd"], abl["future"]["first"]["cd"]),
    ("미래 추적", abl["future"]["zero"]["track"], abl["future"]["first"]["track"]),
]
x = range(len(groups))
w = 0.36
for off, idx, label, color in [(-w / 2, 1, "zero-order만", "#b7c7d3"), (w / 2, 2, "first-order만", C_OURS)]:
    vals = [g[idx] for g in groups]
    bars = ax.bar([i + off for i in x], vals, w, label=label, color=color)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
ax.set_xticks(list(x))
ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylabel("full 대비 오차 증가율 (%)")
ax.set_title("표 3: 희소 최적화 한 단계만 쓸 때 오차 증가 (CD, 추적 오차)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTDIR / "own_ablation.png")
plt.close(fig)

print("saved:", sorted(p.name for p in OUTDIR.glob("own_*.png")))
