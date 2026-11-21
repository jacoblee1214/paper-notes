"""analysis/recompute_results.json과 논문 표 값으로 그래프 세 장을 만든다(우리가 그린 그래프, 논문 그림 아님).

실행: python3 analysis/plot_figures.py  -> report/figures/own_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

KO_FONT = "DejaVu Sans"
for path in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        KO_FONT = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams["font.family"] = KO_FONT
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a978f"

# 1. 행동 전용 대비 이득 (데이터셋별, 검증 / 테스트)
h = RES["headline_gain_vs_action_only"]
sets = ["Droid", "Bridge", "RoboMind"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, split, title in zip(axes, ["val", "ood"], ["검증 (보지 않은 궤적)", "테스트 (환경 이동)"]):
    w = 0.38
    xs = range(len(sets))
    single = [h["Doppel_single_expert"][split]["per_dataset_db"][s] for s in sets]
    d8 = [h["Doppel-8"][split]["per_dataset_db"][s] for s in sets]
    ax.bar([x - w / 2 for x in xs], single, w, color=GREY, label="단일 Doppel (같은 데이터)")
    ax.bar([x + w / 2 for x in xs], d8, w, color=ACC, label="Doppel-8 (8개 데이터셋)")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(list(xs), sets)
    ax.set_title(title, fontsize=10, color=INK)
    for x, v in zip(xs, single):
        ax.text(x - w / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
    for x, v in zip(xs, d8):
        ax.text(x + w / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
axes[0].set_ylabel("행동 전용 대비 PSNR 이득 (dB)")
axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
fig.suptitle("그림 A. 행동 전용 기준선 대비 PSNR 이득 (표 2 값으로 계산)", fontsize=10.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_gain.png", dpi=200)
plt.close(fig)

# 2. 표 10 짝비교 구간 (Doppel-8 - Doppel-3)
rows = RES["table10_paired_doppel8_vs_doppel3"]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
labels = [f"{r['dataset']} {'검증' if r['split']=='val' else '테스트'}" for r in rows]
ys = list(range(len(rows)))[::-1]
for y, r in zip(ys, rows):
    lo, hi = r["ci95"]
    ax.plot([lo, hi], [y, y], color=ACC, lw=2.2)
    ax.plot([r["d_psnr"]], [y], "o", color=INK, ms=5)
    ax.text(hi + 0.02, y, f"{r['d_psnr']:+.2f}  (Better {int(round(100*r['better_share']))}%)", va="center", fontsize=7.5)
ax.axvline(0, color=GREY, lw=0.8, ls="--")
ax.set_yticks(ys, labels)
ax.set_xlabel("Doppel-8 - Doppel-3, PSNR 차이 (dB), 95% 구간 (SE × 1.96)")
ax.set_xlim(-0.4, 0.95)
ax.set_title("그림 B. 표 10 짝비교 (같은 200 클립, SE 기준, 클립 단위 가능성)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_paired_ci.png", dpi=200)
plt.close(fig)

# 3. 드로우 효율 (표 16, 평균 PSNR 여섯 세트)
pts = [("행동 전용 (다중, 배치 64)", 3.356, 23.617, GREY),
       ("Doppel-3 (배치 64)", 1.216, 24.464, ACC)]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for name, x, y, c in pts:
    ax.plot(x, y, "o", color=c, ms=8)
    ax.annotate(f"{name}\n{x:.3f}M 드로우, {y:.2f} dB", (x, y), textcoords="offset points",
                xytext=(8, -4), fontsize=7.5, color=INK)
ax.axvline(1.216, color=GREY, lw=0.6, ls=":")
ax.set_xlabel("총 학습 드로우 (백만)")
ax.set_ylabel("여섯 세트 평균 PSNR (dB)")
ax.set_xlim(0.8, 4.2)
ax.set_ylim(22.8, 25.0)
ax.set_title("그림 C. 드로우 효율 (표 16 평균, 단일 실행)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_draws.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
