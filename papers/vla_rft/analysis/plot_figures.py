"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 여기서 만들지 않는다.

  own_suite_ci.png       표 2 스위트별 Δ와 에피소드 샘플링 Wald 95% CI (가정: 스위트당 500 에피소드)
  own_perturb_cells.png  표 8 섭동 셀별 Δ (VLA-RFT - Base), 셀 수 n 미보고
  own_toy_auc.png        토이: 우회 진폭별 성공 대 실패 AUC (세계모델 오차 σ별)

입력: analysis/recompute_results.json, analysis/toy_reward_validity_results.json
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
TY = json.loads((ROOT / "analysis" / "toy_reward_validity_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
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

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


# 1. 스위트별 Δ와 CI
suites = list(RC["table2_delta_per_suite"].keys())
delta = [RC["table2_delta_per_suite"][s] for s in suites]
ci = [RC["per_suite_wald95_ci_pp"][s] for s in suites]
fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
y = np.arange(len(suites))[::-1]
lo = [d - c[0] for d, c in zip(delta, ci)]
hi = [c[1] - d for d, c in zip(delta, ci)]
ax.errorbar(delta, y, xerr=[lo, hi], fmt="o", color=ACC, ecolor=MUTED, capsize=4, ms=6)
ax.axvline(0, color=MUTED, lw=0.8, ls="--")
ax.set_yticks(y, suites)
ax.set_xlabel("Δ 성공률 (VLA-RFT - Base, pp)", color=INK)
ax.set_title("표 2 스위트별 Δ와 에피소드 샘플링 95% CI (스위트당 500 가정)", color=INK, fontsize=10)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_suite_ci.png")
plt.close(fig)

# 2. 섭동 셀별 Δ
pert = RC["perturbation"]
names = list(pert.keys())
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
for i, k in enumerate(names):
    diffs = pert[k]["cell_diffs"]
    yy = np.full(len(diffs), len(names) - 1 - i) + np.linspace(-0.15, 0.15, len(diffs))
    colors = [ACC if d >= 0 else WARN for d in diffs]
    ax.scatter(diffs, yy, c=colors, s=22, zorder=3)
ax.axvline(0, color=MUTED, lw=0.8, ls="--")
ax.set_yticks(range(len(names))[::-1], names)
ax.set_xlabel("셀별 Δ (VLA-RFT - Base, 성공률 pp)", color=INK)
ax.set_title("표 8 섭동 셀별 차이 (셀 수 n 미보고, 검정 불가)", color=INK, fontsize=10)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_perturb_cells.png")
plt.close(fig)

# 3. 토이 AUC
grid = TY["grid"]
fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
detours = sorted({g["detour"] for g in grid})
sigmas = sorted({g["sigma"] for g in grid})
cols = [ACC, ACC2, WARN]
for c, s in zip(cols, sigmas):
    xs = [g["detour"] for g in grid if g["sigma"] == s]
    ys = [g["auc_detour_success_vs_fail"] for g in grid if g["sigma"] == s]
    ax.plot(xs, ys, "o-", color=c, label=f"σ = {s}")
ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
ax.set_xlabel("우회 진폭 d (유효한 다른 성공 경로)", color=INK)
ax.set_ylabel("AUC: 우회 성공 대 실패", color=INK)
ax.set_ylim(-0.02, 1.02)
ax.set_title("토이: 참조 유사도 보상은 우회 성공을 실패보다 낮게 준다", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=8)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_toy_auc.png")
plt.close(fig)
print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
