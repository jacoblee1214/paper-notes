"""BBF 재계산 결과를 그래프 두 장으로 그린다 (우리가 그린 그래프, 논문 그림 없음).

  report/figures/own_bbf_ablation.png     절제 실험: BBF 대비 IQM 차이와 95% CI
  report/figures/own_bbf_learning_curve.png  RR=2 스티키 학습 곡선 (26게임, 55게임)

입력: analysis/recompute_results.json (recompute.py가 만든다)
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

from matplotlib import font_manager

# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 설치된 첫 폰트를 쓴다
_installed = {f.name for f in font_manager.fontManager.ttflist}
_korean = next((n for n in ("Noto Sans CJK KR", "Apple SD Gothic Neo") if n in _installed), "DejaVu Sans")
plt.rcParams["font.family"] = [_korean, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, GRID, ACC = "#1b1b1a", "#dcdad3", "#1f5fae"

LABELS = {
    "no_resets": "재설정 제거", "n10": "n=10 고정", "no_harder_resets": "강한 재설정 제거",
    "no_annealing": "n-step 감쇠 제거", "no_WD": "가중치 감쇠 제거", "no_SPR": "SPR 손실 제거",
    "gamma099": "γ=0.99 고정",
}

# 1. 절제 실험 (RR=8, RR=2): 점 추정과 CI
rows = []
for rr in ("RR8", "RR2"):
    base = f"BBF_{rr}"
    base_iqm = RES["csv_pooled_runs"][base]["IQM"]
    for key, label in LABELS.items():
        k = f"BBF_{rr}_{key}"
        ci_key = f"{k} - {base}"
        if k not in RES["csv_pooled_runs"] or ci_key not in RES["bootstrap_diff_CI95_IQM"]:
            continue
        point = RES["csv_pooled_runs"][k]["IQM"] - base_iqm
        lo, hi = RES["bootstrap_diff_CI95_IQM"][ci_key]
        rows.append((f"{rr}: {label}", point, lo, hi))

fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=160)
ys = list(range(len(rows)))[::-1]
for y, (name, p, lo, hi) in zip(ys, rows):
    color = ACC if hi < 0 else ("#8a8780" if lo <= 0 <= hi else "#b5432f")
    ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
    ax.plot(p, y, "o", color=color, ms=6)
ax.axvline(0, color=INK, lw=0.9, ls="--")
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
ax.set_xlabel("IQM 차이 (절제 − BBF 기준), 26게임, 실행 단위 95% CI", fontsize=9.5, color=INK)
ax.set_title("절제 실험: 기준 대비 인간 정규화 IQM 차이", fontsize=11, color=INK, loc="left")
ax.grid(axis="x", color=GRID, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.99, -0.13, "파랑: CI가 0보다 낮음 · 회색: CI가 0을 포함 · 빨강: 0보다 높음",
        transform=ax.transAxes, ha="right", fontsize=7.8, color="#55544f")
fig.tight_layout()
fig.savefig(FIG / "own_bbf_ablation.png")
plt.close(fig)

# 2. RR=2 스티키 학습 곡선
steps = ["20k", "40k", "60k", "80k", "100k", "200k", "500k", "1M"]
x = [20, 40, 60, 80, 100, 200, 500, 1000]
c55 = [RES["curve_RR2_sticky_55games"][s]["IQM_55games"] for s in steps]
c26 = [RES["curve_RR2_sticky_26games"][s]["IQM_26"] for s in steps[:5]]

fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
ax.plot(x, c55, "-o", color=ACC, ms=5, lw=2, label="55게임 (스티키 액션)")
ax.plot(x[:5], c26, "--s", color="#55544f", ms=4.5, lw=1.6, label="26게임 (스티키 액션)")
ax.axhline(1.0, color="#b5432f", lw=0.9, ls=":")
ax.text(0.02, 1.0, "인간 수준 1.0", transform=ax.get_yaxis_transform(), fontsize=8, color="#b5432f", va="bottom")
ax.set_xscale("log")
ax.set_xticks([20, 40, 100, 200, 500, 1000])
ax.set_xticklabels(["2만", "4만", "10만", "20만", "50만", "100만"], fontsize=8.5)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("환경 스텝 (로그 축)", fontsize=9.5, color=INK)
ax.set_ylabel("인간 정규화 IQM", fontsize=9.5, color=INK)
ax.set_title("BBF(RR=2) 학습 곡선: 공개 실행별 점수로 재계산", fontsize=11, color=INK, loc="left")
ax.grid(color=GRID, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "own_bbf_learning_curve.png")
plt.close(fig)
print("saved", [p.name for p in FIG.glob("own_bbf_*.png")])
