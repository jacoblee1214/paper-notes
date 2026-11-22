"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 입력: analysis/recompute_results.json.
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FONT_CANDIDATES = ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "NanumGothic", "DejaVu Sans"]
_names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in FONT_CANDIDATES if f in _names), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
ACC, GREY, RED, GREEN, ORANGE = "#1f5fae", "#8a8a85", "#b03a2e", "#2e7d5b", "#c77d1a"

# 1. Table 3 평균 곡선 (네 군집 평균, ∆ = 1..9)
d = list(range(1, 10))
M = R["T3_mean_by_delta"]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
ax.plot(d, M["VLASH"], marker="o", color=GREY, label="VLASH")
ax.plot(d, M["RTC"], marker="s", color=ORANGE, label="RTC")
ax.plot(d, M["Ours"], marker="^", color=ACC, label="Ours")
ax.plot(d, M["Sched"], marker="D", color=GREEN, label="+Sched")
ax.axhline(R["T3_sync_mean4"], color=RED, ls="--", lw=1, label=f"Sync 평균 {R['T3_sync_mean4']:.1f}")
ax.set_xlabel("∆ (VLA 지연, 행동 단계)")
ax.set_ylabel("성공률 평균 (4개 LIBERO 군집, %)")
ax.set_title("Table 3 재배열: ∆별 평균 (논문 값 그대로)", fontsize=10)
ax.set_xticks(d)
ax.legend(fontsize=7, ncol=3, loc="lower left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_t3_curves.png")
plt.close(fig)

# 2. 최종 행 주파수의 버전별 변화 (Orin, Thor). 기준(Naive)은 버전 사이에 변하지 않음
V = R["H1_version_headlines"]
labels = ["v1–v5", "v6", "v7"]
orin = [V["v1-v5"]["orin_final_Hz"], V["v6"]["orin_final_Hz"], V["v7"]["orin_final_Hz"]]
thor = [V["v1-v5"]["thor_final_Hz"], V["v6"]["thor_final_Hz"], V["v7"]["thor_final_Hz"]]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
xs = range(len(labels))
w = 0.36
b1 = ax.bar([x - w / 2 for x in xs], orin, w, color=ACC, label="Orin 최종 행 (Hz)")
b2 = ax.bar([x + w / 2 for x in xs], thor, w, color=GREEN, label="Thor 최종 행 (Hz)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{b.get_height():.2f}", ha="center", fontsize=7)
ax.axhline(0.70, color=ACC, ls=":", lw=1)
ax.text(-0.45, 1.1, "Orin Naive 0.70 Hz (점선)", color=ACC, fontsize=6.5, ha="left")
ax.set_xticks(list(xs))
ax.set_xticklabels([f"arXiv {l}" for l in labels])
ax.set_ylabel("제어 주기 (Hz)")
ax.set_title("같은 방법, 버전별 최종 행 값 (논문 Table 4)", fontsize=10)
ax.set_ylim(0, 19)
ax.legend(fontsize=7, loc="upper left")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_version_drift.png")
plt.close(fig)

# 3. Orin 구성 분해: Naive(목록 합 대 보고 총합의 차이 표시), v6 최종, v7 최종
cases = [
    ("Naive\n(v1–v7 동일)", [152.3, 631.0, 536.8], 1420.8),
    ("최종 행 v6", [71.9, 194.8, 107.8], 374.5),
    ("최종 행 v7", [66.6, 137.0, 67.6], 271.1),
]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
cols = [ACC, "#6fa3d8", GREY]
for i, (name, (v, l, a), tot) in enumerate(cases):
    bottom = 0
    for val, c, lab in zip([v, l, a], cols, ["ViT", "LLM", "행동 전문가"]):
        ax.bar(i, val, 0.55, bottom=bottom, color=c, label=lab if i == 0 else None)
        bottom += val
    gap = tot - (v + l + a)
    if gap > 1:
        ax.bar(i, gap, 0.55, bottom=bottom, color="white", edgecolor=RED, hatch="//", label="목록에 없는 차이" if i == 0 else None)
        bottom += gap
    ax.text(i, bottom + 25, f"총 {tot:.1f} ms\n(목록 합 {v + l + a:.1f})", ha="center", fontsize=7)
ax.set_xticks(range(len(cases)))
ax.set_xticklabels([c[0] for c in cases], fontsize=8)
ax.set_ylabel("지연 (ms, Orin)")
ax.set_ylim(0, 1700)
ax.set_title("Orin 지연 구성: 최종 행 두 번의 재측정", fontsize=10)
ax.legend(fontsize=7, loc="upper right")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_orin_breakdown.png")
plt.close(fig)
print("wrote figures to", FIG)
