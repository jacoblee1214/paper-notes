"""재계산 결과(recompute_results.json)로 그래프 두 장을 만든다.

실행: python3 analysis/plot_figures.py   → report/figures/own_overall_ci.png, own_contrast_ci.png
한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, ACCENT = "#1b1b1a", "#8a877f", "#1f5fae"

# 그림 1: 종합 점수 재계산과 부트스트랩 95% 구간
systems = ["ACT", "OpenVLA", "Octo", "RDT_scratch", "RDT"]
labels = {"ACT": "ACT", "OpenVLA": "OpenVLA", "Octo": "Octo", "RDT_scratch": "RDT (scratch)", "RDT": "RDT (ours)"}
vals = [R["table3_overall_recompute"][s]["overall"] for s in systems]
lo = [R["overall_bootstrap_95ci"][s][0] for s in systems]
hi = [R["overall_bootstrap_95ci"][s][1] for s in systems]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
colors = [MUTED, MUTED, MUTED, "#9db9d9", ACCENT]
bars = ax.bar([labels[s] for s in systems], vals, color=colors, width=0.6)
ax.errorbar(range(len(systems)), vals, yerr=[[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]],
            fmt="none", ecolor=INK, capsize=4, lw=1)
for i, v in enumerate(vals):
    ax.text(i, hi[i] + 2, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
ax.set_ylabel("종합 성공률 (%)", fontsize=9)
ax.set_ylim(0, 90)
ax.tick_params(labelsize=8)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.set_title("표 3 재계산: 부록 H 규칙의 종합 점수 (셀 부트스트랩 95% 구간)", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_overall_ci.png")
plt.close(fig)

# 그림 2: 주요 대비의 Wilson 95% 구간 (RDT 대 비교군)
rows = [
    ("Handover 전체", "RDT", 10, 25, "RDT scratch", 4, 25, 0.114),
    ("Fold Shorts", "RDT", 17, 25, "RDT scratch", 10, 25, 0.088),
    ("Robot Dog 밀기", "RDT", 19, 25, "RDT scratch", 16, 25, 0.538),
    ("Robot Dog 직진", "RDT", 12, 25, "RDT scratch", 8, 25, 0.387),
    ("Pour Water 세 방", "RDT", 18, 24, "RDT scratch", 10, 24, 0.039),
    ("지시 R 정확한 양", "RDT", 6, 8, "RDT scratch", 1, 8, 0.041),
]


def wilson(k, n, z=1.96):
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
ylabels = []
for i, (name, a, ka, na, b, kb, nb, p) in enumerate(rows):
    y_a, y_b = 2 * (len(rows) - i) + 0.35, 2 * (len(rows) - i) - 0.35
    for y, k, n, col, lab in [(y_a, ka, na, ACCENT, a), (y_b, kb, nb, MUTED, b)]:
        pc = 100 * k / n
        l, h = wilson(k, n)
        ax.plot([l, h], [y, y], color=col, lw=2)
        ax.plot(pc, y, "o", color=col, ms=5)
        ax.text(101, y, f"{k}/{n}", va="center", fontsize=7.2, color=INK)
    ylabels.append((2 * (len(rows) - i), f"{name}  (p = {p:.3f})"))
ax.set_yticks([y for y, _ in ylabels])
ax.set_yticklabels([t for _, t in ylabels], fontsize=7.8)
ax.set_xlim(0, 100)
ax.set_xlabel("성공률 (%), Wilson 95% 구간", fontsize=8.5)
ax.tick_params(axis="x", labelsize=8)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.plot([], [], color=ACCENT, lw=2, label="RDT (ours)")
ax.plot([], [], color=MUTED, lw=2, label="RDT (scratch)")
ax.legend(fontsize=7.5, loc="upper right", frameon=False)
ax.set_title("표 3 주요 대비: 시도 수가 적어 구간이 넓습니다", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_contrast_ci.png")
plt.close(fig)
print("wrote", FIG / "own_overall_ci.png", FIG / "own_contrast_ci.png")
