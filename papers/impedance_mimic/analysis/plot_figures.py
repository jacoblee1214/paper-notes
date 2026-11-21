"""논문 수치로 우리가 그린 그래프 세 장 (report/figures/own_*.png).

입력: analysis/recompute_results.json (표 11, 12의 평균 행), 아래 상수(표 14, A.3.7 본문 수치).
오차막대는 시드 5개 평균의 95% 신뢰구간 (t 분포, df=4, 보고된 표준편차 기준)이다.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))["results"]

KO_FONT = next((f.name for f in font_manager.fontManager.ttflist
                if any(k in f.name for k in ("Apple SD Gothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun"))), "DejaVu Sans")
plt.rcParams["font.family"] = KO_FONT
plt.rcParams["axes.unicode_minus"] = False
INK, GREY = "#1b1b1a", "#8a8880"
COL = {"Fixed-PD": "#8a8880", "Torque": "#c9a227", "Muscle": "#7b5ea7", "ImpedanceMimic": "#1f5fae"}
T = stats.t.ppf(0.975, 4)


def ci(mean, sd):
    h = T * sd / math.sqrt(5)
    return mean - h, mean + h


# 그림 1: 표 11(seen), 12(unseen) 평균 행의 생존율과 수직 힘 상관, 95% 구간
rows = RES["V3_welch_mean_row"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
for ax, (split, title) in zip(axes, [("seen", "보던 피험자 (seen)"), ("unseen", "미경험 피험자 (unseen)")]):
    methods = ["Fixed-PD", "Torque", "Muscle", "ImpedanceMimic"]
    base = {}
    for r in rows[split]:
        if r["metric"] == "Ep":
            base.setdefault(r["baseline"], (r["base_mean"], r["base_sd"]))
            base.setdefault("ImpedanceMimic", (r["im_mean"], r["im_sd"]))
    for i, m in enumerate(methods):
        mu, sd = base[m]
        lo, hi = ci(mu, sd)
        ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt="o", color=COL[m], ms=7, capsize=4, lw=1.6)
        ax.text(i + 0.12, hi + 0.8, f"{mu:.1f}", ha="left", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(["Fixed-PD", "Torque", "Muscle", "ImpedanceMimic"], fontsize=8.5)
    ax.set_ylim(60, 102)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_ylabel("생존율 (%)" if split == "seen" else "")
    ax.grid(axis="y", color="#e5e3dc", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("ImpedanceMimic와 대조 행동 공간의 생존율 (표 11, 12 평균 행, 시드 5개 95% 구간)", fontsize=10.5, color=INK, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "own_survival_ci.png", dpi=170)
plt.close(fig)

# 그림 2: 표 14 (G1) 활동 평균 생존율, 방식별 고정 대 학습 임피던스 (2배 priors 포함)
g1 = {
    "명시적 PD": {"Fixed-PD": (56.9, 3.6), "Fixed-PD 2×": (63.3, 5.5), "ImpedanceMimic": (82.1, 4.3), "ImpedanceMimic 2×": (83.5, 3.6)},
    "안정 PD": {"Fixed-PD": (74.5, 2.6), "Fixed-PD 2×": (71.1, 4.9), "ImpedanceMimic": (75.9, 5.0), "ImpedanceMimic 2×": (79.7, 3.5)},
    "암묵 감쇠": {"Fixed-PD": (60.6, 4.0), "Fixed-PD 2×": (65.8, 2.1), "ImpedanceMimic": (75.5, 4.3), "ImpedanceMimic 2×": (77.4, 1.9)},
}
fig, ax = plt.subplots(figsize=(9.2, 4.0))
keys = ["Fixed-PD", "Fixed-PD 2×", "ImpedanceMimic", "ImpedanceMimic 2×"]
colors = {"Fixed-PD": "#8a8880", "Fixed-PD 2×": "#c8c6bd", "ImpedanceMimic": "#1f5fae", "ImpedanceMimic 2×": "#7fa6d6"}
w = 0.2
for gi, (scheme, d) in enumerate(g1.items()):
    for ki, k in enumerate(keys):
        mu, sd = d[k]
        lo, hi = ci(mu, sd)
        x = gi + (ki - 1.5) * w
        ax.bar(x, mu, width=w * 0.92, color=colors[k], label=k if gi == 0 else None)
        ax.errorbar(x, mu, yerr=[[mu - lo], [hi - mu]], fmt="none", ecolor=INK, capsize=3, lw=1.1)
        ax.text(x, hi + 2.0, f"{mu:.1f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(len(g1)))
ax.set_xticklabels(list(g1.keys()))
ax.set_ylim(0, 100)
ax.set_ylabel("생존율 (%), G1 활동 평균")
ax.set_title("표 14: Unitree G1 zero-shot 생존율 (시드 5개 95% 구간)", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.grid(axis="y", color="#e5e3dc", lw=0.7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_g1_survival.png", dpi=170)
plt.close(fig)

# 그림 3: 부록 A.3.7의 GaitDynamics 대 ImpedanceMimic 물리 위반율 (보고된 평균 ± sd, 시드 5개)
viol = [
    ("CoP가 발 밖\n(lateral, %)", (5.7, 1.0), (0.2, 0.1)),
    ("CoP가 발 밖\n(jump, %)", (17.2, 2.7), (0.0, 0.0)),
    ("비행 중 힘\n(lateral, %)", (30, 4), (11, 1)),
    ("비행 중 힘\n(jump, %)", (39, 3), (17, 2)),
]
fig, ax = plt.subplots(figsize=(9.2, 3.9))
for i, (lab, g, im) in enumerate(viol):
    ax.bar(i - 0.18, g[0], width=0.34, color="#c8c6bd", label="GaitDynamics (회귀기)" if i == 0 else None)
    ax.bar(i + 0.18, im[0], width=0.34, color="#1f5fae", label="ImpedanceMimic" if i == 0 else None)
    ax.errorbar(i - 0.18, g[0], yerr=g[1], fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.errorbar(i + 0.18, im[0], yerr=im[1], fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.text(i - 0.18, g[0] + g[1] + 0.8, f"{g[0]:g}", ha="center", fontsize=8.5)
    ax.text(i + 0.18, im[0] + im[1] + 0.8, f"{im[0]:g}", ha="center", fontsize=8.5)
ax.set_xticks(range(len(viol)))
ax.set_xticklabels([v[0] for v in viol], fontsize=8.5)
ax.set_ylabel("위반 또는 비율 (%)")
ax.set_title("부록 A.3.7: 물리 일관성 위반율, 미경험 피험자 1,365 클립 (시드 5개 sd)", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_ylim(0, 48)
ax.grid(axis="y", color="#e5e3dc", lw=0.7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_gd_violation.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
