"""자체 그래프 두 장 (논문 그림은 쓰지 않음; 라이선스는 arXiv 비배타적 배포).

실행: python3 analysis/plot_figures.py  →  report/figures/own_libero_ci.png, own_ablation.png
입력: recompute_results.json의 값과 같은 상수 (표 1, 표 4, 표 5). 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

for name in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "NanumGothic"]:
    try:
        matplotlib.font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.family"] = name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

N = 500
SUITES = ["Goal", "Object", "Spatial", "Long"]
OURS = [86.4, 86.6, 87.6, 57.8]
OFT = [84.0, 74.2, 84.2, 57.0]
UNIVLA = [82.0, 76.2, 84.4, 56.4]


def wilson_halfwidth(pct):
    p = pct / 100
    z = 1.96
    return 100 * z * np.sqrt(p * (1 - p) / N)


fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=160)
x = np.arange(len(SUITES))
w = 0.26
series = [("World-Env (Ours)", OURS, "#1f5f8b"), ("OpenVLA-OFT (SFT)", OFT, "#9aa5b1"), ("UniVLA", UNIVLA, "#d9a441")]
for i, (lab, vals, col) in enumerate(series):
    hw = [wilson_halfwidth(v) for v in vals]
    ax.bar(x + (i - 1) * w, vals, w, yerr=hw, capsize=3, color=col, label=lab, edgecolor="none")
    for xi, v in zip(x, vals):
        pass
ax.set_xticks(x)
ax.set_xticklabels([f"LIBERO-{s}" for s in SUITES])
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 118)
ax.set_title("표 1 (5 demonstrations): 스위트별 성공률, 95% 구간 (n = 500 가정, 시드 분산 미반영)", fontsize=9)
ax.legend(fontsize=7, loc="upper center", ncol=3, frameon=False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_libero_ci.png")
plt.close(fig)

# 두 번째 그림: 표 5 절제 평균 (2x2) + 표 4 종료 규칙에 따른 OFT와 Ours 평균 이동
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), dpi=160, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
labels = ["없음·없음", "데이터·없음", "없음·헤드", "데이터·헤드\n(Ours)"]
avg = [64.75, 71.15, 65.85, 79.6]
cols = ["#c9cfd6", "#8fa9bf", "#8fa9bf", "#1f5f8b"]
ax.bar(range(4), avg, color=cols)
for i, v in enumerate(avg):
    ax.text(i, v + 1, f"{v:.2f}", ha="center", fontsize=7)
ax.axhline(74.85, color="#b04a3a", lw=1, ls="--")
ax.text(3.45, 75.6, "표 1 OFT 74.85", color="#b04a3a", fontsize=6.5, ha="right")
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylim(40, 90)
ax.set_ylabel("4개 스위트 평균 성공률 (%)")
ax.set_title("표 5 절제 (데이터 · 반영기 헤드)", fontsize=8)
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
xx = np.arange(2)
oft_t1 = [74.85, 63.05]
ours_t1 = [79.6, 74.9]
ax.plot(xx, oft_t1, "o-", color="#9aa5b1", label="OFT")
ax.plot(xx, ours_t1, "o-", color="#1f5f8b", label="Ours")
for i in range(2):
    ax.text(i, oft_t1[i] - 3.2, f"{oft_t1[i]:.2f}", ha="center", fontsize=7)
    ax.text(i, ours_t1[i] + 1.6, f"{ours_t1[i]:.2f}", ha="center", fontsize=7)
ax.set_xticks(xx)
ax.set_xticklabels(["표 1\n(자체 종료 규칙)", "표 4\n(최대 스텝까지)"], fontsize=7)
ax.set_ylim(55, 85)
ax.set_title("종료 규칙에 따른 평균 (논문 기재 값)", fontsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_ablation.png")
plt.close(fig)
print("figures written to", FIG)
print("recompute keys:", list(RES.keys())[:4])
