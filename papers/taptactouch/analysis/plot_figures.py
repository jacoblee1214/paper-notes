"""TapTacTouch 분석 그래프 세 장 (우리가 그린 그래프, matplotlib). 실행: python3 analysis/plot_figures.py
결과: report/figures/own_*.png. 입력은 recompute.py 값과 논문 표 II–IV, 식 (3)·(5)의 상수뿐이다.
"""
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    cands = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]
    for c in cands:
        if Path(c).exists():
            return font_manager.FontProperties(fname=c).get_name()
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, GREY, ACC, SOFT = "#1b1b1a", "#8a8880", "#1f5fae", "#9bbbe0"

# ---------------------------------------------------------------- 그래프 1: 표 IV 대상별 CD 차이 (짝비교)
objs = np.arange(1, 7)
ours = np.array([1.590, 1.552, 1.680, 1.166, 1.204, 2.602])
multi = np.array([1.673, 1.828, 1.791, 1.193, 1.229, 2.521])
csp = np.array([1.569, 1.597, 1.684, 1.150, 1.214, 2.594])

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=170)
ax = axes[0]
for name, d, col in [("Ours - Multi-scale", ours - multi, ACC), ("Ours - CSP만", ours - csp, GREY)]:
    ax.plot(objs + (0.08 if col == GREY else -0.08), d, "o", color=col, label=name, ms=5)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(objs)
ax.set_xticklabels([f"객체 {i}" for i in objs], fontsize=7.5)
ax.set_ylabel("CD 차이 (mm, 음수가 좋음)", fontsize=8)
ax.set_title("대상별 차이 (표 IV, fill 후)", fontsize=9, color=INK)
ax.legend(fontsize=7, frameon=False)
ax.tick_params(labelsize=7.5)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)

ax = axes[1]
labels = ["단일 척도", "다중 척도", "다중+PCG", "다중+CSP", "Ours"]
means = [3.766, 1.706, 1.691, 1.635, 1.632]
sds = [np.std([4.456, 3.277, 4.324, 3.362, 2.915, 4.261], ddof=1),
       np.std(multi, ddof=1), np.std([1.674, 1.766, 1.758, 1.205, 1.213, 2.533], ddof=1),
       np.std(csp, ddof=1), np.std(ours, ddof=1)]
ses = np.array(sds) / np.sqrt(6)
ax.bar(range(5), means, yerr=ses, color=[SOFT, SOFT, SOFT, SOFT, ACC], capsize=3, edgecolor=INK, lw=0.4)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=7.2, rotation=0)
ax.set_ylabel("평균 CD (mm)", fontsize=8)
ax.set_title("평균 ± 표준오차 (n = 6 대상)", fontsize=9, color=INK)
ax.tick_params(labelsize=7.5)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "own_paired_cd.png")
plt.close(fig)

# ---------------------------------------------------------------- 그래프 2: 미관측 영역 CD와 F (표 III), 관측 영역 (표 II)
cfg = ["단일 척도", "다중 척도", "다중+PCG", "다중+CSP", "Ours"]
unobs_cd = [4.076, 2.133, 2.102, 2.003, 1.992]
unobs_f = [71.253, 89.705, 90.124, 92.148, 92.605]
obs_cd = [3.448, 1.244, 1.248, 1.229, 1.236]
obs_f = [75.580, 97.504, 97.504, 97.497, 97.509]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=170)
x = np.arange(5)
w = 0.38
ax = axes[0]
ax.bar(x - w / 2, obs_cd, w, color=SOFT, edgecolor=INK, lw=0.4, label="관측 영역")
ax.bar(x + w / 2, unobs_cd, w, color=ACC, edgecolor=INK, lw=0.4, label="미관측 영역")
ax.set_xticks(x)
ax.set_xticklabels(cfg, fontsize=7)
ax.set_ylabel("CD (mm)", fontsize=8)
ax.set_title("영역별 CD (표 II, III 평균)", fontsize=9, color=INK)
ax.legend(fontsize=7, frameon=False)
ax.tick_params(labelsize=7.5)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)

ax = axes[1]
ax.bar(x - w / 2, obs_f, w, color=SOFT, edgecolor=INK, lw=0.4, label="관측 영역")
ax.bar(x + w / 2, unobs_f, w, color=ACC, edgecolor=INK, lw=0.4, label="미관측 영역")
ax.set_ylim(60, 100)
ax.set_xticks(x)
ax.set_xticklabels(cfg, fontsize=7)
ax.set_ylabel("F-score (%)", fontsize=8)
ax.set_title("영역별 F-score (표 II, III 평균)", fontsize=9, color=INK)
ax.tick_params(labelsize=7.5)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "own_region_metrics.png")
plt.close(fig)

# ---------------------------------------------------------------- 그래프 3: 식 (3)·(5)·(4)의 정규화 상수가 물리 길이로 얼마인지
# 정규화: s = L/1.6, L = 대상의 최대 치수. 물리 길이 = 정규화 값 × s.
L_mm = np.linspace(60, 120, 100)           # 최대 치수 (60 mm 높이 ~ 100–120 mm 폭을 포함해 넓게)
s_m = (L_mm / 1000) / 1.6                  # 미터 per 정규화 단위
quant = {
    "부호 오프셋 ε = 0.012": 0.012,
    "미세 길이 ℓf = 0.08": 0.08,
    "거친 길이 ℓc = 0.32": 0.32,
    "허용 하한 0.006/s (정의상 6 mm)": None,
}
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=170)
for name, val in quant.items():
    if val is None:
        ax.plot(L_mm, 6.0 * np.ones_like(L_mm), "--", color=GREY, lw=1.1, label=name)
    else:
        ax.plot(L_mm, val * s_m * 1000, lw=1.5, label=name)
ax.set_xlabel("대상 최대 치수 L (mm)", fontsize=8)
ax.set_ylabel("물리 길이 (mm)", fontsize=8)
ax.set_title("정규화 상수의 물리 크기 (식 3, 4, 5, 6)", fontsize=9, color=INK)
ax.legend(fontsize=6.6, frameon=False, loc="upper left")
ax.tick_params(labelsize=7.5)
ax.grid(color="#e5e3dc", lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "own_scale_constants.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
