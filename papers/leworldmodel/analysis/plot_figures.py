"""우리가 만든 그래프 두 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_diff_ci.png    그림 6 판독값의 LeWM 대비 차이와 95% 구간 (k/50 이항 근사, 독립 가정)
  own_overall.png    표 4 Overall 행: 보고값과 여덟 행 평균의 차이

입력: analysis/recompute_results.json
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
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

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

INK, MUTED, ACC, ACC2 = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b"

# ---- own_diff_ci: 그림 6 판독값, LeWM 대비 차이 (n = 50 가정) ----
cells = [
    ("Two-Room", "PLDM", 87, 97), ("Two-Room", "DINO-WM", 87, 100),
    ("Reacher", "PLDM", 86, 78), ("Reacher", "DINO-WM", 86, 79),
    ("Push-T", "PLDM", 96, 78), ("Push-T", "DINO-WM+prop", 96, 92), ("Push-T", "DINO-WM", 96, 74),
    ("OGBench-Cube", "PLDM", 74, 65), ("OGBench-Cube", "DINO-WM", 74, 86),
]
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
for i, (env, base, a, b) in enumerate(cells):
    pa, pb = a / 100, b / 100
    d = (pa - pb) * 100
    se = np.sqrt(pa * (1 - pa) / 50 + pb * (1 - pb) / 50) * 100
    lo, hi = d - 1.96 * se, d + 1.96 * se
    color = ACC if lo > 0 else (ACC2 if hi < 0 else MUTED)
    ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=color, ms=5, capsize=3, lw=1.4)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(len(cells)))
ax.set_yticklabels([f"{e} · LeWM 대 {bname}" for e, bname, _, _ in cells], fontsize=7.6, color=INK)
ax.invert_yaxis()
ax.set_xlabel("성공률 차이 (%p), 양수면 LeWM이 높음", fontsize=8.5, color=INK)
ax.set_title("그림 6 판독값의 대비와 95% 구간 (에피소드 50개 가정)", fontsize=9.5, color=INK, loc="left")
ax.grid(axis="x", color="#e5e4df", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=7.6, colors=INK)
fig.text(0.01, 0.01, "파란 점: 구간이 0보다 큼(LeWM 우위). 주황: 구간이 0보다 작음(기준선 우위). 회색: 0을 포함. 독립 이항 근사.",
         fontsize=6.8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_diff_ci.png")
plt.close(fig)

# ---- own_overall: 표 4 Overall, 보고값과 행 평균 ----
ov = R["V1_table4_overall"]
models = list(ov.keys())
labels = ["선형 MSE", "선형 r", "MLP MSE", "MLP r"]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.8), dpi=200)
for j, ax in enumerate(axes):
    rep = [ov[m]["reported_overall"][j] for m in models]
    rec = [ov[m]["recomputed_mean_of_8_rows"][j] for m in models]
    x = np.arange(len(models))
    ax.bar(x - 0.18, rep, 0.36, color=ACC, label="보고값")
    ax.bar(x + 0.18, rec, 0.36, color="#c9c8c2", label="여덟 행 평균")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7, rotation=20, color=INK)
    ax.set_title(labels[j], fontsize=8.2, color=INK)
    ax.tick_params(labelsize=6.8, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
axes[-1].legend(fontsize=6.8, frameon=False, loc="upper right")
fig.suptitle("표 4 Overall 행: 보고값과 여덟 속성 평균이 다릅니다", fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "own_overall.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
