"""우리가 만든 그래프 세 장 (논문 그림 미사용). 입력: analysis/recompute_results.json + 본문·표 값.

실행: python3 analysis/plot_figures.py -> report/figures/own_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다(경로는 후보 목록으로만 둔다)
KOREAN_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in KOREAN_FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
C = ["#9bb6d8", "#4f7fbf", "#1f5fae"]

# 1) Table IV PerAct: 평가 100회 가정 Wilson 95% CI
fig, ax = plt.subplots(figsize=(7, 3.4))
labels = ["no aug, no m.v.", "aug, no m.v.", "aug, m.v."]
for gi, task in enumerate(["PushCube", "StackCube"]):
    rows = RES["peract"]["rows"][task]
    for i, key in enumerate(["no_aug_no_mv", "aug_no_mv", "aug_mv"]):
        r = rows[key]
        p = r["k"] / r["n"]
        lo, hi = r["wilson95"]
        x = gi * 4 + i
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", color=C[i], capsize=4)
        ax.text(x, hi + 0.03, f"{p:.2f}", ha="center", fontsize=8)
ax.set_xticks([0, 1, 2, 4, 5, 6])
ax.set_xticklabels(labels * 2, rotation=20, fontsize=8)
ax.set_ylim(0, 1.15)
ax.set_ylabel("best success rate (Table IV)")
ax.set_title("PerAct: PushCube (왼쪽 3) / StackCube (오른쪽 3), Wilson 95% CI (n=100 가정)", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "own_peract_wilson.png", dpi=170)
plt.close(fig)

# 2) Table II vs III: 1000 demos, DP/ACT/BC, state 대 RGB (BC 행 동일 지적)
tasks = ["PickCube", "PushCube", "StackCube", "PegInsertion"]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
data = {"state (Table II)": {"BC": [0.03, 0.81, 0.00, 0.00], "ACT": [1.00, 0.97, 0.97, 0.43], "DP": [1.00, 0.96, 0.99, 0.66]},
        "RGB (Table III)": {"BC": [0.03, 0.81, 0.00, 0.00], "ACT": [0.98, 0.89, 0.80, 0.00], "DP": [1.00, 0.86, 0.81, 0.00]}}
import numpy as np
w = 0.26
for ax, (title, d) in zip(axes, data.items()):
    xs = np.arange(len(tasks))
    for i, (alg, col) in enumerate(zip(["BC", "ACT", "DP"], C)):
        ax.bar(xs + (i - 1) * w, d[alg], w, label=alg, color=col)
    ax.set_xticks(xs)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_title(title + ", 1000 demos", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("success (best during training)")
axes[0].legend(fontsize=8)
fig.text(0.5, 0.01, "BC 행(0.03, 0.81, 0.00, 0.00)이 두 표에서 동일함. 평가 횟수 미보고.", ha="center", fontsize=8)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / "own_lfd_success.png", dpi=170)
plt.close(fig)

# 3) 보고된 메모리 비율과 초록 주장의 범위
fig, ax = plt.subplots(figsize=(7, 3.2))
items = [("본문 III-B: 128 env, 128x128 RGB\n3.5 vs 14.1 GB", 14.1 / 3.5),
         ("부록 XI-C: 최소 카메라 설정\n1.7 vs 4.8 GB", 4.8 / 1.7),
         ("부록 XI-C 본문 서술\n'24x' (값 비교 불가)", 24)]
vals = [v for _, v in items]
bars = ax.barh([t for t, _ in items], vals, color=[C[1], C[1], "#d9d9d9"])
ax.axvspan(2, 3, color="#f2c94c", alpha=0.25, label="초록 '2-3x' 범위")
for b, v in zip(bars, vals):
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}x", va="center", fontsize=9)
ax.set_xlabel("Isaac Lab 메모리 / ManiSkill3 메모리 (보고값에서 계산)")
ax.set_xlim(0, 27)
ax.legend(fontsize=8, loc="lower right")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "own_memory_ratios.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
