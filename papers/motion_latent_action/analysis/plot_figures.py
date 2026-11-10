"""리포트·게시본용 그래프 네 장을 report/figures/에 만든다(직접 만든 그래프; 논문 그림을 복제하지 않음).

실행: python3 analysis/plot_figures.py  (table_check_results.json, motion_toy_results.json을 읽음)
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

INK, MUTED, ACCENT, SOFT = "#1b1b1a", "#8a8880", "#1f5fae", "#c9d6ea"

tc = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
toy = json.loads((ROOT / "analysis" / "motion_toy_results.json").read_text())


def wilson_halfwidth(p, n=200, z=1.96):
    import math
    d = 1 + z * z / n
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * h


# 1. LIBERO 과제군별 성공률과 Wilson 95% 구간 (과제군당 200회 가정)
suites = ["Spatial", "Object", "Goal", "Long"]
rows = {
    "제안": [95.5, 94.0, 93.5, 84.0],
    "villa-x (보고치)": [97.5, 97.0, 91.5, 74.5],
    "UniVLA-Bridge (저자 재현)": [90.0, 86.0, 90.5, 86.0],
    "제안 w/o DINO": [91.0, 81.0, 88.5, 81.0],
}
colors = [ACCENT, "#6b6b66", "#b0874a", "#9ab59a"]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
width = 0.2
for i, (name, vals) in enumerate(rows.items()):
    xs = [j + (i - 1.5) * width for j in range(4)]
    hw = [wilson_halfwidth(v / 100) for v in vals]
    ax.errorbar(xs, vals, yerr=hw, fmt="o", color=colors[i], ms=4, capsize=2, lw=1, label=name)
ax.set_xticks(range(4), [f"LIBERO-{s}" for s in suites])
ax.set_ylabel("성공률 (%)")
ax.set_ylim(65, 102)
ax.grid(axis="y", color="#e6e4dd", lw=0.6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
ax.set_title("LIBERO 과제군별 성공률과 Wilson 95% 구간 (과제군당 200회 가정)", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(FIG / "libero_wilson.png")
plt.close(fig)

# 2. RoboTwin 과제별 성공률 (과제당 100회 가정)
tasks = ["Adjust\nbottle", "Grab\nroller", "Place\nphone stand", "Pick dual\nbottles", "Place\nempty cup",
         "Move\ncan pot", "Handover\nmic", "Open\nlaptop", "Place object\nbasket", "Place\nburger fries"]
robo = {
    "제안": [97, 90, 28, 52, 63, 65, 92, 87, 25, 78],
    "π0": [90, 96, 35, 57, 37, 58, 98, 85, 16, 80],
    "UniVLA": [87, 80, 40, 58, 46, 60, 82, 81, 18, 84],
}
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
w = 0.27
for i, (name, vals) in enumerate(robo.items()):
    ax.bar([j + (i - 1) * w for j in range(10)], vals, width=w, color=[ACCENT, "#6b6b66", "#b0874a"][i], label=name)
ax.set_xticks(range(10), tasks, fontsize=6.5)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 110)
ax.grid(axis="y", color="#e6e4dd", lw=0.6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper center")
ax.set_title("RoboTwin 2.0 과제별 성공률 (과제당 100회 가정; 평균 67.7 / 65.2 / 63.6)", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(FIG / "robotwin_tasks.png")
plt.close(fig)

# 3. 토이: 입력별 판독 (시작 위치 고정, 배경 0.05, K = 16)
fx = toy["fixed_start_pos_tex0.05"]["mechanism_K16"]
inputs = ["full", "fg", "bg", "fg_norm"]
labels = ["전체 차이", "전경 마스크", "배경 마스크", "전경 평균 제거\n(가설, 논문에 없음)"]
metrics = [("action", "행동", ACCENT), ("cam", "카메라 이동", "#b0874a"), ("bright", "밝기", "#6b6b66")]
fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=200)
w = 0.25
for i, (key, name, col) in enumerate(metrics):
    vals = [fx[f][key]["purity"] for f in inputs]
    ax.bar([j + (i - 1) * w for j in range(4)], vals, width=w, color=col, label=name)
    ax.axhline(fx[inputs[0]][key]["chance"], xmin=0.0, xmax=1.0, color=col, lw=0.6, ls="--", alpha=0.6)
ax.set_xticks(range(4), labels, fontsize=7.5)
ax.set_ylabel("판독 정확도(코드 → 라벨)")
ax.set_ylim(0, 1.12)
ax.grid(axis="y", color="#e6e4dd", lw=0.6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper center")
ax.set_title("토이: 프레임 차이 입력별 코드 판독 (K = 16, 점선 = 기준선)", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(FIG / "toy_leakage.png")
plt.close(fig)

# 4. 토이: K에 따른 혼잡도와 행동 판독 (시작 위치 고정)
sw = toy["fixed_start_pos_tex0.05"]["K_sweep"]
Ks = sorted(int(k) for k in sw)
fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=200)
for f, name, col in [("full", "전체 차이", "#6b6b66"), ("fg", "전경 마스크", ACCENT)]:
    ppl = [sw[str(K)][f]["action"]["perplexity"] for K in Ks]
    acc = [sw[str(K)][f]["action"]["purity"] for K in Ks]
    ax.plot(ppl, acc, "o-", color=col, ms=4, lw=1.2, label=name)
    for K, x, y in zip(Ks, ppl, acc):
        ax.annotate(f"K={K}", (x, y), textcoords="offset points", xytext=(4, -9), fontsize=6.5, color=col)
ax.set_xlabel("코드 사용 혼잡도 (perplexity)")
ax.set_ylabel("행동 판독 정확도")
ax.set_ylim(0.3, 1.05)
ax.grid(color="#e6e4dd", lw=0.6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title("토이: 혼잡도 대 행동 판독 (K = 4–64, 시작 위치 고정)", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(FIG / "toy_k_sweep.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("*.png")))
