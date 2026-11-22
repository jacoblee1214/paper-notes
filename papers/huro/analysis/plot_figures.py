"""재계산 결과를 그래프 세 장으로 그린다 (우리가 그린 그래프, 논문 그림 아님).

출력: report/figures/own_scaling_ci.png, own_task_cells.png, own_dataset_epochs.png
실행: python3 analysis/plot_figures.py   (matplotlib, numpy)  — 먼저 python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서
for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
C_ID, C_OOD, C_REF = "#1f5fae", "#c2571a", "#7d7b73"

T2 = {  # 보고 값 (표 2): 앞 7개 셀 + ID 평균, OOD 평균
    "0% PT": [83.3, 25.0, 45.8, 58.3, 75.0, 36.1, 20.0, 68.1, 34.9],
    "10% PT": [75.0, 41.7, 75.0, 60.4, 80.6, 69.4, 66.7, 76.9, 59.5],
    "50% PT": [83.3, 50.0, 79.2, 64.6, 72.2, 94.4, 70.0, 78.2, 69.8],
    "100% PT": [83.3, 50.0, 87.5, 79.2, 94.4, 86.1, 73.3, 88.4, 72.2],
}
ivals = RES["model_average_intervals_rollout_level"]

# 1. 사전학습 비율별 ID/OOD 평균 + 보수적 95% 구간
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
xs = [0, 1, 2, 3]
labels = ["0%", "10%", "50%", "100%"]
names = ["0% PT", "10% PT", "50% PT", "100% PT"]
for which, col, lab, off in (("id", C_ID, "ID 평균 (3과제)", -0.06), ("ood", C_OOD, "OOD 평균 (4과제)", 0.06)):
    ys = [T2[n][7 if which == "id" else 8] for n in names]
    lo = [y - ivals[n][which]["ci95"][0] for y, n in zip(ys, names)]
    hi = [ivals[n][which]["ci95"][1] - y for y, n in zip(ys, names)]
    ax.errorbar([x + off for x in xs], ys, yerr=[lo, hi], fmt="o-", color=col, capsize=3, lw=1.4, ms=5, label=lab)
ax.scatter([3.35], [89.4], marker="s", color=C_ID, facecolors="none", s=40, label="no-overlay ID")
ax.scatter([3.35], [55.7], marker="s", color=C_OOD, facecolors="none", s=40, label="no-overlay OOD")
ax.set_xticks(xs, labels)
ax.set_xlabel("HuRo 사전학습 데이터 비율 (사전학습 스텝 80k, 배치 2048 고정)")
ax.set_ylabel("완료 점수 (%)")
ax.set_ylim(0, 100)
ax.set_title("표 2 보고 값과 보수적 95% 구간 (과제당 롤아웃 수를 표본 수로 둠)", fontsize=10)
ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_scaling_ci.png")
plt.close(fig)

# 2. 과제 셀별 곡선 (0→10→50→100%), Wilson 95% 구간 (롤아웃 수 기준, 부분 점수는 구간이 넓게 나옴)
cells = [("Apple ID", 0, 12), ("Apple OOD", 1, 12), ("Cup stacking ID", 2, 12), ("Cup stacking OOD", 3, 24),
         ("Noodle ID", 4, 12), ("Noodle OOD", 5, 12), ("Microwave OOD", 6, 10)]


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.6), dpi=200, sharey=True)
for ax, (name, j, n) in zip(axes.flat, cells):
    ys = [T2[m][j] for m in names]
    lo = [y - 100 * wilson(y / 100, n)[0] for y in ys]
    hi = [100 * wilson(y / 100, n)[1] - y for y in ys]
    mono = all(b >= a for a, b in zip(ys, ys[1:]))
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=C_ID if "ID" in name else C_OOD, capsize=2, lw=1.2, ms=4)
    ax.set_title(name + ("" if mono else " (단조 아님)"), fontsize=8.5)
    ax.set_xticks(xs, ["0", "10", "50", "100"], fontsize=7)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
axes.flat[-1].axis("off")
axes.flat[-1].text(0.02, 0.6, "구간: Wilson 95%, 표본 수 = 롤아웃 수\n(부분 점수 과제는 보수적)\n세로축: 완료 점수 (%)\n가로축: 사전학습 비율 (%)", fontsize=7.5, va="center")
fig.suptitle("과제 셀별 스케일링 곡선 (표 2)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "own_task_cells.png")
plt.close(fig)

# 3. 데이터 구성과 사전학습 에폭
fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=200, gridspec_kw={"width_ratios": [1, 1.15]})
src = [("EgoDex", 78.9), ("EgoVerse", 37.7), ("Ego4D", 14.8), ("Ego10K", 8.5), ("EPIC-Kitchens", 2.4)]
a.barh([s for s, _ in src][::-1], [v for _, v in src][::-1], color="#1f5fae")
for i, (s, v) in enumerate(src[::-1]):
    a.text(v + 1, i, f"{v}M ({100 * v / 142.2:.1f}%)", va="center", fontsize=8)
a.set_xlim(0, 108)
a.set_xlabel("프레임 수 (백만, 표 5)")
a.set_title("출처별 프레임", fontsize=10)
subs = [("0.7M\n(그림 6)", 0.7), ("3.5M", 3.5), ("7M", 7.0), ("14.2M\n(10%)", 14.22), ("71.1M\n(50%)", 71.1), ("142.2M\n(100%)", 142.2)]
samples = 80_000 * 2048 / 1e6  # 백만 샘플
ep = [samples / v for _, v in subs]
b.bar(range(len(subs)), ep, color=["#7d7b73"] * 3 + ["#c2571a"] * 3)
b.set_yscale("log")
b.set_ylim(0.8, 500)
for i, e in enumerate(ep):
    b.text(i, e * 1.12, f"{e:.1f}", ha="center", fontsize=8)
b.set_xticks(range(len(subs)), [s for s, _ in subs], fontsize=7.5)
b.set_ylabel("사전학습 중 데이터를 도는 횟수 (에폭)")
b.set_title("80k×2048=163.8M 샘플 고정 가정 시 에폭 (회색은 가정)", fontsize=9.5)
b.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_dataset_epochs.png")
plt.close(fig)
print("wrote 3 figures")
