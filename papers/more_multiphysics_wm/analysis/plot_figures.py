"""논문 표 수치(recompute.py와 같은 값)로 우리가 그린 그래프 세 장을 만든다.

출력: report/figures/own_table4_rel.png, own_ablation.png, own_stat_ci.png
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
한글 폰트: Linux Noto CJK -> macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute import T3, T4, wilson  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

INK = "#1b1b1a"
GREY = "#8a8880"
ACC = "#1f5fae"
LIGHT = "#c9d7ea"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e4dd", lw=0.6)
    ax.set_axisbelow(True)


# 1) 표 4: MoRE 대 PointWorld, 지표별 상대 변화 (%) - 음수가 좋음
regs = {"rigid_np": "강체 비파지", "deform_np": "변형체 비파지", "deform_p": "변형체 파지"}
rmse, cd = [], []
for r in regs:
    pw, ou = T4[r]["PointWorld"], T4[r]["Ours"]
    rmse.append(100 * (ou[0] - pw[0]) / pw[0])
    cd.append(100 * (ou[1] - pw[1]) / pw[1])
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
x = range(len(regs))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], rmse, w, color=ACC, label="RMSE")
b2 = ax.bar([i + w / 2 for i in x], cd, w, color=LIGHT, label="CD")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 2, f"{b.get_height():.1f}%",
                ha="center", va="top", fontsize=8, color=INK)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(list(x), list(regs.values()))
ax.set_ylabel("PointWorld 대비 변화 (%)", color=INK, fontsize=9)
ax.set_ylim(-60, 5)
ax.set_title("표 4 (DROID): MoRE 대 PointWorld, 음수가 개선", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)
fig.text(0.01, 0.01, "논문 표 4 값에서 계산. 구간과 시드 수는 논문에 없음.", fontsize=7, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_table4_rel.png")
plt.close(fig)

# 2) 표 3: 절제 변형별, 전체 모델 대비 상대 차이 (RMSE, CD, 지표 평균이 아닌 지표별 점)
variants = ["Single Expert", "Implicit Router", "Implicit MoE", "w/o Self-Rollout"]
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4), dpi=160, sharey=True)
for ax, (reg, label) in zip(axes, [("rigid_np", "강체 비파지"), ("deform_np", "변형체 비파지"), ("deform_p", "변형체 파지")]):
    full = T3[reg]["Ours (Full)"]
    ys = list(range(len(variants)))
    for j, v in enumerate(variants):
        vals = T3[reg][v]
        for k, (a, f, m) in enumerate(zip(vals, full, ["RMSE", "CD", "Lap/Rig"])):
            if f == 0:
                continue
            pct = 100 * (a - f) / f
            ax.scatter(pct, j + (k - 1) * 0.18, s=26, color=[ACC, "#4a7fc1", GREY][k], zorder=3)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ys)
    ax.set_ylim(len(variants) - 0.4, -0.6)
    if ax is not axes[0]:
        ax.tick_params(labelleft=False)
    ax.set_title(label, fontsize=9, color=INK)
    ax.set_xlabel("전체 모델 대비 변화 (%)", fontsize=8, color=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e4dd", lw=0.6)
fig.text(0.01, 0.075, "점 색: 진한 파랑 RMSE, 중간 파랑 CD, 회색 Lap/Rig (전체 모델 값이 0인 Rig 지표는 생략)", fontsize=6.5, color=GREY)
axes[0].set_yticklabels(variants, fontsize=8)
fig.suptitle("표 3 절제: 변형별 차이 (음수가 좋음, 전체 모델이 기준)", fontsize=10, color=INK, x=0.02, ha="left")
fig.text(0.01, 0.01, "논문 표 3 값에서 계산. 지표별 SD가 없어 차이의 유의성은 판단하지 않음.", fontsize=7, color=GREY)
fig.tight_layout(rect=(0, 0.1, 1, 0.92))
fig.savefig(FIG / "own_ablation.png")
plt.close(fig)

# 3) 불확실성: 시뮬 MPC 10회(5/10, 2/10)와 기체 정확도 전체(145/150), 범주당 50회 최근접 k
items = [
    ("시뮬 MPC 성공률: MoRE (5/10)", 5, 10),
    ("시뮬 MPC 성공률: DINO-WM (2/10)", 2, 10),
    ("기체 정확도 전체 (145/150)", 145, 150),
    ("기체 정확도 Push-X (48/50, 최근접)", 48, 50),
    ("기체 정확도 Manipulate Toy (47/50, 최근접)", 47, 50),
    ("기체 정확도 Soft Grasp (50/50, 최근접)", 50, 50),
]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
for i, (lab, k, n) in enumerate(items):
    p = 100 * k / n
    lo, hi = [100 * v for v in wilson(k, n)]
    ax.plot([lo, hi], [i, i], color=ACC, lw=2.2, solid_capstyle="round")
    ax.scatter(p, i, s=34, color=INK, zorder=3)
    ax.text(101, i, f"{k}/{n}  [{lo:.0f}, {hi:.0f}]", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(items)), [it[0] for it in items], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("성공 또는 정확도 (%), Wilson 95% 구간", fontsize=8.5, color=INK)
ax.set_title("표본이 작은 수치의 불확실성", fontsize=10, color=INK, loc="left")
style(ax)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color="#e6e4dd", lw=0.6)
fig.text(0.01, 0.01, "시뮬 MPC 10회는 본문 4.3, 기체 정확도는 표 5와 부록 A.4. 범주별 값은 k/50과 맞지 않아 최근접 k로 표시.",
         fontsize=6.5, color=GREY)
fig.tight_layout(rect=(0, 0.04, 0.86, 1))
fig.savefig(FIG / "own_stat_ci.png")
plt.close(fig)
print("wrote figures to", FIG)
