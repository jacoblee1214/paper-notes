"""리포트 그래프 (CPU, matplotlib). 실행: python3 analysis/plot_figures.py -> report/figures/own_*.png
우리가 계산한 값만 그린다(논문 그림 아님). 팔레트는 색각 이상에 안전한 파랑·주황·보라.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TC = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
LN = json.loads((ROOT / "analysis" / "label_noise_toy_results.json").read_text())
CB = json.loads((ROOT / "analysis" / "collection_toy_results.json").read_text())

for f in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, PURPLE = "#0072B2", "#D55E00", "#7B3FA0"
INK, MUTED, GRID = "#1b1b1a", "#55544f", "#dcdad3"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


# ---- 그림 1: 시간 예산 ----
tb = TC["time_budget"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
parts = [("실행", 24.8, BLUE), ("후퇴", 11.9, "#56B4E9"), ("인지", 7.8, PURPLE), ("계획", 3.8, "#CC79A7")]
hours = [v * 500 / 3600 for _, v, _ in parts]
left = 0
for (lab, _, c), h in zip(parts, hours):
    ax.barh([0], [h], left=left, color=c, edgecolor="white", label=f"{lab} {h:.2f} h")
    left += h
ax.barh([0], [tb["non_loop_h"]], left=left, color="#BBBBBB", edgecolor="white",
        label=f"루프 밖(리셋·개입 등, 역산) {tb['non_loop_h']:.2f} h")
ax.set_yticks([])
ax.set_xlabel("벽시계 시간 (h, 500회)", color=MUTED, fontsize=9)
ax.set_title("AutoDex 500회 시간 구성 (논문 4.1절 값에서 계산)", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=7.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
ax.set_xlim(0, 11)
style(ax)
ax.grid(False)
ax.grid(axis="x", color=GRID, lw=0.6)

ax = axes[1]
labels = ["벽시계 (h)\n8 h/일 가정", "사람 시간 (h)"]
tele = [28.4, 9.3]
auto = [10.3, 0.2]
x = [0, 1]
ax.bar([i - 0.19 for i in x], tele, 0.36, color="#999999", label="teleop")
ax.bar([i + 0.19 for i in x], auto, 0.36, color=BLUE, label="AutoDex")
for i, (t, a) in enumerate(zip(tele, auto)):
    ax.text(i - 0.19, t + 0.4, f"{t}", ha="center", fontsize=8.5, color=INK)
    ax.text(i + 0.19, a + 0.4, f"{a}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title("논문 Fig. 5 좌측 수치", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=8, frameon=False)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_time_budget.png", dpi=200)
plt.close(fig)

# ---- 그림 2: 라벨 오차 토이 ----
settings = [k.split(" | ", 1)[1] for k in LN["settings"] if k.startswith("N=3593")]
short = ["없음", "균일 2%", "균일 5%", "균일 10%", "거짓 성공 10%", "거짓 실패 10%", "x0 의존 거짓 실패"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=False)
for n_tr, c, mk in ((3593, BLUE, "o"), (200, ORANGE, "s")):
    keys = [k for k in LN["settings"] if k.startswith(f"N={n_tr} |")]
    bias = [100 * LN["settings"][k]["calib_bias_pred_minus_true"]["mean"] for k in keys]
    auc = [LN["settings"][k]["auc_clean_test"]["mean"] for k in keys]
    auc0 = LN["settings"][keys[0]]["auc_clean_test"]["mean"]
    axes[0].plot(range(len(keys)), bias, marker=mk, color=c, lw=1.4, label=f"학습 N={n_tr}")
    axes[1].plot(range(len(keys)), [a - auc0 for a in auc], marker=mk, color=c, lw=1.4, label=f"학습 N={n_tr}")
for ax, t, yl in ((axes[0], "예측 평균 - 참 평균 (pp)", "pp"), (axes[1], "시험 AUC - 오라클 AUC", "AUC 차이")):
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=7.6)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_title(t, fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
fig.suptitle("토이 A: 자동 라벨 오차 (설계값, 30 반복)", fontsize=10.5, color=INK, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(FIG / "own_label_noise.png", dpi=200)
plt.close(fig)

# ---- 그림 3: 처리량 대 리셋 실패, 선택 편향 ----
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
ax = axes[0]
qs = [0.0, 0.02, 0.05, 0.10, 0.20]
for every, c in ((1, BLUE), (3, ORANGE), (10, PURPLE)):
    y = [CB["throughput"][f"reset_every={every}, q={q}"]["trials_per_wall_h"] for q in qs]
    ax.plot([100 * q for q in qs], y, marker="o", color=c, lw=1.4, label=f"리셋 {every}회 시도마다")
imp = CB["implied_reset_fail_rate_from_31_interventions"]
ax.set_xlabel("리셋 실패율 q (%)", color=MUTED, fontsize=9)
ax.set_ylabel("시도 / 벽시계 h", color=MUTED, fontsize=9)
ax.set_title("처리량 (개입 600 s 가정)", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=8, frameon=False)
style(ax)
ax = axes[1]
gs = [0.0, 0.5, 1.0, 2.0]
bias = [CB["selection_bias"][f"gamma={g}"]["bias_pp_db_minus_population"] for g in gs]
sd = [CB["selection_bias"][f"gamma={g}"]["bias_pp_sd"] for g in gs]
ax.bar([str(g) for g in gs], bias, yerr=sd, color=ORANGE, capsize=3, width=0.55)
ax.set_xlabel("라벨 가능성과 난이도의 결합 세기 (γ)", color=MUTED, fontsize=9)
ax.set_ylabel("라벨 DB 성공률 - 전체 후보 (pp)", color=MUTED, fontsize=9)
ax.set_title("선택 편향 (라벨 없는 시도 제외)", fontsize=10, color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_throughput_bias.png", dpi=200)
plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("own_*.png")), "implied q:", imp)
