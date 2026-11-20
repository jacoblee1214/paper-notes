"""그래프 세 장을 report/figures/에 만든다 (matplotlib). 모두 우리가 표 수치로 다시 그린 것이며 논문 그림을 쓰지 않는다.

  own_robotwin_horizon.png : RoboTwin horizon 묶음별 평균 (LingBot-VA, pi0.5, Motus), Easy와 Hard
  own_realworld_sr.png     : 실세계 6과제 성공률, 표기 값과 부록 규칙으로 다시 센 값
  own_toy_noisy.png        : 토이: 관측 수준별 행동 오차 (깨끗한 학습 대 증강 학습)

실행: python3 lingbot_va/analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
RW = json.loads((ROOT / "analysis" / "realworld_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_noisy_history_results.json").read_text(encoding="utf-8"))

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"ours": "#1f5fae", "pi05": "#e0741b", "motus": "#8a8a85"}

# 1. RoboTwin horizon
hm = RC["horizon_means_all"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, key, title in [(axes[0], "easy", "Easy"), (axes[1], "hard", "Hard")]:
    x = np.arange(3)
    w = 0.26
    for j, (m, c, lab) in enumerate([("Ours", C["ours"], "LingBot-VA"), ("pi0.5", C["pi05"], "pi0.5"), ("Motus", C["motus"], "Motus")]):
        vals = [hm[str(h)][m][key] for h in (1, 2, 3)]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=c, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, ["horizon 1 (30과제)", "horizon 2 (15)", "horizon 3 (5)"], fontsize=8)
    ax.set_title(f"RoboTwin 2.0 {title}, 과제 묶음별 평균 (%)", fontsize=9)
    ax.set_ylim(50, 100)
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "own_robotwin_horizon.png", dpi=160)
plt.close(fig)

# 2. Real-world success: stated vs rule
tasks = ["Make Breakfast", "Pick Screws", "Fold Clothes", "Unpack Delivery", "Insert Tubes", "Fold Pants"]
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(tasks))
w = 0.2
series = [
    ("Ours, 표기", [RW[f"{t} / Ours"]["stated_success"] / 20 * 100 for t in tasks], C["ours"], 0.0),
    ("Ours, 규칙", [RW[f"{t} / Ours"]["rule_success(progress==max)"] / 20 * 100 for t in tasks], C["ours"], 0.5),
    ("pi0.5, 표기", [RW[f"{t} / π0.5"]["stated_success"] / 20 * 100 for t in tasks], C["pi05"], 0.0),
    ("pi0.5, 규칙", [RW[f"{t} / π0.5"]["rule_success(progress==max)"] / 20 * 100 for t in tasks], C["pi05"], 0.5),
]
for j, (lab, vals, c, hatch_flag) in enumerate(series):
    bars = ax.bar(x + (j - 1.5) * w, vals, w, color=c, alpha=0.45 if hatch_flag == 0 else 1.0,
                  edgecolor="black", linewidth=0.4, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=6.5)
ax.set_xticks(x, tasks, fontsize=8)
ax.set_ylabel("성공률 (%, 시도 20회)", fontsize=8)
ax.set_title("실세계 과제 성공률: 표기 값(흐린 막대)과 부록 규칙으로 다시 센 값(진한 막대)", fontsize=9)
ax.legend(fontsize=7, ncol=4, loc="upper right")
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(OUT / "own_realworld_sr.png", dpi=160)
plt.close(fig)

# 3. Toy
lv = sorted(TOY["summary_by_level"].keys(), key=float)
xs = [float(v) for v in lv]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
c_m = [TOY["summary_by_level"][k]["clean_trained_mse_mean"] for k in lv]
c_s = [TOY["summary_by_level"][k]["clean_trained_mse_sd"] for k in lv]
a_m = [TOY["summary_by_level"][k]["aug_trained_mse_mean"] for k in lv]
a_s = [TOY["summary_by_level"][k]["aug_trained_mse_sd"] for k in lv]
ax.errorbar(xs, c_m, yerr=c_s, marker="o", color="#8a8a85", capsize=3, label="깨끗한 관측으로만 학습")
ax.errorbar(xs, a_m, yerr=a_s, marker="s", color=C["ours"], capsize=3, label="논문식 증강으로 학습")
ax.set_yscale("log")
ax.set_xlabel("관측 수준 (흐름 시간 s, 0 = 노이즈만, 1 = 깨끗함)", fontsize=8)
ax.set_ylabel("행동 MSE (로그, 5 시드 평균 ± sd)", fontsize=8)
ax.set_title("토이: 노이즈 섞인 미래 잠재에서 행동 복원 (합성, 선형)", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "own_toy_noisy.png", dpi=160)
plt.close(fig)
print("ok")
