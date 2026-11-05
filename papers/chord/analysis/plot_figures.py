"""리포트용 그래프 세 장. 실행: python3 analysis/plot_figures.py (table_check.py, wrench_toy.py 결과 JSON을 읽음)

1) reward_curves.png: 같은 위치에서 법선만 틀릴 때, 법선은 맞고 위치만 밀릴 때 보상 세 가지
2) budget.png: PPO 공개 설정 vs FlashCHORD 설정의 전이 수·VOC 일정, 모션당 GPU-h
3) table1_bars.png: Table 1·2의 SP-SR, MP-SR, AUC (시드 3개 평균 ± 표준편차)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, YELLOW, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TC = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
WT = json.loads((ROOT / "analysis" / "wrench_toy_results.json").read_text())


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) 보상 곡선
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200)
w1 = WT["W1_normal_rotation_same_location"]
x = [r["normal_err_deg"] for r in w1]
for key, lab, col, mk in [("location", "위치 기반 (σ = 2 cm, 가정)", MUTED, "s"),
                          ("code", "CHORD 코드 (방향별 exp 평균)", BLUE, "o"),
                          ("eq3", "논문 Eq. 3 (합의 exp)", ORANGE, "^")]:
    axes[0].plot(x, [r[key] for r in w1], marker=mk, ms=4, lw=1.8, color=col, label=lab)
axes[0].set_xlabel("같은 접촉점에서 법선 오차 (도)")
axes[0].set_ylabel("보상")
axes[0].set_title("상자 뚜껑 예: 위치는 같고 미는 방향만 다름", color=INK, fontsize=10)
axes[0].set_ylim(-0.03, 1.05)
style(axes[0])
w2 = WT["W2_location_offset_same_normal"]
x2 = [r["offset_cm"] for r in w2]
for key, col, mk in [("location", MUTED, "s"), ("code", BLUE, "o"), ("eq3", ORANGE, "^")]:
    axes[1].plot(x2, [r[key] for r in w2], marker=mk, ms=4, lw=1.8, color=col)
axes[1].set_xlabel("법선은 같고 접촉점만 이동 (cm)")
axes[1].set_title("받침 접촉: 미는 방향은 같고 위치만 밀림", color=INK, fontsize=10)
axes[1].set_ylim(-0.03, 1.05)
style(axes[1])
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.06))
fig.text(0.5, -0.11, "μ = 0.1, 기저 512개, tolerance 0.1, var 0.1 (공식 설정). analysis/wrench_toy.py W1·W2",
         ha="center", color=INK2, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "reward_curves.png", bbox_inches="tight")
plt.close(fig)

# 2) 예산
c7 = TC["C7_budget"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
ppo_it = c7["ppo_transitions_per_iter"]
sched = [0, 2000, 3500, 5000, 6500, 8000, 9500, 11000, 12500, 14000, 15500, 20000]
voc = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.0, 0.0, 0.0]
ppo_x = np.array(sched) * ppo_it / 1e9
ax.step(ppo_x[:-1].tolist() + [ppo_x[-1]], voc + [0.0], where="post", color=MUTED, lw=2,
        label="PPO 공개 설정 (Isaac Lab, 4096 env × 24 × 20000 iter)")
fl = [0, 19660800, 34406400, 49152000, 63897600, 78643200, 93388800, 108134400, 122880000, 137625600,
      200007680, 250003456]
fvoc = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.0, 0.0, 0.0]
ax.step((np.array(fl) / 1e9).tolist(), fvoc + [0.0], where="post", color=BLUE, lw=2,
        label="FlashCHORD 설정 (FlashSAC, 250M)")
ax.axvline(c7["ppo_voc_zero_transitions"] / 1e9, color=MUTED, ls=":", lw=1)
ax.axvline(c7["flash_voc_zero_transitions"] / 1e9, color=BLUE, ls=":", lw=1)
ax.text(1.23, 0.62, "VOC = 0\n1.23B", color=INK2, fontsize=8)
ax.text(0.14, 0.62, "VOC = 0\n0.123B", color=BLUE, fontsize=8)
ax.set_xlabel("전이 수 (십억)")
ax.set_ylabel("VOC 배율")
ax.set_title("물체 보조(VOC) 감쇠 일정: 같은 모양, 10배 빠름", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
style(ax)
ax = axes[1]
labels = ["CHORD PPO\n(FDR 측정, XHand)", "CHORD PPO\n(FDR 측정, Sharpa)", "CHORD FlashSAC\n(논문 v2 주장)",
          "FlashDexRetarget\n(XHand)", "FlashDexRetarget\n(Sharpa)"]
vals = [c7["fdr_gpu_h_per_motion"]["XHand"], c7["fdr_gpu_h_per_motion"]["Sharpa"], 2.0, 29 / 50, 44 / 50]
cols = [MUTED, MUTED, BLUE, AQUA, AQUA]
y = np.arange(len(labels))[::-1]
ax.barh(y, vals, color=cols, height=0.6)
for yi, v in zip(y, vals):
    ax.text(v * 1.08, yi, f"{v:.2g} h" if v < 10 else f"{v:.1f} h", va="center", fontsize=8, color=INK)
ax.set_xscale("log")
ax.set_xlim(0.3, 300)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("모션당 GPU-h (로그)")
ax.set_title("모션당 비용 (하드웨어 다름: 3090 vs L40S)", color=INK, fontsize=10)
style(ax, "x")
fig.tight_layout()
fig.savefig(OUT / "budget.png", bbox_inches="tight")
plt.close(fig)

# 3) Table 1·2
T = {"DexMachina": [(0.737, 0.053), (0.690, 0.068), (0.569, 0.097)],
     "ManipTrans": [(0.506, 0.006), (0.423, 0.011), (0.323, 0.003)],
     "SPIDER": [(0.804, 0.044), (0.911, 0.039), (0.789, 0.047)],
     "H2S2R": [(0.730, 0.056), (0.686, 0.068), (0.569, 0.085)],
     "No Cont.": [(0.774, 0.036), (0.756, 0.059), (0.612, 0.055)],
     "Pos. Only": [(0.862, 0.019), (0.856, 0.028), (0.780, 0.042)],
     "CHORD": [(0.918, 0.018), (0.926, 0.022), (0.894, 0.032)]}
col = {"DexMachina": MUTED, "ManipTrans": "#c9c9cd", "SPIDER": ORANGE, "H2S2R": "#6b6b70",
       "No Cont.": YELLOW, "Pos. Only": AQUA, "CHORD": BLUE}
fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200)
mets = ["AUC (DexMachina)", "SP-SR (SPIDER)", "MP-SR (ManipTrans)"]
w = 0.11
for j, (name, v) in enumerate(T.items()):
    xs = np.arange(3) + (j - 3) * w
    ax.bar(xs, [a for a, _ in v], w * 0.9, yerr=[b for _, b in v], color=col[name], label=name,
           error_kw={"lw": 0.8, "capsize": 2, "ecolor": INK2})
ax.set_xticks(np.arange(3))
ax.set_xticklabels(mets)
ax.set_ylim(0, 1.05)
ax.set_ylabel("값 (높을수록 좋음)")
ax.set_title("Table 1·2 (60개 과제, 시드 3개). SP-SR에서 CHORD와 SPIDER 차이는 t = 0.58", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=7.5, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.1))
style(ax)
fig.tight_layout()
fig.savefig(OUT / "table1_bars.png", bbox_inches="tight")
plt.close(fig)
print("saved", [p.name for p in sorted(OUT.glob("*.png")) if not p.name.startswith("paper_")])
