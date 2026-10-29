"""리포트 그래프 네 장. 실행: (cd analysis && python3 plot_figures.py)

1) noise_schedules.png: 논문·코드의 σ 스케줄과, 학습되는 σ가 5M step 안에 갈 수 있는 범위(Adam 상한)
2) sim_seed_spread.png: Table I 물체별 평균 ± SD와 본문에 공개된 seed별 값
3) hardware_ci.png: Table III Target 성공(조건당 모델 1개, 20회)과 합산 Wilson 95% 구간
4) exploration_toy.png: exploration_toy.py 결과(near/far regime의 결정론적 성공)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import table_check as tc
from table_check import wilson

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e4e7", "#9a9a9f"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) σ 스케줄
s = np.linspace(0, 5e6, 1001)
eq4 = np.where(s < 2e6, 0.20 - 0.10 * s / 2e6, np.where(s < 4e6, 0.10 - 0.05 * (s - 2e6) / 2e6, 0.04))
screen = np.where(s < 2e6, 0.20 - 0.10 * s / 2e6, np.where(s < 4e6, 0.08 - 0.03 * (s - 2e6) / 2e6, 0.04))
fpo = 0.20 - 0.15 * s / 5e6
init = math.exp(-1.6)
steps = s / 256 * 5  # Adam step 수
lo = init * np.exp(-1e-5 * steps)
fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
ax.fill_between(s / 1e6, lo, init, color=MUTED, alpha=0.18, lw=0)
ax.plot(s / 1e6, eq4, color=BLUE, lw=2, label="PPO + DexPolicy (Eq. 4, 코드 no_mid_jump_5m)")
ax.plot(s / 1e6, screen, color=BLUE, lw=1.2, ls="--", label="초기 screen 곡선 (2M에서 0.10→0.08)")
ax.plot(s / 1e6, fpo, color=ORANGE, lw=2, label="FPO + DexPolicy (선형 0.20→0.05)")
ax.hlines(0.10, 0, 5, color=ORANGE, lw=1.2, ls=":", label="FPO baseline (고정 0.10)")
ax.plot([0, 5], [init, 0.1855], color=INK2, lw=1.6, label="PPO baseline 학습 σ (0.202→0.185, 본문)")
ax.text(0.08, 0.012, "회색: 학습 σ가 5M step에 갈 수 있는 범위\n(Adam lr 1e-5 × 97,656 step ≈ log σ -0.98)",
        ha="left", va="bottom", fontsize=7, color=INK2)
ax.set_xlabel("학습 환경 step (백만)", color=INK2)
ax.set_ylabel("행동 잡음 σ", color=INK2)
ax.set_ylim(0, 0.23)
ax.set_xlim(0, 5)
ax.legend(frameon=False, fontsize=7, loc="upper right")
style(ax)
fig.text(0.01, 0.01, "출처: Eq. 4, Sec. IV-C, V-E, 공개 코드 phase_schedule_core.py · fpo_trainer.py. 상한은 Adam 한 step ≲ lr인 근사.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "noise_schedules.png")
print("wrote", OUT / "noise_schedules.png")

# 2) 시뮬레이션 seed 산포
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), dpi=200, sharey=True)
known = {"FPO": {"mustard": tc.SEEDS["FPO mustard"], "mug": tc.SEEDS["FPO mug"], "tomato": tc.SEEDS["FPO tomato"]},
         "GRPO": {"mustard": tc.SEEDS["GRPO mustard"]}, "PPO": {}}
names = ["머스터드", "머그", "바나나", "설탕", "토마토"]
for ax, m in zip(axes, ("PPO", "GRPO", "FPO")):
    x = np.arange(5)
    for off, cond, col, lab in ((-0.17, "base", MUTED, "baseline"), (0.17, "dex", BLUE, "+ DexPolicy")):
        mean, sd, _ = tc.T1[m][cond]
        ax.errorbar(x + off, mean, yerr=sd, fmt="o", color=col, ms=5, capsize=2, lw=1.2, label=lab)
        for j, obj in enumerate(tc.OBJ5):
            if obj in known[m]:
                vals = known[m][obj][0 if cond == "base" else 1]
                ax.scatter([j + off + 0.07] * 3, vals, s=10, facecolor="white", edgecolor=col, lw=1, zorder=3)
    ov_b, ov_d = tc.T1[m]["base"][2][0], tc.T1[m]["dex"][2][0]
    ax.set_title(f"{m}: Overall {ov_b} → {ov_d}%", fontsize=9.5, color=INK)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(-8, 112)
    style(ax)
axes[0].set_ylabel("결정론적 Target 성공 (%)", color=INK2)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.text(0.01, 0.01, "점과 막대: Table I 평균 ± 표본 SD (seed 3개). 빈 원: 본문에 공개된 seed별 값(FPO 머스터드·머그·토마토, GRPO 머스터드).",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "sim_seed_spread.png")
print("wrote", OUT / "sim_seed_spread.png")

# 3) 하드웨어
fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=200)
objs = ["토마토", "설탕", "머스터드"]
for i, m in enumerate(("PPO", "GRPO", "FPO")):
    for j, (cond, col) in enumerate((("base", MUTED), ("dex", BLUE))):
        ks = [round(v[0] * 20 / 100) for v in tc.T3[m][cond]]
        K = sum(ks)
        lo_, hi_ = wilson(K, 60)
        xpos = i * 1.0 + (j - 0.5) * 0.36
        ax.bar(xpos, 100 * K / 60, width=0.32, color=col, alpha=0.9)
        ax.vlines(xpos, lo_, hi_, color=INK2, lw=1)
        for k_, mk in zip(ks, ("o", "s", "^")):
            ax.scatter(xpos, 100 * k_ / 20, marker=mk, s=14, color="white", edgecolor=INK, lw=0.8, zorder=3)
        ax.text(xpos, hi_ + 2, f"{100 * K / 60:.0f}", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks([0, 1, 2], ["PPO", "GRPO", "FPO"])
ax.set_ylabel("실제 로봇 Target 성공 (%)", color=INK2)
ax.set_ylim(0, 108)
style(ax)
h = [plt.Rectangle((0, 0), 1, 1, color=MUTED), plt.Rectangle((0, 0), 1, 1, color=BLUE),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK), plt.Line2D([], [], marker="s", ls="", mfc="white", mec=INK),
     plt.Line2D([], [], marker="^", ls="", mfc="white", mec=INK)]
ax.legend(h, ["baseline", "+ DexPolicy", "토마토", "설탕", "머스터드"], frameon=False, fontsize=7, loc="upper left", ncol=2)
fig.text(0.01, 0.01, "Table III. 막대: 물체 3개 합산(60회), 선: Wilson 95% 구간(시행 잡음만). 조건마다 학습된 모델은 1개라 seed 변동은 포함되지 않음.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "hardware_ci.png")
print("wrote", OUT / "hardware_ci.png")

# 4) toy
toy = json.loads((ROOT / "analysis" / "exploration_toy_results.json").read_text())
conds = ["learned", "scheduled", "linear", "fixed_0.20", "fixed_0.10", "fixed_0.04"]
labels = ["학습 σ", "Eq. 4 모양", "선형 0.20→0.04", "고정 0.20", "고정 0.10", "고정 0.04"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=200, sharey=True)
for ax, reg, title in zip(axes, ("near", "far"), ("near: 초기 위치에서도 보상 신호가 있음", "far: 넓게 탐색해야 신호를 찾음")):
    per = toy[reg]["per_seed_det_success"]
    for i, c in enumerate(conds):
        v = np.array(per[c])
        col = BLUE if c in ("scheduled", "linear") else (ORANGE if c == "learned" else MUTED)
        ax.bar(i, v.mean(), color=col, width=0.6)
        ax.scatter(i + np.linspace(-0.18, 0.18, len(v)), v, s=8, color="white", edgecolor=INK, lw=0.6, zorder=3)
    ax.set_xticks(range(len(conds)), labels, fontsize=7.5, rotation=20)
    ax.set_title(title, fontsize=9, color=INK)
    style(ax)
axes[0].set_ylabel("결정론적 성공 (평균, 점은 seed)", color=INK2)
fig.text(0.01, 0.01, "exploration_toy.py: 4차원 행동 문맥 밴딧 + PPO-clip, seed 10개, lr × Adam step ≈ 0.9(논문 설정과 같은 log σ 이동 폭). 경향 확인용.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "exploration_toy.png")
print("wrote", OUT / "exploration_toy.png")
