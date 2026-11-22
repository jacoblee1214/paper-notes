"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_real_ci.png       표 1(실로봇 30회 평가)의 성공률과 Wilson 95% 구간
  own_recovery.png      표 2(LIBERO-10 회복)의 절대 성공률과 20 cm 유지율
  own_gate_toy.png      토이 2: 게이트 Δ에 따른 실패/성공 이벤트 비율
  own_reward_scale.png  토이 1: 진행 이벤트 밀도에 따른 보상 합 (공식 코드의 보상 배치 논리 재구현)

입력: analysis/recompute_results.json, toy_gate_results.json, toy_reward_scale_results.json
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
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TG = json.loads((ROOT / "analysis" / "toy_gate_results.json").read_text(encoding="utf-8"))
TR = json.loads((ROOT / "analysis" / "toy_reward_scale_results.json").read_text(encoding="utf-8"))

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
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

# ------------------------------------------------ 그래프 1: 실로봇 성공률과 Wilson 구간
cells = RC["V1_table1_realworld"]["cells"]
tasks = ["Drawer open", "Pick eraser", "Hand over", "Clothes folding"]
fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.0), dpi=200, sharey=True)
off = {"BinR": -0.9, "RoboM": 0.0, "Ours": 0.9}
col = {"BinR": MUTED, "RoboM": ACC2, "Ours": ACC}
for ax, task in zip(axes, tasks):
    for who, key in (("BinR", "binr"), ("RoboM", "robom"), ("Ours", "ours")):
        xs, ys, lo, hi = [], [], [], []
        for c in cells:
            if c["task"] != task:
                continue
            k = c[who]
            ci = c[f"{key}_ci"]
            xs.append(c["budget"] + off[who]); ys.append(k / 30); lo.append(k / 30 - ci[0]); hi.append(ci[1] - k / 30)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", ms=3.5, lw=1.2, capsize=2, color=col[who], label=who)
    ax.set_title(task, fontsize=9)
    ax.set_xticks([30, 60, 90])
    ax.set_xlabel("DSRL 학습 롤아웃 수")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
axes[0].set_ylabel("성공률 (30회 평가, Wilson 95%)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png")
plt.close(fig)

# ------------------------------------------------ 그래프 2: 표 2 절대값과 유지율
T2 = {"GVL": [45.1, 38.7, 26.3, 17.5], "RoboDopamine": [36.7, 31.9, 23.2, 15.1], "RoboMeter": [48.8, 40.5, 34.7, 23.7],
      "TemporalOT": [22.6, 18.6, 15.8, 14.1], "RARM": [95.6, 90.9, 78.9, 48.2]}
names = list(T2)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), dpi=200)
ax = axes[0]
for n in names:
    ax.plot([0, 5, 10, 20], T2[n], "o-", ms=3.5, lw=1.3, color=ACC if n == "RARM" else MUTED, alpha=1 if n == "RARM" else 0.75)
    ax.text(20.6, T2[n][3], n, fontsize=7.5, va="center", color=INK)
ax.set_xlim(-1, 30)
ax.set_xlabel("중간 개입: 팔 변위 (cm)")
ax.set_ylabel("성공률 (%, 4 시드 평균)")
ax.set_title("절대 성공률 (논문 표 2)", fontsize=9)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
ax = axes[1]
ret = [100 * T2[n][3] / T2[n][0] for n in names]
bars = ax.bar(names, ret, color=[ACC if n == "RARM" else "#b7c5d8" for n in names])
for b, v in zip(bars, ret):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_ylim(0, 80)
ax.set_ylabel("20 cm 성공률 / 개입 없음 성공률 (%)")
ax.set_title("유지율 (우리가 계산)", fontsize=9)
ax.tick_params(axis="x", labelrotation=25, labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "own_recovery.png")
plt.close(fig)

# ------------------------------------------------ 그래프 3: 게이트 토이
fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
deltas = [0.25, 0.5, 1.0, 2.0, "inf"]
labels = ["0.25", "0.5", "1.0", "2.0", "게이트 없음"]
pairs = [(0.1, 0.3), (0.1, 0.6), (0.1, 1.0)]
cols = [MUTED, ACC2, ACC]
for (qs, qf), c in zip(pairs, cols):
    ys = [next(g["ratio"] for g in TG["grid"] if g["q_success"] == qs and g["q_fail"] == qf and g["delta"] == d) for d in deltas]
    ax.plot(range(len(deltas)), ys, "o-", color=c, lw=1.4, ms=4, label=f"실패 외형 격차 q_f = {qf}")
ax.axvline(1, color="#c8c6bd", lw=0.8, ls="--")
ax.text(1.05, 0.55, "논문 Δ = 0.5", fontsize=8, color=MUTED)
ax.set_xticks(range(len(deltas)))
ax.set_xticklabels(labels)
ax.set_xlabel("게이트 여유 Δ (점수 척도 단위)")
ax.set_ylabel("실패 이벤트 / 성공 이벤트")
ax.set_ylim(0, 0.65)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "own_gate_toy.png")
plt.close(fig)

# ------------------------------------------------ 그래프 4: 보상 합
fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
lam = sorted(TR["density"], key=float)
ev = [TR["density"][k]["events_mean"] for k in lam]
rm = [TR["density"][k]["rm_sum_mean"] for k in lam]
ax.plot(ev, rm, "o-", color=ACC, lw=1.4, ms=4, label="진행 보상 합 (T = 300 스텝)")
ax.plot(ev, [12 * e for e in ev], ":", color=MUTED, lw=1.0, label="이벤트마다 12 (격리 이벤트의 값)")
ax.axhline(5, color=WARN, lw=1.0, ls="--")
ax.text(ev[0], 14, "성공 보상 5 (한 스텝)", color=WARN, fontsize=8)
ax.axhline(150, color=MUTED, lw=0.8, ls="--")
ax.text(ev[0], 158, "스텝 비용 합의 크기 150 (부호는 음수)", color=MUTED, fontsize=8)
ax.set_xlabel("에피소드당 진행 이벤트 수 (평균)")
ax.set_ylabel("에피소드 합")
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "own_reward_scale.png")
plt.close(fig)
print("저장:", sorted(p.name for p in FIG.glob("own_*.png")))
