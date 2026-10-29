"""리포트 그래프. table_check.py, timing_toy.py를 먼저 실행할 것.

  report/figures/headline_ablation.png  Conveyor·Pivoting 5개 변형 × 6조건 (평균 ± 시드 SD)
  report/figures/standard_tasks.png     표준 6과제 교란 5조건 평균, DP vs Chronos
  report/figures/timing_coverage.png    학습/평가 이미지 간격 분포, time-aware RoPE 시간 커널
실행: python3 analysis/plot_headline.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK2, GRID = "#52514e", "#e6e5e0"
# dataviz 기준 팔레트 슬롯 1–5 (고정 순서). Chronos가 슬롯 1.
COL = {"RoPE (ours)": "#2a78d6", "DP": "#eb6834", "DP + rand.": "#1baf7a",
       "RoPE w/o rand.": "#eda100", "AddTime": "#e87ba4"}
LABEL = {"DP": "Diffusion Policy", "DP + rand.": "DP + 시간 무작위화", "RoPE w/o rand.": "RoPE, 무작위화 없음",
         "AddTime": "AddTime (시간 임베딩 더하기)", "RoPE (ours)": "Chronos (RoPE + 무작위화)"}
CONDS = ["nominal", "small_obs", "wide_sensors", "delay_100", "delay_200", "combined"]
CLAB = ["Nominal", "Small\n(=학습 범위)", "Wide\n(1–50 Hz)", "지연\n0–100 ms", "지연\n0–200 ms", "Combined"]

r = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
sols = r["seeds"]["solutions"]


def seeds(key, n_ep):
    s = sols[key]
    return (np.array(s[0]) / n_ep * 100) if s else None


# ---------------------------------------------------------------- 1. ablation
order = ["DP", "DP + rand.", "RoPE w/o rand.", "AddTime", "RoPE (ours)"]
fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
w = 0.155
for ax, task, title in zip(axes, ["conveyor", "pivoting"], ["Conveyor (동적 과제)", "Pivoting (다중 감각 과제)"]):
    for j, m in enumerate(order):
        xs = np.arange(6) + (j - 2) * w
        means, sds = [], []
        for c in CONDS:
            v = seeds(f"{task}|{m}|{c}", 150)
            means.append(v.mean())
            sds.append(v.std(ddof=1))
        ax.bar(xs, means, w * 0.9, color=COL[m], label=LABEL[m], zorder=2)
        ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#3a3936", elinewidth=0.8, capsize=0, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("성공률 (%)")
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.axvspan(1.5, 2.5, color="#f3f2ee", zorder=0)
    ax.axvspan(4.5, 5.5, color="#f3f2ee", zorder=0)
axes[1].set_xticks(np.arange(6), CLAB)
axes[0].legend(ncol=3, frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 1.32))
axes[1].text(5.0, 92, "회색 배경: 학습 분포 밖 관측 간격", ha="center", fontsize=7.5, color=INK2)
fig.text(0.01, 0.005, "막대 = 3개 학습 시드 평균(시드당 150 에피소드), 선 = 시드 간 표준편차. 출처: 논문 Table 3, 프로젝트 페이지 data.js",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(FIG / "headline_ablation.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. 표준 6과제
tasks = ["Threading", "Assembly", "Transport", "Coffee", "Drawer Cleanup", "Pouring"]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for i, t in enumerate(tasks):
    dp = np.mean([seeds(f"{t}|DP|{c}", 50).mean() for c in CONDS[1:]])
    ch = np.mean([seeds(f"{t}|Chronos Policy|{c}", 50).mean() for c in CONDS[1:]])
    big = ch - dp > 5
    ax.plot([dp, ch], [i, i], color="#b9b8b2" if not big else "#8a8984", lw=2, zorder=1)
    ax.scatter([dp], [i], s=46, color=COL["DP"], zorder=3, edgecolor="white", linewidth=1.5,
               label="Diffusion Policy" if i == 0 else None)
    ax.scatter([ch], [i], s=46, color=COL["RoPE (ours)"], zorder=3, edgecolor="white", linewidth=1.5,
               label="Chronos Policy" if i == 0 else None)
    ax.text(max(dp, ch) + 2, i, f"{ch - dp:+.1f}%p", va="center", fontsize=8,
            color="#0b0b0b" if big else INK2, fontweight="bold" if big else "normal")
ax.set_yticks(range(6), tasks)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("교란 5조건 평균 성공률 (%)")
ax.xaxis.grid(True, color=GRID, zorder=0)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "standard_tasks.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. 시간 분포와 커널
smp = np.load(ROOT / "analysis" / "timing_samples.npz")
kz = np.load(ROOT / "analysis" / "timing_kernel.npz")
tt = json.loads((ROOT / "analysis" / "timing_toy_results.json").read_text())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.1))
for name, lab, col, ls in [("train", "학습 (12.5–33 Hz)", "#0b0b0b", "-"), ("nominal", "Nominal (20 Hz)", COL["DP + rand."], "-"),
                           ("large", "Wide / Combined (1–50 Hz)", COL["DP"], "-")]:
    g = np.sort(smp[f"{name}_gap"]) * 1000
    a1.plot(g, np.linspace(0, 1, len(g)), color=col, lw=2, ls=ls, label=lab)
a1.axvline(tt["train_max"]["image_gap_s"] * 1000, color=INK2, lw=1, ls="--")
a1.text(tt["train_max"]["image_gap_s"] * 1000 * 1.08, 0.08, "학습 최대 간격 80 ms", fontsize=7.5, color=INK2)
a1.set_xscale("log")
a1.set_xlabel("연속 두 이미지의 캡처 간격 (ms, 로그축)")
a1.set_ylabel("누적 비율")
a1.set_title("(a) 이미지 간격 분포: Wide 조건의 23.6%가 학습 범위 밖", loc="left", fontsize=9)
a1.legend(frameon=False, fontsize=7.5, loc="center right")
a1.grid(True, color=GRID)
a2.plot(kz["dt"], kz["kern"], color=COL["RoPE (ours)"], lw=1.6)
tr = tt["coverage"]["train"]["span_image_to_last_action_s"][2]     # 학습 때 토큰 쌍 |Δt| 최대
wd = tt["coverage"]["combined"]["span_image_to_last_action_s"][2]  # Wide/Combined 최대
a2.axvspan(-wd, wd, color="#fbeee8", zorder=0)
a2.axvspan(-tr, tr, color="#e3edfa", zorder=0)
a2.text(-1.0, 0.62, f"학습 범위\n|Δt| ≤ {tr:.2f} s", fontsize=7.5, color=INK2)
a2.text(1.65, 0.62, f"Wide·Combined\n|Δt| ≤ {wd:.1f} s", fontsize=7.5, color=INK2)
a2.axhline(0, color=GRID, lw=1)
a2.set_xlim(-3, 3)
a2.set_ylim(-0.42, 1.05)
a2.set_xlabel("시간 차이 Δt (s)")
a2.set_ylabel("s(Δt) = mean cos(ω_k Δt)")
a2.set_title("(b) time-aware RoPE 커널: 주기 0.1–2 s, 16쌍", loc="left", fontsize=9)
a2.grid(True, color=GRID)
fig.tight_layout()
fig.savefig(FIG / "timing_coverage.png", dpi=200)
plt.close(fig)
print("wrote", *[p.name for p in sorted(FIG.glob("*.png")) if not p.name.startswith("paper_")])
