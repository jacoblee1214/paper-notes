"""우리가 그린 그래프 세 장을 report/figures/own_*.png로 만든다. 논문 그림은 쓰지 않는다.

입력: analysis/table_check_results.json, toy_routing_results.json, toy_power_results.json
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

font_path = next((p for p in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
                              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
                  if Path(p).exists()), None)
if font_path:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACC = "#1b1b1a", "#55544f", "#1f5fae"
TASKS = ["Key", "2-pin", "3-pin", "USB", "Ethernet"]
tc = json.loads((A / "table_check_results.json").read_text(encoding="utf-8"))
wil = tc["table1_wilson"]


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


# 1. 과제별 성공률과 Wilson 95% 구간
fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
methods = [("PACE", ACC), ("ACT + Tactile Concat", "#8c8a82"), ("Vision-only ACT", "#c9c6bd")]
w = 0.26
for i, (m, col) in enumerate(methods):
    ks = [wil[m][t]["k"] for t in TASKS]
    lo = [wil[m][t]["wilson95"][0] * 100 for t in TASKS]
    hi = [wil[m][t]["wilson95"][1] * 100 for t in TASKS]
    xs = [j + (i - 1) * w for j in range(len(TASKS))]
    rate = [100 * k / 30 for k in ks]
    err = [[r - l for r, l in zip(rate, lo)], [h - r for h, r in zip(hi, rate)]]
    ax.bar(xs, rate, width=w, color=col, label=m)
    ax.errorbar(xs, rate, yerr=err, fmt="none", ecolor=INK, capsize=2, lw=0.8)
ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels(TASKS)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
ax.set_title("과제별 성공률과 Wilson 95% 구간 (과제당 30회, 논문 표 1 값)", fontsize=10, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_success_ci.png")
plt.close(fig)

# 2. 토이 1: 상태/사건 분리 대 단일 특징
tr = json.loads((A / "toy_routing_results.json").read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.4), dpi=200, sharey=True)
phases = ["approach", "contact", "insert"]
labels = ["접근", "첫 접촉", "안정 삽입"]
for ax, sig in zip(axes, ["sigma=0.1", "sigma=0.3", "sigma=0.6"]):
    for i, (name, col) in enumerate([("single", "#8c8a82"), ("split_state_scaled", ACC)]):
        vals = [tr[sig][name]["by_phase"][p] for p in phases]
        xs = [j + (i - 0.5) * 0.36 for j in range(3)]
        ax.bar(xs, vals, width=0.34, color=col, label="단일 특징" if i == 0 else "상태 분리")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"잡음 σ={sig.split('=')[1]}", fontsize=9, color=INK)
    style(ax)
axes[0].set_ylabel("사건 후 10스텝 내 검출률")
fig.suptitle("토이 1 (가정 의존, 논문 실험 아님): 구간별 검출률", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(0.98, 0.97))
fig.savefig(FIG / "own_routing.png")
plt.close(fig)

# 3. 토이 2: 30회 검정력과 부풀림
pw = json.loads((A / "toy_power_results.json").read_text(encoding="utf-8"))
ds = [0.10, 0.20, 0.30, 0.45]
power = [pw["power"][f"delta={d}"] for d in ds]
infl = [pw["inflation"][f"delta={d}"] for d in ds]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
ax.plot(ds, power, marker="o", color=ACC, label="검출력 (단측 Fisher, α=0.05)")
ax.axhline(0.8, color=MUTED, lw=0.8, ls="--")
ax.text(0.455, 0.82, "0.8", color=MUTED, fontsize=8)
ax.set_xlabel("참 차이 Δ (기준 46%)")
ax.set_ylabel("검출력")
ax.set_ylim(0, 1.05)
ax2 = ax.twinx()
ax2.plot(ds, infl, marker="s", color="#8c8a82", label="유의한 결과의 평균 관측 차이")
ax2.set_ylabel("관측 차이 (유의한 경우 평균)")
ax2.set_ylim(0, 0.6)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("토이 2: 과제당 30회 비교의 검출력과 부풀림", fontsize=10, color=INK)
style(ax)
ax2.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_power.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
