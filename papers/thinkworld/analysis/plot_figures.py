"""리포트용 그래프 세 장. 실행: python3 analysis/plot_figures.py (table_check.py, reward_toy.py 결과 JSON을 읽음)

1) real_robot_ci.png: Table 6 설정별 성공률과 Wilson 95% 구간
2) component_gain.png: Table 3 구성 요소별 LIBERO-PRO 축 이득
3) reward_weights.png: Eq. 6 motion 가중치와 보상 몫, CoD 보상 차이 대 형식 오류 벌칙
"""
import json
import math
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
RT = json.loads((ROOT / "analysis" / "reward_toy_results.json").read_text())


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


# 1) 실물 로봇
T6 = {
    "Diffusion Policy": [(18, 30), (11, 30), (9, 15), (1, 15), (4, 15), (1, 15)],
    "MolmoAct2": [(24, 30), (8, 30), (11, 15), (14, 15), (12, 15), (3, 15)],
    "LaWAM": [(16, 30), (8, 30), (9, 15), (12, 15), (4, 15), (4, 15)],
    "ThinkWorld": [(30, 30), (24, 30), (15, 15), (15, 15), (15, 15), (13, 15)],
}
COL = {"Diffusion Policy": MUTED, "MolmoAct2": ORANGE, "LaWAM": AQUA, "ThinkWorld": BLUE}
labels = ["색 맞추기\nID (n=30)", "색 맞추기\nOOD (n=30)", "컵 걸기\nID (n=15)", "컵 걸기\nOOD (n=15)",
          "분류\nID (n=15)", "분류\nOOD (n=15)"]
fig, ax = plt.subplots(figsize=(10, 3.8), dpi=200)
off = {"Diffusion Policy": -0.27, "MolmoAct2": -0.09, "LaWAM": 0.09, "ThinkWorld": 0.27}
for m, cells in T6.items():
    xs = np.arange(6) + off[m]
    ps = [100 * k / n for k, n in cells]
    lo = [p - wilson(k, n)[0] for p, (k, n) in zip(ps, cells)]
    hi = [wilson(k, n)[1] - p for p, (k, n) in zip(ps, cells)]
    ax.errorbar(xs, ps, yerr=[lo, hi], fmt="o", ms=6, color=COL[m], ecolor=COL[m], elinewidth=1.6, capsize=0,
                label=m, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
for x in (1.5, 3.5):
    ax.axvline(x, color=GRID, lw=1)
ax.set_xticks(range(6), labels, fontsize=8)
ax.set_ylim(-3, 105)
ax.set_ylabel("성공률 (%) · 막대: Wilson 95%", color=INK2)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
style(ax)
ax.text(3.0, 44, "기준선이 OOD에서\n오히려 오름\n(MolmoAct2 73→93,\nLaWAM 60→80)", fontsize=7.2, color=INK2, ha="center", va="center")
fig.text(0.01, 0.01, "출처: Table 6 (성공/시행). 설정당 5회, 학습 seed 1개(Appendix D). Fisher: 전체 112/120 vs 72/120, p = 6e-10",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "real_robot_ci.png")
print("wrote", OUT / "real_robot_ci.png")

# 2) Table 3 구성 요소
steps = ["+ History", "+ Future prediction", "+ CoD", "+ RL"]
names = {"+ History": "+ 이력 프레임", "+ Future prediction": "+ 미래 예측", "+ CoD": "+ CoD", "+ RL": "+ RL"}
scol = {"+ History": MUTED, "+ Future prediction": BLUE, "+ CoD": ORANGE, "+ RL": AQUA}
axes_l = ["Object", "Position", "Semantic", "Task", "Env", "평균"]
fig, ax = plt.subplots(figsize=(9, 3.4), dpi=200)
w = 0.2
for i, s in enumerate(steps):
    g = TC["C5_table3"]["step_gain"][s]
    vals = g["per_axis"] + [g["avg"]]
    xs = np.arange(6) + (i - 1.5) * w
    ax.bar(xs, vals, width=w - 0.03, color=scol[s], label=names[s])
for x, s in zip(np.arange(6) + 0.5 * w, [None] * 6):
    pass
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(6), axes_l)
ax.set_ylabel("앞 단계 대비 성공률 변화 (%p)", color=INK2)
ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
style(ax)
share = TC["C5_table3"]["share_of_total_gain"]
ax.text(5.45, 23.5, f"평균 이득 15.5%p 중\n미래 예측 {share['+ Future prediction']*100:.0f}% · CoD {share['+ CoD']*100:.0f}%\nRL {share['+ RL']*100:.0f}% · 이력 {share['+ History']*100:.0f}%",
        fontsize=7.5, color=INK2, ha="right", va="top")
fig.text(0.01, 0.01, "출처: Table 3 (LIBERO-PRO, 단일 학습 seed). 누적 추가 순서의 단계별 차이.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "component_gain.png")
print("wrote", OUT / "component_gain.png")

# 3) Eq. 6 가중치와 보상 차이
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200, gridspec_kw={"width_ratios": [1.1, 1]})
q = np.linspace(0, 1, 400)
wq = np.clip(0.2 + 0.8 * (q - 0.5) / 0.4, 0.2, 1.0)
a1.plot(100 * q, wq, color=BLUE, lw=2)
a1.fill_between(100 * q, 0, wq, color=BLUE, alpha=0.08)
a1.set_xlabel("현재→미래 DINO 변화량의 패치 순위 (백분위)", color=INK2)
a1.set_ylabel("motion 가중치 w_p", color=INK2)
a1.set_ylim(0, 1.1)
sh = RT["R1_weight_shares"]["total_reward_share"]
a1.text(3, 0.55, f"보상 전체에서의 몫(196 패치)\n상위 10%: {sh['top10']*100:.1f}% (균등이면 10%)\n상위 25%: {sh['top25']*100:.1f}% (균등이면 25%)\n하위 50%: {sh['bottom50']*100:.1f}% (균등이면 50%)",
        fontsize=7.5, color=INK2)
a1.set_title("Eq. 6 = 0.5·균등 평균 + 0.5·motion 가중 평균", fontsize=9, color=INK)
style(a1)
gaps = RT["R2_reward_gap_from_table2"]
names2 = list(gaps.keys()) + ["형식 오류 벌칙"]
vals2 = [gaps[k]["reward_gap_eq6"] for k in gaps] + [0.05]
cols2 = [BLUE, BLUE, BLUE, ORANGE]
a2.barh(range(4), vals2, color=cols2, height=0.55)
for i, v in enumerate(vals2):
    a2.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
a2.set_yticks(range(4), ["CoD vs Shuffle", "CoD vs Empty", "CoD vs CoD off", "형식 오류 벌칙\n(그룹 최저 - 0.05)"])
a2.invert_yaxis()
a2.set_xlabel("보상 단위 (코사인 유사도)", color=INK2)
a2.set_title("CoD가 바꾸는 보상 차이(Table 2에서 추정)", fontsize=9, color=INK)
a2.set_xlim(0, 0.062)
style(a2, "x")
fig.text(0.01, 0.01, "가정: Table 2의 '상위 25% 움직임 패치' = 보상 가중치와 같은 변화 순위, 그룹 안 차이는 균일. analysis/reward_toy.py",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "reward_weights.png")
print("wrote", OUT / "reward_weights.png")
