"""리포트용 그래프 세 장. 실행: python3 analysis/plot_figures.py (mapping_toy.py를 먼저 실행)

1) success_ci.png: Table II 성공률과 Wilson 95% 구간
2) noise_vs_state_only.png: Fig. 13에서 읽은 잡음 곡선 + Table III의 상태만 모델 기준선
3) mapping_toy.png: 3-DoF 손가락 toy의 Table III식 비교와 잡음 곡선
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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


# 1) Table II + Wilson 구간
labels = {"sim/in_hand_box_flipping": "Box flip\n(sim, n=250)", "sim/fragile_egg_grasping": "Egg grasp\n(sim, n=250)",
          "sim/dish_wiping": "Dish wipe\n(sim, n=250)", "real/jar_opening": "Jar open\n(real, n=15)",
          "real/in_hand_box_flipping": "Box flip\n(real, n=15)"}
colors = (BLUE, ORANGE, AQUA)
fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=200)
for ti, (task, (n, *srs)) in enumerate(tc.TABLE2.items()):
    for mi, sr in enumerate(srs):
        k = round(sr * n)
        lo, hi = tc.wilson(k, n)
        x = ti + (mi - 1) * 0.24
        ax.plot([x, x], [lo, hi], color=colors[mi], lw=2, solid_capstyle="round")
        ax.scatter([x], [sr], s=40, color=colors[mi], edgecolor="white", linewidth=1.2, zorder=3,
                   label=tc.METHODS[mi] if ti == 0 else None)
    p = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())["C2_ci_and_tests"][task]["fisher_p_two_sided"]
    ax.text(ti, 1.02, f"p = {p:.3f}" if p >= 0.001 else "p < 0.001", ha="center", fontsize=7.5, color=INK2)
ax.axvline(2.5, color=GRID, lw=1)
ax.set_xticks(range(5), list(labels.values()), fontsize=8)
ax.set_ylim(0.25, 1.07)
ax.set_ylabel("성공률 (점: 보고값, 선: Wilson 95% 구간)", color=INK2)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
style(ax)
fig.text(0.01, 0.01, "출처: Table II. p는 CGP vs 최강 기준선 Fisher 정확검정. sim 250회는 한 학습 실행의 체크포인트 5개 × 50회라 독립 시행이 아님.",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "success_ci.png")
print("wrote", OUT / "success_ci.png")

# 2) Fig. 13 다시 읽기
fig, ax = plt.subplots(figsize=(6.2, 3.3), dpi=200)
pct = [0, 50]
for name, c in (("S+T residual", BLUE), ("S+T absolute", ORANGE), ("T-only residual", AQUA)):
    a, b = tc.FIG13[name]
    ax.plot(pct, [a, b], color=c, lw=2, marker="o", ms=5, label=f"{name} (Fig. 13에서 읽음)")
s_only = tc.TABLE3["S/-"][2]
ax.axhline(s_only, color=INK2, lw=1.2, ls="--")
ax.text(21, s_only - 1.35, f"State only, residual = {s_only} (Table III, 잡음 없음)", fontsize=7.5, color=INK2)
ax.set_xlabel("촉각 잡음 표준편차 (데이터셋 촉각 표준편차 대비 %)", color=INK2)
ax.set_ylabel("관절각 MAE (×10⁻³ rad)", color=INK2)
ax.set_xlim(-2, 55)
ax.set_ylim(0, 25)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
style(ax)
fig.text(0.01, 0.01, "양 끝점만 표시. 잡음 50%에서 State+Tactile 잔차(10.0)는 상태만 모델(10.64)과 거의 같다.", fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "noise_vs_state_only.png")
print("wrote", OUT / "noise_vs_state_only.png")

# 3) toy 결과
toy = json.loads((ROOT / "analysis" / "mapping_toy_results.json").read_text())
m2 = toy["M2_table3_style"]
rows = [("해석식 (모든 taxel)", toy["M1_analytic_full_sensing_mae"], 0, INK2),
        ("S+T 잔차", *m2["S+T/residual"], BLUE), ("S+T 절대", *m2["S+T/absolute"], BLUE),
        ("S 잔차", *m2["S/residual"], ORANGE), ("S 절대", *m2["S/absolute"], ORANGE),
        ("T 잔차", *m2["T/residual"], AQUA), ("T 절대", *m2["T/absolute"], AQUA),
        ("S+T 잔차, 손끝만 감지", *toy["M3_mlp_tip_only_S+T_residual"], INK2),
        ("S+T 잔차, 속도 항 포함", *toy["M4_with_unobserved_velocity_S+T_residual"], INK2),
        ("a = x (잔차 0)", m2["zero_residual_a=q"], 0, INK2)]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200, gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
ys = list(range(len(rows)))[::-1]
XMAX = 120  # 촉각만·절대(311)는 축 밖이라 잘라서 표시
for y, (name, m, s, c) in zip(ys, rows):
    ax.barh(y, min(m, XMAX), xerr=s if s and m < XMAX else None, color=c, height=0.62,
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    label = f"{m:.2f}" if m < XMAX else f"{m:.0f} (축 밖) →"
    ax.text(min(m, XMAX) + 1.2 + (s or 0) if m < XMAX else XMAX - 1.5, y, label, va="center",
            ha="left" if m < XMAX else "right", fontsize=7, color=INK2 if m < XMAX else "white")
ax.set_xlim(0, XMAX)
ax.set_yticks(ys, [r[0] for r in rows], fontsize=7.5)
ax.set_xlabel("검증 MAE (×10⁻³ rad, 3 seed)", color=INK2)
ax.set_title("(a) Table III식 비교 (toy)", fontsize=9.5, color=INK)
style(ax)
ax = axes[1]
m5 = toy["M5_noise_robustness"]
pcts = [int(p) for p in m5]
ax.plot(pcts, [m5[str(p)]["S+T/residual"] for p in pcts], color=BLUE, lw=2, marker="o", ms=4, label="S+T 잔차")
ax.plot(pcts, [m5[str(p)]["T/residual"] for p in pcts], color=AQUA, lw=2, marker="o", ms=4, label="T 잔차")
ax.axhline(toy["M5_state_only_residual_reference"], color=ORANGE, lw=1.2, ls="--", label="S 잔차 (잡음 무관)")
ax.set_xlabel("촉각 잡음 (데이터셋 촉각 표준편차 대비 %)", color=INK2)
ax.set_ylabel("MAE (×10⁻³ rad)", color=INK2)
ax.set_title("(b) 촉각 잡음 강건성 (toy, Fig. 13과 같은 방식)", fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=7.5)
style(ax)
fig.text(0.01, 0.01, "3-DoF 평면 손가락, 관절 PD, 원형 물체 penalty 접촉, 준정적 평형. analysis/mapping_toy.py. 수치는 논문과 직접 비교 대상이 아니라 경향 확인용.",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "mapping_toy.png")
print("wrote", OUT / "mapping_toy.png")
