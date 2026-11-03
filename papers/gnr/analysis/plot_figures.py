"""리포트·게시본용 그래프 네 장 (직접 계산한 값만). 실행: cd analysis && python3 plot_figures.py

1) table1_split.png   Table 1을 MPC-success(272)·MPC-failure(728) 몫으로 분해, OOD 부분집합의 MPC 성공 비율
2) scaling_fit.png    pass@S 관측값과 이질성 모델(Beta + 0 질량) 적합, 같은 p 독립 가정과 비교
3) cost_per_success.png  시연 1개당·성공 1건당 CPU 비용 (Table 1), Table 14와의 차이
4) manifold_toy.png   잔차 생성기 대용 vs MPC toy
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import scaling_toy as st

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TC = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
MT = json.loads((ROOT / "analysis" / "manifold_toy_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) Table 1 분해
c2 = TC["C2_table1_decomposition"]
names = ["IK", "DIAL-MPC", "MPPI", "iCEM", "CMA-ES", "GNR S=1", "GNR S=64", "GNR S=128", "GNR S=256"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200, gridspec_kw={"width_ratios": [1.7, 1]})
ax = axes[0]
y = np.arange(len(names))[::-1]
ks = [c2[n]["k_success"] for n in names]
kf = [c2[n]["k_failure"] for n in names]
ax.barh(y, ks, color=BLUE, height=0.62, label="MPC-success 272개 중 성공")
ax.barh(y, kf, left=ks, color=ORANGE, height=0.62, label="MPC-failure 728개 중 성공", edgecolor="white", linewidth=1)
for yi, a, b in zip(y, ks, kf):
    ax.text(a + b + 6, yi, f"{(a + b) / 10:.1f}%", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y, names)
ax.set_xlim(0, 790)
ax.set_xlabel("시험 1,000개 중 성공 수", color=INK2)
ax.set_title("(a) Table 1을 두 부분집합 몫으로 분해", fontsize=10, color=INK)
ax.annotate("DIAL-MPC 행 = 시험 집합을 나눈 레이블 자체\n(272/0, 다시 돌린 결과 아님)", xy=(335, y[1] - 0.05), xytext=(330, y[1] + 0.9),
            fontsize=7.2, color=INK2, arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.7})
ax.legend(frameon=False, fontsize=7.5, loc="center right")
style(ax)
ax = axes[1]
c3 = TC["C3_ood_subsets"]
subs = ["OOD object", "OOD grasp", "OOD geometry"]
gnr256 = {"OOD object": 44.33, "OOD grasp": 42.90, "OOD geometry": 71.24}
x = np.arange(3)
ax.bar(x - 0.18, [c3[s]["mpc_success_rate_pct"] for s in subs], width=0.34, color=MUTED, label="DIAL-MPC 성공 비율(역산)")
ax.bar(x + 0.18, [gnr256[s] for s in subs], width=0.34, color=BLUE, label="GNR S=256")
for xi, s in zip(x, subs):
    ax.text(xi - 0.18, c3[s]["mpc_success_rate_pct"] + 1.5, f"{c3[s]['mpc_success_rate_pct']:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.text(xi + 0.18, gnr256[s] + 1.5, f"{gnr256[s]:.0f}", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(x, ["물체", "파지", "기하"])
ax.set_ylim(0, 85)
ax.set_ylabel("성공률 (%)", color=INK2)
ax.set_title("(b) OOD 부분집합 (Fig. 3에서 역산)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
style(ax)
fig.text(0.01, 0.01, "출처: Table 1, Fig. 3, 부록 D.3. 부분집합별 MPC 성공 수 37·69·166의 합이 정확히 272. analysis/table_check.py",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "table1_split.png")
print("wrote table1_split.png")

# 2) scaling fit
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
show = [("HOT3D 360k MPC-failure (D.4)", ORANGE, "HOT3D MPC-failure 264k"),
        ("HOT3D 360k MPC-success (D.4)", BLUE, "HOT3D MPC-success 95k"),
        ("ShapeFilter N=50 (Table 12)", AQUA, "ShapeFilter 시드 50개")]
Sg = np.unique(np.round(np.logspace(0, np.log10(1024), 120)).astype(int))
for key, col, lab in show:
    S, sr = st.CURVES[key]
    pi0, a, b, _ = st.fit(S, sr)
    ax.plot(Sg, 100 * st.pass_at(Sg, pi0, a, b), color=col, lw=2, label=f"{lab} (a={a:.2f}, b={b:.1f})")
    ax.scatter(S, sr, color=col, s=26, zorder=3, edgecolor="white", linewidth=1)
S, sr = st.CURVES["HOT3D 360k MPC-failure (D.4)"]
p1 = sr[0] / 100
ax.plot(Sg, 100 * (1 - (1 - p1) ** Sg), color=ORANGE, lw=1.2, ls="--", label="같은 p로 독립이라면 (MPC-failure)")
ax.axvline(256, color=INK2, lw=0.8, ls=":")
ax.text(280, 78, "논문 최대 S=256", fontsize=7, color=INK2, rotation=90, va="center")
ax.set_xscale("log", base=2)
ax.set_xlim(1, 1024)
ax.set_ylim(0, 102)
ax.set_xlabel("시연당 GNR 표본 수 S (로그축)", color=INK2)
ax.set_ylabel("pass@S (%)", color=INK2)
ax.legend(frameon=False, fontsize=7.3, loc="lower right")
style(ax)
fig.text(0.01, 0.01, "점: 논문 값(부록 D.4, Table 12에서 역산). 선: 성공 확률 p ~ Beta(a,b)(+0 질량) 적합. a<1이면 p가 0 근처에 몰려 표본을 늘려도 느리게 오름. analysis/scaling_toy.py",
         fontsize=6.3, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "scaling_fit.png")
print("wrote scaling_fit.png")

# 3) 비용
c6 = TC["C6_compute_accounting"]
meth = ["IK", "DIAL-MPC", "MPPI", "iCEM", "CMA-ES", "GNR S=64", "GNR S=256"]
per_demo = {"IK": 39.0, "DIAL-MPC": 2689.3, "MPPI": 1849.6, "iCEM": 3862.1, "CMA-ES": 2912.7, "GNR S=64": 90.9, "GNR S=256": 354.1}
per_succ = c6["core_s_per_success_test"]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
x = np.arange(len(meth))
ax.bar(x - 0.19, [per_demo[m] for m in meth], width=0.36, color=MUTED, label="시연 1개당 (Table 1)")
ax.bar(x + 0.19, [per_succ[m] for m in meth], width=0.36, color=BLUE, label="성공 1건당 (= 시연당 / SR)")
for xi, m in zip(x, meth):
    ax.text(xi + 0.19, per_succ[m] * 1.12, f"{per_succ[m]:,.0f}", ha="center", fontsize=6.8, color=INK2)
ax.set_yscale("log")
ax.set_xticks(x, meth, fontsize=7.5)
ax.set_ylabel("CPU core-s (로그축)", color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
style(ax)
fig.text(0.01, 0.01, f"HOT3D 시험 1,000개. 같은 방법의 Table 14 단가는 Table 1보다 1.11–1.52배 큼 → DIAL/GNR 비는 Table 1 기준 7.6배, Table 14 기준 9.9배.",
         fontsize=6.6, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "cost_per_success.png")
print("wrote cost_per_success.png")

# 4) manifold toy
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=200, sharey=True)
gp = MT["gnr_proxy"]
Ns = list(gp.keys())
Ss = [1, 4, 16, 64, 256]
for ax, setname, title in ((axes[0], "in_dist", "(a) 학습 범위 안"), (axes[1], "interp_ood", "(b) 학습 범위 안의 빈 원판(보간 OOD)")):
    for N, col in ((Ns[0], MUTED), (Ns[1], AQUA), (Ns[-1], BLUE)):
        r = gp[N][f"{setname}/residual"]
        a = gp[N][f"{setname}/absolute"]
        ax.plot(Ss, [r[str(s)] for s in Ss], color=col, lw=2, marker="o", ms=4, label=f"잔차, 시드 {N}개")
        ax.plot(Ss, [a[str(s)] for s in Ss], color=col, lw=1.2, ls="--", label=f"절대, 시드 {N}개")
    m = MT[f"mpc_{setname}"]["success_pct"]
    ax.axhline(m, color=ORANGE, lw=1.4)
    tx, ty = (12, 46) if setname == "in_dist" else (1.05, m + 2)
    ax.text(tx, ty, f"주황 선: MPC 3,000표본 {m:.0f}%", fontsize=7.3, color=ORANGE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ss, [str(s) for s in Ss])
    ax.set_ylim(0, 102)
    ax.set_xlabel("생성기 표본 수 S", color=INK2)
    ax.set_title(title, fontsize=10, color=INK)
    style(ax)
axes[0].set_ylabel("성공률 (%)", color=INK2)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.045))
ext = MT["mpc_extrap_ood"]["success_pct"]
fig.text(0.01, 0.01, f"24차원 잔차, 3차원 잠재 manifold + 파지 모드 2개 toy. 생성기는 kNN 모드별 국소 선형 회귀 + 잡음(흐름 정합 아님). 범위 밖 외삽은 MPC {ext:.0f}%, 생성기도 5% 이하. analysis/manifold_toy.py",
         fontsize=6.3, color=INK2)
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(OUT / "manifold_toy.png")
print("wrote manifold_toy.png")
