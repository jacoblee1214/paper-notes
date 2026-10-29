"""리포트용 그래프 세 장 (게시본에도 쓰는 직접 그린 그림). 실행: python3 analysis/plot_figures.py

1) fig3_counts_ci.png: Fig. 3을 정수 개수로 되돌리고 구간별 Wilson 95% 구간 (table_check_results.json)
2) nullspace_eps.png: actuation-consistency 상자 제약의 null 차원·허용 크기와 실현 잔차 (qp_toy_results.json)
3) pose_error_contacts.png: Fig. 6 조건에서 접점 재현율·오검출·법선 오차 (pose_error_toy_results.json)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
A = ROOT / "analysis"


def style(ax):
    ax.grid(color=GRID, lw=0.8)
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
    return c - h, c + h


# 1) Fig. 3 정수 개수 + Wilson
tc = json.loads((A / "table_check_results.json").read_text())["C1_fig3"]
n = [51, 45, 78, 98, 131, 116, 116, 100, 55]
labels = ["2", "3", "4", "5", "6", "7", "8", "9", "≥10"]
fig, ax = plt.subplots(figsize=(8.6, 3.3), dpi=200)
for j, (m, c, lab) in enumerate((("feedforward", BLUE, "Feedforward"), ("impedance", ORANGE, "Impedance"), ("ours", AQUA, "Ours"))):
    ks = [k if isinstance(k, int) else k[0] for k in tc["counts_per_bin"][m]]
    xs = [i + (j - 1) * 0.22 for i in range(len(n))]
    for x, k, ni in zip(xs, ks, n):
        lo, hi = wilson(k, ni)
        ax.plot([x, x], [100 * lo, 100 * hi], color=c, lw=2, solid_capstyle="round")
    ax.scatter(xs, [100 * k / ni for k, ni in zip(ks, n)], s=26, color=c, edgecolor="white", linewidth=1, zorder=3, label=lab)
ax.set_xticks(range(len(n)), [f"{b}\n(n={ni})" for b, ni in zip(labels, n)])
ax.set_ylabel("성공률 (%)", color=INK2)
ax.set_xlabel("초기화 시 MuJoCo 접점 수", color=INK2)
ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
style(ax)
fig.text(0.01, 0.01, "Fig. 3 막대값을 정수 개수 k/n으로 되돌린 값과 Wilson 95% 구간. 790개 전체: Ours 92.3%, Impedance 70.0%, Feedforward 64.7%.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "fig3_counts_ci.png")
print("wrote", OUT / "fig3_counts_ci.png")

# 2) null 차원과 ε 효과
qp = json.loads((A / "qp_toy_results.json").read_text())
e3, e3b = qp["E3_actuation_consistency"], qp["E3b_precision_and_eps"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=200)
ax = axes[0]
Ks = [1, 3, 5]
d = [e3[f"K={k}"]["null_dim"] for k in Ks]
bound = [e3[f"K={k}"]["eps_sqrt_d_N"] for k in Ks]
ax.plot(Ks, bound, color=BLUE, lw=2, marker="o", ms=6, mec="white")
for k, b, dd in zip(Ks, bound, d):
    ax.annotate(f"d = {dd}", (k, b), xytext=(6, -12), textcoords="offset points", fontsize=7.5, color=INK2)
ax.axhline(10, color=INK2, lw=1, ls="--")
ax.text(1.02, 10.2, "f_n,min = 10 N", fontsize=7.5, color=INK2)
ax.set_xticks(Ks)
ax.set_xlabel("링크당 최대 접점 K (toy power grasp)", color=INK2)
ax.set_ylabel("상자 제약이 허용하는 null 힘 크기 ε√d (N)", color=INK2)
ax.set_ylim(0, 12)
ax.set_title("(a) ε = 0.5 N 성분별 상자의 허용 범위", fontsize=9.5, color=INK)
style(ax)
ax = axes[1]
eps = [0.02, 0.1, 0.5]
for name, c, lab in (("power_K5", BLUE, "power grasp, 80 접점"), ("precision_3finger_K5", ORANGE, "세 손가락, 15 접점")):
    v = e3b[name]
    ys = [v[f"eps={e}"]["realized_residual_force_N"] for e in eps]
    ax.plot(eps, ys, color=c, lw=2, marker="o", ms=6, mec="white", label=lab)
    ax.axhline(v["no_AC"]["realized_residual_force_N"], color=c, lw=1, ls=":")
ax.set_xscale("log")
ax.set_xticks(eps, ["0.02", "0.1", "0.5"])
ax.set_xlabel("ε_margin (N, 로그축)  · 점선: 제약 없음", color=INK2)
ax.set_ylabel("실현 잔차 힘 (N)", color=INK2)
ax.set_title("(b) 구동 불가 성분이 남기는 잔차", fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.62))
style(ax)
fig.text(0.01, 0.01, "toy 손 모델(20 DoF)과 ADMM(OSQP 설정). 실현 힘 = range(J_act) 사영(같은 접점 강성, 물체 고정 가정). analysis/qp_toy.py",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "nullspace_eps.png")
print("wrote", OUT / "nullspace_eps.png")

# 3) 자세 오차와 접점 추정
pe = json.loads((A / "pose_error_toy_results.json").read_text())["results"]
conds = list(pe.keys())
xt = ["(0°, 0)", "(10°, 1 mm)", "(20°, 3 mm)", "(30°, 5 mm)"]
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=200)
for th, c, lab in (("thresh_5mm", BLUE, "d_thresh 0.5 cm"), ("thresh_10mm", AQUA, "1 cm"), ("thresh_20mm", ORANGE, "2 cm")):
    axes[0].plot(range(4), [100 * pe[k][th]["true_contact_recall"] for k in conds], color=c, lw=2, marker="o", ms=5, mec="white", label=lab)
    axes[1].plot(range(4), [100 * pe[k][th]["noncontact_false_detect"] for k in conds], color=c, lw=2, marker="o", ms=5, mec="white", label=lab)
axes[2].plot(range(4), [pe[k]["thresh_10mm"]["normal_err_deg_median"] for k in conds], color=AQUA, lw=2, marker="o", ms=5, mec="white", label="중앙값")
axes[2].plot(range(4), [pe[k]["thresh_10mm"]["normal_err_deg_p90"] for k in conds], color=AQUA, lw=1.5, ls="--", marker="o", ms=4, mec="white", label="90분위")
axes[2].axhline(35, color=INK2, lw=1, ls=":")
axes[2].text(0.05, 37, "마찰원뿔 반각 atan 0.7 = 35°", fontsize=7, color=INK2)
titles = ["(a) 실제 접점 재현율 (%)", "(b) 5–20 mm 떨어진 점의 오검출 (%)", "(c) 법선 오차 (°, d_thresh 1 cm)"]
for ax, t in zip(axes, titles):
    ax.set_xticks(range(4), xt, fontsize=7)
    ax.set_title(t, fontsize=9.5, color=INK)
    style(ax)
axes[0].set_ylim(60, 102)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
axes[2].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.text(0.01, 0.01, "6×6×12 cm 상자, 실제 접점 16개 + 비접촉 점 48개, 조건마다 오프셋 400개. 같은 조건에서 논문 성공률은 87–98% (Fig. 6). analysis/pose_error_toy.py",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "pose_error_contacts.png")
print("wrote", OUT / "pose_error_contacts.png")
