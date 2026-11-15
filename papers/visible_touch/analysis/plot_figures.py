"""우리가 그린 그래프 세 장을 report/figures/에 만든다 (논문 그림 아님, CPU, matplotlib).

own_libero_2v.png     : LIBERO BC-Transformer 2-view, 시드별 점과 평균 ± 모집단 SD (App. C.5 표 값)
own_realworld.png     : π0.5 실제 과제, 하위 단계별 성공 시험 수 (App. F.3 표 값)
own_toy_overlay.png   : 토이 A(카메라 기울기별 선형 복원 R²)와 토이 B(헤드라인 상관의 부풀림)
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

KO_FONT = None
for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        KO_FONT = cand
        break
if KO_FONT:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(KO_FONT)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=KO_FONT).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

INK, GRAY, ORANGE, BLUE = "#1b1b1a", "#8c8a84", "#d9822b", "#2c6fbb"

# ---- 1. LIBERO 2-view, per-seed (App. C.5) ----
suites = ["OBJECT", "GOAL", "SPATIAL", "LONG"]
seed = {
    "Plain": np.array([[84.5, 73.5, 90.5], [90.0, 89.5, 81.0], [90.5, 82.0, 82.0], [50.5, 52.5, 55.5]]),
    "Proprio": np.array([[82.5, 77.5, 80.0], [84.0, 75.5, 85.0], [82.0, 79.0, 79.0], [50.0, 38.5, 40.0]]),
    "Visible Touch": np.array([[99.0, 99.0, 96.5], [96.0, 92.5, 90.0], [96.0, 91.5, 89.5], [84.0, 86.0, 89.5]]),
}
colors = {"Plain": GRAY, "Proprio": BLUE, "Visible Touch": ORANGE}
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
w = 0.26
x = np.arange(len(suites))
for k, (name, arr) in enumerate(seed.items()):
    xs = x + (k - 1) * w
    m = arr.mean(axis=1)
    sd = arr.std(axis=1, ddof=0)  # 논문 표의 "±"와 같은 모집단 SD
    ax.bar(xs, m, w * 0.92, color=colors[name], alpha=0.85, label=name)
    ax.errorbar(xs, m, yerr=sd, fmt="none", ecolor=INK, capsize=3, lw=0.8)
    for j in range(3):
        ax.scatter(xs + (j - 1) * 0.05, arr[:, j], s=8, color=INK, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(suites)
ax.set_ylim(30, 105)
ax.set_ylabel("성공률 (%)")
ax.set_title("LIBERO BC-Transformer, 2-view: 시드 3개 (점), 평균과 모집단 SD", fontsize=9.5, loc="left")
ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32))
fig.tight_layout()
fig.savefig(FIG / "own_libero_2v.png")
plt.close(fig)

# ---- 2. Real-world sub-task counts (App. F.3) ----
subtasks = ["Lift\nPick", "Tube\nPick", "Tube\nInsert", "Dish\nPull", "Dish\nPick", "Dish\nPlace", "Charger\nPick", "Charger\nInsert"]
rw = {
    "Baseline": [20, 5, 0, 21, 11, 11, 9, 0],
    "Tac-View": [16, 9, 0, 22, 18, 16, 17, 0],
    "Multi-arrows": [24, 24, 16, 28, 22, 21, 18, 2],
}
rw_colors = {"Baseline": GRAY, "Tac-View": BLUE, "Multi-arrows": ORANGE}
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
x = np.arange(len(subtasks))
w = 0.27
for k, (name, vals) in enumerate(rw.items()):
    ax.bar(x + (k - 1) * w, vals, w * 0.92, color=rw_colors[name], label=name)
ax.axhline(30, color=INK, lw=0.6, ls=":")
ax.text(len(subtasks) - 0.5, 30.6, "30회", fontsize=7.5, ha="right", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(subtasks, fontsize=7.5)
ax.set_ylim(0, 34)
ax.set_ylabel("성공 (30회 중)")
ax.set_title("π0.5 실제 과제: 하위 단계별 성공 시험 수 (조건당 학습 1회로 보임)", fontsize=9.5, loc="left")
ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "own_realworld.png")
plt.close(fig)

# ---- 3. Toys (from JSON results) ----
toy_path = ROOT / "analysis" / "toy_overlay_results.json"
res = json.loads(toy_path.read_text(encoding="utf-8"))
A = res["toy_A_projection_info"]
B = res["toy_B_mechanical_headroom_corr"]["sweep"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]})
tilts = list(A.keys())
labels = [t.replace("cam_tilt_", "").replace("deg", "°") for t in tilts]
xa = np.arange(len(tilts))
wa = 0.26
for k, (axis, col) in enumerate([("Fx", GRAY), ("Fy", BLUE), ("Fz (광축)", ORANGE)]):
    vals = [A[t]["overlay_2d_R2_per_axis_Fx_Fy_Fz"][k] for t in tilts]
    a1.bar(xa + (k - 1) * wa, vals, wa * 0.92, color=col, label=axis)
a1.axhline(1.0, color=INK, lw=0.6, ls=":")
a1.set_xticks(xa)
a1.set_xticklabels(labels)
a1.set_xlabel("카메라 기울기")
a1.set_ylabel("선형 복원 R² (2D 화살표)")
a1.set_ylim(0, 1.1)
a1.set_title("토이 A: 2D 화살표의 힘 복원", fontsize=9, loc="left")
a1.legend(frameon=False, fontsize=7.5, ncol=1, loc="lower right")
sig = [0.02, 0.05, 0.10, 0.15]
keys = [f"acro_sd_{s}" for s in sig]
med = [B[k]["median_r"] for k in keys]
lo = [B[k]["q05"] for k in keys]
hi = [B[k]["q95"] for k in keys]
a2.plot(sig, med, "-o", color=ORANGE, ms=4)
a2.fill_between(sig, lo, hi, color=ORANGE, alpha=0.2)
a2.axhline(0.924, color=INK, lw=0.8, ls="--")
a2.text(0.021, 0.934, "논문 r = 0.924", fontsize=7.5, color=INK)
a2.set_xlabel("오버레이 성공률 산포 σ (무작위 가정)")
a2.set_ylabel("Δ와 headroom의 상관 r")
a2.set_ylim(0.8, 1.01)
a2.set_title("토이 B: 공유 항이 만드는 상관", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_toy_overlay.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
