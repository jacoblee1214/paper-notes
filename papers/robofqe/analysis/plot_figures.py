"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_gt_floor.png    표 1의 RoboFQE·세계모델 ARE와 정답 200궤적 노이즈 바닥(토이 1)
  own_label_toy.png   LRM 라벨 오류에 따른 할인 반환 편향(토이 2): 대칭 정확도 곡선, 실로봇 설정
  own_rank_toy.png    정답 노이즈가 정할 순위 상관 천장(토이 1)

입력: analysis/toy_gt_noise_results.json, analysis/toy_label_noise_results.json
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
GN = json.loads((ROOT / "analysis" / "toy_gt_noise_results.json").read_text(encoding="utf-8"))
LN = json.loads((ROOT / "analysis" / "toy_label_noise_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
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

SUITES = ["Spatial", "Object", "Goal", "Long"]
ARE = {  # 표 1 (RoboFQE, Cosmos 2.5-AC, Ctrl-World, WorldGym), 평균 ARE(%)
    "RoboFQE": [3.40, 4.56, 2.34, 5.63],
    "Ctrl-World": [9.13, 8.17, 8.71, 58.75],
    "Cosmos 2.5-AC": [8.44, 7.55, 8.03, 54.37],
    "WorldGym": [24.20, 22.21, 50.04, 69.18],
}

# ------------------------------------------------ 그래프 1: ARE와 정답 노이즈 바닥
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
x = np.arange(len(SUITES))
w = 0.2
colors = {"RoboFQE": ACC, "Ctrl-World": "#8fa9c9", "Cosmos 2.5-AC": "#b7c5d8", "WorldGym": "#dcdfe6"}
for k, (name, vals) in enumerate(ARE.items()):
    ax.bar(x + (k - 1.5) * w, vals, w, label=name, color=colors[name], edgecolor="white")
# 정답 노이즈 바닥: 평균 절대 오차 = 0.798 * SE/V, c = 0.15–0.35 범위를 음영으로
lo = [GN["gt_noise_floor"][s]["c=0.15"]["ARE_floor_mean_abs_pct"] for s in SUITES]
hi = [GN["gt_noise_floor"][s]["c=0.35"]["ARE_floor_mean_abs_pct"] for s in SUITES]
for i in range(len(SUITES)):
    ax.fill_between([i - 0.46, i + 0.46], lo[i], hi[i], color=WARN, alpha=0.18, lw=0)
ax.plot([], [], color=WARN, alpha=0.4, lw=8, label="정답 200궤적 노이즈 바닥(토이 1)")
ax.set_yscale("log")
ax.set_yticks([2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
ax.set_ylabel("ARE (%), 로그 척도")
ax.set_xticks(x)
ax.set_xticklabels([f"LIBERO-{s}" for s in SUITES])
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
ax.set_title("표 1의 ARE와 정답 노이즈 바닥: Long에서 RoboFQE의 5.6%는 정답 자체의 노이즈 아래입니다", fontsize=8.6, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_gt_floor.png")
plt.close(fig)

# ------------------------------------------------ 그래프 2: 라벨 오류와 편향
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=200, gridspec_kw={"width_ratios": [1.1, 1]})
curve = LN["spatial_bias_vs_acc_closed_form"]
accs = [c["acc"] for c in curve]
bias = [c["mean_rel_bias_pct"] for c in curve]
a1.plot([100 * a for a in accs], bias, color=ACC, marker="o", ms=3.5, lw=1.4)
mc = [(0.974, LN["libero_symmetric"]["Spatial"]["acc=0.974"]["bias_relative_pct_mc_mean"]),
      (0.90, LN["libero_symmetric"]["Spatial"]["acc=0.9"]["bias_relative_pct_mc_mean"])]
a1.scatter([100 * m[0] for m in mc], [m[1] for m in mc], color=ACC2, zorder=3, s=22, label="몬테카를로(n=400k)")
a1.axvline(97.4, color=MUTED, lw=0.8, ls=":")
a1.text(97.6, -0.5, "LIBERO 97.4%", fontsize=7.2, color=MUTED)
a1.axvline(83.3, color=MUTED, lw=0.8, ls=":")
a1.text(83.5, -0.5, "실로봇 83.3%", fontsize=7.2, color=MUTED)
a1.set_xlabel("대칭 라벨 정확도 (%)")
a1.set_ylabel("평균 상대 편향 (%)")
a1.set_title("LIBERO-Spatial, γ=0.99 (정책 4개 평균)", fontsize=8, loc="left")
a1.legend(frameon=False, fontsize=7)
a1.invert_xaxis()

real = LN["real_world_gamma1"]["acc=0.833"]
pol = ["SR 58.5%", "SR 81.0%", "SR 33.0%"]
xs = np.arange(3)
vals = real["abs_bias_pp"]
a2.bar(xs, vals, color=[ACC if v < 0 else WARN for v in vals], width=0.55)
for i, v in enumerate(vals):
    a2.text(i, v + (0.6 if v >= 0 else -1.6), f"{v:+.1f}", ha="center", fontsize=7.5)
a2.axhline(0, color=INK, lw=0.6)
a2.set_xticks(xs)
a2.set_xticklabels(pol, fontsize=7.5)
a2.set_ylabel("절대 편향 (pp)")
a2.set_ylim(-14, 9)
a2.set_title(f"실로봇(γ=1), 대칭 83.3%: 평균 상대 오차 {real['mean_relative_error_pct']}%\n(논문 Table 3 ARE 7.5 ± 5.0)", fontsize=7.8, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_label_toy.png")
plt.close(fig)

# ------------------------------------------------ 그래프 3: 순위 상관 천장
fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=200)
spreads = list(GN["rank_ceiling"]["Spatial"].keys())
short = ["좁음 (0.84–0.73)", "중간 (0.84–0.30)", "넓음 (0.90–0.05)"]
xs = np.arange(len(SUITES))
w = 0.26
cols = ["#b7c5d8", "#7fa3cf", ACC]
for k, sp in enumerate(spreads):
    vals = [GN["rank_ceiling"][s][sp]["ceiling_r_median_of_noisy_GT_vs_true"] for s in SUITES]
    ax.bar(xs + (k - 1) * w, vals, w, color=cols[k], label=short[k], edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(xs[i] + (k - 1) * w, v + 0.005, f"{v:.2f}", ha="center", fontsize=6.3, color=INK)
ax.set_ylim(0.6, 1.02)
ax.set_xticks(xs)
ax.set_xticklabels([f"LIBERO-{s}" for s in SUITES])
ax.set_ylabel("천장 r (정답 노이즈 대 참값)")
ax.legend(frameon=False, fontsize=7.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), title="체크포인트 SR 간격 (4개)", title_fontsize=7.2)
ax.set_title("정답 200궤적으로 얻을 수 있는 순위 상관의 상한(토이 1, 정책 4개 가정)", fontsize=8.6, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_rank_toy.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
