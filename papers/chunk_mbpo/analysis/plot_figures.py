"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림 조각은 만들지 않는다.

  own_final_returns.png  19개 과제의 마지막 지점 반환(그림 7 판독 값): CMPO, TD-MPC2, SAC
  own_grad_horizon.png   그림 4(b) 판독 값: 호라이즌별 기울기 노름의 평균과 시드별 최소–최대 평균
  own_walltime.png       표 1의 학습 시간을 추론 시간(가정)과 나머지로 나눈 산술
  own_toy_chunk.png      그림 5(b) 판독 값(왼쪽)과 토이의 예측 스텝별 MSE(오른쪽)

입력: analysis/curves_fig7_summary.json, fig45_values.json, recompute_results.json, toy_chunk_vs_recurrent_results*.json
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
A = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
S = json.loads((A / "curves_fig7_summary.json").read_text(encoding="utf-8"))
F = json.loads((A / "fig45_values.json").read_text(encoding="utf-8"))
R = json.loads((A / "recompute_results.json").read_text(encoding="utf-8"))
T = json.loads((A / "toy_chunk_vs_recurrent_results.json").read_text(encoding="utf-8"))
TL = json.loads((A / "toy_chunk_vs_recurrent_results_long.json").read_text(encoding="utf-8"))

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False
INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


# 1. 과제별 마지막 지점
tasks = [t for t in S if t != "Average"]
order = sorted(tasks, key=lambda t: -S[t]["CMPO"]["y_final"])
fig, ax = plt.subplots(figsize=(8.6, 5.4))
y = np.arange(len(order))
for off, m, col in ((-0.27, "CMPO", ACC), (0, "TD-MPC2", ACC2), (0.27, "SAC", MUTED)):
    v = [max(S[t][m]["y_final"], 1) for t in order]
    ax.barh(y + off, v, height=0.25, color=col, label=m)
lo = [max(S[t]["CMPO"]["band_lo_end"], 1) for t in order]
hi = [S[t]["CMPO"]["band_hi_end"] for t in order]
ax.hlines(y - 0.27, lo, hi, color=INK, lw=1.0)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("마지막 지점 반환 (그림 7 판독 값, 로그 축). 검은 선은 CMPO 음영(95% 구간)", fontsize=8)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title("19개 과제의 마지막 지점: CMPO는 16개에서 가장 높음", fontsize=10, color=INK)
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "own_final_returns.png", dpi=170)
plt.close(fig)

# 2. 그림 4(b)
g = F["fig4b_grad_norm_by_horizon"]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for m, col, dx in (("MAAC", ACC2, -0.05), ("CMPO", ACC, 0.05)):
    hs = sorted(int(h) for h in g[m])
    mean = [g[m][str(h)]["mean"] for h in hs]
    mx = [g[m][str(h)]["max"] for h in hs]
    mn = [g[m][str(h)]["min"] for h in hs]
    xs = np.array(hs) + dx
    ax.errorbar(xs, mean, yerr=[np.array(mean) - mn, np.array(mx) - mean], fmt="o-", color=col, capsize=3, label=m, lw=1.2, ms=4)
ax.set_xticks([2, 3, 4])
ax.set_xlabel("호라이즌 H", fontsize=9)
ax.set_ylabel("기울기 노름 (그림 4(b) 판독 값)", fontsize=9)
ax.set_title("평균은 비슷하고, 시드별 최대값이 MAAC에서 2.2–4.1배", fontsize=10, color=INK)
ax.legend(fontsize=8, frameon=False)
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "own_grad_horizon.png", dpi=170)
plt.close(fig)

# 3. 표 1 시간 산술
v1 = R["V1_table1"]["Walk"]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
labels = ["TD-MPC2", "CMPO"]
inf = [v1["assumed_inference_hours_tdmpc2"], v1["assumed_inference_hours_cmpo"]]
res = [v1["residual_hours_tdmpc2"], v1["residual_hours_cmpo"]]
ax.barh(labels, res, color=MUTED, label="나머지(갱신 등, 산술)")
ax.barh(labels, inf, left=res, color=ACC2, label="추론 시간 x 195,000 스텝(가정)")
for i, (a, b) in enumerate(zip(res, inf)):
    ax.text(a + b + 0.05, i, f"{a + b:.2f} h", va="center", fontsize=8, color=INK)
ax.set_xlabel("Walk, 200k 스텝 학습 시간 (표 1)", fontsize=9)
ax.set_xlim(0, 4.3)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.set_title("시간 감소의 대부분이 추론 쪽 (가정에 따른 산술)", fontsize=10, color=INK)
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "own_walltime.png", dpi=170)
plt.close(fig)

# 4. 그림 5(b) 판독 값 + 토이
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
e = F["fig5b_latent_mse_x1e-3"]
ax = axs[0]
for k, col in (("Recurrent", INK), ("Chunk h=2", "#6ab0de"), ("Chunk h=3", "#3d8cc4"), ("Chunk h=4", ACC), ("Chunk h=5", "#0f3a73")):
    st = sorted(int(s) for s in e[k])
    ax.plot(st, [e[k][str(s)] for s in st], "o-", color=col, ms=3, lw=1.2, label=k)
ax.set_xlabel("예측 스텝", fontsize=9)
ax.set_ylabel("잠재 MSE x 1e-3 (그림 5(b) 판독 값)", fontsize=8)
ax.set_title("논문 그림 5(b)", fontsize=10, color=INK)
ax.legend(fontsize=7, frameon=False)
clean(ax)
ax = axs[1]
steps = np.arange(1, 6)
for key, d, col, ls in (("iid_n20000_steps1500", T, ACC, "-"), ("AR1_n20000_steps1500", T, "#6ab0de", "--"), ("iid_n20000_steps5000", TL, "#0f3a73", ":")):
    ax.plot(steps, np.array(d[key]["chunk"]["mse_by_step_mean"]) * 1e3, "o" + ls, color=col, ms=3, lw=1.2,
            label=f"청크 {key.replace('_n20000', '')}")
    ax.plot(steps, np.array(d[key]["recurrent"]["mse_by_step_mean"]) * 1e3, "s" + ls, color=ACC2, ms=3, lw=1.2,
            label=f"재귀 {key.replace('_n20000', '')}")
ax.set_yscale("log")
ax.set_xlabel("예측 스텝", fontsize=9)
ax.set_ylabel("상태 MSE x 1e-3 (토이, 시드 3개 평균)", fontsize=8)
ax.set_title("토이(이중 진자): 이 설정에서는 재귀가 낮음", fontsize=10, color=INK)
ax.legend(fontsize=6, frameon=False, ncol=1)
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "own_toy_chunk.png", dpi=170)
plt.close(fig)
print("saved", [p.name for p in FIG.glob("own_*.png")])
