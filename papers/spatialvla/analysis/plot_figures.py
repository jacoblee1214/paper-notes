"""우리가 그린 그래프 세 장 (논문 그림은 쓰지 않음, 라이선스가 CC BY가 아님).

  report/figures/own_widowx_real.png : 실로봇 제로샷 성공률 (Wilson 95%)과 과제별 성공 횟수
  report/figures/own_libero.png      : LIBERO 스위트별 SR (± 표기값)
  report/figures/own_adaptive_grid.png : 토이 T1 (적응 격자 대 균일 격자)

입력: analysis/recompute_results.json, analysis/toy_adaptive_grid_results.json
실행: python3 analysis/plot_figures.py
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
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
font_path = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
if font_path:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

INK, MUTED, ACC, GREY = "#1b1b1a", "#55544f", "#1f5fae", "#b9b7b0"
rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
toy = json.loads((ANA / "toy_adaptive_grid_results.json").read_text(encoding="utf-8"))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


# ---------------------------------------------------------------- 1. 실로봇 제로샷
real = rec["widowx_real_zeroshot"]
pols = ["SpatialVLA", "OpenVLA", "RoboVLM", "Octo-Base", "RT-1-X"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
rates = [real[p]["rate"] for p in pols]
lo = [real[p]["rate"] - real[p]["wilson95"][0] for p in pols]
hi = [real[p]["wilson95"][1] - real[p]["rate"] for p in pols]
x = np.arange(len(pols))
a1.bar(x, rates, color=[ACC] + [GREY] * 4, width=0.62)
a1.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=3, lw=1)
for i, r in enumerate(rates):
    a1.text(i, r + hi[i] + 2, f"{r:.1f}%", ha="center", color=INK, fontsize=8)
a1.set_xticks(x, pols, rotation=20, ha="right")
a1.set_ylabel("성공률 (%), 77회 합산")
a1.set_ylim(0, 100)
a1.set_title("전체 성공률 (Wilson 95%)", color=INK, fontsize=10)
style(a1)

tasks = ["microwave", "chili", "eggplant", "purple_white", "green_stove", "purple_sink", "carrot"]
labels = ["마이크로웨이브", "고추", "가지", "보라 컵(흰 접시)", "초록 컵(스토브)", "보라 컵(싱크)", "당근"]
per = real["per_task_successes"]
for p, c, off in [("SpatialVLA", ACC, -0.18), ("OpenVLA", "#8a8a85", 0.18)]:
    vals = [per[p][t] for t in tasks]
    a2.barh(np.arange(len(tasks)) + off, vals, height=0.34, color=c, label=p)
a2.set_yticks(np.arange(len(tasks)), labels)
a2.invert_yaxis()
a2.set_xlim(0, 11.5)
a2.set_xlabel("성공 횟수 (과제당 11회)")
a2.set_title("과제별 성공 횟수", color=INK, fontsize=10)
a2.legend(frameon=False, fontsize=8, loc="lower right")
style(a2)
fig.tight_layout()
fig.savefig(FIG / "own_widowx_real.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. LIBERO
suites = ["Spatial", "Object", "Goal", "Long"]
sr = {
    "SpatialVLA": ([88.2, 89.9, 78.6, 55.5], [0.5, 0.7, 0.6, 1.0]),
    "OpenVLA": ([84.7, 88.4, 79.2, 53.7], [0.9, 0.8, 1.0, 1.3]),
    "Octo": ([78.9, 85.7, 84.6, 51.1], [1.0, 0.9, 0.9, 1.0]),
    "TraceVLA": ([84.6, 85.2, 75.1, 54.1], [0.2, 0.4, 0.3, 1.0]),
    "Diffusion Policy": ([78.3, 92.5, 68.3, 50.5], [1.1, 0.7, 1.2, 1.3]),
}
fig, ax = plt.subplots(figsize=(9.2, 3.5))
w = 0.16
cols = [ACC, "#8a8a85", "#b5b3ab", "#d6d4cc", "#e9e7e0"]
for i, (name, (v, e)) in enumerate(sr.items()):
    xs = np.arange(len(suites)) + (i - 2) * w
    ax.bar(xs, v, width=w, color=cols[i], label=name, yerr=e, capsize=2, error_kw={"lw": 0.8, "ecolor": INK})
ax.set_xticks(np.arange(len(suites)), [f"LIBERO-{s}" for s in suites])
ax.set_ylim(40, 100)
ax.set_ylabel("성공률 (%)")
ax.set_title("LIBERO 스위트별 성공률 (표 III, ± 표기값)", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_libero.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. 토이 T1 적응 격자
vt = toy["vars"]["theta"]
from scipy.stats import norm  # noqa: E402

mu, sg = 1.6157435178756714, 0.5976657271385193  # gs_fractal.json의 theta (토이가 읽은 값과 같음)
lo_, hi_ = 0.0, np.pi
n = 16
pp = np.linspace(norm.cdf(lo_, mu, sg), norm.cdf(hi_, mu, sg), n + 1)
ea = np.clip(norm.ppf(pp, mu, sg), lo_, hi_)
eu = np.linspace(lo_, hi_, n + 1)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
xs = np.linspace(0, np.pi, 400)
b1.plot(xs, norm.pdf(xs, mu, sg), color=INK, lw=1.2, label="가우시안 밀도 (theta)")
for e in ea[1:-1]:
    b1.axvline(e, color=ACC, lw=0.8, alpha=0.8)
for e in eu[1:-1]:
    b1.axvline(e, color="#8a8a85", lw=0.6, ls=":")
b1.plot([], [], color=ACC, lw=0.8, label="적응 경계 (등확률)")
b1.plot([], [], color="#8a8a85", lw=0.6, ls=":", label="균일 경계")
b1.set_xlabel("theta (rad)")
b1.set_ylabel("밀도")
b1.legend(frameon=False, fontsize=7.5, loc="upper right")
b1.set_title("격자 경계 비교 (theta, 16 bin)", color=INK, fontsize=10)
style(b1)
labels_v = list(toy["vars"].keys())
ratio_in = [toy["vars"][v]["mse_in_adaptive"] / toy["vars"][v]["mse_in_uniform"] for v in labels_v]
ratio_sh = [toy["vars"][v]["mse_shift_adaptive"] / toy["vars"][v]["mse_shift_uniform"] for v in labels_v]
xx = np.arange(len(labels_v))
b2.bar(xx - 0.2, ratio_in, width=0.38, color=ACC, label="같은 분포")
b2.bar(xx + 0.2, ratio_sh, width=0.38, color="#8a8a85", label="다른 데이터셋 분포")
b2.axhline(1.0, color=INK, lw=0.8, ls="--")
b2.set_xticks(xx, labels_v, rotation=20)
b2.set_ylabel("MSE 비 (적응 / 균일)")
b2.set_title("재구성 오차 비 (1보다 크면 적응이 나쁨)", color=INK, fontsize=9.5)
b2.legend(frameon=False, fontsize=7.5)
style(b2)
fig.tight_layout()
fig.savefig(FIG / "own_adaptive_grid.png", dpi=200)
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
