"""우리가 그린 그래프 네 장 (report/figures/own_*.png). 논문 수치와 analysis/recompute.py의 결과만 쓴다.

  own_pref.png    : 사람 선호 승·무·패 (쌍 50개), 부호 검정 p (V2)
  own_policy.png  : 정책 평가 성공률 대 시뮬레이터, Wilson 95% (V1)
  own_pusht.png   : Push-T 평균 IoU 대 K, P별 선 (V4, 감소 구간 표시)
  own_real.png    : 실로봇 성공률 k/15, Wilson 95% (V3)

한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if font_path:
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False
INK, GREY, TEAL, RED, LIGHT = "#1b1b1a", "#55544f", "#0f8a85", "#d7263d", "#c9cdd3"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 사람 선호 (V2)
pref = RES["figure4_preference"]
labels = list(pref.keys())
fig, ax = plt.subplots(figsize=(8.2, 4.6))
y = list(range(len(labels)))[::-1]
for yi, lab in zip(y, labels):
    w, t, l = pref[lab]["win_tie_loss_of_50"]
    ax.barh(yi, w, color=TEAL, height=0.6)
    ax.barh(yi, t, left=w, color=LIGHT, height=0.6)
    ax.barh(yi, l, left=w + t, color=RED, height=0.6)
    p = pref[lab]["sign_test_p_two_sided"]
    ax.text(51, yi, f"{w}/{t}/{l}  p = {p:.2g}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 50)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel("쌍 수 (50쌍 중, 승·무·패)", fontsize=9, color=GREY)
ax.set_title("사람 선호: IRASim-Frame-Ada 기준 승·무·패 (V2, 그림 4 환산)", fontsize=10.5, color=INK, loc="left")
style(ax)
ax.text(0.0, -0.22, "쌍 수는 그림 4 백분율을 참가자 5명 × 클립 10개 = 50쌍으로 환산했다. p는 동점을 뺀 양측 부호 검정. 우리가 그린 그래프.",
        transform=ax.transAxes, fontsize=7.5, color=GREY)
save(fig, "own_pref.png")

# ---------------------------------------------------------------- 정책 평가 (V1)
t4 = RES["table4"]
gt = [0.18, 0.50, 0.80, 1.00]
ir = [0.28, 0.48, 0.74, 0.96]
gt_ci = t4["gt_wilson95"]
ir_ci = t4["irasim_wilson95"]
fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.plot([0, 1.05], [0, 1.05], color=GREY, lw=0.8, ls="--")
for i, (x, yv) in enumerate(zip(gt, ir)):
    ax.errorbar(x, yv, xerr=[[x - gt_ci[i][0]], [gt_ci[i][1] - x]], yerr=[[yv - ir_ci[i][0]], [ir_ci[i][1] - yv]],
                fmt="o", color=TEAL, ecolor=LIGHT, capsize=3, ms=6)
    ax.text(x + 0.02, yv - 0.05, f"체크포인트 {i + 1}", fontsize=8, color=INK)
ax.set_xlabel("시뮬레이터 성공률 (50회, Wilson 95%)", fontsize=9, color=GREY)
ax.set_ylabel("IRASim 성공률 (50회, Wilson 95%)", fontsize=9, color=GREY)
ax.set_xlim(0, 1.08)
ax.set_ylim(0, 1.08)
ax.set_title(f"정책 평가 상관: r = {t4['pearson_r']:.2f}, n = 4 (V1)", fontsize=10.5, color=INK, loc="left")
style(ax)
ax.text(0.0, -0.2, f"r의 Fisher z 95% 구간 {t4['pearson_fisher_95ci_n4'][0]:.2f}–{t4['pearson_fisher_95ci_n4'][1]:.2f}. 우리가 그린 그래프.",
        transform=ax.transAxes, fontsize=7.5, color=GREY)
save(fig, "own_policy.png")

# ---------------------------------------------------------------- Push-T (V4, 표 5 값)
K = [1, 5, 10, 50]
push = {
    "P=0": [0.637, 0.679, 0.572, 0.418],
    "P=100": [0.637, 0.847, 0.878, 0.888],
    "P=200": [0.637, 0.866, 0.916, 0.912],
    "P=500": [0.637, 0.907, 0.906, 0.938],
    "P=1000": [0.637, 0.886, 0.945, 0.961],
}
fig, ax = plt.subplots(figsize=(6.6, 4.4))
colors = ["#8a8f98", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
for (name, vals), c in zip(push.items(), colors):
    ax.plot(K, vals, marker="o", color=c, lw=1.5, ms=5, label=name)
    for k, a, b in zip(K[:-1], vals[:-1], vals[1:]):
        pass
    # 감소 구간을 빨간 점으로 표시
    for j in range(len(K) - 1):
        if vals[j + 1] < vals[j] and name != "P=0":
            ax.scatter([K[j + 1]], [vals[j + 1]], s=110, facecolors="none", edgecolors=RED, lw=1.5, zorder=5)
ax.axhline(0.637, color=GREY, lw=0.8, ls="--")
ax.text(1.3, 0.60, "계획 없음 0.637", fontsize=8, color=GREY)
ax.set_xscale("log")
ax.set_xticks(K)
ax.set_xticklabels([str(k) for k in K])
ax.set_xlabel("K (후보 궤적 수, 로그 축)", fontsize=9, color=GREY)
ax.set_ylabel("평균 IoU (셀당 100회)", fontsize=9, color=GREY)
ax.set_title("Push-T 계획: P별 평균 IoU (V4)", fontsize=10.5, color=INK, loc="left")
ax.legend(title="사후 롤아웃 P", fontsize=8, title_fontsize=8, frameon=False, loc="lower right")
style(ax)
ax.text(0.0, -0.2, "빨간 원: 같은 P에서 K를 늘렸을 때 또는 P를 늘렸을 때 값이 내려간 지점. 분산이 없어 차이의 유의성은 판단하지 않음. 우리가 그린 그래프.",
        transform=ax.transAxes, fontsize=7.2, color=GREY)
save(fig, "own_pusht.png")

# ---------------------------------------------------------------- 실로봇 (V3)
tasks = ["Close Drawer", "Place Mandarin\non Green Plate", "Place Mandarin\non Red Plate"]
methods = [("무작위", "Random", "#8a8f98"), ("ResNet50", "IRASim (ResNet50)", "#9ecae1"), ("MSE", "IRASim (MSE)", "#08306b")]
rates = {
    "Random": [3, 1, 2],
    "IRASim (ResNet50)": [9, 11, 9],
    "IRASim (MSE)": [13, 12, 13],
}
from math import sqrt


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, ax = plt.subplots(figsize=(7.4, 4.4))
width = 0.25
for mi, (lab, key, col) in enumerate(methods):
    xs = [i + (mi - 1) * width for i in range(len(tasks))]
    ks = rates[key]
    ps = [k / 15 for k in ks]
    lo = [p - wilson(k, 15)[0] for p, k in zip(ps, ks)]
    hi = [wilson(k, 15)[1] - p for p, k in zip(ps, ks)]
    ax.bar(xs, ps, width=width, color=col, label=lab, yerr=[lo, hi], capsize=3, ecolor=GREY)
    for x, p, k in zip(xs, ps, ks):
        ax.text(x, p + 0.05, f"{k}/15", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(tasks, fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("성공률 (15회, Wilson 95%)", fontsize=9, color=GREY)
ax.set_title("실로봇 계획 성공률 (V3, 셀당 15회로 역산)", fontsize=10.5, color=INK, loc="left")
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
style(ax)
ax.text(0.0, -0.25, "셀당 시행 수 15는 표 6의 값에서 역산한 것이며, 논문에 명시되지 않았다. 우리가 그린 그래프.",
        transform=ax.transAxes, fontsize=7.5, color=GREY)
save(fig, "own_real.png")
