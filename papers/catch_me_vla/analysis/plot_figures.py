"""재계산 결과로 그래프를 그린다 (CPU, 수 초). 논문 그림은 쓰지 않는다.

입력: analysis/recompute_results.json, analysis/toy_feedback_param_results.json (먼저 각 스크립트를 실행할 것)
출력: report/figures/own_sim_ci.png, own_real_ablation.png, own_latency.png, own_toy_param.png
실행: python3 analysis/plot_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
T = json.loads((ROOT / "analysis" / "toy_feedback_param_results.json").read_text(encoding="utf-8"))

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, LINE = "#1b1b1a", "#8a877f", "#dcdad3"
C = {"gr00t": "#b9b5ab", "fb": "#1f5fae", "warn": "#c2410c", "alt": "#7a9cc6"}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=LINE, linewidth=0.6)
    ax.set_axisbelow(True)


def bars_with_ci(ax, labels, ks, ns, color, offset, width):
    xs = [i + offset for i in range(len(labels))]
    ys = [100 * k / n for k, n in zip(ks, ns)]
    lo = [wilson(k, n)[0] for k, n in zip(ks, ns)]
    hi = [wilson(k, n)[1] for k, n in zip(ks, ns)]
    ax.bar(xs, ys, width, color=color, edgecolor="none")
    ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]], fmt="none",
                ecolor=INK, elinewidth=0.8, capsize=2)


# 1) 시뮬레이션 동적 과제: 표 1, 과제당 40회, Wilson 95%
rows = [r for r in R["gr00t_vs_vlafb"] if r["n"] == 40]
labels = [r["col"].replace("*", "") for r in rows]
kg = [int(r["gr00t"].split("/")[0]) for r in rows]
kf = [int(r["vlafb"].split("/")[0]) for r in rows]
fig, ax = plt.subplots(figsize=(10, 3.9))
w = 0.38
bars_with_ci(ax, labels, kg, [40] * len(rows), C["gr00t"], -w / 2, w)
bars_with_ci(ax, labels, kf, [40] * len(rows), C["fb"], w / 2, w)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(["Pick 1D", "Pick 2D", "Drop Ball", "Catch Block\n(미학습)", "1D Speed\n(미학습)", "Ball Color\n(미학습)", "Ball Speed\n(미학습)"], fontsize=8.5)
ax.set_ylim(0, 112)
ax.set_ylabel("성공률 (%)", color=INK)
ax.set_title("동적 시뮬레이션 과제: GR00T 대 VLA-Feedback (과제당 40회, Wilson 95%)", color=INK, fontsize=10)
ax.bar([0], [0], color=C["gr00t"], label="GR00T")
ax.bar([0], [0], color=C["fb"], label="VLA-Feedback")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
style(ax)
fig.text(0.01, 0.01, "막대: 논문 표 1 값. Catch Block(p=0.64)만 두 방법을 구분하기 어렵다. 우리가 그린 그래프.", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "own_sim_ci.png", dpi=170)
plt.close(fig)

# 2) 실로봇(그림 4 막대 읽기)과 절제(그림 5 막대 읽기)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1.0, 1.1]})
tasks = ["Pick bread\n(정적)", "Drop lemon", "Catch can"]
kg = [R["real_bars"]["GR00T"]["counts"][t] for t in ("bread", "lemon", "can")]
kf = [R["real_bars"]["VLA-Feedback"]["counts"][t] for t in ("bread", "lemon", "can")]
bars_with_ci(a1, tasks, kg, [20] * 3, C["gr00t"], -0.19, 0.38)
bars_with_ci(a1, tasks, kf, [20] * 3, C["fb"], 0.19, 0.38)
a1.set_xticks(range(3))
a1.set_xticklabels(tasks, fontsize=8.5)
a1.set_ylim(0, 125)
a1.set_ylabel("성공률 (%)", color=INK)
a1.set_title("실로봇 (과제당 20회, Wilson 95%)", color=INK, fontsize=10)
a1.axhline(51, color=C["warn"], linewidth=1, linestyle="--", label="보고된 GR00T 평균 51")
a1.axhline(R["real_bars"]["GR00T"]["mean_of_three"], color=MUTED, linewidth=1, linestyle=":", label="GR00T 세 값의 평균 53.3")
a1.legend(frameon=False, fontsize=8, loc="upper right")
style(a1)
abl = R["fig5a_dynamic_static"]
names = ["GR00T\n(기준)", "No Obs", "Δa\n잔차", "Trans→MLP", "VLA-\nFeedback"]
dyn = [R["fig5_checks"]["gr00t_dyn3_mean"], abl["No Obs"][0], abl["Delta-a (direct residual)"][0], abl["Transformer->MLP"][0], abl["VLA-Feedback"][0]]
cols = [C["gr00t"], C["alt"], C["alt"], C["alt"], C["fb"]]
a2.bar(range(5), dyn, 0.6, color=cols, edgecolor="none")
for i, v in enumerate(dyn):
    a2.text(i, v + 1.5, f"{v:.1f}" if i == 0 else f"{v:.0f}", ha="center", fontsize=8.5, color=INK)
a2.set_xticks(range(5))
a2.set_xticklabels(names, fontsize=8.5)
a2.set_ylim(0, 100)
a2.set_title("동적 과제 평균 성공률: 절제 (그림 5a 막대 읽기)", color=INK, fontsize=10)
style(a2)
fig.text(0.01, 0.01, "실로봇·절제 막대는 PDF의 눈금 기준 읽은 값(±0.5 %p). GR00T 기준은 표 1의 세 과제 평균. 우리가 그린 그래프.", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "own_real_ablation.png", dpi=170)
plt.close(fig)

# 3) 반응 지연: 10 Hz에서 T_update + R/2 dt, 하드웨어 감지·통신 69 ms를 더하면
t2 = R["table2_at_10Hz"]
names = ["OpenVLA", "GR00T", "FiS-VLA", "VLA-Feedback"]
tu = [t2[n]["T_update_ms"] for n in names]
wait = [t2[n]["R"] / 2 * 100 for n in names]
sense = R["table5"]["hw_non_inference_ms"]
fig, ax = plt.subplots(figsize=(8.8, 3.9))
x = range(len(names))
ax.bar(x, tu, 0.55, color=C["fb"], label="모듈 실행 시간 T_update")
ax.bar(x, wait, 0.55, bottom=tu, color=C["alt"], label="갱신 대기 R/2·Δt (Δt=100 ms)")
ax.bar(x, [sense] * 4, 0.55, bottom=[a + b for a, b in zip(tu, wait)], color=C["gr00t"], alpha=0.8,
       label="하드웨어 감지·통신 69 ms (표 5, 식 (9)에는 없음)")
for i, (a, b) in enumerate(zip(tu, wait)):
    ax.text(i, a + b + sense + 18, f"식 (9): {a + b:.0f} ms", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 1100)
ax.set_ylabel("ms", color=INK)
ax.set_title("기대 반응 지연: 식 (9)와 하드웨어 감지 지연을 더한 값", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper right")
style(ax)
fig.text(0.01, 0.01, "표 2의 T_update·R, 표 5의 합계를 다시 계산해 쌓았다. 감지·통신 69 ms는 방법과 무관하다고 가정. 우리가 그린 그래프.", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "own_latency.png", dpi=170)
plt.close(fig)

# 4) 장난감: dk에 따른 최적 오차와 Adam 오차, 관측 유무
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
dks = ["dk=0.25", "dk=0.5", "dk=1.0"]
cf = [T["closed_form_mse"][k] * 1e6 for k in dks]
ad = [T["adam_mse_300steps"][k] * 1e6 for k in dks]
x = range(3)
a1.bar([i - 0.19 for i in x], cf, 0.38, color=C["fb"], label="닫힌형 최적해")
a1.bar([i + 0.19 for i in x], ad, 0.38, color=C["alt"], label="Adam 300 스텝")
a1.set_xticks(list(x))
a1.set_xticklabels(["Δk=0.25", "Δk=0.5", "Δk=1.0 (잔차)"], fontsize=8.5)
a1.set_ylabel("시험 MSE (1e-6 m²)", color=INK)
a1.set_title("선형 머리: Δk가 달라도 최적해는 같다", color=INK, fontsize=10)
a1.legend(frameon=False, fontsize=8)
style(a1)
vals = [T["planner_only_mse"] * 1e6, T["closed_form_mse_no_obs"]["dk=0.5"] * 1e6, T["closed_form_mse"]["dk=0.5"] * 1e6]
a2.bar(range(3), vals, 0.55, color=[C["gr00t"], C["alt"], C["fb"]])
for i, v in enumerate(vals):
    a2.text(i, v * 1.03, f"{v:.0f}" if v > 10 else f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
a2.set_xticks(range(3))
a2.set_xticklabels(["계획기만\n(낡은 청크)", "피드백\n관측 없음", "피드백\n관측 사용"], fontsize=8.5)
a2.set_title("관측 없는 모듈도 계획기 편향을 줄인다", color=INK, fontsize=10)
style(a2)
fig.text(0.01, 0.01, "1차원 낡은 청크 장난감(수 초). 식 (6), (8)의 대수적 성질만 확인하며 VLA 결과를 재현하지 않는다. 우리가 그린 그래프.", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "own_toy_param.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
