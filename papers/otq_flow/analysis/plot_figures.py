"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real.png     : Fig. 5(벡터 판독값) 곡선과 base 선, 표 2 비율과 Wilson 95% 구간
own_offline.png  : 표 1에서 OTQL과 QAM-F의 환경별 차이, 방법별 평균
own_online.png  : Fig. 2·3 벡터 판독값으로 다시 그린 OTQL 대 기준선 최종값·최고값
own_toy.png      : 장난감(2D two-moons)의 NFE별 분포 거리와 배치 유효 크기
실행: python3 plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
입력: curves_extracted.json, recompute_results.json, toy_wcot_results.json
"""
import json
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


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#b5372a"
cur = json.load(open(ANA / "curves_extracted.json"))
rec = json.load(open(ANA / "recompute_results.json"))
toy = json.load(open(ANA / "toy_wcot_results.json"))


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---- own_real.png
fig, ax = plt.subplots(1, 4, figsize=(13, 3.1), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
for a, (t, v) in zip(ax[:3], cur["fig5"].items()):
    x = [0, 10, 20, 30, 40, 50]
    a.plot(x, v["otql"], "o-", color=ACC, label="OTQL")
    a.axhline(v["base"], color=RED, ls="--", label="base")
    for xi, yi in zip(x, v["otql"]):
        lo, hi = wilson(round(yi * 10), 10)
        a.plot([xi, xi], [lo, hi], color=ACC, alpha=0.25, lw=4)
    a.set_title(t, fontsize=10); a.set_ylim(0, 1.02); a.set_xlabel("에피소드")
    a.grid(alpha=0.25)
ax[0].set_ylabel("성공률 (판독값, 10회 가정)")
ax[0].legend(fontsize=8, loc="lower right")
b = ax[3]
names = ["tape in\npenholder", "move\nknight", "합계"]
sft = [14, 9, 23]; ot = [24, 22, 46]; n = [30, 30, 60]
xs = np.arange(3)
for i, (s, o, m) in enumerate(zip(sft, ot, n)):
    for off, k, col in [(-0.18, s, GREY), (0.18, o, ACC)]:
        lo, hi = wilson(k, m)
        b.bar(i + off, k / m, 0.32, color=col, yerr=[[k / m - lo], [hi - k / m]], capsize=3, ecolor=INK)
        b.text(i + off, hi + 0.02, f"{k}/{m}", ha="center", fontsize=8)
b.set_xticks(xs); b.set_xticklabels(names, fontsize=9); b.set_ylim(0, 1.1)
b.set_title("표 2 SmolVLA (회색 SFT, 파랑 OTQL)", fontsize=10); b.grid(alpha=0.25, axis="y")
fig.suptitle("실로봇: Fig. 5 곡선과 표 2 (우리가 다시 그림)", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "own_real.png", dpi=170); plt.close(fig)

# ---- own_offline.png
envs = ["antmaze-L", "antmaze-G", "humanoid-M", "humanoid-L", "cube-2", "cube-3", "scene", "puzzle-3x3"]
T = {
    "OTQL": [0.88, 0.19, 0.82, 0.19, 0.68, 0.04, 0.99, 0.99],
    "QAM-F": [0.83, 0.12, 0.65, 0.12, 0.65, 0.03, 0.95, 0.99],
    "FQL": [0.76, 0, 0.68, 0.09, 0.46, 0.03, 0.78, 0.7],
    "QAM": [0.81, 0.18, 0.67, 0.11, 0.64, 0.03, 0.97, 1.0],
}
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.4))
d = np.array(T["OTQL"]) - np.array(T["QAM-F"])
ax[0].bar(range(8), d, color=[ACC if x > 0 else GREY for x in d])
ax[0].axhline(0, color=INK, lw=0.8)
ax[0].set_xticks(range(8)); ax[0].set_xticklabels(envs, rotation=35, ha="right", fontsize=8)
ax[0].set_ylabel("OTQL - QAM-F (표 1)"); ax[0].set_title("환경별 차이 (7곳 +, 1곳 동률)", fontsize=10); ax[0].grid(alpha=0.25, axis="y")
avg = rec["V1_table1_avg"]
order = sorted(avg, key=lambda k: -avg[k]["recomputed"])
ax[1].barh(order[::-1], [avg[k]["recomputed"] for k in order[::-1]], color=[ACC if k == "OTQL" else GREY for k in order[::-1]])
for i, k in enumerate(order[::-1]):
    ax[1].text(avg[k]["recomputed"] + 0.005, i, f'{avg[k]["recomputed"]:.3f} (논문 {avg[k]["reported"]:.2f})', va="center", fontsize=8)
ax[1].set_xlim(0, 0.8); ax[1].set_title("8환경 평균 재계산 (논문 값은 버림 표기)", fontsize=10); ax[1].grid(alpha=0.25, axis="x")
fig.tight_layout(); fig.savefig(FIG / "own_offline.png", dpi=170); plt.close(fig)

# ---- own_online.png
fig, ax = plt.subplots(1, 4, figsize=(13, 3.0))
panels = [("fig2", "cube-double-task4", "cube-double task4"), ("fig2", "humanoidmaze-task1", "humanoidmaze task1"),
          ("fig3", "OpenMicrowave-v0", "OpenMicrowave"), ("fig3", "ArrangeBoxes-v0", "ArrangeBoxes")]
cols = {"OTQL": ACC, "DSRL": "#555", "QAM-F": "#888", "FQL": "#bbb", "QC": "#555", "EWFM": "#999"}
for a, (f, k, t) in zip(ax, panels):
    for m, s in cur[f][k].items():
        v = np.array(s["values"]); x = np.linspace(0, 100 if f == "fig2" else 8.0, len(v))
        a.plot(x, v, "-", color=cols[m], lw=2.2 if m == "OTQL" else 1.2, label=m)
    a.set_title(t, fontsize=10); a.set_ylim(-0.02, 1.02); a.grid(alpha=0.25)
    a.set_xlabel("x축 (논문 Fig. 2는 x1k 표기, Fig. 3은 x1k 표기)" if False else "환경 스텝 (x1k, 논문 표기)", fontsize=8)
    a.legend(fontsize=7)
ax[0].set_ylabel("성공률 (벡터 판독값)")
fig.suptitle("Fig. 2·3 곡선을 PDF 벡터에서 읽어 다시 그림 (음영 구간은 생략)", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "own_online.png", dpi=170); plt.close(fig)

# ---- own_toy.png
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3))
nf = [1, 2, 3, 100]
for m, col, lab in [("ewfm", GREY, "EWFM (독립 결합 + 가중 손실)"), ("wcot", ACC, "wCOT (가중 질량 OT 결합)")]:
    s = toy["summary_mean_std_over_3_seeds"][m]
    mu = [s[f"swd_nfe{k}"][0] for k in nf]; sd = [s[f"swd_nfe{k}"][1] for k in nf]
    ax[0].errorbar(range(4), mu, yerr=sd, marker="o", color=col, label=lab, capsize=3)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["1", "2", "3", "100"]); ax[0].set_yscale("log")
from matplotlib.ticker import FixedLocator, FixedFormatter
ax[0].yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1.0])); ax[0].yaxis.set_major_formatter(FixedFormatter(["0.1", "0.2", "0.5", "1.0"])); ax[0].yaxis.set_minor_locator(FixedLocator([]))
ax[0].set_xlabel("Euler 스텝 수 (NFE)"); ax[0].set_ylabel("슬라이스 W2 (기울인 목표와의 거리)")
ax[0].set_title("장난감: 2D two-moons, 시드 3개 평균 ± SD", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
ess = toy["ess_table_batch256"]
for s_, col in [(0.1, "#9bb7d9"), (0.3, "#5d8bc4"), (1.0, ACC)]:
    ls = [0.4, 1.0, 2.0, 5.0]
    ax[1].plot(ls, [ess[f"adv_std={s_},lambda={l}"]["ess_noclamp"] for l in ls], "o-", color=col, label=f"어드밴티지 SD {s_}")
    ax[1].plot(ls, [ess[f"adv_std={s_},lambda={l}"]["ess_wmax10"] for l in ls], "x--", color=col)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks([0.4, 1, 2, 5]); ax[1].set_xticklabels(["0.4", "1", "2", "5"]); ax[1].minorticks_off(); ax[1].set_yticks([3, 10, 30, 100, 250]); ax[1].set_yticklabels(["3", "10", "30", "100", "250"]); ax[1].set_xlabel("λ"); ax[1].set_ylabel("배치 256의 유효 크기 1/Σp²")
ax[1].set_title("가중치 집중: 실선 클램프 없음, 점선 w_max=10", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIG / "own_toy.png", dpi=170); plt.close(fig)
print("saved", [p.name for p in FIG.glob("own_*.png")])
