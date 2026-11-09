"""리포트용 그래프 네 장. 먼저 table_check.py와 cost_model.py를 실행해 결과 JSON을 만든다.
실행: python3 analysis/plot_figures.py

1) arch_ratio.png      : 메모리 한계 모델의 16K 대 256K 비용 비율과 s0/k(요청당 상태 ÷ 토큰당 KV), 관측 3.6·5.6·7배
2) r_sensitivity.png   : EORC 절감 배수가 r에 따라 어떻게 줄어드는지 (r = 1이면 지금의 평탄 단가)
3) cost_three_ways.png : 설정마다 평탄 단가 비율, 논문 EORC 절감, 같은 규칙(기준선도 실제 길이로) 절감
4) accuracy_delta.png  : 장기 과제 22개 설정의 정확도 차이와 95% 구간
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TC = json.loads((HERE / "table_check_results.json").read_text())
CM = json.loads((HERE / "cost_model_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# ---------------------------------------------------------------- 1) 구조별 비율
fig, ax = plt.subplots(figsize=(8.6, 4.4))
x = np.linspace(0, 64_000, 400)
ratio = (x + 131_584) / (x + 8_704)
ax.plot(x / 1000, ratio, color=INK, lw=1.6, label="메모리 한계 모델: (s0 + k·128K) / (s0 + k·8K)")
pts = [("순수 어텐션", 0, 15.12, INK), ("Qwen3-Next 구조 가정\n(DeltaNet 상태 FP32)", 3.144, 11.37, ORANGE),
       ("Nemotron 3.5 Super VL\nBF16 KV · SSM FP16", 10.54, 7.39, BLUE),
       ("Super VL FP8 KV · SSM FP16", 21.08, 5.13, BLUE), ("Super VL FP8 KV · SSM FP32", 41.56, 3.44, BLUE)]
for lab, xx, yy, c in pts:
    ax.plot(xx, yy, "o", color=c, ms=6, zorder=5)
    dy = -14 if "FP16" in lab and "FP8" in lab else 4
    ax.annotate(lab, (xx, yy), xytext=(6, dy), textcoords="offset points", fontsize=8, color=c)
for name, ob, c in (("GLM 5.2 관측", 3.61, MUTED), ("Kimi K2.6 관측", 5.63, MUTED), ("Qwen 3 Coder Next 관측", 6.95, ORANGE)):
    ax.axhline(ob, color=c, lw=0.9, ls="--")
    ax.text(63.5, ob + 0.15, f"{name} {ob:.1f}배", ha="right", fontsize=8, color=INK2)
ax.axhline(1, color=AQUA, lw=1.2)
ax.text(63.5, 1.2, "순수 SSM 1배", ha="right", fontsize=8, color=AQUA)
ax.set_xlabel("s0 / k  (요청당 고정 상태를 KV 토큰 수로 환산, 천 토큰)")
ax.set_ylabel("256K 창 대 16K 창 토큰당 비용 비율")
ax.set_ylim(0, 17)
ax.set_xlim(-1, 64)
ax.set_title("같은 '창 16배 축소'라도 구조에 따라 1–15배. 관측값은 메모리 한계보다 낮음", loc="left", fontsize=10)
style(ax)
ax.legend(frameon=False, loc="upper center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "arch_ratio.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2) r 민감도
r_vals = CM["M4_r_sensitivity"]["r_values"]
sv = CM["M4_r_sensitivity"]["savings"]
order = np.argsort(r_vals)
rs = np.array(r_vals)[order]
fig, ax = plt.subplots(figsize=(8.6, 4.2))
for key, m, c, ls in (("SWE-bench Verified 48K", "opus", BLUE, "-"), ("SWE-bench Verified 24K", "opus", BLUE, "--"),
                      ("SWE-bench Verified 48K", "gpt", ORANGE, "-"), ("SWE-bench Verified 24K", "gpt", ORANGE, "--"),
                      ("FRAMES 16K", "opus", INK2, ":"), ("FRAMES 16K", "gpt", MUTED, ":")):
    y = np.array(sv[key][m])[order]
    ax.plot(rs, y, ls, marker="o", ms=3.5, color=c, lw=1.4, label=f"{key} · {'Opus' if m == 'opus' else 'GPT'}")
ax.axhline(1, color=INK, lw=0.8)
ax.axvline(0.20, color=MUTED, lw=0.8, ls="--")
ax.text(0.205, 11.0, "논문 r = 0.20", fontsize=8, color=INK2)
ax.axvspan(0.19, 0.29, color=BLUE, alpha=0.08)
ax.text(0.30, 9.6, "Super VL형 하이브리드\n메모리 한계 r 0.19–0.29", fontsize=8, color=BLUE)
ax.text(0.98, 0.3, "r = 1: 지금의 평탄 단가", ha="right", fontsize=8, color=INK2)
ax.set_ylim(0, 11.6)
ax.set_xlabel("r = 16K 창 비용 / 최대 창 비용")
ax.set_ylabel("기준선 대비 비용 절감 배수")
ax.set_title("절감 배수의 대부분은 토큰 감소가 아니라 가정한 r에서 나옴", loc="left", fontsize=10)
style(ax)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "r_sensitivity.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3) 세 가지 비용 셈
m5 = CM["M5_same_rule"]["rows"]
lh = TC["C4_savings"]["long_horizon"]
abbr = {"SWE-bench Verified": "SWE-V", "SWE-bench Multilingual": "SWE-M", "Terminal-Bench 2.1": "TB", "FRAMES": "FRAMES"}
fig, axs = plt.subplots(1, 2, figsize=(9.8, 5.2), sharey=True)
for ax, m, title in ((axs[0], "opus", "Claude Opus 4.5"), (axs[1], "gpt", "GPT 5.6 Sol")):
    keys = [k for k in m5 if k.endswith(m)]
    ys = np.arange(len(keys))[::-1]
    for y, k in zip(ys, keys):
        bench_w = k.rsplit(" ", 1)[0]
        nom = lh[bench_w][f"{m}_nominal_ratio"]
        pap = m5[k]["paper_savings"]
        same = m5[k]["same_rule_savings"]
        ax.plot([min(nom, same, pap), max(nom, same, pap)], [y, y], color=GRID, lw=2, zorder=1)
        ax.plot(nom, y, "s", color=MUTED, ms=5, zorder=3, label="평탄 단가(명목)" if y == ys[0] else None)
        ax.plot(same, y, "D", color=AQUA, ms=5, zorder=4, label="같은 규칙(기준선도 실제 길이)" if y == ys[0] else None)
        ax.plot(pap, y, "o", color=BLUE if m == "opus" else ORANGE, ms=5.5, zorder=5, label="논문 EORC" if y == ys[0] else None)
    b, w = zip(*[k.rsplit(" ", 2)[:2] for k in keys])
    ax.set_yticks(ys, [f"{abbr[bb]} {ww}" for bb, ww in zip(b, w)])
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax.set_xlabel("기준선 대비 비용 절감 배수 (1 미만 = 압축이 더 비쌈)")
    ax.set_title(title, loc="left", fontsize=10)
    style(ax)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, ncol=3, loc="upper right", bbox_to_anchor=(0.99, 0.995))
fig.suptitle("같은 토큰 합계라도 값 매기는 규칙에 따라 0.5-8.8배", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "cost_three_ways.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4) 정확도 차이와 95% 구간
N = {"SWE-bench Verified": 500, "SWE-bench Multilingual": 300, "Terminal-Bench 2.1": 89, "FRAMES": 824,
     "BFCLv3": 200, "tau2-bench": 184}
import sys
sys.path.insert(0, str(HERE))
from table_check import LH  # noqa: E402  표 값은 한 곳에서만 관리
ACC = {}
for bench, (n, data) in LH.items():
    b = data[0]
    for r in data[1:]:
        for m, i in (("opus", 1), ("gpt", 2)):
            p1, p2 = b[i] / 100, r[i] / 100
            ACC[(bench, r[0], m)] = (round(r[i] - b[i], 1), 100 * math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n))
fig, ax = plt.subplots(figsize=(9.2, 5.6))
labels = []
y = 0
for bench in N:
    ws = sorted({k[1] for k in ACC if k[0] == bench}, reverse=True)
    for w in ws:
        for m, c, off in (("opus", BLUE, 0.18), ("gpt", ORANGE, -0.18)):
            d, se = ACC[(bench, w, m)]
            ax.errorbar(d, y + off, xerr=1.96 * se, fmt="o", color=c, ms=4, lw=1, capsize=2,
                        label=("Opus" if m == "opus" else "GPT") if y == 0 else None)
        labels.append((y, f"{abbr.get(bench, bench.replace('tau2-bench', 'τ²'))} {w}K"))
        y -= 1
    y -= 0.5
ax.axvline(0, color=INK, lw=0.8)
ax.axvline(-2, color=MUTED, lw=0.8, ls="--")
ax.text(-2.1, 0.6, "-2%p", ha="right", fontsize=8, color=INK2)
ax.set_yticks([l[0] for l in labels], [l[1] for l in labels], fontsize=8)
ax.set_xlabel("정확도 차이 (압축 - 기준선, %p). 막대 = 비대응 두 비율 95% 구간, 단일 실행")
ax.set_title("22개 설정 대부분이 0을 포함. 유의한 것은 GPT SWE-V 48K(+4.4)와 GPT FRAMES 24K·16K(-3.3·-3.1)", loc="left", fontsize=9.5)
style(ax)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "accuracy_delta.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png") if not p.name.startswith("paper_")))
