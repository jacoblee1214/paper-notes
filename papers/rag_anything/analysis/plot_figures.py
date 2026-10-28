"""리포트용 그래프 세 장 (논문 수치와 그림 판독값으로 직접 그림). 실행: cd analysis && python3 plot_figures.py

1) overall_consistency.png: 표의 Overall vs 영역별 정확도를 문항 수로 가중 평균한 값 (Table 2·3·4)
2) docbench_gap.png: DocBench에서 RAG-Anything − 기준선 차이 (영역, 유형, Overall)
3) length_bins_ci.png: Fig. 2 구간별 정확도와 Wilson 95% 구간 (그림 픽셀 판독값)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import table_check as tc

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


def style(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) Overall 정합성
rows = []
for name, (dom, _, ov) in tc.T2.items():
    rows.append((f"DocBench · {name}", ov, tc.wavg(dom, tc.DOC_Q)))
for name, (dom, ov) in tc.T3.items():
    rows.append((f"MMLongBench · {name}", ov, tc.wavg(dom, tc.MML_Q)))
for name, (dom, _, ov) in tc.T4.items():
    if name != "RAGAnything":
        rows.append((f"DocBench 절제 · {name}", ov, tc.wavg(dom, tc.DOC_Q)))
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
ys = np.arange(len(rows))[::-1]
for y, (label, rep, w) in zip(ys, rows):
    bad = abs(rep - w) > 1.0
    c = ORANGE if bad else BLUE
    ax.plot([w, rep], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter([w], [y], s=40, facecolor="white", edgecolor=c, linewidth=2, zorder=3)
    ax.scatter([rep], [y], s=40, color=c, zorder=3)
    if bad:
        ax.annotate(f"표 {rep:.1f} vs 영역 가중 {w:.1f}", (min(w, rep), y), xytext=(-6, 8), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="right")
ax.set_yticks(ys, [r[0] for r in rows], fontsize=7.5)
ax.set_xlabel("정확도 (%)", color=INK2)
h1 = ax.scatter([], [], s=40, color=INK2, label="표의 Overall")
h2 = ax.scatter([], [], s=40, facecolor="white", edgecolor=INK2, linewidth=2, label="영역별 정확도의 문항 가중 평균")
ax.legend(handles=[h1, h2], frameon=False, fontsize=7.5, loc="upper left")
style(ax, "x")
fig.text(0.01, 0.01, "출처: Table 2·3·4와 문항 수 Table 5·6. analysis/table_check.py C1", fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "overall_consistency.png")
print("wrote", OUT / "overall_consistency.png")

# 2) DocBench 차이
cats = tc.DOC_DOMAINS + ["Txt", "Mm", "Una", "Overall"]
ra = tc.T2["RAGAnything"]
vals = {}
for base in ("LightRAG", "MMGraphRAG"):
    b = tc.T2[base]
    vals[base] = [a - c for a, c in zip(ra[0] + ra[1] + [ra[2]], b[0] + b[1] + [b[2]])]
fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=200)
x = np.arange(len(cats))
wd = 0.36
for i, (base, c) in enumerate((("LightRAG", BLUE), ("MMGraphRAG", ORANGE))):
    bars = ax.bar(x + (i - 0.5) * wd, vals[base], width=wd - 0.04, color=c, label=f"RAG-Anything - {base}")
    for xi, v in zip(x + (i - 0.5) * wd, vals[base]):
        ax.text(xi, v + (0.6 if v >= 0 else -0.6), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5, color=INK2)
ax.axhline(0, color=INK2, lw=0.8)
ax.axvline(4.5, color=GRID, lw=1)
ax.axvline(7.5, color=GRID, lw=1)
ax.set_xticks(x, cats)
ax.set_ylabel("정확도 차이 (%p)", color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
style(ax, "y")
fig.text(0.01, 0.01, "출처: Table 2 (DocBench, GPT-4o-mini 판정). 영역 | 유형 | Overall. MMGraphRAG Overall 61.0은 영역 가중 평균 56.7과 불일치(C1)",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "docbench_gap.png")
print("wrote", OUT / "docbench_gap.png")

# 3) 길이 구간별 정확도와 Wilson 구간
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200, sharey=True)
for ax, bench in zip(axes, ("DocBench", "MMLongBench")):
    d = tc.FIG2[bench]
    cnt = np.array(d["count"], dtype=float)
    if bench == "MMLongBench":  # 작은 구간은 정확도가 k/n이 되는 n으로 (C5)
        cnt[0], cnt[4] = 16, 5
    xs = np.arange(len(tc.BINS))
    for off, m, c in ((-0.12, "RAGAnything", BLUE), (0.12, "MMGraphRAG", ORANGE)):
        acc = np.array(d[m])
        lo, hi = zip(*[tc.wilson(round(a / 100 * n), int(n)) for a, n in zip(acc, cnt)])
        ax.errorbar(xs + off, acc, yerr=[acc - np.array(lo), np.array(hi) - acc], fmt="o", color=c, ms=5,
                    elinewidth=1.4, capsize=2.5, label=m)
    ax.set_xticks(xs, [f"{b}\n(n≈{int(n)})" for b, n in zip(tc.BINS, cnt)], fontsize=7.5)
    ax.set_title(bench, fontsize=10, color=INK)
    ax.set_ylim(0, 100)
    style(ax, "y")
axes[0].set_ylabel("정확도 (%) · 95% Wilson 구간", color=INK2)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.text(0.01, 0.01, "출처: Fig. 2를 다시 렌더링해 픽셀로 판독(정확도 ±0.3%p, 문항 수 ±2). MMLongBench 1–10·200+ 구간 n은 정확도가 k/n이 되는 값",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "length_bins_ci.png")
print("wrote", OUT / "length_bins_ci.png")
