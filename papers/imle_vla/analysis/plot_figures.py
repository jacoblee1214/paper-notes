"""IMLE-VLA 직접 그린 그래프 네 장 (CPU). 실행: python3 analysis/plot_figures.py

1) success_ci.png: Table II(LIBERO)와 Table V(실물) 성공률 차이와 Wilson 95% 구간 (IMLE 대 인용 π0.5)
2) latency_decomp.png: 빈도별 패스 시간, 3.67배의 엔진·헤드 분해, 접두부 P에 따른 헤드 배율
3) toy_scatter.png: 4봉우리 toy에서 시드 0의 생성 샘플(방법별), 봉우리 표시
4) toy_m_sweep.png: cIMLE m에 따른 봉우리 사이 비율과 덮음률, 흐름 1·10단계 기준선
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, GRID = "#0b0b0b", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
T = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
TOY = json.loads((ROOT / "analysis" / "imle_toy_results.json").read_text())
SAMP = np.load(ROOT / "analysis" / "imle_toy_samples.npz")
SIZE = (6.5, 3.6)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1) 성공률 차이와 구간
t1 = T["T1_table2"]["IMLE H10 vs pi0.5 H10 (cited)"]
rows = [(r["suite"], r["imle_k"], r["pi05_k"], 500) for r in t1["rows"]]
rows.append(("LIBERO 전체", t1["pooled_imle_k"], t1["pooled_pi05_k"], 2000))
t5 = T["T4_table5"]["rows"]
for r in t5:
    ka = int(r["imle"].split("/")[0])
    kb = int(r["pi05"].split("/")[0])
    rows.append((r["task"], ka, kb, 20))
fig, ax = plt.subplots(figsize=SIZE)
labels, diffs, los, his, cols = [], [], [], [], []
for name, ka, kb, n in rows:
    pa, pb = ka / n, kb / n
    # 차이의 Wilson 구간 대신 두 구간의 겹침을 보기 쉽게: 차이 점과 두 점의 Newcombe 근사 구간
    la, ha = wilson(ka, n)
    lb, hb = wilson(kb, n)
    d = pa - pb
    lo = d - math.sqrt((pa - la) ** 2 + (hb - pb) ** 2)
    hi = d + math.sqrt((ha - pa) ** 2 + (pb - lb) ** 2)
    labels.append(name)
    diffs.append(100 * d)
    los.append(100 * lo)
    his.append(100 * hi)
    cols.append(ORANGE if n == 2000 else (BLUE if n == 500 else AQUA))
y = np.arange(len(rows))[::-1]
for yi, d, lo, hi, c in zip(y, diffs, los, his, cols):
    ax.plot([lo, hi], [yi, yi], color=c, lw=2)
    ax.plot(d, yi, "o", color=c, ms=5)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("IMLE - π0.5 (pp), Newcombe 95% 구간")
ax.set_title("LIBERO(H=10, 인용 π0.5)와 실물(과제당 20회)의 성공률 차이", fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "success_ci.png", dpi=200)
plt.close(fig)

# 2) 지연 분해
fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))
ax = axes[0]
names = ["π0.5 JAX", "PyTorch\n+compile", "Triton π0.5", "IMLE-VLA"]
hz = [15, 20, 25, 55]
ms_ = [1000 / h for h in hz]
ax.bar(names, ms_, color=[MUTED, MUTED, BLUE, ORANGE], width=0.6)
for i, (m, h) in enumerate(zip(ms_, hz)):
    ax.text(i, m + 1.2, f"{h} Hz\n{m:.1f} ms", ha="center", fontsize=7.5)
ax.set_ylabel("패스당 시간 (ms)")
ax.set_ylim(0, 85)
ax.tick_params(axis="x", labelsize=7.5)
ax.set_title("Table I: 빈도", fontsize=9)
style(ax)
ax = axes[1]
P, a = T["T6_decomp"]["P_ms"], T["T6_decomp"]["a_ms"]
ax.bar(["Triton π0.5\n(10단계)", "IMLE\n(1단계)"], [P + 10 * a, P + a], color=[BLUE, ORANGE], width=0.5, label="총합")
ax.bar(["Triton π0.5\n(10단계)", "IMLE\n(1단계)"], [P, P], color=INK, alpha=0.55, width=0.5, label="접두부 P")
ax.text(0, 40 + 1.2, "40 ms", ha="center", fontsize=7.5)
ax.text(1, P + a + 1.2, f"{P + a:.1f} ms", ha="center", fontsize=7.5)
ax.text(1, P / 2, f"P={P:.1f}", ha="center", color="white", fontsize=7.5)
ax.set_ylabel("청크당 시간 (ms)")
ax.set_ylim(0, 48)
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title("같은 엔진 가정의 분해", fontsize=9)
style(ax)
ax = axes[2]
sens = T["T6_sensitivity"]
Ps = [s["P_ms"] for s in sens]
sp = [s["speedup_x"] for s in sens]
grid = np.linspace(2, 32, 100)
ax.plot(grid, 40 / (grid + (40 - grid) / 10), color=BLUE, lw=1.6)
ax.plot(Ps, sp, "o", color=ORANGE, ms=4)
ax.axhline(3.67, color=MUTED, ls="--", lw=0.9)
ax.text(31, 3.8, "논문 3.67배(엔진 포함)", ha="right", fontsize=7, color=MUTED)
ax.axvline(P, color=INK, lw=0.6, ls=":")
ax.set_xlabel("접두부 P (ms), 총 40 ms 고정")
ax.set_ylabel("헤드 배율 (x)")
ax.set_title("접두부 비중에 따른 헤드 배율", fontsize=9)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "latency_decomp.png", dpi=200)
plt.close(fig)

# 3) 토이 샘플 산점도
MU = np.stack([np.cos(np.arange(4) * np.pi / 2), np.sin(np.arange(4) * np.pi / 2)], 1)
def skey(name):
    # imle_toy.py가 npz 키를 만드는 규칙과 같다
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")


panels = [("cIMLE m=2 (무작위 초기화)", "cIMLE m=2 (zd=2)"), ("cIMLE m=2 (흐름 가중치 초기화)", "cIMLE m=2 warm"),
          ("cIMLE m=5", "cIMLE m=5 (zd=2)"), ("cIMLE m=20", "cIMLE m=20 (zd=2)"),
          ("흐름 1단계", "flow 1-step"), ("흐름 10단계", "flow 10-step")]
fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2))
for ax, (title, key) in zip(axes.ravel(), panels):
    s = SAMP[skey(key)]
    ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.35, color=BLUE, lw=0)
    ax.scatter(MU[:, 0], MU[:, 1], s=40, marker="x", color=ORANGE, lw=1.2, zorder=3)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)
    style(ax)
fig.suptitle("4봉우리 링 목표에서 시드 0의 샘플 1000개 (x: 목표 봉우리 중심)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "toy_scatter.png", dpi=200)
plt.close(fig)

# 4) m 스윕
fig, ax = plt.subplots(figsize=SIZE)
ms = [5, 10, 20]
between = [TOY["summary"][f"cIMLE m={m} (zd=2)"]["between_frac"] for m in ms]
ax.errorbar(ms, [100 * b["mean"] for b in between], yerr=[100 * b["sd"] for b in between],
            marker="o", color=ORANGE, lw=1.6, capsize=3, label="cIMLE 봉우리 사이 비율")
for m, key in [(1, "L2 regression (m=1)"), (2, "cIMLE m=2 (zd=2)")]:
    b = TOY["summary"][key]["between_frac"]
    ax.errorbar([m], [100 * b["mean"]], yerr=[100 * b["sd"]], marker="o", color=ORANGE, capsize=3)
f1 = TOY["summary"]["flow 1-step"]["between_frac"]["mean"]
f10 = TOY["summary"]["flow 10-step"]["between_frac"]["mean"]
ax.axhline(100 * f1, color=MUTED, ls="--", lw=1)
ax.text(19.5, 100 * f1 + 2, "흐름 1단계", ha="right", fontsize=7.5, color=MUTED)
ax.axhline(100 * f10, color=AQUA, ls="--", lw=1)
ax.text(19.5, 100 * f10 + 2, "흐름 10단계", ha="right", fontsize=7.5, color=AQUA)
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20])
ax.set_xticklabels(["1", "2", "5", "10", "20"])
ax.set_xlabel("cIMLE 샘플 인자 m (로그 눈금, 시드 3개)")
ax.set_ylabel("봉우리 사이 샘플 (%)")
ax.set_ylim(-3, 110)
ax.set_title("m이 커질수록 봉우리 사이 샘플이 줄지만 m=2에서는 줄지 않음", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="center right")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "toy_m_sweep.png", dpi=200)
plt.close(fig)
print("wrote figures to", OUT)
