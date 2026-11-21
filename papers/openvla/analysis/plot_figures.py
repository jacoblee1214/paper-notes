"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다: 실사 사진이 들어 있음).

own_bridge_category.png : 표 4 범주별 성공률과 Wilson 95% 구간 (BridgeData V2, 네 방법)
own_franka_peft.png     : 표 7 Franka 평균(테이블탑, DROID)과 표 8 미세조정 방식별 평균, 이항 SE
own_quant.png           : 표 5(비차단)와 표 11(차단 제어)의 양자화 성공률과 Wilson 95% 구간
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, GREY = "#1b1b1a", "#55544f"
COL = {"RT-1-X": "#c9c6bd", "Octo": "#9a9890", "RT-2-X": "#6f8fb8", "OpenVLA": "#1f5fae"}


def wilson(p, n, z=1.959964):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def binom_se(p, n):
    return 100 * math.sqrt(p * (1 - p) / n)


# 1. 표 4 범주별 (범주당 과제 수 x 10회)
CATS = ["시각 일반화", "움직임 일반화", "물리 일반화", "의미 일반화", "언어 근거화"]
CAT_N = [50, 20, 30, 40, 30]
import json  # noqa: E402

RES = json.loads((Path(__file__).resolve().parent / "recompute_results.json").read_text(encoding="utf-8"))
CAT_KEYS = ["Visual", "Motion", "Physical", "Semantic", "Language"]
# 성공 횟수 = 범주별 성공률 (%) x 시도 수 / 100 (표 4 재계산 결과에서 가져옴)
CAT_K = {m: [RES["bridge_category_pct"][c][m] / 100 * n for c, n in zip(CAT_KEYS, CAT_N)] for m in COL}
fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200)
w = 0.2
x = np.arange(len(CATS))
for j, (m, ks) in enumerate(CAT_K.items()):
    ps = [k / n for k, n in zip(ks, CAT_N)]
    vals = [100 * p for p in ps]
    cis = [wilson(p, n) for p, n in zip(ps, CAT_N)]
    err = np.array([[v - lo for v, (lo, _) in zip(vals, cis)], [hi - v for v, (_, hi) in zip(vals, cis)]])
    ax.bar(x + (j - 1.5) * w, vals, w, yerr=err, color=COL[m], label=m, capsize=2, error_kw=dict(lw=0.8, ecolor=GREY))
ax.set_xticks(x)
ax.set_xticklabels(CATS, fontsize=9)
ax.set_ylabel("성공률 (%)", color=INK)
ax.set_ylim(0, 110)
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("BridgeData V2 WidowX, 범주별 (표 4 재합산, Wilson 95% 구간)", fontsize=10, pad=24, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_bridge_category.png")
plt.close(fig)

# 2. Franka 평균과 PEFT 방식별 평균 (두 패널)
FR_M = ["DP", "DP (matched)", "Octo", "OpenVLA scratch", "OpenVLA"]
FR_TAB = [48.5, 43.4, 43.4, 43.4, 67.2]
FR_DROID = [35.0, 26.7, 38.3, 21.7, 58.3]
FR_N_TAB, FR_N_DROID = 99, 30
PEFT_M = ["Full FT", "마지막 층만", "비전 동결", "샌드위치", "LoRA r=32", "LoRA r=64"]
PEFT_V = [69.7, 30.3, 47.0, 62.1, 68.2, 68.2]
PEFT_N = 33
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=200, gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axes[0]
x = np.arange(len(FR_M))
w = 0.38
e1 = [binom_se(v / 100, FR_N_TAB) for v in FR_TAB]
e2 = [binom_se(v / 100, FR_N_DROID) for v in FR_DROID]
ax.bar(x - w / 2, FR_TAB, w, yerr=e1, color="#1f5fae", capsize=2, label="테이블탑 (99회)", error_kw=dict(lw=0.8, ecolor=GREY))
ax.bar(x + w / 2, FR_DROID, w, yerr=e2, color="#c9c6bd", capsize=2, label="DROID (30회)", error_kw=dict(lw=0.8, ecolor=GREY))
ax.set_xticks(x)
ax.set_xticklabels(FR_M, fontsize=8, rotation=15)
ax.set_ylabel("평균 성공률 (%)", color=INK)
ax.set_ylim(0, 85)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Franka 평균 (표 7, 이항 SE)", fontsize=10, color=INK)
ax = axes[1]
x = np.arange(len(PEFT_M))
e = [binom_se(v / 100, PEFT_N) for v in PEFT_V]
ax.bar(x, PEFT_V, 0.6, yerr=e, color=["#1f5fae", "#c9c6bd", "#c9c6bd", "#9a9890", "#6f8fb8", "#6f8fb8"], capsize=2, error_kw=dict(lw=0.8, ecolor=GREY))
ax.set_xticks(x)
ax.set_xticklabels(PEFT_M, fontsize=7.5, rotation=25, ha="right")
ax.set_ylim(0, 85)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("미세조정 방식 (표 8, 33회, 이항 SE)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_franka_peft.png")
plt.close(fig)

# 3. 양자화: 표 5 (비차단, 80회)와 표 11 (차단, 80회)
Q_M = ["bfloat16", "int8", "int4"]
Q_NB = [71.25, 58.125, 71.875]  # 표 5 (57, 46.5, 57.5 / 80)
Q_B = [70.0, 74.375, 68.75]  # 표 11 (56, 59.5, 55 / 80)
fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=200)
x = np.arange(len(Q_M))
w = 0.36
for off, vals, lab, col in [(-w / 2, Q_NB, "비차단 제어 (표 5)", "#1f5fae"), (w / 2, Q_B, "차단 제어 (표 11)", "#c9c6bd")]:
    ps = [v / 100 for v in vals]
    cis = [wilson(p, 80) for p in ps]
    err = np.array([[v - lo for v, (lo, _) in zip(vals, cis)], [hi - v for v, (_, hi) in zip(vals, cis)]])
    ax.bar(x + off, vals, w, yerr=err, color=col, capsize=2, label=lab, error_kw=dict(lw=0.8, ecolor=GREY))
ax.set_xticks(x)
ax.set_xticklabels(Q_M)
ax.set_ylabel("성공률 (%)", color=INK)
ax.set_ylim(0, 100)
ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("양자화 추론, 8과제 x 10회 (Wilson 95% 구간)", fontsize=10, pad=22, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_quant.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
