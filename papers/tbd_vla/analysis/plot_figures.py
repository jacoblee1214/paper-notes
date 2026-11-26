"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_cells.png : 실로봇 12셀별 성공 수 차이(TBD-VLA + RTC 빼기 pi0.5 + RTC)와 합계 분해
own_liberoplus.png : LIBERO-Plus 평균의 집계 방식(보고값, 단순 평균, 가중 추정)
own_mask_count.png : 블록(28 토큰) 마스크 수 분포, 코드(블록당 t) 대 식 1 문자 해석(토큰당 t)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Noto CJK -> Apple SD Gothic Neo 순서)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

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
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#b23a2e"
rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def plot_real():
    d = rec["V9b_tbd_vs_pi05"]["TBD-VLA w/ RTC vs pi0.5 w/ RTC"]["cell_diff"]
    labels = list(d.keys())
    vals = [d[k] for k in labels]
    short = [k.replace("Everything in Bin", "Bin").replace("Bread in Toaster", "Bread").replace("Transfer the Liquid", "Liquid") for k in labels]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    cols = [ACC if v > 0 else RED if v < 0 else GREY for v in vals]
    ax.bar(range(12), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.5 if v >= 0 else -1.6), f"{v:+d}", ha="center", fontsize=8)
    ax.set_xticks(range(12))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("성공 수 차이 (20회 중)")
    ax.set_title("실로봇 셀별 차이: TBD-VLA + RTC 빼기 π0.5 + RTC (합계 +41)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_cells.png", dpi=170)
    plt.close(fig)


def plot_plus():
    b = rec["V5b_liberoplus_weighting"]
    names = ["OpenVLA-OFT", "UniVLA", "pi0-FAST", "pi0", "RIPT-VLA", "TBD-VLA w/o pretrain", "TBD-VLA w/ pretrain"]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(names))
    w = 0.26
    ax.bar(x - w, [b[n]["reported"] for n in names], w, color=GREY, label="보고 Avg")
    ax.bar(x, [b[n]["simple_mean"] for n in names], w, color="#8fb0d9", label="7차원 단순 평균")
    ax.bar(x + w, [b[n]["weighted"] for n in names], w, color=ACC, label="가중 추정 (리더보드 적합 가중치)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("pi0", "π0").replace(" w/", "\nw/") for n in names], fontsize=8)
    ax.set_ylim(35, 90)
    ax.set_ylabel("LIBERO-Plus 평균 (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("LIBERO-Plus 평균의 집계 방식 (기준선 행은 프로젝트 페이지 값)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_liberoplus.png", dpi=170)
    plt.close(fig)


def plot_mask():
    N = 28
    k = np.arange(N + 1)
    code = np.full(N + 1, 1 / (N + 1))
    lit = np.array([math.comb(N, i) * 0.5 ** N for i in k])
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.bar(k - 0.2, code, 0.4, color=ACC, label="코드: 블록당 t, 마스크 수가 균일")
    ax.bar(k + 0.2, lit, 0.4, color=RED, label="식 1 문자 해석: 토큰당 t, Binomial(28, 0.5)")
    ax.set_xlabel("블록 안 마스크된 토큰 수 (전체 28)")
    ax.set_ylabel("확률")
    ax.legend(fontsize=8)
    ax.set_title("학습에서 보이는 마스크 수 분포 (해석 계산)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_mask_count.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_real()
    plot_plus()
    plot_mask()
    print("wrote 3 figures")
