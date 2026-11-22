"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sr_ci.png      : 표 1 과제별 SR(DeCAL, DECO, 과제별 최고 기준선)과 Wilson 95% 구간, 합산 비교
own_ablation.png   : 표 2 절제의 SR과 Wilson 95% 구간 (Assemble Parts, Twist Cap)
own_psr_agg.png    : PSR 집계 비교 (표 1 평균 대 표 5 Avg 열, 그림 1 라벨)
own_toy_gate.png   : 토이 게이트 결과 (체제별 MSE, 시드 5개 평균과 SD)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
입력: analysis/recompute_results.json, analysis/toy_gate_results.json
"""
import json
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#d9822b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def load(name):
    return json.loads((ANA / name).read_text(encoding="utf-8"))


def plot_sr():
    rec = load("recompute_results.json")
    rows = rec["V10_per_task_sr"]
    tasks = [r["task"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(tasks))
    series = [("DeCAL", [r["decal_k"] for r in rows], ACC, -0.22),
              ("DECO", [round(rec["V10_per_task_sr"][i]["decal_k"] - rec["V10_per_task_sr"][i]["vs_DECO_diff_pp"] / 5) for i in range(6)], ORG, 0.0),
              ("과제별 최고 기준선", [r["best_k"] for r in rows], GREY, 0.22)]
    for label, ks, col, off in series:
        ci = [wilson(k, 20) for k in ks]
        pct = [5 * k for k in ks]
        ax.errorbar(x + off, pct, yerr=[[p - c[0] for p, c in zip(pct, ci)], [c[1] - p for p, c in zip(pct, ci)]], fmt="o", color=col, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("성공률 % (20회, Wilson 95%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("표 1 재계산: 과제별로는 구간이 크게 겹칩니다", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_sr_ci.png", dpi=200)
    plt.close(fig)


def plot_ablation():
    labels = ["전체", "촉각 게이트 제거", "촉각 생성 제거", "시각 생성 제거", "시각+촉각 생성 제거", "Factorized FM 제거"]
    asm = [65, 50, 55, 35, 20, 25]
    twi = [80, 70, 60, 45, 30, 35]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, vals, title in ((axes[0], asm, "Assemble Parts"), (axes[1], twi, "Twist Cap")):
        y = np.arange(len(labels))[::-1]
        ci = [wilson(v / 5, 20) for v in vals]
        ax.errorbar(vals, y, xerr=[[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]], fmt="o", color=ACC, capsize=3)
        ax.axvline(vals[0], color=GREY, lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_xlabel("성공률 % (20회, Wilson 95%)")
    axes[0].set_yticks(np.arange(len(labels))[::-1])
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("표 2 절제: 구간으로 보면 촉각 게이트와 촉각 생성 제거는 전체와 겹칩니다", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=200)
    plt.close(fig)


def plot_psr():
    rec = load("recompute_results.json")["V6_averages"]
    ms = list(rec.keys())
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = np.arange(len(ms))
    a = [rec[m]["psr_mean_from_stages_exact"] for m in ms]
    b = [100 * rec[m]["table5_avg_reported"] for m in ms]
    c = [100 * rec[m]["table5_avg_all21_mean"] for m in ms]
    ax.bar(x - 0.27, a, 0.26, color=ACC, label="표 1 PSR의 과제 평균 (초록 83.4%의 출처)")
    ax.bar(x, c, 0.26, color=GREY, label="표 5 값 21개의 평균 (재계산)")
    ax.bar(x + 0.27, b, 0.26, color=ORG, label="표 5 Avg 열 = 그림 1 \"Average PSR\"")
    for xi, v in zip(x + 0.27, b):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(ms, fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 125)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("진행률 평균의 세 가지 값: 표 5 Avg 열은 표 5 값에서 재현되지 않습니다", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_psr_agg.png", dpi=200)
    plt.close(fig)


def plot_toy():
    res = load("toy_gate_results.json")
    keys = [("drift_sfree0.3", "비접촉 촉각 잡음 0.3"), ("drift_sfree1.0", "잡음 1.0"), ("drift_sfree3.0", "잡음 3.0\n(접촉과 구분 불가)"), ("clean_sfree3.0", "촉각이 항상 깨끗")]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = np.arange(len(keys))
    for i, (kind, col, lab) in enumerate((("vision", GREY, "시각만"), ("ungated", ORG, "게이트 없음"), ("gated", ACC, "스칼라 게이트"))):
        m = [res[k][kind]["all"][0] for k, _ in keys]
        s = [res[k][kind]["all"][1] for k, _ in keys]
        ax.bar(x + (i - 1) * 0.26, m, 0.25, yerr=s, color=col, capsize=2, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in keys], fontsize=8)
    ax.set_ylabel("시험 MSE (로그 축, 시드 5개 평균 ± SD)")
    ax.legend(fontsize=8)
    ax.set_title("토이: 게이트는 비접촉 촉각이 잡음이고 접촉을 읽을 수 있을 때만 이깁니다", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_gate.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_sr()
    plot_ablation()
    plot_psr()
    plot_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
