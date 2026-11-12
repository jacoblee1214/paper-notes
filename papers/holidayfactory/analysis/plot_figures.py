"""리포트·게시본 그래프 (CPU, matplotlib). 실행: python3 analysis/plot_figures.py → report/figures/*.png
입력은 table_check_results.json, power_toy_results.json이다. 논문 그림은 그리지 않는다(우리가 그린 그래프만).
팔레트: 작업 단위(naive) #A8560B, 실행 단위(run) #0072B2 (contactbench와 같은 계열).
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
T = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
P = json.loads((ROOT / "analysis" / "power_toy_results.json").read_text())

for f in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

NAIVE, RUN = "#A8560B", "#0072B2"
INK, MUTED, GRID = "#1b1b1a", "#55544f", "#dcdad3"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def fig_generation():
    """Table 13: 수정 전략별 Success@t (n = 32, 같은 1차 자산). Wilson 95% 구간."""
    rounds = [1, 2, 3]
    data = {
        "독립 재생성": [20, 24, 27],
        "거친 피드백": [20, 22, 26],
        "E RIS (측정 피드백)": [20, 31, 32],
    }
    colors = [MUTED, NAIVE, RUN]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    width = 0.26
    for i, (label, ks) in enumerate(data.items()):
        xs = np.array(rounds) + (i - 1) * width
        ps = np.array(ks) / 32
        lo, hi = zip(*[wilson(k, 32) for k in ks])
        ax.bar(xs, ps, width=width * 0.92, color=colors[i], alpha=0.85, label=label)
        ax.errorbar(xs, ps, yerr=[ps - np.array(lo), np.array(hi) - ps], fmt="none", ecolor=INK, lw=0.8, capsize=2)
        for x, p, k in zip(xs, ps, ks):
            ax.text(x, 0.03, f"{k}/32", ha="center", va="bottom", fontsize=6.5, color="white", rotation=90)
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Success@{r}" for r in rounds])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("성공 비율", color=MUTED, fontsize=9)
    ax.set_title("수정 전략별 성공 비율(논문 Table 13 집계, Wilson 95% 구간)", fontsize=10, color=INK, loc="left", pad=22)
    ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "hf_generation_revision.png")
    plt.close(fig)


def fig_dependence():
    """작업 독립 검정 대 실행 단위 검정: 1종 오류(좌)와 포함 확률(우). 정책당 실행 10회, 실행당 작업 20개(가정)."""
    sigmas = [0.0, 0.5, 1.0]
    labels = ["0", "0.5", "1.0"]
    key = [f"sigma={s}" for s in sigmas]
    naive = [P["type1_and_power"][k]["R=10"]["delta=0.0"]["naive_reject"] for k in key]
    run = [P["type1_and_power"][k]["R=10"]["delta=0.0"]["run_reject"] for k in key]
    cov_naive = [P["ci_coverage_R10"][k]["wilson_naive_jobs_coverage"] for k in key]
    cov_run = [P["ci_coverage_R10"][k]["run_cluster_t_coverage"] for k in key]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), dpi=200)
    x = np.arange(len(sigmas))
    w = 0.36
    ax = axes[0]
    ax.bar(x - w / 2, naive, w, color=NAIVE, label="작업 독립 z 검정")
    ax.bar(x + w / 2, run, w, color=RUN, label="실행 평균 t 검정")
    ax.axhline(0.05, color=INK, lw=0.8, ls="--")
    ax.text(2.45, 0.06, "0.05", fontsize=7, color=INK, ha="right")
    for xi, v in zip(x - w / 2, naive):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    for xi, v in zip(x + w / 2, run):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("실행 효과 s (로짓 표준편차, 가정)", color=MUTED, fontsize=9)
    ax.set_ylabel("1종 오류(두 정책 성공률 같음)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 0.42)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("1종 오류", fontsize=10, color=INK, loc="left")
    style(ax)

    ax = axes[1]
    ax.bar(x - w / 2, cov_naive, w, color=NAIVE, label="작업 독립 Wilson 구간")
    ax.bar(x + w / 2, cov_run, w, color=RUN, label="실행 군집 t 구간")
    ax.axhline(0.95, color=INK, lw=0.8, ls="--")
    for xi, v in zip(x - w / 2, cov_naive):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    for xi, v in zip(x + w / 2, cov_run):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("실행 효과 s (로짓 표준편차, 가정)", color=MUTED, fontsize=9)
    ax.set_ylabel("95% 구간의 포함 비율", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("구간 포함 비율", fontsize=10, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "hf_dependence_error.png")
    plt.close(fig)


def fig_power():
    """정책당 실행 수에 따른 검정력(Δ = 0.10, 0.05). 실선은 실행 검정, 점선은 작업 독립 검정."""
    runs = [3, 5, 10, 20, 40, 80]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=200, sharey=True)
    for ax, delta in zip(axes, ["0.1", "0.05"]):
        for s, col in zip([0.0, 0.5, 1.0], ["#7a7a74", RUN, NAIVE]):
            run = [P["type1_and_power"][f"sigma={s}"][f"R={r}"][f"delta={delta}"]["run_reject"] for r in runs]
            ax.plot(runs, run, "-o", color=col, ms=3.5, lw=1.4, label=f"s = {s}")
        ax.axhline(0.8, color=INK, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(runs)
        ax.set_xticklabels([str(r) for r in runs], fontsize=8)
        ax.set_xlabel("정책당 실행 수 R (log)", color=MUTED, fontsize=9)
        ax.set_title(f"차이 {int(float(delta) * 100)} pp, 실행 단위 검정", fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0].set_ylabel("검정력 (α = 0.05, 4000회)", color=MUTED, fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right", title="실행 효과 s", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "hf_power_runs.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_generation()
    fig_dependence()
    fig_power()
    print("wrote figures to", FIG)
