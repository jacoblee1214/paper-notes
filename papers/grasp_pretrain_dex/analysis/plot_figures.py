"""우리가 그린 그래프 세 장을 만든다. 데이터는 recompute.py의 표 값을 그대로 쓴다.

실행: python3 analysis/plot_figures.py  -> report/figures/own_sim_avg.png, own_real_trigger.png, own_ablation.png
한글 폰트는 macOS Apple SD Gothic Neo, 그다음 Linux Noto CJK 순서로 찾는다.
"""
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recompute as rc  # noqa: E402  (표 값 상수와 wilson 함수)

OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def pick_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"]:
        if cand in names:
            return cand
    return "DejaVu Sans"


plt.rcParams["font.family"] = pick_font()
plt.rcParams["axes.unicode_minus"] = False
COL = {"Ours": "#1f5fae", "HP-Scratch": "#7a9cc6", "GraspXL-PT": "#b9a46a", "DP3-PT": "#9b9b96", "DP3": "#c9c7bf"}


def fig_sim_avg():
    """Table 1 평균과 셀 분산 기반 95% 구간(과제 독립 가정, 시드 분산 제외)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=160)
    xs = rc.BUDGETS
    for m in ["DP3", "DP3-PT", "GraspXL-PT", "HP-Scratch", "Ours"]:
        means, errs = [], []
        for n in xs:
            v = [x / 100 for x in rc.T1[m][n]]
            mean = sum(v) / len(v)
            se = math.sqrt(sum(p * (1 - p) / rc.N_SIM for p in v)) / len(v)
            means.append(100 * mean)
            errs.append(100 * 1.96 * se)
        ax.errorbar(xs, means, yerr=errs, marker="o", ms=4, lw=1.8 if m == "Ours" else 1.2,
                    color=COL[m], capsize=3, label=m)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("시연 수 (demos)")
    ax.set_ylabel("과제 평균 성공률 (%)")
    ax.set_title("시뮬레이션 6과제 평균 (Table 1), 95% 구간은 샘플링 오차만", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "own_sim_avg.png")
    plt.close(fig)


def fig_real_trigger():
    """Table 4: 단계별 풀링 성공률과 Wilson 95% 구간 (과제 3개 x 18회 = 54회 풀링)."""
    stages = ["Grasp", "Align", "Trigger"]
    methods = ["DP3", "HP-Scratch", "Ours"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    w = 0.26
    for j, m in enumerate(methods):
        vals, lo, hi = [], [], []
        for s in stages:
            k = sum(rc.T4[m][s])
            n = rc.N_REAL * len(rc.T4[m][s])
            p = 100 * k / n
            ci = rc.wilson(k, n)
            vals.append(p)
            lo.append(p - ci[0])
            hi.append(ci[1] - p)
        xs = [i + (j - 1) * w for i in range(len(stages))]
        ax.bar(xs, vals, width=w, color=COL.get(m, "#999"), label=m, yerr=[lo, hi], capsize=2,
               error_kw={"lw": 0.8})
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(["잡기 (Grasp)", "정렬 (Align)", "작동 (Trigger)"])
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("실로봇 3과제 풀링 (과제당 18회, Wilson 95% 구간)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "own_real_trigger.png")
    plt.close(fig)


def fig_ablation():
    """Table 3: Ours 대비 절제 차이 (과제 평균, 95% 구간, 샘플링 오차만)."""
    rows = []
    for k in ["PT-215k", "PT w/o Appr.", "PT w/o Lift"]:
        d = rc.diff_ci_avg(rc.T3["Ours"], rc.T3[k])
        rows.append((k, d[0], d[1]))
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=160)
    for i, (k, d, ci) in enumerate(rows):
        ax.errorbar([d], [i], xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", color=COL["Ours"], capsize=3)
    ax.axvline(0, color="#55544f", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Ours 대비 성공률 차이 (pp), 양수면 Ours가 높음")
    ax.set_title("사전학습 데이터 절제 (Table 3, 50 demos, 95% 구간)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "own_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_sim_avg()
    fig_real_trigger()
    fig_ablation()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
