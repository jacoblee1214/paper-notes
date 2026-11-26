"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_check.png : 표 1 값 대 공식 저장소 기록 데이터(시드 1)의 재계산 값, BC 분포 주요 행의 궤적 부트스트랩 구간
own_fig3_check.png   : 그림 3 점 36개 재현 (사이트 JSON), 시리즈별 색, 합산 r과 시리즈 군집 부트스트랩 구간
own_scaling_vs_aim.png : 그림 5의 AIM 점과 그림 6의 수동 수집 곡선(그림 판독 값)을 같은 축에 놓은 것 (H=10, BC 평가)
own_toy_scope.png    : 토이 A: 데이터 상호정보 I_D, 평가 정책 상호정보, 모델별 I_hat
입력: analysis/recompute_results.json, analysis/toy_theory_results.json, repo/docs/static/data/*.json
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
DATA = ROOT / "repo" / "docs" / "static" / "data"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True
INK, ACC, GREY, RED, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#c0392b", "#2e7d32"
rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
toy = json.loads((ANA / "toy_theory_results.json").read_text(encoding="utf-8"))


def plot_table1():
    rows = rec["V3_table1_vs_site"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    col = {"BC": ACC, "BC-Failure": GRN, "Push-Away": RED}
    for d in ("BC", "BC-Failure", "Push-Away"):
        xs = [r["paper"] for r in rows if r["dist"] == d]
        ys = [r["site_seed1"] for r in rows if r["dist"] == d]
        ax.scatter(xs, ys, s=16, color=col[d], label=d, alpha=0.85)
    ax.plot([0.1, 0.7], [0.1, 0.7], color=GREY, lw=1)
    ax.set_xlabel("논문 표 1 평균 (시드 3 평균 추정)")
    ax.set_ylabel("사이트 기록 데이터 재계산 (시드 1)")
    ax.set_title(f"99칸: 평균 절대차 {rec['V3_table1_vs_site']['mean_abs_diff']:.4f}, 최대 {rec['V3_table1_vs_site']['max_abs_diff']:.4f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    names = ["PPO-300K", "PPO-3K", "PPO-3K+AIM500", "PPO-3K+Recovery500", "PPO-3K+ICM500", "Recovery-3K", "Recovery-3K+AIM500", "AIM w/o re-estimation"]
    y = np.arange(len(names))[::-1]
    for k, (metric, c, off) in enumerate((("ee", ACC, 0.18), ("joint", RED, -0.18))):
        for yy, n in zip(y, names):
            r = next(x for x in rows if x["row"] == n and x["dist"] == "BC" and x["metric"] == metric)
            ax.errorbar(r["site_seed1"], yy + off, xerr=[[r["site_seed1"] - r["site_ci"][0]], [r["site_ci"][1] - r["site_seed1"]]],
                        fmt="o", color=c, ms=4, capsize=2, label=("EE 재현율" if metric == "ee" else "Joint 재현율") if yy == y[0] else None)
            ax.plot(r["paper"], yy + off, marker="|", color=INK, ms=9, ls="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("BC 평가 분포 (H=33). 점: 시드 1 재계산 ± 궤적 부트스트랩 95%, 검은 막대: 논문 값")
    ax.xaxis.label.set_fontsize(8)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_check.png", dpi=170)
    plt.close(fig)


def plot_fig3():
    d = json.loads((DATA / "fourrooms_influence_fidelity.json").read_text())["panels"][0]["series"]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    cols = [ACC, GRN, "#e08e0b", RED]
    for s, c in zip(d, cols):
        ax.scatter(s["x"], s["y"], s=22, color=c, label=s["key"])
    v = rec["V5_fig3"]
    ax.set_xlabel("행동 영향 (모델이 설명한 행동 정보)")
    ax.set_ylabel("시뮬 충실도 (%)")
    lo, hi = v["series_cluster_bootstrap_r_2.5_50_97.5"][0], v["series_cluster_bootstrap_r_2.5_50_97.5"][2]
    ax.set_title(f"점 {v['points']}개 합산 r = {v['pooled_r']:.3f}\n시리즈 군집 부트스트랩 구간 [{lo:.2f}, {hi:.2f}]", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig3_check.png", dpi=170)
    plt.close(fig)


def plot_scaling():
    bc = rec["V7_fig6_read_values_BC_EE"]
    x = bc["x_trajectories"]
    f5 = json.loads((DATA / "composite_pusht_controllability.json").read_text())["panels"][0]
    aim = f5["series"][0]["y"][-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, c, m in (("PPO expert", ACC, "o"), ("Random Walk", "#7b4fa6", "^"), ("Uniform Random", "#a0522d", "D"), ("Recovery", GRN, "s")):
        ax.plot(x, bc[k], marker=m, color=c, ms=4, lw=1.3, label=f"{k} 단독 (수동)")
    ax.scatter([3505], [aim], marker="*", s=170, color=RED, zorder=5, label="3k expert + AIM 500 (사이트 JSON)")
    ax.scatter([3005], [f5["refs"][1]["y"]], marker="x", s=50, color=INK, zorder=5, label="3k expert (사이트 JSON)")
    ax.set_xscale("log")
    ax.set_xlabel("학습 궤적 수 (로그)")
    ax.set_ylabel("EE 재현율 (%), BC 평가, H=10")
    ax.set_title("수동 곡선 값은 그림 6 판독 값(±0.7p), AIM 점은 사이트 JSON", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_scaling_vs_aim.png", dpi=170)
    plt.close(fig)


def plot_toy():
    rows = toy["A_theorem1_scope"]["rows"][1:]   # sigma_a = 0 행은 로그 축에 못 그림
    sa = [r["sigma_a"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    ax.plot(sa, [r["I_D"] for r in rows], "k--", label="데이터의 I_D (정리 1 우변)")
    ax.axhline(rows[0]["I_eval_true"], color=GREY, lw=1, label="평가 정책의 참 I")
    ax.plot(sa, [r["linear"]["I_hat_on_data_distribution"] for r in rows], "o-", color=ACC, label="선형 모델, 데이터 분포 평가")
    ax.plot(sa, [r["linear"]["I_hat_on_eval_policy"] for r in rows], "s-", color=RED, label="선형 모델, 다른 학습자 평가")
    ax.set_xscale("log")
    ax.set_ylim(-1, 5.5)
    ax.set_xlabel("데이터 정책의 행동 잡음 σ_a")
    ax.set_ylabel("설명된 행동 정보 (nats)")
    ax.set_title("외삽하는 모델: 데이터 분포에서는 I_D 이하, 밖에서는 I_D보다 큼", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    ax.plot(sa, [r["I_D"] for r in rows], "k--", label="I_D")
    ax.plot(sa, [r["knn"]["I_hat_on_data_distribution"] for r in rows], "o-", color=ACC, label="kNN, 데이터 분포 평가")
    ax.plot(sa, [max(r["knn"]["I_hat_on_eval_policy"], -50) for r in rows], "s-", color=RED, label="kNN, 다른 학습자 평가 (-50에서 자름)")
    ax.set_xscale("log")
    ax.set_xlabel("데이터 정책의 행동 잡음 σ_a")
    ax.set_title("외삽하지 않는 모델: 데이터 밖에서 크게 음수", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_scope.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_fig3()
    plot_scaling()
    plot_toy()
    print("wrote 4 figures to", FIG)
