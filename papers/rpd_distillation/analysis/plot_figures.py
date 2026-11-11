"""우리가 만든 그래프 3장 (공개 가능): 토이 결과와 논문 곡선 판독값.

실행: OMP_NUM_THREADS=1 python3 analysis/plot_figures.py → report/figures/own_*.png
입력: toy_teacher_pull_results.json, fig_reading.json
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
for name in ["Apple SD Gothic Neo", "Noto Sans CJK KR"]:
    try:
        from matplotlib import font_manager
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.family"] = name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

R = json.loads((HERE / "toy_teacher_pull_results.json").read_text(encoding="utf-8"))
RD = json.loads((HERE / "fig_reading.json").read_text(encoding="utf-8"))
LAMS = [0.1, 0.3, 1.0, 3.0]
COLORS = {"good": "tab:green", "mediocre": "tab:orange", "wrong": "tab:red", "irrelevant": "tab:purple"}
LABEL = {"good": "좋은 교사", "mediocre": "편향 교사(+0.4)", "wrong": "반대 방향 교사", "irrelevant": "무관한 교사(평균 0)"}


def fig_bias():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    lam_grid = np.linspace(0.1, 3, 60)
    for k in ["mediocre", "wrong", "irrelevant"]:
        tbar = {"mediocre": 1.4, "wrong": -2.0, "irrelevant": 0.0}[k]
        ax.plot(lam_grid, lam_grid / (1 + lam_grid) * (tbar - 1.0), "-", color=COLORS[k], lw=1.2, label=f"이론 {LABEL[k]}")
        sim = [R["dense"][f"{k}|const|lam={l}"]["final_bias_mean"] for l in LAMS]
        ax.plot(LAMS, sim, "o", color=COLORS[k], ms=4, label=f"시뮬 {LABEL[k]}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("MSE 가중치 λ (상수, log)")
    ax.set_ylabel("고정점 편향 μ - a*")
    ax.set_title("토이: 상수 λ에서 고정점 편향 (dense, 30 시드)", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_bias.png", dpi=170)
    plt.close(fig)


def fig_schedule():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    for sched, ls in [("const", "-"), ("linear_half", "--"), ("cosine", ":")]:
        for k in ["irrelevant"]:
            y = [R["dense"][f"{k}|{sched}|lam={l}"]["final_abs_err_mean"] for l in LAMS]
            ax.plot(LAMS, y, ls, marker="o", ms=3, color="tab:purple", label=f"{sched}")
    ax.set_xscale("log")
    ax.set_xlabel("λ (log)")
    ax.set_ylabel("최종 |μ - a*| (평균)")
    ax.set_title("무관한 교사: 스케줄별 최종 오차", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    kinds = ["good", "mediocre", "wrong", "irrelevant"]
    x = np.arange(len(kinds))
    w = 0.25
    for i, lam in enumerate([0.0, 1.0, 3.0]):
        vals = []
        for k in kinds:
            key = f"{k}|const|lam={lam}"
            vals.append(R["sparse"][key]["success_prob_mean"])
        ax.bar(x + (i - 1) * w, vals, w, label=f"λ={lam}")
    ax.set_xticks(x)
    ax.set_xticklabels(["좋은", "편향(+0.4)", "반대", "무관"], fontsize=7.5)
    ax.set_ylabel("최종 성공 확률 P(|a-a*|<0.1)")
    ax.set_title("희소 보상 토이 (상수 λ, 800 반복)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_schedule.png", dpi=170)
    plt.close(fig)


def fig_reading():
    tasks = RD["fig4_octo_dense_vs_sparse"]["tasks"]
    names = list(tasks)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    series = [("RPD dense", "tab:orange", -0.3), ("PPO dense", "tab:blue", -0.1),
              ("RPD sparse", "tab:red", 0.1), ("PPO sparse", "tab:green", 0.3)]
    x = np.arange(len(names))
    for label, color, off in series:
        vals = [tasks[n].get(label) for n in names]
        ys = [v if v is not None else np.nan for v in vals]
        ax.bar(x + off, ys, 0.18, color=color, label=label)
        for i, v in enumerate(vals):
            if v is None:
                ax.text(x[i] + off, 0.2, "x", ha="center", fontsize=7, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("0.5 성공에 걸린 스텝 (×10^6, log)")
    ax.set_title("Fig. 4 육안 판독: 성공률 0.5까지 걸린 스텝 (x = 도달 못함, 판독치 ±10%)", fontsize=8.5)
    ax.legend(fontsize=7, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "own_fig4_reading.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_bias()
    fig_schedule()
    fig_reading()
    print("wrote own_toy_bias.png, own_toy_schedule.png, own_fig4_reading.png")
