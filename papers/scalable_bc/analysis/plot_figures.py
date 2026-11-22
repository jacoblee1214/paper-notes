"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1.png   : 표 1 아키텍처 비교, 과제별 progress와 strict (모델 6종)
own_corr.png     : 그림 8, 9의 상관계수와 Fisher z 95% 구간, 유효 표본 수를 줄였을 때 p값
own_sim_curve.png: 프로젝트 페이지 SVG에서 읽은 sim 진행률 곡선(그림 12 원자료)
own_compute.png  : 표 2 샘플당 TFLOPs, 추론 지연 단계별 감소(그림 23, 24)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
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
INK, ACC, RED, GREY = "#1b1b1a", "#1f5fae", "#b94a3a", "#9a9890"
R = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))

# 표 1 값 (recompute.py 와 같은 값)
MODELS = ["VLA\nPooled adaLN", "VLA\nFAST+X-Attn", "VLA\nX-Attn", "DiT\nDINOv3-xattn", "DiT\nCLIP-adaln", "DiT\nCLIP-xattn"]
PROG = {"Bottles": [83.0, 44.0, 5.2, 93.1, 53.6, 74.1], "Dishrack": [75.7, 47.2, 25.0, 74.1, 72.3, 81.3], "Mugs": [25.5, 6.5, 5.0, 35.3, 15.9, 21.0]}
STRICT = {"Bottles": [67.3, 8.0, 0.0, 73.5, 11.8, 38.9], "Dishrack": [30.0, 2.8, 0.0, 23.2, 28.6, 34.7], "Mugs": [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]}


def plot_table1():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=False)
    cols = [ACC, RED, GREY]
    for ax, data, title in ((axes[0], PROG, "작업 진행률 (progress, %)"), (axes[1], STRICT, "엄격 성공률 (strict, %)")):
        x = np.arange(6)
        for j, (task, vals) in enumerate(data.items()):
            ax.bar(x + (j - 1) * 0.27, vals, width=0.26, color=cols[j], label=task)
        mean = np.mean(list(data.values()), axis=0)
        ax.plot(x, mean, "k_", ms=14, mew=2, label="3과제 평균")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.text(0.5, -0.01, "논문 표 1 (200k 스텝, 과제당 50회라고 서술, 신뢰구간 없음). 값은 논문에서 옮긴 것이며 막대에 오차 막대는 없다.", ha="center", fontsize=7, color="#55544f")
    fig.tight_layout()
    fig.savefig(FIG / "own_table1.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_corr():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    rows = R["V4_fig8_offline_vs_real"] + [dict(metric="sim strict", target="vs real", r=R["V5_fig9_sim_real"][0]["r"], r_ci95_fisher_z=R["V5_fig9_sim_real"][0]["r_ci95_fisher_z"]),
                                         dict(metric="sim progress", target="vs real", r=R["V5_fig9_sim_real"][1]["r"], r_ci95_fisher_z=R["V5_fig9_sim_real"][1]["r_ci95_fisher_z"])]
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        lo, hi = row["r_ci95_fisher_z"]
        c = ACC if row["target"] != "vs real" else RED
        ax.plot([lo, hi], [yi, yi], color=c, lw=2)
        ax.plot(row["r"], yi, "o", color=c)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r["metric"]} → 실제 {r["target"]}' if r["target"] != "vs real" else f'{r["metric"]} → 실제' for r in rows], fontsize=7.5)
    ax.set_xlabel("Pearson r (95% 구간, Fisher z)")
    ax.set_title("오프라인 지표(n=16)와 sim(n=12)의 상관", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ns = [16, 12, 8, 6]
    sens = R["V4_fig8_effective_n_sensitivity"]
    ax.plot(ns, [sens[str(n)]["val_action_error_strict_r-0.89_p"] for n in ns], "o-", color=ACC, label="검증 행동 오차 (r=-0.89)")
    ax.plot(ns, [sens[str(n)]["train_loss_strict_r-0.84_p"] for n in ns], "s-", color=GREY, label="학습 손실 (r=-0.84)")
    s5 = R["V5_fig9_effective_n_sensitivity"]
    n2 = [12, 8, 6, 5]
    ax.plot(n2, [s5[str(n)]["strict_r0.85_p"] for n in n2], "^-", color=RED, label="sim strict (r=0.85)")
    ax.axhline(0.05, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["1e-5", "1e-4", "1e-3", "0.01", "0.1"])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel("유효 표본 수 (가정)")
    ax.set_ylabel("p값 (log)")
    ax.set_title("표본이 독립이 아니라면: 유효 n별 p", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_corr.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_sim_curve():
    v = R["V12_fig12_sim_progress_from_project_page"]
    x = np.arange(1, 21) * 10
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(x, v["dit"], "-o", ms=3, color=ACC, label="ABC-DiT (배치 9.2k)")
    ax.plot(x, v["vla"], "-o", ms=3, color=RED, label="ABC-VLA (배치 9.2k)")
    ax.set_xlabel("체크포인트 스텝 (k, 균등 간격 가정)")
    ax.set_ylabel("sim 진행률 (%), 3과제 평균")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("프로젝트 페이지 SVG에서 역산한 사전학습 중 sim 진행률", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_curve.png", dpi=170)
    plt.close(fig)


def plot_compute():
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))
    ax = axes[0]
    ax.bar(["ABC-DiT", "ABC-VLA"], [0.329, 6.957], color=GREY, label="백본")
    ax.bar(["ABC-DiT", "ABC-VLA"], [0.349, 0.063], bottom=[0.329, 6.957], color=ACC, label="행동 헤드")
    ax.set_ylabel("샘플당 학습 TFLOPs")
    ax.set_title("표 2: 10.4배 차이", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ms = [63.0, 47.5, 41.3, 36.3]
    names = ["eager", "분리 compile", "단일 그래프\n+autotune", "+CUDA\ngraph"]
    ax.bar(range(4), ms, color=ACC)
    for i, m in enumerate(ms):
        ax.text(i, m + 1, f"{m}", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("ms / 청크 (RTX 5090)")
    ax.set_title("ABC-DiT 추론 최적화 (1.74배)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    ms = [47.8, 22.6, 17.5]
    ax.bar(range(3), ms, color=RED)
    for i, m in enumerate(ms):
        ax.text(i, m + 1, f"{m}", ha="center", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["eager", "분리 compile", "전체 compile"], fontsize=7)
    ax.set_title("ABC-VLA 추론 최적화 (2.73배)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_compute.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_corr()
    plot_sim_curve()
    plot_compute()
    print("wrote figures to", FIG)
