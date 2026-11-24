"""우리가 만든 그래프를 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_env.png     : 표 8을 다시 집계한 환경별 실로봇 성공률(30회)과 전체(120회), Wilson 95% 구간
own_sim_table1.png   : 표 1 오라클 성공률(칸별)과 표에 인쇄된 ± 반폭
own_ablation_read.png: 그림 5와 그림 8(b)에서 읽은 실로봇 성공률(30회 가정)과 Wilson 구간 (그림 판독 값)
own_version_diff.png : v1 대 v2·v3 시뮬 평균 (같은 정책의 시뮬 평균이 판본 사이에 얼마나 바뀌었나)
own_sim_real.png     : Pick v1 시뮬 성공률 대 실로봇 전체 성공률 (모델 6개)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 python3 analysis/recompute.py 를 실행해 recompute_results.json 을 만들어 둘 것.
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
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"
COL = {"MolmoBot(H=1)": "#c2185b", "MolmoBot-Img": "#e57399", "MolmoBot-Pi0": "#f2b3c9", "pi0.5-DROID": "#6b7280", "pi0-DROID": "#c7cad1"}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def plot_real_env():
    v1 = REC["V1_table8_real_world"]
    pols = ["pi0-DROID", "pi0.5-DROID", "MolmoBot-Pi0", "MolmoBot-Img", "MolmoBot(H=1)"]
    envs = ["Workroom", "Kitchen", "Bedroom", "Office", "전체 (120회)"]
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    w = 0.16
    for j, pol in enumerate(pols):
        ys, lo, hi = [], [], []
        for env in envs:
            if env.startswith("전체"):
                k, n = v1["total"][pol]["k"], 120
            else:
                k, n = v1["env_avg_recomputed_pct"][env][pol][0], 30
            y = 100 * k / n
            a, b = wilson(k, n)
            ys.append(y)
            lo.append(y - a)
            hi.append(b - y)
        x = np.arange(len(envs)) + (j - 2) * w
        ax.bar(x, ys, w * 0.95, color=COL[pol], label=pol, yerr=[lo, hi], error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": INK})
    ax.set_xticks(np.arange(len(envs)))
    ax.set_xticklabels(envs)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=5, fontsize=8, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.12))
    ax.set_title("표 8을 다시 집계한 실로봇 성공률 (환경당 30회, 전체 120회; 막대 위 선은 Wilson 95%)", fontsize=9.5, pad=22)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_env.png", dpi=170)
    plt.close(fig)


T1_COLS = ["Pick v1", "Pick v2", "Pick v2 (F)", "Pick v2 RC (F)", "PnP Into (F)", "PnP Next-To (F)", "PnP Color"]
T1 = {
    "π0.5": [(18.1, 2.4), (6.4, 1.5), (7.0, 1.6), (8.0, 1.9), (11.7, 2.1), (8.2, 2.2), (10.4, 1.9)],
    "π0.5-Finetune": [(48.0, 3.1), (28.3, 2.8), (25.8, 2.7), (29.7, 2.9), (43.5, 6.0), (28.4, 3.2), (48.3, 3.1)],
    "MolmoBot-Pi0": [(66.2, 2.9), (35.7, 3.0), (33.3, 2.9), (39.8, 3.1), (44.7, 3.1), (24.7, 3.2), (46.2, 3.1)],
    "MolmoBot-Img": [(92.2, 1.7), (63.5, 3.0), (61.4, 3.0), (62.1, 3.0), (63.0, 3.0), (21.0, 2.5), (67.8, 2.9)],
    "MolmoBot (H=1)": [(93.5, 1.5), (66.8, 2.9), (64.0, 3.0), (63.7, 3.0), (66.4, 2.9), (26.4, 2.7), (67.8, 2.9)],
    "MolmoBot (H=2)": [(91.3, 1.8), (63.8, 3.0), (59.0, 3.0), (62.7, 3.0), (65.4, 2.9), (28.3, 2.8), (66.1, 2.9)],
}


def plot_sim_table1():
    fig, ax = plt.subplots(figsize=(10, 3.9))
    cols = ["#6b7280", "#a3a8b3", "#f2b3c9", "#e57399", "#c2185b", "#7d1040"]
    w = 0.13
    for j, (m, cells) in enumerate(T1.items()):
        x = np.arange(7) + (j - 2.5) * w
        ax.bar(x, [c[0] for c in cells], w * 0.95, yerr=[c[1] for c in cells], color=cols[j], label=m,
               error_kw={"elinewidth": 0.7, "capsize": 1.2, "ecolor": INK})
    ax.set_xticks(range(7))
    ax.set_xticklabels(T1_COLS, fontsize=8.5)
    ax.set_ylabel("오라클 성공률 (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=6, fontsize=7.5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.13))
    ax.set_title("표 1 시뮬 평가 (막대 위 선은 표에 인쇄된 95% 반폭; PnP 칸은 오라클 값)", fontsize=9.5, pad=22)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_table1.png", dpi=170)
    plt.close(fig)


def plot_ablation_read():
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharey=True, gridspec_kw={"width_ratios": [3, 4, 4, 2]})
    n = 30
    data = [
        ("(a) 궤적 수", ["10k", "25k", "50k"], [22, 53, 80]),
        ("(b) 물체 클래스 수", ["10", "25", "50", "100"], [25, 67, 64, 60]),
        ("(c) 환경 수", ["50", "100", "5k", "50k"], [70, 70, 80, 63]),
        ("그림 8(b) 행동 표현", ["delta", "absolute"], [47, 76]),
    ]
    for ax, (title, labs, ys) in zip(axes, data):
        ks = [round(y * n / 100) for y in ys]
        pts = [100 * k / n for k in ks]
        err = [[p - wilson(k, n)[0] for p, k in zip(pts, ks)], [wilson(k, n)[1] - p for p, k in zip(pts, ks)]]
        ax.errorbar(range(len(labs)), pts, yerr=err, fmt="o-", color="#c2185b", ecolor=INK, elinewidth=0.9, capsize=2.5, ms=5)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.4, len(labs) - 0.6)
    axes[0].set_ylabel("실로봇 성공률 (%)")
    axes[0].set_ylim(0, 100)
    fig.suptitle("그림 5와 8에서 읽은 실로봇 값(±3 pp)을 30회로 환산하고 Wilson 95%를 다시 그림 (그림 판독 값, 시행 수 30 가정)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "own_ablation_read.png", dpi=170)
    plt.close(fig)


def plot_version_diff():
    names = ["π0.5", "π0.5-Finetune", "MolmoBot-Pi0", "MolmoBot-Img", "MolmoBot\n(F=2 → H=1)"]
    v1 = [20.6, 36.6, 41.4, 66.4, 66.4]
    v23 = [10.0, 36.0, 41.5, 61.6, 64.1]
    v1m = [36.4, 47.0, 65.5, 92.8, 92.0]
    v23m = [18.1, 48.0, 66.2, 92.2, 93.5]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    x = np.arange(len(names))
    for ax, a, b, title in [(axes[0], v1m, v23m, "Pick MSProc(v1) 칸"), (axes[1], v1, v23, "표의 평균(Avg.)")]:
        ax.bar(x - 0.19, a, 0.36, color="#a3a8b3", label="v1 (2026-03-17)")
        ax.bar(x + 0.19, b, 0.36, color="#c2185b", label="v2·v3 (같은 값)")
        for xi, (p, q) in enumerate(zip(a, b)):
            ax.text(xi - 0.19, p + 1, f"{p:g}", ha="center", fontsize=7.5)
            ax.text(xi + 0.19, q + 1, f"{q:g}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 105)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[0].set_ylabel("시뮬 성공률 (%)")
    fig.suptitle("같은 모델의 시뮬 성공률이 v1 → v2에서 바뀐 정도 (실로봇 수치는 세 판본이 같음)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "own_version_diff.png", dpi=170)
    plt.close(fig)


def plot_sim_real():
    sim = {"π0-DROID": 13.5, "π0.5-DROID": 18.1, "MolmoBot-Pi0": 66.2, "MolmoBot-Img": 92.2, "MolmoBot (H=1)": 93.5, "MolmoBot (H=2)": 91.3}
    real = {"π0-DROID": 9.2, "π0.5-DROID": 39.2, "MolmoBot-Pi0": 46.7, "MolmoBot-Img": 72.5, "MolmoBot (H=1)": 79.2, "MolmoBot (H=2)": 75.0}
    c = REC["V12_sim_real_rank"]["pickv1_sim_vs_allscene_real"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for k in sim:
        ax.scatter(sim[k], real[k], color="#c2185b" if "Molmo" in k else "#6b7280", s=40, zorder=3)
        dx, dy, ha = 1.2, -3.2, "left"
        if k == "MolmoBot (H=1)":
            dx, dy, ha = -1.0, 2.6, "right"
        if k == "MolmoBot (H=2)":
            dx, dy, ha = -1.4, -0.8, "right"
        if k == "MolmoBot-Img":
            dx, dy, ha = 1.4, -3.6, "left"
        ax.text(sim[k] + dx, real[k] + dy, k, fontsize=7.5, ha=ha)
    ax.set_xlabel("Pick v1 시뮬 성공률 (%, 표 11)")
    ax.set_ylabel("실로봇 전체 성공률 (%, 120회)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 95)
    ax.set_title(f"시뮬 대 실로봇: Spearman {c['spearman']} (정확 순열 p = {c['exact_perm_p']}), 모델 {c['n']}개", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_real.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_real_env()
    plot_sim_table1()
    plot_ablation_read()
    plot_version_diff()
    plot_sim_real()
    print("wrote figures to", FIG)
