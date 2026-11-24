"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_wilson.png : 표 1 실로봇 물체별 성공 수(5회)와 Wilson 95% 구간, 구간별 합계
own_sim_ablation.png: 표 4 시뮬 물체별 w/o 대 w/ 사전학습 (50회, Wilson 95%)
own_sim_vs_real.png : 시뮬 w/ 사전학습 성공률(표 4) 대 실로봇 성공 수(표 1)
own_toy_event.png   : 토이 이벤트 보정(analysis/toy_event_calibration.py)의 보정 전후 지표
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py와 toy_event_calibration.py를 실행해 json을 만들어 둘 것.
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
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#c8641e", "#9a9890"

SEEN = ["Cube", "Ball", "Box", "Cross", "CubeBall", "Egg", "Big Egg", "H-shape", "Hollow Cube", "Hourglass"]
UNSEEN = ["Hash-Shape", "C-Shape", "E-Shape", "T-Shape", "Cylinder", "Fork", "Ring", "Snowman", "Tetrahedral", "Triple"]
REAL = dict(zip(SEEN, [4, 0, 1, 1, 1, 1, 1, 2, 4, 1]))
REAL.update(dict(zip(UNSEEN, [1, 1, 1, 2, 0, 3, 1, 1, 1, 0])))
SIM = {  # (w/o, w/) %
    "Cube": (60, 62), "Ball": (8, 56), "Box": (30, 48), "Cross": (52, 66), "CubeBall": (38, 100), "Egg": (16, 24), "Big Egg": (0, 8),
    "H-shape": (44, 76), "Hollow Cube": (70, 98), "Hourglass": (44, 66), "Hash-Shape": (22, 42), "C-Shape": (18, 34), "E-Shape": (18, 38),
    "T-Shape": (28, 56), "Cylinder": (2, 14), "Fork": (20, 62), "Ring": (50, 80), "Snowman": (22, 48), "Tetrahedral": (8, 28), "Triple": (12, 30),
}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_real():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2.6, 1]})
    ax = axes[0]
    objs = SEEN + UNSEEN
    x = np.arange(len(objs)) + np.r_[np.zeros(10), np.ones(10)]
    for xi, o in zip(x, objs):
        lo, hi = wilson(REAL[o], 5)
        c = ACC if o in SEEN else ORG
        ax.plot([xi, xi], [lo, hi], color=c, lw=2, alpha=0.55)
        ax.plot(xi, 100 * REAL[o] / 5, "o", color=c, ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(objs, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("성공률 % (물체당 5회)")
    ax.set_ylim(-3, 103)
    ax.axhline(27, color=GREY, lw=1, ls="--")
    ax.text(0.2, 29.5, "전체 27%", color=GREY, fontsize=8)
    ax.set_title("표 1 물체별 실로봇 성공률과 Wilson 95% 구간", fontsize=10)
    ax.plot([], [], "o", color=ACC, label="seen")
    ax.plot([], [], "o", color=ORG, label="unseen")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    groups = [("seen", 16, 50, ACC), ("unseen", 11, 50, ORG), ("전체", 27, 100, INK)]
    for i, (name, k, n, c) in enumerate(groups):
        lo, hi = wilson(k, n)
        ax.plot([i, i], [lo, hi], color=c, lw=3, alpha=0.55)
        ax.plot(i, 100 * k / n, "o", color=c, ms=7)
        ax.text(i + 0.12, 100 * k / n, f"{k}/{n}", fontsize=8, va="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylim(0, 60)
    ax.set_title("구간별 합계 (Wilson 95%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_wilson.png", dpi=170)
    plt.close(fig)


def plot_sim():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    objs = SEEN + UNSEEN
    x = np.arange(len(objs)) + np.r_[np.zeros(10), np.ones(10)]
    for xi, o in zip(x, objs):
        a, b = SIM[o]
        for v, c, dx in [(a, GREY, -0.12), (b, ACC if o in SEEN else ORG, 0.12)]:
            lo, hi = wilson(v / 2, 50)
            ax.plot([xi + dx] * 2, [lo, hi], color=c, lw=2, alpha=0.5)
            ax.plot(xi + dx, v, "o", color=c, ms=5)
        ax.plot([xi - 0.12, xi + 0.12], [a, b], color=GREY, lw=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(objs, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("시뮬 성공률 % (물체당 50회)")
    ax.set_ylim(-3, 105)
    ax.plot([], [], "o", color=GREY, label="w/o pretrain")
    ax.plot([], [], "o", color=ACC, label="w/ pretrain (seen)")
    ax.plot([], [], "o", color=ORG, label="w/ pretrain (unseen)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("표 4 물체별 시뮬 성공률과 Wilson 95% 구간 (물체당 50회 독립 시행 가정)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ablation.png", dpi=170)
    plt.close(fig)


def plot_sim_real():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V5_sim_vs_real"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    rng = np.random.default_rng(1)
    for o in SEEN + UNSEEN:
        c = ACC if o in SEEN else ORG
        jitter = rng.uniform(-0.07, 0.07)
        ax.plot(SIM[o][1], REAL[o] + jitter, "o", color=c, ms=6, alpha=0.8)
        if REAL[o] >= 3 or SIM[o][1] >= 90 or o in ("Ball", "Ring"):
            ax.annotate(o, (SIM[o][1], REAL[o] + jitter), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("시뮬 성공률 % (w/ pretrain, 표 4)")
    ax.set_ylabel("실로봇 성공 수 / 5 (표 1)")
    ax.set_ylim(-0.5, 5.2)
    ax.set_xlim(0, 105)
    ax.plot([], [], "o", color=ACC, label="seen")
    ax.plot([], [], "o", color=ORG, label="unseen")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    rho = rec["spearman_all_rho_p"]
    ax.set_title(f"시뮬 대 실로봇 (Spearman ρ={rho[0]}, p={rho[1]:.3f}, n=20)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_vs_real.png", dpi=170)
    plt.close(fig)


def plot_toy():
    d = json.loads((ANA / "toy_event_calibration_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    keys = [("onset_ms", "접촉 시작 오차 (ms)"), ("F1_pct", "활성 F1 (%)"), ("FPR_pct", "FPR (%)")]
    for ax, (k, title) in zip(axes, keys):
        for j, (mode, label) in enumerate([("A_same_form", "A 같은 형태"), ("B_different_form", "B 다른 형태")]):
            nom = np.mean([r["nominal_test"][k] for r in d[mode]])
            cal = np.mean([r["calibrated_test"][k] for r in d[mode]])
            ax.bar(j * 3, nom, color=GREY, width=1)
            ax.bar(j * 3 + 1, cal, color=ACC, width=1)
            ax.text(j * 3, nom, f"{nom:.0f}" if k == "onset_ms" else f"{nom:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(j * 3 + 1, cal, f"{cal:.0f}" if k == "onset_ms" else f"{cal:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks([0.5, 3.5])
        ax.set_xticklabels(["A 같은 형태", "B 다른 형태"], fontsize=8)
        ax.set_title(title, fontsize=9)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=GREY, label="보정 전(임의의 공칭 값)"), Patch(color=ACC, label="보정 후")], frameon=False, fontsize=7)
    fig.suptitle("토이: 보정에 쓰지 않은 궤적 30개의 지표 (시드 3개 평균)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_event.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_real()
    plot_sim()
    plot_sim_real()
    plot_toy()
    print("wrote figures to", FIG)
