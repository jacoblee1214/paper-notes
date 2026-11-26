"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1.png        : 표 1 범주별 LPIPS (다섯 학습 혼합)
own_six_policies.png  : 그림 6 원그래프의 정책 6개, 실제(20회)와 세계모델(50회) 성공률과 Wilson 95% 구간
own_noise_ceiling.png : 토이 T1, 시행 수 잡음만으로 생기는 r과 RMSE (보고 값 표시)
own_finetune_ci.png   : 그림 9 판독 값(체크포인트당 20회 가정)과 Wilson 95% 구간
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 analysis/recompute.py와 analysis/toy_noise_ceiling.py를 실행해 결과 json을 만들어 둘 것.
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
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#d9822b", "#9a9890"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def plot_table1():
    cats = ["Success", "Missed\nGrasp", "Slide", "Slip", "Deformation", "Collision"]
    rows = {
        "Human Demo 6h": [0.084, 0.080, 0.090, 0.090, 0.108, 0.086],
        "Human Play 6h": [0.086, 0.071, 0.089, 0.088, 0.100, 0.080],
        "Robot Play 6h": [0.082, 0.066, 0.077, 0.078, 0.099, 0.074],
        "Robot Play 30h": [0.071, 0.064, 0.073, 0.072, 0.094, 0.076],
        "Robot Play 곡선": [0.070, 0.066, 0.071, 0.070, 0.093, 0.072],
    }
    colors = ["#8fbcd4", "#a89bd6", "#f0c75e", "#e0a030", "#c9731f"]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    w = 0.16
    x = np.arange(len(cats))
    for i, (name, v) in enumerate(rows.items()):
        ax.bar(x + (i - 2) * w, v, w, label=name, color=colors[i], edgecolor=INK, linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0.05, 0.115)
    ax.set_ylabel("LPIPS (낮을수록 좋음)")
    ax.legend(ncol=5, fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("표 1 다시 그림: 범주별 LPIPS (값은 논문 표 그대로, 축은 0.05부터)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "own_table1.png", dpi=170); plt.close(fig)


def plot_six():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V6"]
    real = [round(s * 20) for s in rec["success_rate"]["Real"]]
    pw = [round(s * 50) for s in rec["success_rate"]["PlayWorld"]]
    bl = [round(s * 50) for s in rec["success_rate"]["Baseline"]]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    for ks, n, col, lab, off in [(pw, 50, ORG, "PlayWorld (50회)", 0.008), (bl, 50, "#6b86c9", "Baseline (50회)", -0.008)]:
        for kr, k in zip(real, ks):
            lo, hi = wilson(k, n); xl, xh = wilson(kr, 20)
            ax.plot([kr / 20 + off] * 2, [lo, hi], color=col, lw=1.2, alpha=0.8)
            ax.plot([xl + off, xh + off], [k / n] * 2, color=col, lw=0.8, alpha=0.4)
        ax.scatter([kr / 20 + off for kr in real], [k / n for k in ks], color=col, s=34, edgecolor=INK, linewidth=0.6, label=lab, zorder=3)
    ax.set_xlabel("실제 성공률 (20회, 가로 선은 Wilson 95%)")
    ax.set_ylabel("세계모델 성공률 (50회, 세로 선은 Wilson 95%)")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("그림 6 원그래프의 정책 6개 (표기된 개수에서 계산)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "own_six_policies.png", dpi=170); plt.close(fig)


def plot_noise():
    res = json.loads((ANA / "toy_noise_ceiling_results.json").read_text(encoding="utf-8"))
    g = res["sigma_grid"]
    s = np.array([float(k) for k in g])
    rm = np.array([g[k]["r_mean"] for k in g]); lo = np.array([g[k]["r_5_95"][0] for k in g]); hi = np.array([g[k]["r_5_95"][1] for k in g])
    mm = np.array([g[k]["rmse_mean"] for k in g])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax = axes[0]
    ax.fill_between(s, lo, hi, color=ACC, alpha=0.18)
    ax.plot(s, rm, color=ACC, lw=2, label="시뮬 평균 (5-95% 띠)")
    for name, (r, _), col in [("PlayWorld", res["reported"]["PlayWorld"], ORG), ("Human Demo", res["reported"]["Human Demo"], "#6b86c9"), ("Human Play", res["reported"]["Human Play"], "#8a6bc9")]:
        ax.axhline(r, color=col, ls="--", lw=1)
        dy = {"PlayWorld": 0.008, "Human Demo": 0.012, "Human Play": -0.035}[name]
        ax.text(0.003, r + dy, f"{name} r={r}", color=col, fontsize=8, ha="left")
    ax.set_xlabel("세계모델 오차 σ (정책별 성공률 오차의 표준편차)"); ax.set_ylabel("Pearson r (정책 18개)")
    ax.set_ylim(0.3, 1.0); ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(s, mm, color=ACC, lw=2, label="시뮬 평균 RMSE")
    ax.axhline(res["analytic_rmse_floor"], color=GREY, ls=":", lw=1.2)
    ax.text(0.003, res["analytic_rmse_floor"] + 0.006, f"완벽한 모델의 바닥 {res['analytic_rmse_floor']}", fontsize=8, color=INK)
    for name, (_, m), col in [("PlayWorld", res["reported"]["PlayWorld"], ORG), ("Human Demo", res["reported"]["Human Demo"], "#6b86c9"), ("Human Play", res["reported"]["Human Play"], "#8a6bc9")]:
        ax.axhline(m, color=col, ls="--", lw=1)
        ax.text(0.003, m + 0.006, f"{name} RMSE={m}", color=col, fontsize=8)
    ax.set_xlabel("세계모델 오차 σ"); ax.set_ylabel("RMSE")
    ax.set_ylim(0.05, 0.4); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("토이 T1: 시행 수(20회, 50회) 잡음만으로 생기는 상관 천장과 RMSE 바닥", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "own_noise_ceiling.png", dpi=170); plt.close(fig)


def plot_finetune():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V7"]
    f9 = {"당근 꺼내기": ([0, 200, 400, 800, 1200], [.30, .30, .40, .45, .25], [.30, .50, .65, .75, .75]),
          "수건 접기": ([0, 800, 1200, 2000, 4000], [.10, .25, .40, .35, .25], [.10, .40, .50, .75, .50])}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    for ax, (title, (steps, b, p)) in zip(axes, f9.items()):
        xs = np.arange(len(steps))
        for vals, col, lab, off in [(b, "#6b86c9", "Baseline", -0.05), (p, ORG, "PlayWorld", 0.05)]:
            ci = [wilson(round(v * 20), 20) for v in vals]
            ax.errorbar(xs + off, vals, yerr=[[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]],
                        color=col, marker="D", ms=5, lw=1.4, capsize=2.5, label=lab)
        ax.set_xticks(xs); ax.set_xticklabels(steps)
        ax.axhline(b[0], color="#6a9f58", ls="--", lw=1)
        ax.set_title(title, fontsize=10); ax.set_xlabel("미세조정 스텝"); ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("실로봇 성공률 (20회 가정, Wilson 95%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("그림 9 판독 값 다시 그림 (그림 판독, 시행 수 20 가정)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "own_finetune_ci.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    plot_table1(); plot_six(); plot_noise(); plot_finetune()
    print("wrote 4 figures to", FIG)
