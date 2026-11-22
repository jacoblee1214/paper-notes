"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_multitrial.png : 다중 시도 실패율, 보고값 대 독립 가정 대 베타 혼합 적합 (표 5)
own_scaling.png    : 표 5 모델 규모·객체 규모 곡선과 95% 구간(가정 n = 160,000), 기본 모델 87.96 표시
own_real_ci.png    : 실로봇 성공률과 Wilson 95% 구간, 본문 값 대 표 합계
own_collision.png  : 토이 T2, 환경 점 밀도별 충돌 검출률 대 관통 깊이
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
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#c8541a", "#9a9890"
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ANA / "toy_checks_results.json").read_text(encoding="utf-8"))


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def plot_multitrial():
    m = REC["V8_multi_trial"]
    k = np.arange(1, 5)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(k, m["observed_fail_pct"], "o-", color=ACC, label="보고값 (표 5)")
    ax.plot(k, m["independent_pred_fail_pct"], "s--", color=GREY, label="시도 독립 가정 (12.04%의 k제곱)")
    ax.plot(k, m["beta_fit"]["pred_fail_pct_k1_to_4"], "^:", color=ORG, label="장면별 성공 확률 베타 혼합 적합")
    for x, y in zip(k, m["observed_fail_pct"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=ACC)
    ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xlabel("시도 횟수 (하나라도 성공하면 성공)")
    ax.set_ylabel("실패율 (%)")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_multitrial.png", dpi=200)
    plt.close(fig)


def plot_scaling():
    n = REC["V5_scaling_table5"]["assumed_n"]
    params = [("10M", 79.63), ("50M", 86.77), ("200M", 87.96), ("1B", 88.07)]
    objs = [("12K", 83.65), ("24K", 84.15), ("36K", 85.29), ("48K", 86.24)]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, data, title in ((axes[0], params, "모델 크기"), (axes[1], objs, "학습 객체 수")):
        x = np.arange(len(data))
        y = np.array([v for _, v in data])
        err = 1.959964 * 100 * np.sqrt(y / 100 * (1 - y / 100) / n)
        ax.errorbar(x, y, yerr=err, fmt="o-", color=ACC, capsize=3)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.axhline(87.96, color=ORG, lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([d for d, _ in data])
        ax.set_title(title, fontsize=10)
        style(ax)
    axes[1].text(0.02, 88.15, "기본 모델 87.96 (표 2, 학습 객체 47,737)", color=ORG, fontsize=8)
    axes[0].set_ylabel("성공률 (%)")
    axes[0].set_ylim(78, 90)
    fig.tight_layout()
    fig.savefig(FIG / "own_scaling.png", dpi=200)
    plt.close(fig)


def plot_real():
    rows = [("XHand\n(표 7)", 85, 100, 85), ("LEAP\n(표 7)", 83, 100, 82), ("Wuji\n(표 8)", 340, 414, 82.1),
            ("XHand\n(표 8)", 456, 548, 90.6)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (lab, k, n, claim) in enumerate(rows):
        lo, hi = wilson(k, n)
        p = 100 * k / n
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color=ACC, capsize=4)
        ax.plot(i + 0.18, claim, "x", color=ORG, ms=8)
        ax.annotate(f"{k}/{n}", (i, hi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(70, 96)
    ax.plot([], [], "o", color=ACC, label="표 합계와 Wilson 95% 구간")
    ax.plot([], [], "x", color=ORG, label="본문이 말한 값")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def plot_collision():
    t2 = TOY["T2_collision_rule"]
    d = t2["depths_mm"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cols = [GREY, ORG, ACC, INK]
    for c, (name, v) in zip(cols, t2["by_density"].items()):
        rho = name.replace("rho_", "").replace("_per_cm2", "")
        ax.plot(d, [100 * x for x in v["detect_rate"]], "o-", color=c, ms=4, label=f"환경 점 {rho}개/cm²")
    ax.set_xlabel("구가 평면을 뚫고 들어간 깊이 (mm)")
    ax.set_ylabel("충돌로 판정된 비율 (%)")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_collision.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_multitrial()
    plot_scaling()
    plot_real()
    plot_collision()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
