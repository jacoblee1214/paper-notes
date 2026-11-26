"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_ci.png   : 표 3, 4, D.1의 실로봇 성공률과 Wilson 95% 구간
own_table1.png    : 표 1의 시뮬레이터별 성공률(20시드, Wilson 구간)과 한 스텝 시간
own_toy_mppi.png  : 토이 T1, 비용 배율별 MPPI 보정 크기와 유효 표본 수
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c8641e"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def bars(ax, labels, ks, n, colors):
    x = np.arange(len(labels))
    pct = [100 * k / n for k in ks]
    lo = [wilson(k, n)[0] for k in ks]
    hi = [wilson(k, n)[1] for k in ks]
    ax.bar(x, pct, color=colors, width=0.62)
    ax.errorbar(x, pct, yerr=np.array([np.array(pct) - lo, np.array(hi) - pct]), fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, k in enumerate(ks):
        ax.text(i, 3, f"{k}/{n}", ha="center", va="bottom", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 105)


def plot_real():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.1, 1.1, 1.1]}, sharey=True)
    for ax, title, ks in [(axes[0], "표 3 단일 팔 대각 접기 (10회)", [9, 6, 5, 3]),
                          (axes[1], "표 3 이중 팔 대칭 접기 (10회)", [8, 3, 4, 1])]:
        bars(ax, ["전체", "추정기→DPM", "백엔드→Isaac", "MPPI→기본 정책"], ks, 10, [ACC, GREY, GREY, GREY])
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("성공률 (%), Wilson 95% 구간")
    bars(axes[2], ["긴팔", "반바지", "수건 반대 대각", "수건 중간 교란"], [8, 7, 8, 8], 10, [ORG] * 4)
    axes[2].set_title("표 4 일반화·교란 회복 (기준선 없음)", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def plot_table1():
    T = {"Newton": ([40, 40, 60, 35], [41.0, 55.1, 104.3, 232.8]),
         "Isaac Sim": ([55, 45, 55, 60], [11.7, 22.2, 41.1, 80.7]),
         "FLASH": ([100, 100, 95, 95], [19.4, 36.0, 68.9, 137.3])}
    Ks = [64, 128, 256, 512]
    cols = {"Newton": GREY, "Isaac Sim": ORG, "FLASH": ACC}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for name, (sr, st) in T.items():
        ks = [round(s * 0.2) for s in sr]
        lo = [wilson(k, 20)[0] for k in ks]
        hi = [wilson(k, 20)[1] for k in ks]
        x = np.arange(4) + {"Newton": -0.2, "Isaac Sim": 0, "FLASH": 0.2}[name]
        axes[0].errorbar(x, sr, yerr=[np.array(sr) - lo, np.array(hi) - sr], fmt="o", color=cols[name], capsize=3, label=name)
        axes[1].plot(Ks, st, "o-", color=cols[name], label=name)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(Ks)
    axes[0].set_xlabel("K (병렬 환경 수)")
    axes[0].set_ylabel("성공률 (%) 20시드, Wilson 95%")
    axes[0].set_title("표 1 성공률", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("시뮬 한 스텝 시간 (ms)")
    axes[1].set_title("표 1 스텝 시간", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1.png", dpi=200)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_mppi_refine_results.json").read_text())
    rows = res["rows"]
    sc = [r["cost_scale"] for r in rows]
    corr = [r["mean_correction_mm"] for r in rows]
    ana = [r["analytic_sigma2_over_lambda_grad_mm"] for r in rows]
    ess = [r["ESS_of_256"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.arange(len(sc))
    axes[0].bar(x - 0.18, corr, 0.36, color=ACC, label="가중 평균의 첫 행동 변화(300회 평균)")
    axes[0].bar(x + 0.18, ana, 0.36, color=GREY, label="1차 근사 (σ²/λ)·기울기")
    axes[0].axhline(10, color=ORG, ls="--", lw=1)
    axes[0].text(0, 10.8, "공칭 한 스텝 10 mm (가정)", color=ORG, fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"×{s}" for s in sc])
    axes[0].set_xlabel("비용 배율 (논문 거리항 가중 1.0 = ×1, 단위 m 가정)")
    axes[0].set_ylabel("보정 크기 (mm)")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].plot(x, ess, "o-", color=ACC)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"×{s}" for s in sc])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("비용 배율")
    axes[1].set_ylabel("유효 표본 수 (256개 중)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_mppi.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_real()
    plot_table1()
    plot_toy()
    print("wrote", [p.name for p in FIG.glob("own_*.png")])
