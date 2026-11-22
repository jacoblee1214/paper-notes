"""우리가 그린 그래프 세 장 (논문 그림은 쓰지 않는다: arXiv 라이선스가 CC BY가 아님).

실행: python3 analysis/plot_figures.py   → report/figures/own_*.png (numpy, matplotlib, scipy)
값은 analysis/recompute.py가 쓰는 표 값과 같다(import 시 recompute.py가 한 번 실행된다, 1초 안팎).
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from recompute import wilson, F10, F13, T6, T7, R, cats, fisher  # noqa: E402  (같은 폴더)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

available = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    if name in available:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, LBLUE, ORANGE, GREY, RED = "#1f5fae", "#8fb4e3", "#d08a3a", "#8a8a85", "#b84a3a"


def fig_real_robot():
    """그림 10 막대 라벨과, 범주당 시도 수가 N일 때의 유의성(합산 Fisher)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=160, gridspec_kw={"width_ratios": [1.35, 1]})
    names = ["전문가", "범용", "π0.5", "무사전학습 전문가", "무사전학습 범용"]
    cols = [BLUE, LBLUE, ORANGE, GREY, "#c9c9c4"]
    x = np.arange(len(cats))
    width = 0.16
    N = 20
    for j, nm in enumerate(names):
        ps = np.array([F10[c][j] / 100 for c in cats])
        lo = np.array([wilson(round(p * N), N)[0] for p in ps])
        hi = np.array([wilson(round(p * N), N)[1] for p in ps])
        xs = x + (j - 2) * width
        ax.bar(xs, ps, width, color=cols[j], label=nm)
        ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt="none", ecolor="#333", elinewidth=0.7, capsize=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["공간", "장기", "양손", "일반화"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("성공률", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.8, ncol=3, loc="upper center", frameon=False)
    ax.set_title("그림 10 막대 라벨 (오차막대: 범주당 20회 가정 Wilson 95%)", fontsize=8.8)

    Ns = np.array([20, 40, 80, 160, 320])
    def pooled(i, j, N):
        a = sum(round(F10[c][i] * N / 100) for c in cats)
        b = sum(round(F10[c][j] * N / 100) for c in cats)
        return fisher(a, 4 * N, b, 4 * N)
    series = [("전문가 대 π0.5", 0, 2, BLUE), ("범용 대 π0.5", 1, 2, LBLUE), ("범용 대 무사전학습 범용", 1, 4, RED),
              ("전문가 대 범용", 0, 1, GREY)]
    for lab, i, j, c in series:
        bx.plot(Ns, [max(pooled(i, j, int(n)), 1e-12) for n in Ns], marker="o", ms=3.5, lw=1.4, color=c, label=lab)
    bx.axhline(0.05, color="#444", lw=0.8, ls="--")
    bx.text(Ns[-1], 0.06, "0.05", fontsize=7, ha="right")
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "1e%d" % round(math.log10(v))))
    bx.set_ylim(5e-13, 3)
    bx.set_xticks(Ns)
    bx.set_xticklabels([str(int(n)) for n in Ns], fontsize=7.5)
    bx.tick_params(labelsize=7.5)
    bx.set_xlabel("범주당 시도 수 N (네 범주 합산 4N)", fontsize=8)
    bx.set_ylabel("Fisher p (합산, 독립 가정)", fontsize=8)
    bx.legend(fontsize=6.8, frameon=False, loc="lower left")
    bx.set_title("시도 수에 따른 유의성 (막대 값 고정)", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_power.png")
    plt.close(fig)


def fig_ablation():
    """표 6, 7: 사전학습 이득과 기준선(native VLM) 평균."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=160)
    cfgs = list(T6)
    lab = ["Und+Proj+ViT", "Und+ViT", "Und만", "Proj+ViT", "전체 미세조정"]
    x = np.arange(len(cfgs))
    w = 0.38
    for off, T, nm, c in ((-w / 2, T6, "표 6 단일 과제 스위트", BLUE), (w / 2, T7, "표 7 다중 과제 스위트", LBLUE)):
        g = np.array([T[k]["gain"][4] for k in cfgs])
        se = []
        for k in cfgs:
            p1, p2 = T[k]["hc"][4] / 100, T[k]["native"][4] / 100
            se.append(1.96 * math.sqrt(p1 * (1 - p1) / 2000 + p2 * (1 - p2) / 2000) * 100)
        ax.bar(x + off, g, w, color=c, label=nm)
        ax.errorbar(x + off, g, yerr=se, fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7.2)
    ax.set_ylabel("평균 이득 (%p)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.2, frameon=False)
    ax.set_title("사전학습 이득 (평균, 95% 구간: 스위트당 500회 가정)", fontsize=8.8)

    for T, nm, c, m in ((T6, "표 6", BLUE, "o"), (T7, "표 7", LBLUE, "s")):
        bx.plot(x, [T[k]["native"][4] for k in cfgs], marker=m, color=c, lw=1.4, label=f"{nm} native VLM 시작")
        bx.plot(x, [T[k]["hc"][4] for k in cfgs], marker=m, color=c, lw=1.0, ls="--", label=f"{nm} 사전학습 시작")
    bx.set_xticks(x)
    bx.set_xticklabels(lab, fontsize=7.2)
    bx.set_ylabel("평균 성공률 (%)", fontsize=8.5)
    bx.tick_params(labelsize=7.5)
    bx.legend(fontsize=6.8, frameon=False, loc="lower right")
    bx.set_title("기준선이 일정하지 않다 (표 6: 51.3~78.2)", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_gain.png")
    plt.close(fig)


def fig_mpg():
    """MPG 게이트 장난감: D와 g가 정렬도에 어떻게 반응하는가 (식 14-15, 공식 코드 계산)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.7), dpi=160)
    D = np.linspace(0, 4, 100)
    for tau, c in ((1.0, BLUE), (2.0, ORANGE)):
        ax.plot(D, np.exp(-D / tau), color=c, lw=1.6, label=("τ = 1 (공개 학습 스크립트 값)" if tau == 1 else "τ = 2 (공개 체크포인트 설정 값)"))
    g = R["V13_mpg_toy"]["gate_vs_alignment"]
    for rho, v in g.items():
        ax.scatter([v["D"]], [v["g_tau1"]], color=BLUE, s=18, zorder=3)
        ax.scatter([v["D"]], [v["g_tau2"]], color=ORANGE, s=18, zorder=3)
        ax.annotate(f"ρ={rho}", (v["D"], v["g_tau2"]), textcoords="offset points", xytext=(4, 5), fontsize=6.8)
    ax.set_xlabel("D (LayerNorm 뒤 평균 제곱 거리 ÷ 차원)", fontsize=8)
    ax.set_ylabel("게이트 g = exp(-D/τ)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.set_title("정렬도 ρ가 0에서 0.8로 가도 g는 0.14→0.67 (τ=1)", fontsize=8.8)

    sc = R["V13_mpg_toy"]["D_after_rescaling_features"]
    bx.bar(range(len(sc)), list(sc.values()), color=GREY)
    bx.set_xticks(range(len(sc)))
    bx.set_xticklabels([f"특징 크기 ×{k}" for k in sc], fontsize=7.5)
    bx.set_ylim(0, 1.5)
    bx.set_ylabel("D", fontsize=8)
    bx.tick_params(labelsize=7.5)
    bx.set_title("특징 크기를 100배 바꿔도 D 불변 (LayerNorm이 지움)", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(FIG / "own_mpg_gate.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_real_robot()
    fig_ablation()
    fig_mpg()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
