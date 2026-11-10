"""직접 만든 그래프 네 장 (논문 그림을 쓰지 않는다). 실행: python3 analysis/plot_figures.py → report/figures/

- own_table1_ci.png     : Table I 평균 성공률과 Wilson 95% 구간(n = 150 근사), 9셀 × 3방법
- own_coupling_ratio.png: 결합 비용 비율 1 - φ^d (몬테카를로 점 포함), 논문 비율 0.121 표시
- own_mode_split.png    : 두 봉우리 목표(h = 0.5), 끝점이 어느 모드에 떨어지는지 단계 수별
- own_nfe_error.png     : 단봉 가우시안 오라클, 단계 수별 이산화 오차와 직진성 비율(Eq.8 대 코드 길이)
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                         "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})

TC = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
TT = json.loads((ROOT / "analysis" / "transport_toy_results.json").read_text())
BLUE, GREY, ORANGE = "#1f5fae", "#8a8a85", "#c95a24"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def table1():
    cells = TC["T1_table_I"]["cells"]
    labels = [f"{c['task']} {c['split']}" for c in cells]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(cells))
    offs = {"DP": -0.22, "CFM": 0.0, "TP": 0.22}
    colors = {"DP": GREY, "CFM": "#4b4b48", "TP": ORANGE}
    for m in ("DP", "CFM", "TP"):
        mean = np.array([c[f"{m}_avg"] for c in cells])
        lo = np.array([c[f"{m}_avg_wilson95"][0] for c in cells])
        hi = np.array([c[f"{m}_avg_wilson95"][1] for c in cells])
        ax.errorbar(mean, y + offs[m], xerr=[mean - lo, hi - mean], fmt="o", ms=4,
                    color=colors[m], label={"DP": "Diffusion Policy", "CFM": "CFM", "TP": "Temporal Policy"}[m],
                    lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("평균 성공률 (Table I Average) 와 Wilson 95% 구간, n = 150 가정")
    ax.set_xlim(0.45, 1.02)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("직접 계산: 9개 셀, 방법 간 차이는 대부분 구간 안에 있음", fontsize=9)
    save(fig, "own_table1_ci.png")


def coupling():
    rows = TT["A_coupling_identity"]["table"]
    phi = np.array([r["phi"] for r in rows])
    mc = np.array([r["ratio_MC"] for r in rows])
    form = np.array([r["ratio_formula_1_minus_rho"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    grid = np.linspace(0.5, 0.995, 300)
    ax.plot(grid, 1 - grid ** 14, color=BLUE, lw=1.5, label="1 - φ^14 (단위 분산 AR(1), d = 14)")
    ax.scatter(phi, mc, color=ORANGE, zorder=3, s=18, label="몬테카를로 (20,000 쌍)")
    ax.axhline(0.121, color=GREY, ls="--", lw=1)
    ax.text(0.52, 0.14, "논문 표 II 비율 1.21/10.00 = 0.121 (약 1/8.3)", fontsize=7.5, color="#4b4b48")
    ax.set_xlabel("상태 과정의 φ (lag-1 상관)")
    ax.set_ylabel("J_TP / J_indep")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("직접 계산: 0.121을 얻으려면 lag-14 상관 ≈ 0.88이 필요", fontsize=9)
    save(fig, "own_coupling_ratio.png")


def mode_split():
    rows = [r for r in TT["C_mode_switch"]["rows"] if r["h"] == 0.5]
    rows = sorted(rows, key=lambda r: r["n"])
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    series = [("CFM_noise", "잡음 원천 CFM", GREY),
              ("TP_history_corrected", "이력 원천, 결정적 ODE", ORANGE),
              ("TP_history_SDE_g1.0", "이력 원천, SDE g = 1 (eval.py 기본)", BLUE)]
    for ax, key in zip(axes, ["frac_plus", "frac_between"]):
        for k, lab, col in series:
            ax.plot(ns, [r[k][key] for r in rows], marker="o", ms=3.5, color=col, label=lab, lw=1.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("적분 단계 수 n")
        ax.grid(alpha=0.3)
    axes[0].axhline(0.5, color="#4b4b48", ls=":", lw=1)
    axes[0].set_ylabel("끝점 비율")
    axes[0].set_title("양 모드 중 '+' 쪽 (진실 0.5)", fontsize=8.5)
    axes[1].set_title("두 모드 사이 (평균에 머묾, 진실 0)", fontsize=8.5)
    axes[0].legend(fontsize=6.8, loc="center right")
    fig.suptitle("직접 계산(toy): 두 봉우리 목표, h = 0.5", fontsize=9)
    save(fig, "own_mode_split.png")


def nfe_error():
    rows = TT["B_gaussian_oracle_phi0.991"]["per_n"]
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axes[0]
    ax.plot(ns, [r["CFM_noise"]["err_vs_own_400step_per_dim_mse"] for r in rows], marker="o", ms=3.5,
            color=GREY, label="잡음 원천 CFM")
    ax.plot(ns, [r["TP_history_corrected"]["err_vs_own_400step_per_dim_mse"] + 1e-12 for r in rows],
            marker="o", ms=3.5, color=ORANGE, label="이력 원천 TP (0 근처, 부동소수 하한)")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("적분 단계 수 n")
    ax.set_ylabel("기준 해와의 MSE (차원당)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    ax.set_title("이산화 오차 (φ = 0.991)", fontsize=8.5)
    ax = axes[1]
    ax.plot(ns, [r["CFM_noise"]["eq8_ratio_squared"] for r in rows], marker="o", ms=3.5, color=GREY,
            label="CFM: Eq.8 (제곱 기준)")
    ax.plot(ns, [r["TP_history_corrected"]["eq8_ratio_squared"] for r in rows], marker="o", ms=3.5,
            color=ORANGE, label="TP: Eq.8 (제곱 기준)")
    ax.plot(ns, [r["CFM_noise"]["code_length_ratio"] for r in rows], marker="s", ms=3.5, color="#4b4b48",
            ls="--", label="CFM: 코드 길이 기준")
    ax.plot(ns, [r["TP_history_corrected"]["code_length_ratio"] for r in rows], marker="s", ms=3.5,
            color=BLUE, ls="--", label="TP: 코드 길이 기준")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("적분 단계 수 n")
    ax.set_ylabel("직진성 비율 (1이면 직선)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)
    ax.set_title("직진성 정의에 따라 값이 1/n 또는 1 (논문 1.02)", fontsize=8.5)
    fig.suptitle("직접 계산(toy): 단봉 가우시안 오라클", fontsize=9)
    save(fig, "own_nfe_error.png")


if __name__ == "__main__":
    table1()
    coupling()
    mode_split()
    nfe_error()
