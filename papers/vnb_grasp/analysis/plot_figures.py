"""우리가 그린 그래프 세 장. 논문 그림은 쓰지 않는다(IEEE 고지, 04 §5).
입력: analysis/table_check_results.json, analysis/cvar_toy_results.json (둘 다 같은 폴더)
출력: report/figures/*.png
실행: python3 analysis/plot_figures.py   (matplotlib, numpy, 한글 폰트는 Apple SD Gothic Neo 또는 Noto CJK 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREY = "#1f5fae", "#d9822b", "#7a7a75"


def fig_calibration_sign():
    """Table III: P_bel 과 P_emp 를 체제별로. 점이 대각선 아래면 실패를 과소 예측."""
    r = json.loads((AN / "table_check_results.json").read_text())
    regimes = ["nominal", "wide", "bimodal"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, reg in zip(axes, regimes):
        for m, col in (("Gauss", GREY), ("VNB", BLUE), ("CEM", ORANGE)):
            row = [x for x in r["calibration_rows"][m] if x["regime"] == reg][0]
            ax.scatter(row["Pemp"], row["Pbel"], s=60, color=col, label=m, zorder=3)
        ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
        ax.set_title(reg, fontsize=11)
        ax.set_xlabel("경험적 실패율 P_emp (1 - PertSurv)")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("믿음 예측 실패율 P_bel")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("그래프: Table III의 보정 방향. 점선 위는 과대, 아래는 과소 예측", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "calib_sign.png", dpi=170)
    plt.close(fig)


def fig_w_sweep():
    """토이: 낮은 마찰 봉우리 질량 w 에 따른 선택 전이(10 시드 최빈값)."""
    r = json.loads((AN / "cvar_toy_results.json").read_text())
    rows = [x for x in r["part1_w_sweep_muf1.0_geom_10seeds"] if x["beta"] == 0.95]
    labels = {"A": 0, "B": 1, "C": 2, "D": 3}
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ws = [x["w_low"] for x in rows]
    for key, col, ls in (("GMM_CVaR", BLUE, "-"), ("Gauss_CVaR", ORANGE, "--"), ("Gauss_mean", GREY, ":")):
        ys = [labels[x[key][0]] for x in rows]
        ax.step(ws, ys, where="mid", color=col, ls=ls, lw=2, label={"GMM_CVaR": "GMM + CVaR (VNB 형태)",
                                                                  "Gauss_CVaR": "단일 가우시안 + CVaR",
                                                                  "Gauss_mean": "단일 가우시안 + 기대비용"}[key])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["A 넓은 파지", "B 빠른 파지", "C 중간 파지", "D 강한 파지"], fontsize=9)
    ax.set_xlabel("낮은 마찰 봉우리(mu_o=0.18)의 질량 w")
    ax.set_title("그래프: 믿음 모양과 목적에 따른 선택 (β=0.95, mu_f=1.0, 식 22)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "cvar_w_sweep.png", dpi=170)
    plt.close(fig)


def fig_softplus():
    """토이: softplus 근사 CVaR 과 정확 CVaR 의 차이(β별 평균과 표준편차)."""
    r = json.loads((AN / "cvar_toy_results.json").read_text())["part3_softplus_bias"]
    betas = [x["beta"] for x in r]
    mean = [x["softplus_minus_exact_mean"] for x in r]
    sd = [x["softplus_minus_exact_sd"] for x in r]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.errorbar(betas, mean, yerr=sd, fmt="o-", color=BLUE, capsize=4)
    for b, m in zip(betas, mean):
        ax.annotate(f"+{m:.2f}", (b, m), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("β")
    ax.set_ylabel("softplus - exact CVaR")
    ax.set_title("그래프: 논문 Eq.(12) 근사의 편향 (κ=5, N=256, 비용 N(0,1))", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "softplus_bias.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_calibration_sign()
    fig_w_sweep()
    fig_softplus()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
