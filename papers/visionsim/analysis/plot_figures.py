"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_pileup.png : 토이 T1, 신호 광자 수에 따른 파일업 편향과 검출 효율
own_itof.png   : 토이 T2, 깊이에 따른 iToF 오차 (잡음, 주변광) 와 35 MHz 되접힘
own_checks.png : 재계산 V5, V7, V10 (키프레임 탐색 V*K, 마이크로볼로미터 계단 응답, 디지털 트윈 RMSE 분해)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

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
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#b5472f"
PLAIN = FuncFormatter(lambda x, _: f"{x:g}")  # 로그 축의 유니코드 마이너스 글리프 문제를 피한다


def plain_log(ax, axis="x"):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(PLAIN)
    a.set_minor_formatter(NullFormatter())


def plot_pileup():
    res = json.loads((ANA / "toy_dtof_pileup_results.json").read_text(encoding="utf-8"))
    rows = res["rows"]
    phi = np.array([r["phi_sig_per_cycle"] for r in rows])
    bias = np.array([r["centroid_bias_cm"] for r in rows])
    frac = np.array([r["signal_photon_detection_fraction"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax[0].semilogx(phi, bias, "o-", color=RED)
    plain_log(ax[0])
    ax[0].axhline(0, color=GREY, lw=0.8)
    ax[0].set_xlabel("신호 광자 수 / 레이저 주기")
    ax[0].set_ylabel("무게중심 편향 (cm, 음수 = 더 가깝게 읽음)")
    ax[0].set_title("거리 걸음: 강한 반사일수록 얕게 읽음", fontsize=10)
    ax[1].semilogx(phi, 100 * frac, "o-", color=ACC)
    plain_log(ax[1])
    ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("신호 광자 수 / 레이저 주기")
    ax[1].set_ylabel("검출된 신호 광자 비율 (%)")
    ax[1].set_title("데드타임이 버리는 신호", fontsize=10)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_pileup.png", dpi=200)
    plt.close(fig)


def plot_itof():
    res = json.loads((ANA / "toy_itof_failures_results.json").read_text(encoding="utf-8"))
    prof = res["distance_profile_20MHz"]
    mid = np.array([np.mean(p["range_m"]) for p in prof])
    noise = np.array([p["noise_only_rmse_m"] for p in prof])
    amb = np.array([p["ambient_eq_active_rmse_m"] for p in prof])
    # 35 MHz 되접힘 곡선
    C = 299_792_458.0
    d = np.linspace(0.5, 7.0, 600)
    wrap = C / (2 * 35e6)
    read = np.where(d < wrap, d, d - wrap)
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax[0].plot(d, d, color=GREY, lw=1, label="참 깊이")
    ax[0].plot(d, read, color=RED, lw=1.6, label="35 MHz로 읽은 깊이")
    ax[0].axvline(wrap, color=INK, ls=":", lw=0.8)
    ax[0].text(wrap + 0.08, 0.7, f"c/2f = {wrap:.2f} m", fontsize=8)
    ax[0].set_xlabel("참 깊이 (m)")
    ax[0].set_ylabel("복호 깊이 (m)")
    ax[0].set_title("위상 모호: 한계를 넘으면 되접힘", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot(mid[1:], noise[1:], "o-", color=ACC, label="샷+읽기 잡음, 12비트")
    ax[1].plot(mid[1:], amb[1:], "s-", color=RED, label="+ 주변광 = 활성광")
    ax[1].set_xlabel("참 깊이 (m), 20 MHz (0.5-1.5 m 구간은 포화라 뺌)")
    ax[1].set_ylabel("깊이 RMSE (m)")
    ax[1].set_title("먼 점일수록 주변광 오차가 커짐", fontsize=10)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_itof.png", dpi=200)
    plt.close(fig)


def plot_checks():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4))
    # (a) V5: V*K
    k = np.array([5.0, 30.05, 11.17])
    v = np.array([6.01, 0.37, 0.99])
    ax[0].loglog(k, v, "o-", color=ACC, label="논문 탐색 경로 (V의 값은 역산)")
    kk = np.logspace(np.log10(4), np.log10(40), 50)
    ax[0].loglog(kk, 30.05 / kk, "--", color=GREY, label="V ∝ 1/K (K=5 기준)")
    ax[0].axhline(1.0, color=INK, ls=":", lw=0.8)
    for i, (a, b) in enumerate(zip(k, v)):
        ax[0].annotate(f"K{i}={a:g}", (a, b), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plain_log(ax[0], "x")
    plain_log(ax[0], "y")
    ax[0].set_xlabel("키프레임 배수 K")
    ax[0].set_ylabel("95퍼센타일 광류 V (px)")
    ax[0].set_title("식 S8: V·K가 30에서 11로 변함", fontsize=10)
    ax[0].legend(fontsize=7)
    # (b) V7: 마이크로볼로미터
    n = np.linspace(0, 6, 200)
    for fps, col in [(60, ACC), (100, RED)]:
        a = np.exp(-(1 / fps) / 0.010)
        ax[1].plot(n, 1 - a ** n, color=col, label=f"{fps} fps, τ=10 ms (a={a:.2f})")
    ax[1].axhline(0.9, color=GREY, ls=":", lw=0.8)
    ax[1].axvline(3, color=INK, ls="--", lw=0.8)
    ax[1].text(3.05, 0.35, "논문: 약 3프레임에\n90%", fontsize=8)
    ax[1].set_xlabel("프레임 수")
    ax[1].set_ylabel("계단 입력에 대한 응답")
    ax[1].set_title("식 S20: 90%는 1.4프레임, 3프레임이면 99.3%", fontsize=10)
    ax[1].legend(fontsize=7, loc="lower right")
    # (c) V10: RMSE 분해
    r = rec["V10_digital_twin_rmse"]["suppl_fig4_rmse_m"]
    labels = ["일반", "IRF 학습", "+ 파일업", "+ 글레어"]
    vals = [r["generic"], r["irf_learned"], r["pileup_modeled"], r["bloom_modeled"]]
    ax[2].bar(labels, vals, color=[GREY, "#8fb0d9", "#5f8fc9", ACC])
    for i, val in enumerate(vals):
        ax[2].text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    ax[2].set_ylabel("RMSE (m, 보충 그림 4)")
    ax[2].set_title("총 0.33 m 감소 (본문은 1.53 m)", fontsize=10)
    ax[2].tick_params(axis="x", labelsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_checks.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_pileup()
    plot_itof()
    plot_checks()
    print("wrote figures to", FIG)
