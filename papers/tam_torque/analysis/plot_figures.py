"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_ci.png     : 실로봇 성공률(상자, flip)과 Wilson 95% 구간, 공 MPC 평균 ± SD
own_sim_ci.png      : 표 2 실로봇 추종 오차 (Welch 검정 표시)와 표 3 교차 로봇 비율
own_toy_residual.png: 토이 T1, 조건별 RMSE와 부트스트랩 95% 구간
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
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_real():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    labels = ["Direct", "Online SysID", "RMA", "TAM (w/ RMA)", "TAM"]
    counts = [rec["box_counts"][k][0] for k in labels]
    x = np.arange(len(labels))
    pct = [100 * c / 21 for c in counts]
    lo = [wilson(c, 21)[0] for c in counts]
    hi = [wilson(c, 21)[1] for c in counts]
    yerr = np.array([np.array(pct) - np.array(lo), np.array(hi) - np.array(pct)])
    ax.bar(x, pct, color=[GREY, GREY, GREY, "#8fb0d9", ACC], width=0.62)
    ax.errorbar(x, pct, yerr=yerr, fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, c in enumerate(counts):
        ax.text(i, 2, f"{c}/21", ha="center", va="bottom", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("상자 밀기 (실로봇 21회, Wilson 95%)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    names = ["Direct", "Online SysID", "TAM"]
    means = [8.8, 13.1, 13.6]
    sds = [2.4, 2.3, 2.9]
    ax.bar(range(3), means, yerr=sds, color=[GREY, GREY, ACC], width=0.6, capsize=4, ecolor=INK)
    ax.axhline(15.9, color="#c44e3a", ls="--", lw=1)
    ax.text(-0.4, 16.5, "이상 모델 15.9", fontsize=7.5, color="#c44e3a", ha="left")
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("15 s 동안 목표 수")
    ax.set_ylim(0, 19)
    ax.set_title("공 위 막대 MPC (20회, 평균 ± SD 가정)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def plot_sim():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    conds = ["Direct", "Online\nSysID", "PhysPred", "TAM"]
    # 표 2 실로봇 추종 오차 (cm, 평균 ± SD, 10회)
    hidden_reset = [4.09, 2.67, 1.74, 1.51]
    hidden_sd = [0.39, 0.63, 0.50, 0.30]
    hidden_carried = [np.nan, 2.75, 1.64, 1.04]
    hidden_carried_sd = [np.nan, 0.58, 0.56, 0.24]
    x = np.arange(4)
    w = 0.38
    ax.bar(x - w / 2, hidden_reset, w, yerr=hidden_sd, color="#b9b6ad", capsize=3, ecolor=INK, label="Reset")
    ax.bar(x + w / 2, hidden_carried, w, yerr=hidden_carried_sd, color=ACC, capsize=3, ecolor=INK, label="Carried")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylabel("OSC 목표 위치 RMSE (cm)")
    ax.set_title("실로봇 추종 (숨은 파라미터, 10회)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    robots = ["Panda", "Piper", "iiwa14", "Flexiv", "RBY-1", "Google*", "Z1*"]
    direct = [4.16, 4.88, 3.69, 4.06, 3.80, 4.69, 6.11]
    sysid = [1.43, 0.83, 0.99, 1.33, 1.39, 1.65, 1.05]
    tam = [0.59, 0.52, 0.56, 0.84, 0.84, 1.05, 0.93]
    xx = np.arange(len(robots))
    ax.plot(xx, direct, "o-", color=GREY, label="Direct", ms=4)
    ax.plot(xx, sysid, "s-", color="#6f8fb8", label="Online SysID", ms=4)
    ax.plot(xx, tam, "^-", color=ACC, label="TAM", ms=4)
    ax.set_xticks(xx)
    ax.set_xticklabels(robots, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("관절 RMSE (deg, 100회 평균)")
    ax.set_title("시뮬레이션 교차 로봇 (* 테스트 로봇)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png", dpi=200)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_tam_2link_results.json").read_text(encoding="utf-8"))["results_rmse_deg"]
    conds = list(res.keys())
    methods = [m for m in res[conds[0]] if isinstance(res[conds[0]][m], dict) and not m.startswith("Diag")]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    w = 0.16
    x = np.arange(len(conds))
    colors = ["#b9b6ad", "#1f5fae", "#8fb0d9", "#c9dcf0", "#e0b36b", "#6a9a6a"][:len(methods)]
    for i, m in enumerate(methods):
        vals = [res[c][m]["rmse_deg_mean"] for c in conds]
        lo = [res[c][m]["rmse_deg_mean"] - res[c][m]["ci95"][0] for c in conds]
        hi = [res[c][m]["ci95"][1] - res[c][m]["rmse_deg_mean"] for c in conds]
        ax.bar(x + (i - 2) * w, vals, w, yerr=[lo, hi], color=colors[i], capsize=2, ecolor=INK, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(["학습 분포", "속도 2배", "위치 이동", "마찰 2배", "마찰 3배", "질량 1.3배"][:len(conds)], fontsize=8)
    ax.set_ylabel("관절 RMSE vs 이상 팔 (deg)")
    ax.set_title("토이 T1: 토크 잔차 보정 (2-링크, 120 롤아웃, 부트스트랩 95%)", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, max(res[c][m]["ci95"][1] for c in conds for m in methods) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_residual.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_real()
    plot_sim()
    plot_toy()
    print("wrote figures to", FIG)
