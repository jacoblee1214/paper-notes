"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_success_ci.png  : 표 III와 표 IV의 성공률과 Wilson 95% 구간 (평가 시퀀스 N = 10000 가정)
own_failure_mean.png: 표 III의 성공 시퀀스 평균 대 역산한 실패 시퀀스 평균 (Eg-mpjpe)
own_kernel_toy.png  : 보상 커널 토이. 같은 오차에서 논문 식(두 해석)과 코드 식의 보상 값

먼저 python3 analysis/recompute.py 를 실행한다 (recompute_results.json을 읽는다).
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c4622d"
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def plot_success():
    ci = REC["V4_success_rate_inference"]["ci"]
    left = ["Privileged", "H2O-w/o-sim2data", "H2O-reduced", "H2O"]
    right = ["0.1%", "1%", "10%", "H2O"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, names, title in ((axes[0], left, "표 III: 방법 비교"), (axes[1], right, "표 IV: 학습 데이터 크기 (정리 집합 비율)")):
        x = np.arange(len(names))
        rate = np.array([ci[n]["rate"] for n in names])
        err = np.array([ci[n]["half_width"] for n in names])
        cols = [GREY if n == "Privileged" else (ACC if n == "H2O" else "#8fb0d9") for n in names]
        ax.bar(x, rate, color=cols, width=0.6)
        ax.errorbar(x, rate, yerr=err, fmt="none", ecolor=INK, capsize=3, lw=1)
        for i, v in enumerate(rate):
            ax.text(i, v + 2.2, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names if names is left else ["0.1%", "1%", "10%", "100%"], fontsize=8, rotation=12 if names is left else 0)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("성공률 (%)")
    fig.text(0.5, 0.005, "구간 반폭은 약 ±1%p (N = 10000 시퀀스 가정). 학습 시드 변동은 논문에 없어 포함되지 않음.", ha="center", fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_success_ci.png", dpi=170)
    plt.close(fig)


def plot_failure_mean():
    v1 = REC["V1_implied_failure_means"]
    names = ["Privileged policy", "H2O-w/o-sim2data", "H2O-reduced", "H2O"]
    ok = {"Privileged policy": 46.0, "H2O-reduced": 182.5, "H2O-w/o-sim2data": 163.1, "H2O": 151.0}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(names))
    w = 0.36
    fail = [v1[n]["Eg-mpjpe"]["fail_mean"] for n in names]
    ax.bar(x - w / 2, [ok[n] for n in names], w, color=ACC, label="성공 시퀀스 평균 (표 III 오른쪽)")
    ax.bar(x + w / 2, fail, w, color=ORG, label="실패 시퀀스 평균 (역산)")
    for i, n in enumerate(names):
        ax.text(i - w / 2, ok[n] + 4, f"{ok[n]:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, fail[i] + 4, f"{fail[i]:.0f}", ha="center", fontsize=8)
    ax.axhline(500, color=GREY, ls="--", lw=0.8)
    ax.text(3.45, 506, "실패 판정 거리 500 mm", ha="right", fontsize=7.5, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Eg-mpjpe (mm)")
    ax.set_ylim(0, 560)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("표 III 전역 추적 오차: 실패 시퀀스 평균은 성공 시퀀스의 1.2–1.6배", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_failure_mean.png", dpi=170)
    plt.close(fig)


def plot_kernel():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    e = np.linspace(0, 1.0, 400)
    w_pos = np.array([2.0, .5, .5, .5, .5] * 2 + [2.0] + [2.0] * 8)
    # 관절 위치: 모든 관절 같은 오차
    ax = axes[0]
    ax.plot(e, np.exp(-0.25 * 19 * e ** 2), color=ACC, label="논문 식, 19차원 합")
    ax.plot(e, np.exp(-0.25 * e ** 2), color=ACC, ls=":", label="논문 식, 평균으로 읽음")
    ax.plot(e, np.exp(-np.mean(w_pos) * e ** 2 / 0.5), color=ORG, label="코드 (sigma = 0.5, 가중 평균)")
    ax.set_title("관절 위치 (모든 관절 같은 오차, rad)", fontsize=9)
    ax.set_xlabel("오차 (rad)")
    ax.set_ylabel("보상 (0–1)")
    ax.legend(fontsize=7)
    # 몸체 위치
    ax = axes[1]
    e2 = np.linspace(0, 0.8, 400)
    ax.plot(e2, np.exp(-0.5 * 22 * e2 ** 2), color=ACC, label="논문 식, 22개 몸체 합")
    ax.plot(e2, np.exp(-0.5 * e2 ** 2), color=ACC, ls=":", label="논문 식, 몸체 평균")
    ax.plot(e2, np.exp(-(e2 ** 2 / 3) / 0.03), color=ORG, label="코드 상체 (sigma = 0.03)")
    ax.plot(e2, np.exp(-(e2 ** 2 / 3) / 0.5), color=ORG, ls=":", label="코드 하체 (sigma = 0.5)")
    ax.set_title("몸체 위치 (몸체당 같은 오차, m)", fontsize=9)
    ax.set_xlabel("오차 (m)")
    ax.legend(fontsize=7)
    # 속도
    ax = axes[2]
    e3 = np.linspace(0, 1.5, 400)
    ax.plot(e3, np.exp(-10 * 22 * e3 ** 2), color=ACC, label="논문 식, 22개 몸체 합")
    ax.plot(e3, np.exp(-10 * e3 ** 2), color=ACC, ls=":", label="논문 식, 몸체 평균")
    ax.plot(e3, np.exp(-(e3 ** 2 / 3) / 10.0), color=ORG, label="코드 (sigma = 10)")
    ax.set_title("몸체 선속도 (몸체당 같은 오차, m/s)", fontsize=9)
    ax.set_xlabel("오차 (m/s)")
    ax.legend(fontsize=7)
    for a in axes:
        a.set_ylim(0, 1.03)
    fig.tight_layout()
    fig.savefig(FIG / "own_kernel_toy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_success()
    plot_failure_mean()
    plot_kernel()
    print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
