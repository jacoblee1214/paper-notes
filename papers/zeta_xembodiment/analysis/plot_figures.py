"""우리가 그린 그래프 세 장 (논문 그림은 쓰지 않는다: arXiv 라이선스가 CC BY-NC-ND 4.0).

실행: python3 analysis/plot_figures.py   → report/figures/own_*.png (numpy, matplotlib)
값은 analysis/recompute.py가 부록 표에서 읽은 것이다(import 시 recompute.py가 한 번 실행된다, 1초 안팎).
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute import OUT, T, RR, RQ4_ROWS, rq3, RQ3_COLS  # noqa: E402  (같은 폴더)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

available = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    if name in available:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
BLUE, LBLUE, ORANGE, GREY, RED, GREEN = "#1f5fae", "#8fb4e3", "#d08a3a", "#8a8a85", "#b84a3a", "#3a8a5a"


def fig_rq1_per_robot():
    """표 2의 범주 평균 뒤에 숨은 로봇별 점수 (표 8-10 재계산)."""
    names = ["Abs/World", "Abs/EEF", "EEF/World", "EEF/EEF"]
    labels = {"Abs/World": "절대 상태 · 월드 델타", "Abs/EEF": "절대 상태 · EEF 델타",
              "EEF/World": "EEF 델타 상태 · 월드 델타", "EEF/EEF": "EEF 델타 상태 · EEF 델타"}
    robots = ["Franka+logo", "Franka+green", "Franka+UMI", "UR5e+Franka", "UR5e+UMI", "Google+Franka", "GoogleRobot"]
    short = ["Franka\n+로고", "Franka\n+초록", "Franka\n+UMI", "UR5e\n+Franka 손", "UR5e\n+UMI", "Google\n+Franka 손", "Google\nRobot"]
    cols = [GREY, "#c9c9c4", LBLUE, BLUE]
    fig, ax = plt.subplots(figsize=(10.4, 4.2), dpi=160)
    x = np.arange(len(robots))
    w = 0.2
    for j, n in enumerate(names):
        ax.bar(x + (j - 1.5) * w, [OUT["V2_rq1_per_robot"][n][r] for r in robots], w, color=cols[j], label=labels[n])
    ax.axvspan(4.5, 6.5, color="#f3e3e0", alpha=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7.5)
    ax.set_ylabel("과제 진행도 평균 (%)", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    ax.set_title("RQ1: 로봇별 점수 (표 8–10 재계산). 절대 상태는 Google 두 변형에서만 ≈1%로 붕괴", fontsize=9)
    ax.text(5.5, 98, "Google 계열", ha="center", fontsize=7.5, color=RED)
    fig.tight_layout()
    fig.savefig(FIG / "own_rq1_per_robot.png")
    plt.close(fig)


def fig_rq4_oracle():
    """표 17-19: 목표 로봇 노출 비율별 과제 점수와 오라클."""
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.7), dpi=160, sharey=True)
    xs = [0, 5, 30, 100]
    tasks = ["소시지 서빙", "그릇 쌓기", "큐브 쌓기"]
    tcol = [ORANGE, BLUE, GREEN]
    for ax, tgt in zip(axs, ["UR5e+UMI", "GoogleRobot"]):
        for k in range(3):
            ys = [OUT["V6_rq4"][RQ4_ROWS[i]][tgt]["per_task"][k] for i in range(4)]
            ax.plot(range(4), ys, "-o", color=tcol[k], ms=4, lw=1.4, label=tasks[k])
            orc = OUT["V6_rq4"][RQ4_ROWS[4]][tgt]["per_task"][k]
            ax.axhline(orc, color=tcol[k], ls=":", lw=1)
            ax.text(3.08, orc, f"{orc:.0f}", color=tcol[k], fontsize=7, va="center")
        avg = [OUT["V6_rq4"][RQ4_ROWS[i]][tgt]["mean"] for i in range(4)]
        ax.plot(range(4), avg, "-s", color="#222", ms=4, lw=2, label="3과제 평균")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{v}%" for v in xs], fontsize=8)
        ax.set_xlabel("사전학습에 섞은 목표 로봇 데이터 비율", fontsize=8)
        ax.set_title(f"{tgt} (점선: 목표 로봇 후학습 오라클)", fontsize=8.8)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(-0.2, 3.4)
    axs[0].set_ylabel("과제 진행도 (%)", fontsize=8.5)
    axs[0].legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_rq4_oracle.png")
    plt.close(fig)


def fig_rq3_rq2real():
    """왼쪽: RQ3 실행 3회 집계값(점)과 평균. 오른쪽: RQ2 실세계 셀(대상 x 과제)별 1, 8, 512 소스 점수."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=160, gridspec_kw={"width_ratios": [1, 1.1]})
    labels = ["공동학습\n없음", "언어-행동\n(EEF)", "언어-행동\n(두 프레임)", "하위 목표", "BBox"]
    for i, n in enumerate(RQ3_COLS):
        ys = rq3[n]["Average"]
        ax.scatter(i + np.array([-0.08, 0, 0.08]), ys, color=BLUE if i else GREY, s=22, zorder=3)
        ax.hlines(ys.mean(), i - 0.25, i + 0.25, color="#222", lw=1.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("4범주 평균 진행도 (%), 실행 3회", fontsize=8.5)
    ax.set_title("RQ3: 실행별 집계값 (표 14–16 재계산)", fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.set_ylim(72, 85)
    xs = np.arange(3)
    for r in RR[1:]:
        for t, ls in ((22, "-"), (23, "--")):
            ys = T[t][r]
            hl = r in ("Piper-on-Quadruped", "Humanoid robot", "UR")
            bx.plot(xs, ys, ls, color=RED if hl else "#b9b9b3", lw=1.4 if hl else 0.8, marker="o", ms=3)
    avg8 = [OUT["V7_rq2_real"][n]["Average"] for n in ("1", "8", "512")]
    bx.plot(xs, avg8, "-s", color="#222", lw=2.2, ms=5, label="범주 평균(표 기준)")
    bx.set_xticks(xs)
    bx.set_xticklabels(["1 소스", "8 소스", "512 소스"], fontsize=8)
    bx.set_ylabel("진행도 (%)", fontsize=8.5)
    bx.set_title("RQ2 실세계: 14개 셀 (실선 fryer, 점선 water; 붉은색 UR·휴머노이드·Piper)", fontsize=8.8)
    bx.tick_params(labelsize=7.5)
    bx.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_rq3_rq2real.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_rq1_per_robot()
    fig_rq4_oracle()
    fig_rq3_rq2real()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
