"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다: 라이선스가 CC BY가 아님).

own_sim_ci.png : 표 I 과제별 성공률과 Wilson 95% 구간 (ACT, VITaL, Ours), 값 직접 표기
own_scale.png  : 표 III 데이터 규모별 과제 평균과 과제별 선 (회색)
own_real_ci.png: 표 IV 실로봇 성공률과 Wilson 95% 구간 (시각 단독 대 시각+UniVTAC)

색: dataviz 기준 팔레트 슬롯 1–3 (blue, orange, aqua; 라이트 모드). aqua는 대비 3:1 미만이라
값을 직접 적는다 (validate_palette.js: CVD 통과, 대비 WARN -> 직접 라벨 필수).
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recompute as rc  # noqa: E402  (표 값과 Wilson 구간을 한 곳에서 가져옴)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, GREY = "#1b1b1a", "#55544f", "#e4e2dc", "#b5b3ab"
SLOT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
N = rc.N_SIM


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


# 1) 표 I: 과제별 점과 Wilson 구간. 세로축 = 과제, 가로축 = 성공률.
def fig_sim():
    methods = [("ACT", "ACT (시각만)", SLOT["blue"], 0.26),
               ("VITaL", "VITaL", SLOT["orange"], 0.0),
               ("Ours", "ACT + UniVTAC Encoder", SLOT["aqua"], -0.26)]
    tasks = rc.TASKS
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    y = np.arange(len(tasks))[::-1]
    for key, label, col, off in methods:
        vals = rc.T1[key]
        for i, v in enumerate(vals):
            lo, hi = rc.wilson(round(v), N)
            ax.plot([lo * 100, hi * 100], [y[i] + off, y[i] + off], color=col, lw=1.4, alpha=0.85, solid_capstyle="round")
            ax.plot(v, y[i] + off, "o", ms=5.5, color=col, mec="white", mew=0.6)
            ax.annotate(f"{v}", (v, y[i] + off), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=7.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.set_xlim(0, 118)
    ax.set_xlabel("성공률 (%) 과 Wilson 95% 구간, 과제당 100회", color=INK2, fontsize=9)
    style(ax)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=1.4, label=l) for _, l, c, _ in methods]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=3, fontsize=8, frameon=False)
    ax.set_title("표 I: 과제별 성공률 (논문 보고값을 그대로 사용)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png")
    plt.close(fig)


# 2) 표 III: 데이터 규모. 과제별 선은 회색, 평균은 파랑 (단일 시리즈 강조).
def fig_scale():
    cols = rc.SCALE_COLS
    x = np.arange(len(cols))
    avg = rc.T3_REPORTED_AVG
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    for t in rc.TASKS:
        ax.plot(x, rc.T3[t], color=GREY, lw=0.9, alpha=0.9, zorder=1)
    ax.plot(x, avg, color=SLOT["blue"], lw=2.4, marker="o", ms=5.5, mec="white", mew=0.6, zorder=3)
    for i, v in enumerate(avg):
        ax.annotate(f"{v:.1f}", (x[i], v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7.8, color=INK, fontweight="bold")
    ax.text(x[-1] + 0.12, rc.T3["Grasp Classify"][-1], "Grasp Classify", fontsize=7, color=INK2, va="center")
    ax.text(x[-1] + 0.12, rc.T3["Insert Hole"][-1] - 2, "Insert Hole", fontsize=7, color=INK2, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_xlabel("사전학습 샘플 수 (논문 표 III 열)", color=INK2, fontsize=9)
    ax.set_ylabel("성공률 (%)", color=INK2, fontsize=9)
    ax.set_ylim(-2, 108)
    ax.set_xlim(-0.3, len(cols) - 0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("표 III: 과제 평균(파랑)과 과제별 선(회색)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_scale.png")
    plt.close(fig)


# 3) 표 IV: 실로봇 20회. 두 조건을 같은 행에 놓는다.
def fig_real():
    tasks = list(rc.T4.keys())
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    y = np.arange(len(tasks))[::-1]
    for key, off, col, lab in [(0, 0.16, SLOT["blue"], "시각만"), (1, -0.16, SLOT["orange"], "시각 + UniVTAC")]:
        for i, t in enumerate(tasks):
            v = rc.T4[t][key]
            k = round(v / 100 * rc.N_REAL)
            lo, hi = rc.wilson(k, rc.N_REAL)
            ax.plot([lo * 100, hi * 100], [y[i] + off] * 2, color=col, lw=1.4, alpha=0.85, solid_capstyle="round")
            ax.plot(v, y[i] + off, "o", ms=5.5, color=col, mec="white", mew=0.6)
            ax.annotate(f"{v}% ({k}/20)", (v, y[i] + off), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=7.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.set_xlim(0, 120)
    ax.set_xlabel("성공률 (%) 과 Wilson 95% 구간, 조건당 20회", color=INK2, fontsize=9)
    style(ax)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=1.4, label=l)
               for c, l in [(SLOT["blue"], "시각만"), (SLOT["orange"], "시각 + UniVTAC")]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)
    ax.set_title("표 IV: 실로봇 성공률 (논문 보고값, 20회)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_sim()
    fig_scale()
    fig_real()
    for n in ["own_sim_ci.png", "own_scale.png", "own_real_ci.png"]:
        print("wrote", FIG / n)
