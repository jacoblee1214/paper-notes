"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table2.png    : 표 2 재계산. 동작별 NDM 최선 칸 수, Holm 보정 검정 결과, 직접 전이 대비 감소율
own_real.png      : 표 4 실로봇 추종 오차 감소율과 Welch p (논문의 3회 반복 기준)
own_explore.png   : 표 5 흔들림 비율(논문)과 토이 T1 비율 비교, 토이 파라미터 식별 오차
own_sigma.png     : 식 1-2의 σ가 DFT 규약에 따라 시간 영역 RMS 몇 도인가 (T = 8 s, kc = 1 Hz)
실행: python3 plot_figures.py   (numpy, matplotlib; 한글 폰트는 Noto Sans CJK KR, Apple SD Gothic Neo 순서)
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from table_data import MOTIONS, T4, T5  # noqa: E402

FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#b5432f", "#3b7d4f"
R = json.load(open(HERE / "recompute_results.json"))
TOY = json.load(open(HERE / "toy_fence_results.json"))


def fig_table2():
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    bc = R["V6_per_motion_best"]
    x = np.arange(len(MOTIONS))
    ax[0].bar(x, [bc[m]["ndm_best_among_4"] for m in MOTIONS], color=ACC)
    ax[0].axhline(12, color=GREY, lw=0.8, ls=":")
    ax[0].set_xticks(x, MOTIONS)
    ax[0].set_ylabel("NDM이 4개 방법 중 최소인 칸 수 (12칸 중)")
    ax[0].set_title("표 2: 동작별 NDM 최선 칸 (합계 40/60)", fontsize=9.5)
    bet, wor = R["V4_ndm_vs_best_baseline"]["per_motion_holm_better"], R["V4_ndm_vs_best_baseline"]["per_motion_holm_worse"]
    ax[1].bar(x - 0.2, [bet[m] for m in MOTIONS], 0.4, color=GRN, label="NDM이 낮음")
    ax[1].bar(x + 0.2, [wor[m] for m in MOTIONS], 0.4, color=RED, label="NDM이 높음")
    ax[1].set_xticks(x, MOTIONS)
    ax[1].set_ylabel("Holm 보정 p < 0.05인 칸 수")
    ax[1].set_title("NDM 대 ASAP·RWM 중 낮은 쪽 (Welch, 5회)", fontsize=9.5)
    ax[1].legend(fontsize=8, frameon=False)
    red = R["V2_reduction_vs_direct"]
    names = ["ASAP", "RWM", "NDM"]
    w = 0.35
    xs = np.arange(3)
    ax[2].bar(xs - w / 2, [red[n]["link_pos"]["mean_of_ratios_reduction_pct"] for n in names], w, color=ACC, label="Elink pos")
    ax[2].bar(xs + w / 2, [red[n]["link_rot"]["mean_of_ratios_reduction_pct"] for n in names], w, color=GREY, label="Elink rot")
    for i, n in enumerate(names):
        ax[2].text(i - w / 2, red[n]["link_pos"]["mean_of_ratios_reduction_pct"] + 1, f'{red[n]["link_pos"]["mean_of_ratios_reduction_pct"]:.0f}', ha="center", fontsize=8)
        ax[2].text(i + w / 2, red[n]["link_rot"]["mean_of_ratios_reduction_pct"] + 1, f'{red[n]["link_rot"]["mean_of_ratios_reduction_pct"]:.0f}', ha="center", fontsize=8)
    ax[2].set_xticks(xs, names)
    ax[2].set_ylabel("직접 전이 대비 평균 감소율 (%)")
    ax[2].set_title("15개 설정 비율의 평균 (논문: 30·47, 23·27, 9·35)", fontsize=9.5)
    ax[2].legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_table2.png", dpi=170)
    plt.close(fig)


def fig_real():
    tasks = list(T4.keys())
    lab = {"swallow_balance": "Swallow\nBalance", "arm_swing": "Arm\nSwing", "pen_capping": "Pen\nCapping",
           "lateral_raise": "Lateral\nRaise", "walking": "Walking", "bowing": "Bowing"}
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(tasks))
    w = 0.38
    imp = R["V9_table4_tracking_reduction_pct"]
    lp = [imp[t]["link_pos"] for t in tasks]
    qp = [imp[t]["qpos"] for t in tasks]
    ax.bar(x - w / 2, lp, w, color=ACC, label="Elink pos")
    ax.bar(x + w / 2, qp, w, color=GREY, label="Eqpos")
    for i, t in enumerate(tasks):
        pl = R["V9_table4"][f"{t}|NDM vs Direct|link_pos"]["p"]
        pq = R["V9_table4"][f"{t}|NDM vs Direct|qpos"]["p"]
        for off, v, p in ((-w / 2, lp[i], pl), (w / 2, qp[i], pq)):
            txt = "SD 0" if p is None else f"p={p:.2f}" if p >= 0.001 else "p<.001"
            ax.text(i + off, v + 1, txt, ha="center", fontsize=7)
    ax.set_xticks(x, [lab[t] for t in tasks])
    ax.set_ylabel("직접 전이 대비 감소율 (%)")
    ax.set_title("표 4 실로봇: 미세조정 후 추종 오차 감소율 (3회 반복 Welch p)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real.png", dpi=170)
    plt.close(fig)


def fig_explore():
    meths = ["Action Gaussian", "Obs Gaussian", "Action OU", "FENCE", "No Noise"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    x = np.arange(len(meths))
    w = 0.38
    paper = [T5[m]["sqpos"][0] / T5["FENCE"]["sqpos"][0] for m in meths]
    toy = [TOY[m]["s_q"] / TOY["FENCE"]["s_q"] for m in meths]
    ax[0].bar(x - w / 2, paper, w, color=ACC, label="논문 표 5 Sqpos (실로봇)")
    ax[0].bar(x + w / 2, toy, w, color=GREY, label="토이 T1 관절 흔들림")
    ax[0].set_yscale("log")
    ax[0].axhline(1, color=INK, lw=0.6)
    ax[0].set_xticks(x, [m.replace(" ", "\n") for m in meths], fontsize=8)
    ax[0].set_ylabel("FENCE 대비 비율 (로그)")
    ax[0].set_title("같은 크기 편차에서 흔들림 비율", fontsize=9.5)
    ax[0].legend(fontsize=8, frameon=False)
    errs = [abs(TOY[m]["I_err_pct"]) for m in meths]
    gerr = [abs(TOY[m]["G_err_pct"]) for m in meths]
    ax[1].bar(x - w / 2, errs, w, color=RED, label="관성 I 식별 오차")
    ax[1].bar(x + w / 2, gerr, w, color=GRN, label="중력 토크 G 식별 오차")
    ax[1].set_xticks(x, [m.replace(" ", "\n") for m in meths], fontsize=8)
    ax[1].set_ylabel("|상대 오차| (%)")
    ax[1].set_title("토이 T1: 20궤적 최소제곱 식별 (엔코더 잡음 0.002 rad)", fontsize=9.5)
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_explore.png", dpi=170)
    plt.close(fig)


def fig_sigma():
    sig = np.array([1, 2, 3, 4, 6])
    Tn, kc = 400, 8
    un = np.degrees(sig * np.sqrt(2 * kc) / Tn)
    orth = np.degrees(sig * np.sqrt(2 * kc) / np.sqrt(Tn))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(sig, un, "o-", color=ACC, label="식 (1)(2) 그대로 (비정규화 DFT)")
    ax.plot(sig, orth, "s-", color=RED, label="정규화(ortho) DFT로 읽을 때")
    toy = TOY["fence_sigma_to_rms_dev_rad"]
    ax.plot([1, 3, 6], [np.degrees(toy[k]) for k in ("1", "3", "6")], "^", color=GRN, label="토이 T1 롤아웃 편차 (T = 10 s, PD 추종)")
    ax.set_yscale("log")
    ax.set_xlabel("σ (그림 10의 고정 값 1, 2, 3, 4, 6)")
    ax.set_ylabel("시간 영역 RMS 섭동 (도, 로그)")
    ax.set_title("σ가 뜻하는 크기는 DFT 규약에 따라 20배 다르다 (T = 8 s, kc = 1 Hz)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sigma.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_table2()
    fig_real()
    fig_explore()
    fig_sigma()
    print("saved", sorted(p.name for p in FIG.glob("own_*.png")))
