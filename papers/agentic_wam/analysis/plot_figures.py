"""리포트 그래프. table_check.py, noise_toy.py를 먼저 실행할 것.

  report/figures/robotwin_headline.png   Table 1 + RoboTwin 공식 리더보드 π0.5 (clean / random)
  report/figures/robotwin_pertask.png    과제별: 논문 π0.5 vs 리더보드 π0.5, AW-0 기본 vs 하이브리드(random)
  report/figures/ablation_ladder.png     실제 로봇 8과제 평균: 변형별 점수와 표본 오차 상한
  report/figures/noise_sigma.png         학습 video σ 분포(사전학습·혼합·균등)와 추론 단계별 σ
실행: python3 analysis/plot_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK, INK2, GRID, MUTED = "#1f1e1c", "#52514e", "#e6e5e0", "#a9a8a2"
# dataviz 기준 팔레트 슬롯 1–5 (고정 순서). AW-0가 슬롯 1.
C_AW, C_PI, C_HY, C_DZ, C_WLA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

r = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
lb = r["L1_leaderboard"]
T11 = r["_table11"]


def robotwin_headline():
    methods = [("π0.5 (논문 재현)", 54.9, 27.7, C_PI, None), ("π0.5 (RoboTwin 리더보드)", lb["pi05_leaderboard"]["clean"],
               lb["pi05_leaderboard"]["rand"], C_PI, "//"),
               ("DreamZero", 33.6, 17.6, C_DZ, None), ("WLA", 64.8, 3.6, C_WLA, None),
               ("AW-0", 80.8, 20.6, C_AW, None), ("AW-0 hybrid", 63.1, 29.2, C_HY, None)]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    w = 0.13
    x = np.arange(2)
    for i, (name, c, rr, col, hatch) in enumerate(methods):
        xs = x + (i - (len(methods) - 1) / 2) * (w + 0.012)
        bars = ax.bar(xs, [c, rr], w, color="white" if hatch else col, edgecolor=col, hatch=hatch,
                      linewidth=1.2, label=name, zorder=3)
        for b, v in zip(bars, [c, rr]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2, color=INK2)
    ax.set_xticks(x, ["Clean (학습과 같은 장면)", "Random (도메인 무작위화)"])
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 95)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("RoboTwin 2.0 (50과제, clean 시연 50개/과제로 학습). 빗금 = 같은 프로토콜의 공식 리더보드 π0.5",
                 fontsize=8.5, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "robotwin_headline.png", dpi=200)
    plt.close(fig)


def robotwin_pertask():
    names = list(T11.keys())
    arr = np.array([T11[n] for n in names], dtype=float)
    lbc, lbr = np.array(r["_leaderboard_pi05_per_task"]["lb_clean"]), np.array(r["_leaderboard_pi05_per_task"]["lb_rand"])
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
    jit = lambda n: np.random.default_rng(1).uniform(-0.8, 0.8, n)
    a = axs[0]
    a.plot([0, 100], [0, 100], color=MUTED, lw=1, zorder=1)
    a.scatter(lbc + jit(50), arr[:, 0], s=16, color=C_PI, label="clean", zorder=3, edgecolor="white", linewidth=0.5)
    a.scatter(lbr + jit(50), arr[:, 1], s=16, color=C_PI, marker="^", facecolor="white", label="random", zorder=3, linewidth=1)
    a.set_xlabel("리더보드 π0.5 (%)")
    a.set_ylabel("논문 재현 π0.5 (%)")
    a.set_title(f"(a) π0.5 과제별: 논문 값이 낮은 과제 clean {lb['tasks_paper_lower_clean']}/50,"
                f" random {lb['tasks_paper_lower_rand']}/50", fontsize=8, color=INK2, loc="left")
    a.legend(frameon=False, fontsize=7.5, loc="upper left")
    a.set_xlim(-3, 103); a.set_ylim(-3, 103)
    b = axs[1]
    b.plot([0, 100], [0, 100], color=MUTED, lw=1, zorder=1)
    d = arr[:, 11] - arr[:, 9]
    b.scatter(arr[:, 9], arr[:, 11], s=16, color=C_HY, zorder=3, edgecolor="white", linewidth=0.5)
    for i in np.argsort(-d)[:3]:
        b.annotate(names[i], (arr[i, 9], arr[i, 11]), xytext=(5, 5), textcoords="offset points", fontsize=6.8, color=INK2)
    b.set_xlabel("AW-0 기본, random (%)")
    b.set_ylabel("AW-0 hybrid, random (%)")
    s1 = r["S1_robotwin"]["hybrid vs AW-0 rand"]
    b.set_title(f"(b) hybrid가 나은 과제 {s1['wins']}, 못한 과제 {s1['losses']} (부호 검정 p={s1['sign_p']:.2f})",
                fontsize=8, color=INK2, loc="left")
    b.set_xlim(-3, 103); b.set_ylim(-3, 103)
    for a_ in axs:
        a_.grid(True, color=GRID, zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "robotwin_pertask.png", dpi=200)
    plt.close(fig)


def ablation_ladder():
    rows = [("V → A (코드 전문가 없음, ViT도 없음)", 32.9, MUTED), ("V → C → A (코드가 video 뒤)", 48.3, MUTED),
            ("AR 코드 생성", 54.5, MUTED), ("코드 전문가 입력을 VAE 특징으로", 55.6, MUTED),
            ("주석 없는 코드", 62.8, C_DZ), ("스킬 없는 코드", 63.3, C_DZ), ("자연어 subgoal", 63.6, C_DZ),
            ("AW-0 (C → V → A, 전체 코드)", 71.1, C_AW)]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    y = np.arange(len(rows))
    for i, (name, v, col) in enumerate(rows):
        se = math.sqrt(sum(0.25 for _ in range(8)) / 64 / 48) * 100   # 분산 상한 0.25(최악)일 때 8과제 평균의 SE
        ax.barh(i, v, 0.62, color=col, zorder=3)
        ax.errorbar(v, i, xerr=1.96 * se, color=INK, capsize=2.5, lw=1, zorder=4)
        ax.text(v + 1.96 * se + 0.8, i, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ax.axvline(46.0, color=C_PI, ls="--", lw=1, zorder=2)
    ax.text(46.6, -0.75, "가장 강한 기준선 DreamZero 46.0", fontsize=7.3, color=C_PI)
    ax.set_yticks(y, [n for n, _, _ in rows], fontsize=8)
    ax.set_xlim(0, 85)
    ax.set_ylim(-1.1, len(rows) - 0.4)
    ax.set_xlabel("실제 로봇 8과제 평균 루브릭 점수 (%), 오차 막대 = ±1.96 SE 상한(시드 변동 제외)")
    ax.xaxis.grid(True, color=GRID, zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_ladder.png", dpi=200)
    plt.close(fig)


def noise_sigma():
    z = np.load(ROOT / "analysis" / "noise_samples.npz")
    n = json.loads((ROOT / "analysis" / "noise_toy_results.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.5, 1]})
    a = axs[0]
    bins = np.linspace(0, 1, 51)
    for key, lab, col in [("s_pre", "사전학습 p_pre (Beta(7,1), shift 5)", C_PI), ("s_q", "혼합 q (α = 0.5)", C_AW),
                          ("s_uni", "균등 U(0,1) (action·code 손실의 목표)", C_HY)]:
        a.hist(z[key], bins=bins, density=True, histtype="step", lw=1.6, color=col, label=lab)
    for s in n["N4"]["4"]["sigmas_read_by_action"]:
        a.axvline(s, color=INK2, ls=":", lw=1)
    a.text(0.55, 1.5e-2, "점선: 추론 4단계에서\naction이 읽는 σ", fontsize=7, color=INK2, ha="right")
    a.set_yscale("log")
    a.set_xlabel("video noise σ (1 = 순수 잡음)")
    a.set_ylabel("밀도 (log)")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    a.set_title(f"(a) p_pre에서 σ > 0.9 비율 {n['N1']['P_pre(sigma>0.9)'] * 100:.1f}%", fontsize=8, color=INK2, loc="left")
    b = axs[1]
    steps = [4, 10, 20, 40]
    mins = [n["N4"][str(s)]["min_sigma_read"] for s in steps]
    b.bar(range(4), mins, 0.55, color=[C_AW, MUTED, MUTED, MUTED], zorder=3)
    for i, v in enumerate(mins):
        b.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    b.set_xticks(range(4), fontsize=7.5, labels=["4\n(논문)", "10\n(실물 기본)", "20\n(RoboTwin)", "40\n(hybrid)"])
    b.set_ylabel("action이 마지막에 읽는 video σ")
    b.set_ylim(0, 0.8)
    b.yaxis.grid(True, color=GRID, zorder=0)
    b.set_title("(b) joint denoising 단계 수별", fontsize=8, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "noise_sigma.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    robotwin_headline()
    robotwin_pertask()
    ablation_ladder()
    noise_sigma()
    print("saved:", sorted(p.name for p in FIG.glob("*.png")))
