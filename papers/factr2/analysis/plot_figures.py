"""우리가 그린 그래프 세 장. 논문 그림은 쓰지 않는다(라이선스 arXiv non-exclusive, CC BY 아님).
입력: 논문 표 1의 값(03 §2.1에 옮긴 값), toy_residual_history_w*_results.json, toy_upsampling_results.json
출력: report/figures/own_*.png
실행: python3 analysis/plot_figures.py  (matplotlib, numpy; 한글 폰트는 Apple SD Gothic Neo → Noto CJK KR 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

BLUE, ORANGE, GREY, GREEN = "#1f5fae", "#d9822b", "#7a7a75", "#3d8c5a"


def fig_table1():
    """논문 표 1: 접촉과 자유 공간의 L1 오차 평균 ± 표준편차 표기값."""
    contact = [("FILIC", 4.395, 1.531), ("DO", 1.471, 0.761), ("NEXT", 0.547, 0.348)]
    free = [("센서(Franka)", 0.449, 0.208), ("FILIC", 2.460, 1.358), ("DO", 2.429, 1.298), ("NEXT", 0.414, 0.278)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [3, 4]})
    for ax, rows, title in [(axes[0], contact, "접촉 구간 (기준: 센서 추정)"), (axes[1], free, "자유 공간 (기준: 0 Nm)")]:
        names = [r[0] for r in rows]
        means = [r[1] for r in rows]
        sds = [r[2] for r in rows]
        colors = [GREY if n not in ("NEXT",) else BLUE for n in names]
        if "센서(Franka)" in names:
            colors[names.index("센서(Franka)")] = GREEN
        ax.bar(names, means, yerr=sds, color=colors, capsize=4, alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("평균 L1 오차 (Nm)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for i, m in enumerate(means):
            ax.text(i, m + sds[i] + 0.05, f"{m:.3f}", ha="center", fontsize=8)
    fig.suptitle("논문 표 1 수치 (막대 위 숫자 = 평균, 오차 막대 = 논문 ± 표기값)", fontsize=9, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1.png", dpi=200)
    plt.close(fig)


def fig_toy_residual():
    """토이 1: 이력 길이 H와 외부 토크 피드포워드 진폭 W에 따른 접촉 L1과 자유 RMS."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    Hs = [1, 10, 50]
    colors = {"0": GREY, "0.5": ORANGE, "2": BLUE}
    for w in ("0", "0.5", "2"):
        d = json.loads((AN / f"toy_residual_history_w{w}_results.json").read_text())
        axes[0].plot(Hs, [d["L_H"][f"H{h}"]["contact_l1_nm"] for h in Hs], "o-", color=colors[w], label=f"W = {w} Nm")
        axes[1].plot(Hs, [d["L_H"][f"H{h}"]["free_rms_nm"] for h in Hs], "o-", color=colors[w], label=f"W = {w} Nm")
    d0 = json.loads((AN / "toy_residual_history_w0_results.json").read_text())
    axes[0].axhline(d0["contact_abs_tau_ext_mean_nm"], ls="--", color=GREY, lw=1)
    axes[0].text(1, d0["contact_abs_tau_ext_mean_nm"] + 0.03, "접촉 참값 |τ_ext| 평균", fontsize=7, color=GREY)
    axes[0].set_title("접촉 구간 추정 오차 (L1, Nm)", fontsize=10)
    axes[1].set_title("자유 구간 추정 RMS (Nm)", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(Hs)
        ax.set_xticklabels([str(h) for h in Hs])
        ax.set_xlabel("이력 길이 H (스텝)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("토이 1 (가정 설정). 순수 PD(W = 0)에서는 접촉이 보이지 않습니다", fontsize=9, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_residual.png", dpi=200)
    plt.close(fig)


def fig_toy_upsampling():
    """토이 2: 업샘플링 배율과 단계별 테스트 MSE."""
    d = json.loads((AN / "toy_upsampling_results.json").read_text())
    rows = d["rows"]

    def pick(scheme):
        pts = [r for r in rows if r["scheme"] == scheme]
        base = [r for r in rows if r["scheme"] == "none"][0]
        return [1] + [p["w_PC"] if scheme != "C_only" else p["w_C"] for p in pts], [base] + pts

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for scheme, label, style in [("PC", "PC 업샘플링", "o-"), ("C_only", "C만 업샘플링", "s--")]:
        xs, pts = pick(scheme)
        axes[0].plot(xs, [p["mse_PC"] for p in pts], style, color=BLUE if scheme == "PC" else ORANGE, label=f"PC 오차 ({label})")
        axes[1].plot(xs, [p["mse_C"] for p in pts], style, color=BLUE if scheme == "PC" else ORANGE, label=f"C 오차 ({label})")
        axes[1].plot(xs, [p["mse_F"] for p in pts], style, color=GREY, alpha=0.6, label=f"자유 오차 ({label})")
    axes[0].set_title("PC 구간 테스트 MSE", fontsize=10)
    axes[1].set_title("C와 자유 구간 테스트 MSE", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("업샘플링 배율 m (1 = 없음)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle("토이 2 (가정 설정). 한 단계를 올리면 다른 단계 오차가 늘어납니다", fontsize=9, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_upsampling.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_table1()
    fig_toy_residual()
    fig_toy_upsampling()
    print("wrote own_table1.png, own_toy_residual.png, own_toy_upsampling.png")
