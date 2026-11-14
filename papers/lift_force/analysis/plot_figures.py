"""우리가 그린 그래프 세 장. 논문 그림은 쓰지 않는다(라이선스 CC BY-SA 4.0, 04 §5).
입력: analysis/table_check_results.json, analysis/toy_init_equiv_results.json
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
plt.rcParams["mathtext.fontset"] = "dejavusans"

BLUE, ORANGE, GREY = "#1f5fae", "#d9822b", "#7a7a75"


def fig_hanoi():
    r = json.loads((AN / "table_check_results.json").read_text())["hanoi_counts"]
    names = [("LIFT\n(peak)", "LIFT_peak_0.6"), ("π0.5 Online\nDAgger (peak)", "OnlineDAgger_peak_0.3"),
             ("LIFT w/o RFI\n(peak)", "LIFT_wo_RFI_peak_0.5"), ("LIFT\n(final)", "LIFT_final_0.6_at_2.0K"),
             ("π0.5 Online\nDAgger (final)", "OnlineDAgger_final_0.2"), ("LIFT w/o RFI\n(final)", "LIFT_wo_RFI_final_0.1")]
    rates = [r[k]["rate"] for _, k in names]
    lo = [r[k]["wilson95"][0] for _, k in names]
    hi = [r[k]["wilson95"][1] for _, k in names]
    err = np.array([np.array(rates) - lo, np.array(hi) - np.array(rates)])
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=150)
    colors = [BLUE, GREY, GREY, BLUE, GREY, GREY]
    x = np.arange(len(names))
    ax.bar(x, rates, color=colors, alpha=0.85, width=0.6)
    ax.errorbar(x, rates, yerr=err, fmt="none", ecolor="black", capsize=3, lw=1)
    for xi, (_, k) in zip(x, names):
        ax.text(xi, rates[xi] + 0.11, f"{r[k]['k']}/{r[k]['n']}", ha="center", fontsize=8)
    ax.set_xticks(x, [n for n, _ in names], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hanoi 성공률 (이진, n=30 가정)")
    ax.set_title("Hanoi: 본문 값의 Wilson 95% 구간 (롤아웃 표본만 반영)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_hanoi_ci.png")
    plt.close(fig)


def fig_init():
    t = json.loads((AN / "toy_init_equiv_results.json").read_text())
    labels = ["같은 노이즈\n(0 초기화)", "독립 노이즈\n(0 초기화)", "음성 대조:\n마스크 오류", "출력 투영\n0.001", "출력 투영\n0.01", "출력 투영\n0.1"]
    vals = [t["init_same_noise_max_abs_diff"], t["init_indep_noise_max_abs_diff"],
            t["negative_control_full_base_mask_same_noise_diff"],
            t["nonzero_cross_output_sensitivity_max_abs_diff"]["0.001"],
            t["nonzero_cross_output_sensitivity_max_abs_diff"]["0.01"],
            t["nonzero_cross_output_sensitivity_max_abs_diff"]["0.1"]]
    vals = [max(v, 1e-16) for v in vals]
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=150)
    colors = [BLUE, ORANGE, GREY, GREY, GREY, GREY]
    ax.bar(range(len(vals)), vals, color=colors, alpha=0.9, width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 60)
    ax.set_xticks(range(len(vals)), labels, fontsize=7.5)
    ax.set_ylabel("반응 스트림과 기본 스트림의 최대 절대차")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.6, f"{v:.1e}", ha="center", fontsize=7.5)
    ax.set_title("토이 C: 초기화 동일성은 같은 노이즈일 때만 성립 (numpy, 차원 16)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "own_init_equiv.png")
    plt.close(fig)


def fig_latency():
    """제어 10 Hz, 힘 센서 1000 Hz 중 10 Hz로 내림, 지연 L=3. 행동 i는 힘 프레임 i-3까지만 본다."""
    fig, ax = plt.subplots(figsize=(7.6, 2.9), dpi=150)
    H, L = 10, 3
    for i in range(H):
        ax.add_patch(plt.Rectangle((i, 0.55), 0.92, 0.3, color=BLUE, alpha=0.25))
        ax.text(i + 0.46, 0.7, f"a{i}", ha="center", va="center", fontsize=7.5)
        if i - L >= 0:
            ax.annotate("", xy=(i + 0.46, 0.5), xytext=(i - L + 0.46, 0.2),
                        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
        ax.text(i + 0.46, -0.12, f"F{i}", ha="center", fontsize=7.5, color=GREY)
    ax.text(-0.2, 0.9, "행동(10 Hz)", fontsize=8)
    ax.text(-0.2, 0.05, "힘(10 Hz)", fontsize=8, va="center")
    ax.text(H + 0.1, 0.2, "주황 화살표: 행동 i가 보는 가장 최신 힘 (i-3)\n i=0,1,2는 힘을 쓰지 못함 (마스크 전부 0)",
            fontsize=7.5, va="center")
    ax.set_xlim(-0.4, H + 4.2)
    ax.set_ylim(-0.2, 1.0)
    ax.axis("off")
    ax.set_title("지연 정렬 마스크: 힘에서 행동까지 최소 3 스텝(0.3초) (코드 s3=3, 논문 Table 1)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_force_latency.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_hanoi()
    fig_init()
    fig_latency()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
