"""extra_public_data_metrics_results.json의 집계값으로 그래프 한 장을 만든다 (matplotlib).

실행: .venv/bin/python -I phuma/analysis/extra_public_data_plots.py phuma/analysis/extra_public_data_metrics_results.json phuma/report/figures/own_public_metrics.png
패널 (a): 공개 G1·H1-2 모션의 통과율 대 논문 표 1 PhySINK 행 (바닥 z=0과 클립별 바닥 두 가지).
패널 (b): 2차 질량중심 필터 검사 (G1 클립별 최저 질량중심 높이와 발 볼록 껍질 거리의 백분위 범위).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 한글 글꼴: Linux 후보를 먼저, 다음 macOS 후보 (plot_figures.py와 같은 순서)
FONT = next((f for f in ("Noto Sans CJK KR", "Apple SD Gothic Neo")
             if any(f == x.name for x in font_manager.fontManager.ttflist)), "DejaVu Sans")

BLUE = "#2a78d6"  # 범주 슬롯 1 (라이트)
GRAY = "#8a8985"  # 비교 값 (중립)
INK = "#1b1b1a"
INK2 = "#55544f"

PAPER = {  # 논문 표 1 PhySINK (G1), 표 37 PhySINK (H1-2). 관절, 접지, 관통, 미끄럼 순서로 맞춤
    "g1": {"non_floating_pct": 99.9, "non_penetration_pct": 96.8, "non_skating_pct": 89.7,
           "joint_feasibility_pct": 100.0},
    "h1_2": {"non_floating_pct": 97.7, "non_penetration_pct": 99.7, "non_skating_pct": 87.7,
             "joint_feasibility_pct": 99.9},
}
LABELS = [("non_floating_pct", "비접지\n(뜸)"), ("non_penetration_pct", "비관통"),
          ("non_skating_pct", "비미끄럼"), ("joint_feasibility_pct", "관절 한계")]


def main(js, out):
    r = json.load(open(js, encoding="utf-8"))
    rb = r["robots"]
    plt.rcParams.update({"font.family": FONT, "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})

    # 패널 (a): 4개 지표 x (G1, H1-2) 각각 본 계산 대 논문
    x = []
    h_meas, h_paper, names = [], [], []
    for robot in ("g1", "h1_2"):
        for key, lab in LABELS:
            names.append(f"{'G1' if robot == 'g1' else 'H1-2'}\n{lab}")
            h_meas.append(rb[robot]["pass_rates_main_band"].get(key, float("nan")))
            h_paper.append(PAPER[robot][key])
    xs = list(range(len(names)))
    w = 0.38
    ax1.bar([i - w / 2 for i in xs], h_meas, w, color=BLUE, label="공개 데이터 재계산 (바닥 z=0)")
    ax1.bar([i + w / 2 for i in xs], h_paper, w, color=GRAY, label="논문 표 1 PhySINK")
    for i, v in zip(xs, h_meas):
        ax1.text(i - w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=6.5, color=INK)
    for i, v in zip(xs, h_paper):
        ax1.text(i + w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=6.5, color=INK2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, fontsize=6.3)
    ax1.set_ylim(0, 122)
    ax1.set_ylabel("통과율 (%)")
    ax1.set_title("(a) 발 지표와 관절 한계 통과율", loc="left", fontsize=9.5)
    ax1.legend(loc="upper center", ncol=2, fontsize=7.5, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    # 패널 (b): 로봇별 2차 필터의 최저 질량중심 높이 백분위 범위와 각자의 임계
    foot = []
    for i, (key, label) in enumerate((("g1", "G1"), ("h1_2", "H1-2"))):
        f = rb[key][f"robot_filter_{key}"]
        thr_h, thr_b = f.get("thresholds_com_min_h_m_bos_m", (0.40, 0.16))
        hc = f["chunk_min_com_height_m"]
        hb = f["chunk_mean_bos_distance_m"]
        ax2.errorbar([i], [hc["p50"]], yerr=[[hc["p50"] - hc["p5"]], [hc["p95"] - hc["p50"]]],
                     fmt="o", color=BLUE, capsize=4, ms=5)
        ax2.hlines(thr_h, i - 0.3, i + 0.3, color=INK2, lw=1, ls="--")
        ax2.text(i + 0.32, thr_h, f"임계 {thr_h:.2f} m", fontsize=7, color=INK2, va="center")
        foot.append(f"{label}: BoS 거리 중앙값 {hb['p50']:.3f} m (임계 {thr_b:.2f} m), 실패 창 {f['chunks_fail_any']:,}/{f['chunks_total']:,}")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["G1", "H1-2"], fontsize=8)
    ax2.set_xlim(-0.6, 1.9)
    ax2.set_ylim(0.3, 1.1)
    ax2.set_ylabel("창별 최저 질량중심 높이 (m)", fontsize=7.5)
    ax2.set_title("(b) 2차 필터: 질량중심 높이 (점 중앙값, 선 5–95%)", loc="left", fontsize=9.5)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.text(0.53, 0.01, " · ".join(foot), ha="center", fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
