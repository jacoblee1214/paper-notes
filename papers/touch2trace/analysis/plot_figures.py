"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sr_ci.png     : 주요 조건의 SR@10, SR@20과 Wilson 95% 구간 (시행 수 n = 30 가정)
own_two_levels.png: 같은 비교를 시행 단위(Fisher, n = 30 가정)와 시드 단위(Welch, 시드 3개)로 본 p 값
own_dist_bounds.png: 보고된 평균 거리와 SR@10, SR@20이 허용하는 범위
own_smooth.png    : Smooth 지표의 제어 주기 의존(합성 신호)과 보정 비율

실행: python3 analysis/plot_figures.py   (numpy, matplotlib. 먼저 recompute.py를 실행)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))

for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GREY, RED = "#1f5fae", "#d9822b", "#8a8984", "#b3372f"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def sr_ci():
    # (이름, SR@10 %, SR@20 %)  표 1, 3, 5, 9, 2의 보고 값. k = round(p * 30 / 100)
    rows = [
        ("고유수용만 (TF-GMM)", 0, 0), ("고유수용 + 촉각 (기본)", 93, 57), ("촉각 16차원 요약", 30, 0),
        ("DP 60 Hz + 촉각", 0, 0), ("DP 30 Hz + 촉각", 100, 0),
        ("인코더 미세조정", 0, 0), ("인코더 무작위 초기화", 0, 0), ("데이터 3.5분", 17, 7), ("데이터 6.7분", 37, 0),
        ("15 Hz", 0, 0), ("30 Hz", 23, 0), ("프레임 5", 11, 0), ("프레임 10", 28, 17), ("프레임 20", 50, 37),
        ("해상도 8×8", 17, 7), ("해상도 16×16", 41, 24),
        ("USB-1 직선", 73, 38), ("USB-2 직선", 47, 13), ("Ethernet 링", 97, 73), ("USB-0 링 (학습)", 96, 73),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    y = np.arange(len(rows))[::-1]
    for yi, (name, p10, p20) in zip(y, rows):
        for p, c, off, lab in ((p10, BLUE, 0.17, "SR@10"), (p20, ORANGE, -0.17, "SR@20")):
            k = round(p * 30 / 100)
            lo, hi = wilson(k, 30)
            ax.plot([lo, hi], [yi + off, yi + off], color=c, lw=2, alpha=0.55)
            ax.plot([p], [yi + off], "o", color=c, ms=5, label=lab if yi == y[0] else None)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("성공률 (%) — 보고 값(점)과 Wilson 95% 구간(선), 시행 수 30 가정")
    ax.set_xlim(-3, 103)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="center right")
    for yi in (y[4] - 0.5, y[8] - 0.5, y[11 - 2] - 0.5):
        pass
    ax.set_title("조건별 성공률과 구간 (구간은 n = 30 가정)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_sr_ci.png", dpi=160)
    plt.close(fig)


def two_levels():
    f = R["V6"]["fisher"]
    w = R["V7"]
    pairs = [
        ("고유수용만 대 촉각 (표 1)", f["SR@10 TF tactile 28/30 vs proprio 0/30"], w["Tactile vs proprio (T1)"]["p_sd_as_sample"]),
        ("촉각 16차원 대 조밀 (표 5)", f["SR@10 TF tactile 28 vs compact 9"], w["Dense vs compact (T5)"]["p_sd_as_sample"]),
        ("60 Hz 대 30 Hz (표 9)", f["SR@10 TF 60Hz 28 vs 30Hz 7"], w["60 Hz vs 30 Hz (T9)"]["p_sd_as_sample"]),
        ("프레임 15 대 20 (표 9)", f["SR@10 frame15 28 vs frame20 15"], w["frame15 vs frame20 (T9)"]["p_sd_as_sample"]),
        ("프레임 15 대 10 (표 9)", f["SR@10 frame15 28 vs frame10 8"], w["frame15 vs frame10 (T9)"]["p_sd_as_sample"]),
        ("해상도 32 대 16 (표 9)", f["SR@10 32x32 28 vs 16x16 12"], w["32x32 vs 16x16 (T9)"]["p_sd_as_sample"]),
        ("해상도 32 대 8 (표 9)", f["SR@10 32x32 28 vs 8x8 5"], w["32x32 vs 8x8 (T9)"]["p_sd_as_sample"]),
        ("동결 대 미세조정 (표 3)", f["SR@10 frozen 28 vs finetuned 0"], w["frozen vs finetuned (T3)"]["p_sd_as_sample"]),
        ("Ethernet 직선 대 USB-2 (표 2)", f["SR@10 Ethernet straight 28 vs USB-2 14"], w["Ethernet straight vs USB-2 (T2)"]["p_sd_as_sample"]),
        ("Ethernet 링 대 직선 SR@20 (표 2)", f["SR@20 Ethernet ring 22 vs straight 17"], w["Ethernet ring vs straight (T2)"]["p_sd_as_sample"]),
        ("TF-GMM 60 대 DP 30 (표 1)", f["SR@10 DP30 30 vs TF-GMM60 28"], w["TF-GMM60 vs DP30 tactile (T1)"]["p_sd_as_sample"]),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    y = np.arange(len(pairs))[::-1]
    for yi, (n, p1, p2) in zip(y, pairs):
        ax.plot([max(p1, 1e-6), p2], [yi, yi], color=GREY, lw=1, alpha=0.6)
        ax.plot([max(p1, 1e-6)], [yi], "o", color=BLUE, label="시행 단위 Fisher (n = 30 가정)" if yi == y[0] else None)
        ax.plot([p2], [yi], "s", color=ORANGE, label="시드 단위 Welch (시드 3개)" if yi == y[0] else None)
    ax.axvline(0.05, color=RED, ls="--", lw=1)
    ax.text(0.057, y[-1] + 0.35, "p = 0.05", color=RED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(5e-7, 1.5)
    ax.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["1e-6", "1e-5", "1e-4", "1e-3", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in pairs], fontsize=9)
    ax.set_xlabel("p 값 (로그 축, 1e-6 미만은 1e-6 위치에 표시)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("같은 비교를 시행 단위와 시드 단위로 보면 결론이 갈리는 곳", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_two_levels.png", dpi=160)
    plt.close(fig)


def dist_bounds():
    rows = R["V4"]["rows"]
    names = [f"{r['row']}" for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        hi = r["upper"] if r["upper"] is not None else 32
        ax.plot([r["lower"], hi], [yi, yi], color=BLUE, lw=5, alpha=0.25, solid_capstyle="butt")
        ax.plot([r["dist"]], [yi], "o", color=ORANGE, ms=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(-0.5, 33)
    ax.set_xlabel("거리 (cm). 파란 띠: SR@10, SR@20에서 계산한 평균의 허용 범위(오른쪽 끝이 열린 띠는 상한 없음), 점: 보고된 평균")
    ax.xaxis.label.set_size(8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("보고된 평균 거리는 성공률이 허용하는 범위 안에 있다 (27행 모두)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_dist_bounds.png", dpi=160)
    plt.close(fig)


def smooth():
    k = R["K1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    hz = [60, 30, 15]
    for key, c, lab in (("norm", BLUE, "‖Δa‖"), ("squared_norm", ORANGE, "‖Δa‖²")):
        v = [k[key]["mean_by_hz"][str(h)] for h in hz]
        ax.plot(hz, np.array(v) / v[0], "o-", color=c, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(hz)
    ax.set_xticklabels(["60", "30", "15"])
    ax.invert_xaxis()
    ax.set_xlabel("표본화 주기 (Hz)")
    ax.set_ylabel("같은 신호의 지표 / 60 Hz 값")
    ax.set_title("합성 사인 신호의 한 스텝 차이 지표", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.25)
    ax = axes[1]
    labels = ["보고 비율\n(DP30 / TF60)", "주기 보정\n(‖Δa‖ 가정)", "주기 보정\n(‖Δa‖² 가정)"]
    vals = [k["paper_ratio_dp30_over_tf60"], k["adjusted_ratio_if_norm"], k["adjusted_ratio_if_squared"]]
    ax.bar(labels, vals, color=[GREY, BLUE, ORANGE])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v:.1f}×", ha="center", fontsize=9)
    ax.set_ylabel("Smooth 비율")
    ax.set_title("0.028 / 0.004 = 7×를 주기로 보정하면", fontsize=10)
    ax.set_ylim(0, 8.2)
    fig.tight_layout()
    fig.savefig(FIG / "own_smooth.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sr_ci()
    two_levels()
    dist_bounds()
    smooth()
    print("wrote 4 figures in", FIG)
