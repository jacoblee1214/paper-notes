"""우리가 만든 그래프를 report/figures/ 에 저장한다.

own_table2.png    : 표 2 물체별 성공 수와 Wilson 95% 구간 (pick-and-place, squeeze)
own_fig5_check.png: 그림 5 인쇄 값으로 계산한 손가락별 오차 감소율과 PR AUC 비
own_mae_values.png: 같은 모델의 MAE 로 문서 안에 나오는 값들, 손가락별 논문 대 저장소 참고 값
own_toy_calibration.png : 토이 T1, 보정 길이별 시험 R²
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo)
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
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#d0651b", "#9a9890"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fig_table2():
    objs = ["Mustard", "Face Wash", "Toothpaste", "BBQ Small", "Chips", "BBQ", "Mustard Small", "Chocolate", "Ketchup"]
    cont = [(7, 0), (6, 4), (4, 2), (6, 0), (3, None), (8, 0), (3, 0), (0, 0), (7, 0)]
    binr = [(10, 0), (10, 0), (8, 0), (9, 0), (10, None), (7, 0), (10, 0), (10, 0), (10, 0)]
    fb = [(10, 10), (10, 10), (9, 9), (8, 8), (7, None), (6, 6), (10, 10), (10, 9), (10, 10)]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, idx, title in ((axes[0], 0, "pick-and-place 성공률 (각 10회)"), (axes[1], 1, "squeeze 성공률 (각 10회, Chips 제외)")):
        for off, data, col, lab in ((-0.22, cont, GREY, "Continuous gripper"), (0.0, binr, ACC, "Binary gripper"), (0.22, fb, ORG, "ForceBand")):
            xs, ys, lo, hi = [], [], [], []
            for i, row in enumerate(data):
                k = row[idx]
                if k is None:
                    continue
                l, h = wilson(k, 10)
                xs.append(i + off); ys.append(k / 10); lo.append(max(0.0, k / 10 - l)); hi.append(max(0.0, h - k / 10))
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=col, ms=5, capsize=2, lw=1, label=lab)
        ax.set_xticks(range(9)); ax.set_xticklabels(objs, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylim(-0.05, 1.08); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("성공률 (Wilson 95% 구간)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("표 2 다시 그리기: 칸마다 시행 10회라 구간이 넓습니다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_table2.png", dpi=170)
    plt.close(fig)


def fig_fig5():
    r = json.loads((ANA / "recompute_results.json").read_text())["V6_relative_error_reduction"]
    labels = ["손 전체\nMAE", "손 전체\nRMSE", "엄지", "검지", "중지", "약지", "소지"]
    vals = [r["MAE"], r["RMSE"], r["Thumb_MAE"], r["Index_MAE"], r["Middle_MAE"], r["Ring_MAE"], r["Pinky_MAE"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    cols = [ORG if v >= 0.5 else ACC for v in vals]
    ax.bar(range(7), [100 * v for v in vals], color=cols)
    for i, v in enumerate(vals):
        ax.text(i, 100 * v + 1, f"{100 * v:.1f}", ha="center", fontsize=8)
    ax.axhline(50, color=INK, ls="--", lw=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("FEEL 대비 오차 감소율 (%)"); ax.set_ylim(0, 75)
    ax.set_title("그림 5(B, D) 인쇄 값으로 계산한 감소율 (주황: 50% 이상)", fontsize=9.5)
    ax = axes[1]
    fing = ["약지", "소지"]
    text_r = [0.763 / 0.398, 0.590 / 0.314]
    fig_r = [0.76 / 0.38, 0.59 / 0.27]
    x = np.arange(2)
    ax.bar(x - 0.18, text_r, 0.36, color=ACC, label="본문 값 (0.398, 0.314)")
    ax.bar(x + 0.18, fig_r, 0.36, color=ORG, label="그림 5C 인쇄 값 (0.38, 0.27)")
    ax.axhline(2, color=INK, ls="--", lw=1)
    for i in range(2):
        ax.text(i - 0.18, text_r[i] + 0.03, f"{text_r[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.18, fig_r[i] + 0.03, f"{fig_r[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(fing); ax.set_ylim(0, 2.6)
    ax.set_ylabel("ForceBand PR AUC / FEEL PR AUC"); ax.legend(fontsize=7.5, loc="lower center")
    ax.set_title('"2배 이상" 문장과 비교', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig5_check.png", dpi=170)
    plt.close(fig)


def fig_mae():
    d = json.loads((ANA / "recompute_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    items = [
        ("표 1 오른쪽, 8ch ours (30분 자료)", 0.77),
        ("그림 5B, 손 전체", 0.80),
        ("표 1 왼쪽, 8ch ours", 0.85),
        ("그림 9, 전체 모델 (epoch 50 검증)", 0.92),
        ("저장소 참고 검증 (17.63 ÷ 18)", round(17.63 / 18, 3)),
    ]
    for i, (lab, v) in enumerate(items):
        ax.plot(v, i, "o", color=ORG if "저장소" not in lab else ACC, ms=8)
        ax.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=8.5)
    ax.set_xlim(0.6, 1.15); ax.invert_yaxis(); ax.set_xlabel("MAE (N)"); ax.grid(axis="x", alpha=0.25)
    ax.set_title("같은 8채널 모델에 붙은 MAE 값들 (평가 자료는 각각 다름)", fontsize=9.5)
    ax = axes[1]
    fing = ["엄지", "검지", "중지", "약지", "소지"]
    paper = list(d["V10_paper_fig5D_ours_per_finger_N"].values())
    repo = list(d["V10_repo_per_finger_N"].values())
    x = np.arange(5)
    ax.bar(x - 0.18, paper, 0.36, color=ORG, label="논문 그림 5D (인쇄 값)")
    ax.bar(x + 0.18, repo, 0.36, color=ACC, label="저장소 참고 검증 (센서 단위 ÷ 18)")
    ax.set_xticks(x); ax.set_xticklabels(fing); ax.set_ylabel("MAE (N)"); ax.legend(fontsize=7.5)
    ax.set_title("손가락별 MAE", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_mae_values.png", dpi=170)
    plt.close(fig)


def fig_toy():
    d = json.loads((ANA / "toy_calibration_results.json").read_text())["T1_calibration"]
    ls = [int(k) for k in d["calibrated_r2_by_cal_seconds"]]
    med = [v["median"] for v in d["calibrated_r2_by_cal_seconds"].values()]
    lo = [v["p10"] for v in d["calibrated_r2_by_cal_seconds"].values()]
    hi = [v["p90"] for v in d["calibrated_r2_by_cal_seconds"].values()]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ls, med, "o-", color=ORG, label="보정 뒤 시험 R² (중앙값, 10–90% 구간)")
    ax.fill_between(ls, lo, hi, color=ORG, alpha=0.2)
    ax.axhline(d["pearson_r_squared"], color=INK, ls="--", lw=1, label=f"r² = {d['pearson_r_squared']:.2f} (이득만 맞출 때 상한)")
    ax.axhline(d["zero_shot_r2_median"], color=GREY, ls=":", lw=1, label=f"보정 전 R² = {d['zero_shot_r2_median']:.1f}")
    ax.set_xscale("log"); ax.set_ylim(-1, 0.5); ax.set_xlabel("보정에 쓴 짝 데이터 길이 (초)"); ax.set_ylabel("시험 R²")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("토이 T1: 이득 불일치(≈1.7배)를 두 모수로 보정", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_calibration.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_table2(); fig_fig5(); fig_mae(); fig_toy()
    print("wrote figures to", FIG)
