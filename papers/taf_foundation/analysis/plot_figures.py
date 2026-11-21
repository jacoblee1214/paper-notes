"""우리가 그린 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다: 라이선스가 CC BY가 아님).

own_bench.png    : 표 2 단일프레임 힘 RMSE와 검색 R@1 (3 시드 평균 ± SD)
own_outcomes.png : 표 4·5 평균 성공률과 Wilson 95% 구간 (시뮬 150회, 실로봇 4과제 80회, 3과제 60회)
own_ablation.png : 표 3 절제 차이를 표 2 "Ours" 시드 SD 단위로 표시
own_fewshot.png  : 표 S4 시간 결합 적응 곡선 (데이터 비율별 정규화 RMSE)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, PALE = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


# 표 2 (평균, SD; 3 시드)
T2 = {
    "DINOv2": (3.032, .053, 14.00, .27), "Sparsh": (2.886, .155, 14.21, .59), "T3": (2.820, .131, 17.38, 1.05),
    "AnyTouch 2": (2.799, .203, 21.27, .80), "HTT": (2.971, .024, 14.78, .46), "Ours": (2.321, .084, 34.38, 1.35),
}


def plot_bench():
    names = list(T2)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, (idx, title, ylab, lower) in zip(axes, [(0, "단일프레임 힘 RMSE (N, 낮을수록 좋음)", "N", True),
                                                   (2, "검색 R@1 (%, 높을수록 좋음)", "%", False)]):
        vals = [T2[n][idx] for n in names]
        sds = [T2[n][idx + 1] for n in names]
        cols = [ACC if n == "Ours" else GREY for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, yerr=sds, color=cols, width=0.62, capsize=3, ecolor=INK, error_kw={"lw": 1})
        for i, v in enumerate(vals):
            ax.text(i, v + (sds[i] + 0.02 * max(vals)), f"{v:.2f}" if idx == 0 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.set_ylim(0, 3.6)
        else:
            ax.set_ylim(0, 42)
    fig.text(0.01, 0.01, "막대: 3 시드 평균, 오차 막대: 표준편차 (논문 표 2). 우리가 다시 그렸습니다.", fontsize=7.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "own_bench.png", dpi=170)
    plt.close(fig)


def plot_outcomes():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    sim = {"UniVTAC": 71, "T3": 62, "AnyTouch 2": 64, "Ours": 77}  # 합 / 150
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, (title, counts, n) in zip(axes, [
        ("시뮬레이션 3과제 합산 (150회, 과제당 50)", sim, 150),
        ("실로봇 3과제 합산 (60회, 과제당 20)", {"T3": 11, "AnyTouch 2": 11, "Ours": 24}, 60),
    ]):
        names = list(counts)
        pct = [100 * counts[k] / n for k in names]
        lo = [wilson(counts[k], n)[0] for k in names]
        hi = [wilson(counts[k], n)[1] for k in names]
        yerr = np.array([np.array(pct) - np.array(lo), np.array(hi) - np.array(pct)])
        x = np.arange(len(names))
        ax.bar(x, pct, color=[ACC if k == "Ours" else GREY for k in names], width=0.6)
        ax.errorbar(x, pct, yerr=yerr, fmt="none", ecolor=INK, capsize=3, lw=1)
        for i, k in enumerate(names):
            ax.text(i, 3, f"{counts[k]}/{n}", ha="center", va="bottom", color="white", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 80)
        ax.set_ylabel("성공률 (%)")
        ax.set_title(title, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    p = rec["table5_3task_fisher_ours_vs_T3"]["fisher_p"]
    fig.text(0.01, 0.01, f"막대: 합산 성공률, 오차: Wilson 95% 구간. 실로봇 합산 Fisher p = {p:.3f} (과제 독립 근사). 우리가 다시 그렸습니다.",
             fontsize=7.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "own_outcomes.png", dpi=170)
    plt.close(fig)


def plot_ablation():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    ab = rec["table3_ablation_delta_vs_full"]
    rows = [("직접 렌치 지도", "DirectSup"), ("정렬 손실 제거", "woAlign"),
            ("센서별 MLP 제거", "woSensorMLP"), ("센서 식별 토큰 제거", "woSensorIDToken")]
    metrics = [("F1", "단일 F"), ("T1", "단일 T"), ("F2", "시간 F"), ("T2", "시간 T"), ("RET1", "R@1"), ("RET5", "R@5")]
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    y = np.arange(len(rows) * len(metrics))
    labels, vals, cols = [], [], []
    for (rn, key) in rows:
        for mk, ml in metrics:
            d = ab[key][mk]["delta_in_seed_sd"]
            # 낮을수록 좋은 지표는 부호를 뒤집어, 양수가 "전체 모델보다 나쁨"을 뜻하게 한다
            sign = 1 if mk in ("F1", "T1", "F2", "T2") else -1
            labels.append(f"{rn} · {ml}")
            vals.append(sign * d)
            cols.append(ACC if sign * d > 0 else GREY)
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=cols, height=0.7)
    ax.axvspan(-1, 1, color="#eef3fa", zorder=0)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("전체 모델 대비 차이 (표 2 시드 SD 단위, 양수 = 전체 모델보다 나쁨)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("표 3 절제: 단일 시드 차이 (음영 = ±1 SD, 단일 시드라 해석 주의)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=170)
    plt.close(fig)


def plot_fewshot():
    budgets = ["5%", "10%", "25%", "50%", "100%"]
    tmp = {
        "DINOv2": [.4485, .4014, .3862, .3862, .3414], "Sparsh": [.4439, .4123, .3901, .3767, .3520],
        "T3": [.4339, .4026, .3818, .3658, .3328], "AnyTouch 2": [.4254, .4058, .3957, .3691, .3486],
        "HTT": [.4120, .3859, .3767, .3617, .3381], "Ours": [.4187, .3987, .3736, .3582, .3294],
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(budgets))
    for k, v in tmp.items():
        ax.plot(x, v, marker="o", ms=3.5, lw=2.4 if k == "Ours" else 1.2,
                color=ACC if k == "Ours" else None, alpha=1 if k == "Ours" else 0.75, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.set_xlabel("미지 센서 라벨 데이터 비율")
    ax.set_ylabel("정규화 RMSE (낮을수록 좋음)")
    ax.set_title("시간 결합 적응 (단일 시드, 표 S4 값)", fontsize=9.5)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "논문 표 S4 값을 다시 그렸습니다. 25–100%에서 제안이 가장 낮습니다.", fontsize=7.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "own_fewshot.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_bench()
    plot_outcomes()
    plot_ablation()
    plot_fewshot()
    print("wrote figures to", FIG)
