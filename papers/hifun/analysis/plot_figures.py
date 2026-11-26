"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_ci.png   : 표 1 과제별 최종 성공률과 Wilson 95% 구간 (시행 수 50 가정, 반 점은 분수 성공 수로 처리)
own_ablation_ci.png : 표 2 절제의 Wilson 구간 (시행 수 20과 50 두 가정)
own_cost.png        : 과제별 시간 회계 (표 1, 표 5에 적힌 것만)
own_toy_iaw.png     : 토이, 비평가 오차에 따른 IAW 대 균등 모방 오차
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
import math
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c4651a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


TASKS = ["드릴 작동", "각도 확장기\n꺼내기", "얇은 손잡이\n꺼내기", "드릴 비트\n빼기", "피펫\n분주", "얇은 축\n꺼내기"]
HIFUN = [100, 100, 100, 100, 90, 100]
INTERV = [30, 0, 0, 20, 32, 0]
BC = [15, 30, 35, 30, 12, 20]
DP = [30, 60, 40, 36, 26, 58]


def plot_table1():
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    series = [("HiFun", HIFUN, ACC), ("HIL-SERL (Ours Interv.)", INTERV, ORG), ("BC", BC, GREY), ("DP", DP, "#5a5a55")]
    x = np.arange(6)
    w = 0.2
    for i, (name, vals, col) in enumerate(series):
        pos = x + (i - 1.5) * w
        lo = [wilson(v / 2, 50)[0] for v in vals]
        hi = [wilson(v / 2, 50)[1] for v in vals]
        yerr = np.clip(np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]), 0, None)
        ax.bar(pos, vals, width=w * 0.92, color=col, label=name)
        ax.errorbar(pos, vals, yerr=yerr, fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=8)
    ax.set_ylabel("최종 성공률 (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=170)
    plt.close(fig)


def plot_ablation():
    labels = ["Full", "단일 헤드\n(w/o multi-head)", "균등 개입 BC\n(Uniform)", "개입 BC 없음\n(w/o Interv. BC)"]
    data = {"각도 확장기 꺼내기": [100, 85, 60, 40], "드릴 비트 빼기": [100, 75, 60, 40]}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (task, vals) in zip(axes, data.items()):
        x = np.arange(4)
        for off, n, col in ((-0.17, 20, ACC), (0.17, 50, ORG)):
            lo = [wilson(v / 100 * n, n)[0] for v in vals]
            hi = [wilson(v / 100 * n, n)[1] for v in vals]
            yerr = np.clip(np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]), 0, None)
            ax.errorbar(x + off, vals, yerr=yerr, fmt="o", color=col, capsize=3, ms=4, lw=1, label=f"시행 수 {n} 가정")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(task, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("성공률 (%) 과 Wilson 95% 구간")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_ci.png", dpi=170)
    plt.close(fig)


def plot_cost():
    names = ["드릴 작동", "각도 확장기", "얇은 손잡이", "드릴 비트", "피펫", "얇은 축"]
    clf = [4, 13, 11, 7, 10, 7.5]
    dmp = [34, 13, 17, 21, 14, 14]
    onl = [60, 55, 60, 22, 60, 20]
    il = [150, 60, 80, 80, 100, 100]
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    x = np.arange(6)
    ax.bar(x - 0.2, il, width=0.36, color=GREY, label="IL 시연 수집 (200회, 표 1)")
    ax.bar(x + 0.2, onl, width=0.36, color=ACC, label="HiFun 온라인 학습 (사람 개입)")
    ax.bar(x + 0.2, clf, width=0.36, bottom=onl, color="#8fb0d9", label="보상 분류기 데이터 (표 5)")
    b2 = [a + b for a, b in zip(onl, clf)]
    ax.bar(x + 0.2, dmp, width=0.36, bottom=b2, color="#d9d6cc", hatch="//", edgecolor=GREY, label="DMP 롤아웃 (자동, 표 5)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("분 (표에 적힌 항목만)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.text(0.0, -0.22, "손 기술 잔차 RL 시간, 기준 시연 1회, 교시(과제당 5분 미만)는 표 1과 표 5에 없어 포함하지 않았습니다.",
            transform=ax.transAxes, fontsize=7.5, color="#55544f")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_cost.png", dpi=170)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_iaw_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
    for ax, fb in zip(axes, (0.3, 0.5)):
        rows = [r for r in res["grid"] if abs(r["f_bad"] - fb) < 1e-9]
        eps = [r["eps"] for r in rows]
        ax.errorbar(eps, [r["uniform"] for r in rows], yerr=[r["uniform_se"] for r in rows], color=GREY, marker="o", ms=4, label="균등 모방")
        ax.errorbar(eps, [r["iaw"] for r in rows], yerr=[r["iaw_se"] for r in rows], color=ACC, marker="o", ms=4, label="IAW (식 7–8)")
        ax.errorbar(eps, [r["oracle_good_only"] for r in rows], yerr=[r["oracle_good_only_se"] for r in rows], color=ORG, marker="o", ms=4, label="좋은 개입만 (상한)")
        ax.set_title(f"주저하는 개입 비율 {int(fb * 100)}%", fontsize=10)
        ax.set_xlabel("비평가 오차 크기 (가치 표준편차 대비)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("정책 오차 ||Θ-W|| / ||W||")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_iaw.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_ablation()
    plot_cost()
    plot_toy()
    print("wrote 4 figures to", FIG)
