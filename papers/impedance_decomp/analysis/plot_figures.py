"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sim_ratios.png : 표 1, 표 10 시뮬레이션의 Ours/Iso 비율(힘 임펄스, 힘 피크)과 Oracle 회복률
own_real_ratios.png: 표 2 실로봇의 Ours/Iso 비율 (과제별 네 지표)과 평균의 비
own_toy_ensemble.png: 공식 저장소 예시 앙상블 4개의 부분표본 SE 대 N (우리 구현), 논문 표 11과 나란히
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 recompute.py, toy_ensemble_estimator.py를 실행)
한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서로 찾는다.
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
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#c4651b"


def load(name):
    return json.loads((ANA / name).read_text(encoding="utf-8"))


def plot_sim():
    rec = load("recompute_results.json")
    T1, T10 = rec["tables"]["T1"], rec["tables"]["T10"]
    tasks = list(T1)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(tasks))
    w = 0.2
    series = [("힘 임펄스 ICF, 단일", T1, "ICF", ACC), ("힘 피크 PCF, 단일", T1, "PCF", "#8fb0d9"),
              ("ICF, 다중 과제", T10, "ICF", WARM), ("PCF, 다중 과제", T10, "PCF", "#e6b48a")]
    for i, (lab, T, m, c) in enumerate(series):
        v = [T[t][m][1] / T[t][m][0] for t in tasks]
        ax.bar(x + (i - 1.5) * w, v, w, color=c, label=lab)
        for xi, vi in zip(x + (i - 1.5) * w, v):
            ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(1, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("Ours / Iso")
    ax.set_title("구속 방향 하중 비 (1 아래가 감소)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    ax.set_ylim(0, 0.75)
    ax = axes[1]
    for i, (lab, T, c) in enumerate([("단일", T1, ACC), ("다중 과제", T10, WARM)]):
        v = [min((T[t]["ICF"][0] - T[t]["ICF"][1]) / (T[t]["ICF"][0] - T[t]["ICF"][2]),
                 1.5) for t in tasks]
        ax.bar(x + (i - 0.5) * 0.35, v, 0.35, color=c, label=lab)
        for xi, vi in zip(x + (i - 0.5) * 0.35, v):
            ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(1, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=7, rotation=20)
    ax.set_ylim(0.6, 1.22)
    ax.set_title("Oracle 회복률 (ICF)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ratios.png", dpi=180)
    plt.close(fig)


def plot_real():
    rec = load("recompute_results.json")
    T2 = rec["tables"]["T2"]
    tasks = list(T2)
    mets = [("ICF", "힘 임펄스"), ("PCF", "힘 피크"), ("ICM", "모멘트 임펄스"), ("PCM", "모멘트 피크")]
    cols = [ACC, "#8fb0d9", WARM, "#e6b48a"]
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    x = np.arange(len(tasks) + 1)
    w = 0.2
    for i, ((m, lab), c) in enumerate(zip(mets, cols)):
        v = [T2[t][m][1] / T2[t][m][0] for t in tasks] + [rec["V2_table2"]["avg"][m]["ratio_of_means"]]
        ax.bar(x + (i - 1.5) * w, v, w, color=c, label=lab)
        for xi, vi in zip(x + (i - 1.5) * w, v):
            ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(1, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks + ["평균의 비"], fontsize=8)
    ax.set_ylabel("Ours / Iso (과제당 10회 평균)")
    ax.set_title("실로봇 FR3: 구속 방향 하중 비 (논문 표 2의 평균값으로 다시 그림, 분산 정보 없음)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper left")
    ax.set_ylim(0, 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ratios.png", dpi=180)
    plt.close(fig)


def plot_toy():
    toy = load("toy_ensemble_estimator_results.json")
    rec = load("recompute_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    Ns = [1, 2, 4, 8, 16, 32, 64]
    ax = axes[0]
    cmap = {"revolute": "#8fb0d9", "cylindrical": "#5a8cc4", "planar": ACC, "universal": WARM}
    for name, c in cmap.items():
        v = [toy["K2"][name][str(n)]["se_p95"] for n in Ns]
        ax.plot(Ns, v, "-o", ms=3, color=c, lw=1.2, label=f"{name} (우리 구현)")
    T11 = rec["tables"]["T11"]
    ax.plot(T11["N"][:7], T11["se95"][:7], "-s", ms=3.5, color=INK, lw=1.6, label="논문 표 11 (4개 합산)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("앙상블 크기 N")
    ax.set_ylabel("SE 95번째 백분위 (도)")
    ax.set_title("차원을 참값으로 고정한 SE", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    for name, c in cmap.items():
        v = [100 * toy["K2"][name][str(n)]["correct_dim_frac"] for n in Ns]
        ax.plot(Ns, v, "-o", ms=3, color=c, lw=1.2, label=name)
    ax.plot(T11["N"][:7], T11["correct_dim_pct"][:7], "-s", ms=3.5, color=INK, lw=1.6, label="논문 표 11")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("앙상블 크기 N")
    ax.set_ylabel("올바른 차원을 고른 비율 (%)")
    ax.set_title("차원 규칙 (v_abs = 5 mm/s, kappa_rel = 4)", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_ensemble.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_sim()
    plot_real()
    plot_toy()
    print("wrote figures in", FIG)
