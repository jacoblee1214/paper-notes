"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_ci.png : 표 1 데이터 수집 성공률(방법 5개 x 과제 3개)과 Wilson 95% 구간 (각 10회)
own_table2_ci.png : 표 2 정책 성공률(데이터 3종 x 가드 유무 x 과제 3개)과 Wilson 95% 구간 (각 10회)
own_selection.png : 라운드별 성공 횟수와 최댓값 선택의 정확한 편향 (토이 S)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 recompute.py와 toy_selection_and_filter.py를 실행)
한글 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서로 찾는다.
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
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#c2571a"
TASK_LABEL = {"tomato": "토마토 접시 옮기기", "marker": "마커 넘기고 세우기", "wine": "와인 따르기"}


def load():
    return json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def ci_rows(ax, labels, entries, colors, ylabel_visible):
    y = np.arange(len(labels))[::-1]
    for yi, e, c in zip(y, entries, colors):
        lo, hi = e["wilson"]
        ax.plot([100 * lo, 100 * hi], [yi, yi], color=c, lw=2)
        ax.plot(100 * e["rate"], yi, "o", color=c, ms=6)
        ax.text(103, yi, f'{e["k"]}/{e["n"]}', va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    if ylabel_visible:
        ax.set_yticklabels(labels, fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(-3, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_table1():
    rec = load()["V1_wilson"]["table1"]
    methods = ["OpenTV", "Manual", "G(0)", "G(1)", "G*"]
    labels = ["OpenTV (가드 없음)", "Manual (전문가 가드)", "GLIDE 라운드 0", "GLIDE 라운드 1", "GLIDE G*"]
    colors = [GREY, GREY, "#8fb0d9", "#8fb0d9", ACC]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))
    for ax, t in zip(axes, ["tomato", "marker", "wine"]):
        ci_rows(ax, labels, [rec[m][t] for m in methods], colors, t == "tomato")
        ax.set_title(TASK_LABEL[t], fontsize=10)
        ax.set_xlabel("성공률 (%)", fontsize=8.5)
    fig.suptitle("표 1 다시 그림: 데이터 수집 성공률과 Wilson 95% 구간 (각 10회)", fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_table2():
    rec = load()["V1_wilson"]["table2"]
    rows = list(rec["tomato"].keys())
    pretty = {"Vanilla/unguarded": "Vanilla 가드 없음", "Vanilla/guarded": "Vanilla 가드 있음",
              "Success-only/unguarded": "성공만 가드 없음", "Success-only/guarded": "성공만 가드 있음",
              "Mixed/unguarded": "섞은 데이터 가드 없음", "Mixed/guarded": "섞은 데이터 가드 있음"}
    colors = ["#c9c7bf", GREY, "#b8cde8", "#6f9bd1", "#e9b48c", WARM]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    for ax, t in zip(axes, ["tomato", "marker", "wine"]):
        ci_rows(ax, [pretty[r] for r in rows], [rec[t][r] for r in rows], colors, t == "tomato")
        ax.set_title(TASK_LABEL[t], fontsize=10)
        ax.set_xlabel("성공률 (%)", fontsize=8.5)
    fig.suptitle("표 2 다시 그림: 정책 성공률과 Wilson 95% 구간 (각 10회)", fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "own_table2_ci.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_selection():
    toy = json.loads((ANA / "toy_selection_and_filter_results.json").read_text(encoding="utf-8"))["S_selection"]
    rec = load()["V1_wilson"]["table1"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    x = np.arange(3)
    for t, c, dx in [("tomato", ACC, -0.12), ("marker", WARM, 0.0), ("wine", "#3a8a5c", 0.12)]:
        keys = ["G(0)", "G(1)", "G*"]
        ks = [rec[m][t]["k"] for m in keys]
        lo = [100 * rec[m][t]["wilson"][0] for m in keys]
        hi = [100 * rec[m][t]["wilson"][1] for m in keys]
        ax.errorbar(x + dx, [10 * k for k in ks], yerr=[np.array([10 * k for k in ks]) - lo, np.array(hi) - [10 * k for k in ks]],
                    fmt="o-", color=c, capsize=3, lw=1.2, ms=5, label=TASK_LABEL[t])
        ax.axhline(100 * toy["tasks"][t]["expected_max_rate_if_equal_rounds"], color=c, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["라운드 0", "라운드 1", "라운드 2 (G*)"], fontsize=8.5)
    ax.set_ylabel("성공률 (%) , 각 10회", fontsize=8.5)
    ax.set_ylim(-5, 122)
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=3)
    ax.set_title("라운드별 성공률 (점선: 라운드 간 차이가 없을 때 세 라운드 최댓값의 기댓값)", fontsize=8.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax = axes[1]
    ps = sorted(float(p) for p in toy["bias_curve"])
    bias = [100 * toy["bias_curve"][str(p)]["bias"] for p in ps]
    ax.plot([100 * p for p in ps], bias, "o-", color=INK, ms=5)
    ax.set_xlabel("참 성공률 p (%)", fontsize=8.5)
    ax.set_ylabel("세 라운드 최댓값의 상향 편향 (%p)", fontsize=8.5)
    ax.set_title("정확한 이항 계산: 10회 x 3라운드의 최댓값 빼기 p", fontsize=8.6)
    ax.grid(color="#e5e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_selection.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_table2()
    plot_selection()
    print("wrote 3 figures")
