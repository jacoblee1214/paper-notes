"""우리가 그린 그래프 두 장. 입력은 recompute.py의 결과(논문 Table II, III)다.

실행: python3 analysis/plot_figures.py  → report/figures/own_success_ci.png, own_failures.png
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((Path(__file__).resolve().parent / "recompute_results.json").read_text(encoding="utf-8"))

for cand in ("Noto Sans CJK KR", "Apple SD Gothic Neo"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

DEV = ["GELLO", "VR", "3D Mice"]
COLOR = {"GELLO": "#1f5fae", "VR": "#8a8a85", "3D Mice": "#c9a227"}
TASKS = ["Hat", "Mask", "Banana", "Towel", "USB"]
COUNTS = {d: [round(v * 12) for v in vals] for d, vals in {
    "GELLO": [0.92, 0.92, 1.0, 0.92, 0.83],
    "3D Mice": [0.75, 0.58, 0.67, 0.58, 0.58],
    "VR": [0.92, 0.83, 0.75, 0.58, 0.5]}.items()}


def wilson(k, n=12, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def success_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [3, 1.4]})
    width = 0.26
    for i, dev in enumerate(DEV):
        xs = [t + (i - 1) * width for t in range(len(TASKS))]
        ks = COUNTS[dev]
        ps = [k / 12 for k in ks]
        lo = [p - wilson(k)[0] for p, k in zip(ps, ks)]
        hi = [wilson(k)[1] - p for p, k in zip(ps, ks)]
        ax1.bar(xs, ps, width, color=COLOR[dev], label=dev, alpha=0.9)
        ax1.errorbar(xs, ps, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=2.5, lw=0.8)
        for x, k in zip(xs, ks):
            ax1.text(x, 0.03, f"{k}/12", ha="center", fontsize=6.5, color="white", rotation=90)
    ax1.set_xticks(range(len(TASKS)))
    ax1.set_xticklabels(TASKS)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("성공률 (12명)")
    ax1.set_title("과제별 성공률 (논문 Table II) · 오차 막대: Wilson 95%", fontsize=10)
    ax1.legend(fontsize=8, ncol=3, loc="upper center", frameon=False)
    ax1.grid(axis="y", lw=0.4, alpha=0.5)

    pooled = RES["wilson_pooled"]
    for i, dev in enumerate(DEV):
        r = pooled[dev]
        ax2.bar(i, r["p"], color=COLOR[dev], width=0.6)
        ax2.errorbar(i, r["p"], yerr=[[r["p"] - r["ci95"][0]], [r["ci95"][1] - r["p"]]], fmt="none",
                     ecolor="#333", capsize=3, lw=0.9)
        ax2.text(i + 0.33, r["p"], f'{r["k"]}/{r["n"]}', ha="left", va="center", fontsize=8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(DEV, fontsize=8)
    ax2.set_ylim(0, 1.12)
    ax2.set_title("5과제 풀링 (60시도)", fontsize=10)
    ax2.grid(axis="y", lw=0.4, alpha=0.5)
    fig.text(0.01, 0.01, "그래프: 우리가 그림. 논문 Table II 값을 12명 기준 개수로 바꿔 Wilson 구간을 계산했습니다. "
             "사용자 단위 쌍체 정보가 없어 과제 간 검정은 하지 않습니다.", fontsize=6.8, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "own_success_ci.png", dpi=170)
    plt.close(fig)


def failure_plot():
    marks = {
        "GELLO": {"Timeout": 1, "Self Collision": 1, "Env Collision": 3, "Other": 0},
        "3D Mice": {"Timeout": 9, "Self Collision": 5, "Env Collision": 6, "Other": 2},
        "VR": {"Timeout": 5, "Self Collision": 5, "Env Collision": 6, "Other": 1},
    }
    modes = ["Timeout", "Self Collision", "Env Collision", "Other"]
    shades = ["#4b6fa5", "#8fb0d9", "#c6d6ec", "#e0e0dc"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, dev in enumerate(DEV):
        bottom = 0
        for m, s in zip(modes, shades):
            v = marks[dev][m]
            ax.barh(i, v, left=bottom, color=s, edgecolor="white", label=m if i == 0 else None)
            if v:
                ax.text(bottom + v / 2, i, str(v), ha="center", va="center", fontsize=8)
            bottom += v
        ax.text(bottom + 0.3, i, f"합 {bottom} (실패 {60 - RES['wilson_pooled'][dev]['k']})", va="center", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(DEV)
    ax.invert_yaxis()
    ax.set_xlim(0, 30)
    ax.set_xlabel("실패 개수 (5과제 x 12명 = 60시도)")
    ax.set_title("실패 유형별 개수 (논문 Table III)", fontsize=10)
    ax.legend(fontsize=7.5, ncol=4, loc="lower center", bbox_to_anchor=(0.45, -0.42), frameon=False)
    fig.text(0.01, 0.01, "그래프: 우리가 그림. 합계는 Table II의 실패 수(60 - 성공 수)와 같습니다.", fontsize=6.8, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_failures.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    success_plot()
    failure_plot()
    print("wrote", FIG / "own_success_ci.png", FIG / "own_failures.png")
