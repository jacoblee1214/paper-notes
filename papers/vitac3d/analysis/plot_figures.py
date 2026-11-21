"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_task_ci.png    : 표 1 과제별 전체 과제 성공률과 Wilson 95% 구간 (다섯 방법)
own_camera_gain.png: 표 2 카메라 절제, 촉각 이득(점군 + 촉각 - 점군)과 Fisher p
own_power.png      : n = 20 설계의 검정력 곡선과 필요 시도 수
실행: /opt/anaconda3/bin/python3 analysis/plot_figures.py   (numpy, scipy, matplotlib)
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
N = 20


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#c4622d"


def wilson(k, n=N, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fig_task_ci():
    res = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    cells = res["table1_cells"]
    tasks = [("egg", "계란 찜"), ("fruit", "과일 손질"), ("hex", "육각 렌치"), ("sandwich", "샌드위치")]
    methods = [("RGB Only", "RGB"), ("RGB w/ Tactile Image", "RGB+촉각 이미지"),
               ("PC. Only", "점군"), ("PC. w/ Tactile Image", "점군+촉각 이미지"),
               ("PC. w/ Tactile Points (Ours)", "점군+촉각 점 (Ours)")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, (tk, tname) in zip(axes, tasks):
        for i, (m, _) in enumerate(methods):
            key = f"T1|{tk}|{m}|Whole Task"
            k = cells[key]["k"]
            lo, hi = wilson(k)
            color = WARM if "Ours" in m else (ACC if "Tactile" in m else GREY)
            ax.errorbar(k / N, i, xerr=[[k / N - lo], [hi - k / N]], fmt="o", color=color,
                        ms=6, capsize=3, lw=1.4)
            ax.text(min(hi, 0.99) + 0.02, i, f"{k}/{N}", va="center", fontsize=7.5, color=INK)
        ax.set_title(tname, fontsize=10, color=INK)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(axis="x", color="#e5e3dc", lw=0.6)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([m[1] for m in methods])
        ax.tick_params(axis="y", labelleft=(ax is axes[0]))
        ax.invert_yaxis()
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_xlabel("전체 과제 성공률 (20회)", fontsize=8.5)
    fig.suptitle("표 1: 과제별 전체 과제 성공률과 Wilson 95% 구간 (그래프: 우리)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "own_task_ci.png", dpi=170)
    plt.close(fig)


def fig_camera_gain():
    res = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    gain = res["table2_gain"]
    did = res["table2_diff_in_gain"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    groups = [("egg", "계란 찜"), ("hex", "육각 렌치")]
    x = np.arange(len(groups))
    w = 0.34
    for j, (cam, label, color) in enumerate([("Single Cam", "단일 카메라", GREY), ("Multi Cam", "다중 카메라", ACC)]):
        vals, errs, ps = [], [], []
        for g, _ in groups:
            item = gain[f"{g}|{cam}"]
            vals.append(item["gain"])
            ps.append(item["p_fisher"])
            errs.append(0.0)
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=color, label=label)
        for b, v, p in zip(bars, vals, ps):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"+{v:.2f}\n(p = {p:.2f})",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    diff_txt = "; ".join(f"{name}: 차이 {did[g]['diff_in_gain_single_minus_multi']:+.2f} "
                         f"[{did[g]['boot95'][0]:+.2f}, {did[g]['boot95'][1]:+.2f}]"
                         for g, name in groups)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in groups])
    ax.set_ylabel("촉각 이득 (점군 + 촉각 - 점군)", fontsize=8.5)
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("표 2: 카메라 수별 촉각 이득 (그래프: 우리)", fontsize=10, color=INK)
    fig.text(0.5, 0.01, "단일 - 다중 이득 차이의 부트스트랩 95% 구간 " + diff_txt, ha="center", fontsize=7, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "own_camera_gain.png", dpi=170)
    plt.close(fig)


def fig_power():
    ns = np.arange(10, 130)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    pairs = [(0.85, 0.50, ACC), (0.85, 0.65, WARM), (0.85, 0.70, GREY)]
    for p1, p2, color in pairs:
        za = stats.norm.ppf(0.975)
        power = []
        for n in ns:
            pbar = (p1 + p2) / 2
            # 정규근사 검정력 (양측 0.05), n은 군당
            se0 = np.sqrt(2 * pbar * (1 - pbar) / n)
            se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
            d = abs(p1 - p2)
            power.append(stats.norm.cdf((d - za * se0) / se1) + stats.norm.cdf((-d - za * se0) / se1))
        ax.plot(ns, power, color=color, lw=1.8, label=f"{p1:.2f} 대 {p2:.2f}")
    ax.axvline(N, color=INK, lw=0.9, ls="--")
    ax.text(N + 2, 0.06, "표 1, 3의 시도 수 20", fontsize=7.5, color=INK)
    ax.axhline(0.8, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("기준선 1군당 시도 수", fontsize=8.5)
    ax.set_ylabel("검정력 (양측 0.05)", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("n에 따른 검정력 (정규근사, 그래프: 우리)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_power.png", dpi=170)
    plt.close(fig)


def main():
    fig_task_ci()
    fig_camera_gain()
    fig_power()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))


if __name__ == "__main__":
    main()
