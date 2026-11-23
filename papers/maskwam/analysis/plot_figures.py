"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_libero.png    : 표 1 스위트별 성공률과 Wilson 95% 구간 (스위트당 500회 가정)
own_ambiguous.png : 표 5 설정별 평균(4과제 x 60회 = 240회)과 표 4 절제(960회), Wilson 95% 구간
own_fig8.png      : 그림 8 판독 값(그림 판독 값, 조건당 60회 가정)과 Wilson 95% 구간
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORANGE = "#1b1b1a", "#1f5fae", "#9a9890", "#c4701b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_libero():
    rec = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))["V1_libero_rows"]
    methods = ["pi0.5", "Motus", "FastWAM", "Ours-RGB", "Ours-Mask", "Ours"]
    labels = ["π0.5", "Motus", "FastWAM", "RGB만 예측", "마스크만 예측", "MaskWAM"]
    colors = [GREY, GREY, GREY, "#8fb0d9", "#8fb0d9", ACC]
    suites = ["Spatial", "Object", "Goal", "Long"]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.4), sharey=True)
    for i, (ax, s) in enumerate(zip(axes, suites)):
        for j, m in enumerate(methods):
            k = rec[m]["counts_n500"][i]
            lo, hi = wilson(k, 500)
            p = 100 * k / 500
            ax.errorbar(j, p, yerr=[[p - lo], [hi - p]], fmt="o", color=colors[j], ecolor=colors[j], capsize=2, ms=5)
        ax.set_title(s, fontsize=10)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
        ax.set_ylim(88, 100.6)
        ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    axes[0].set_ylabel("성공률 (%)")
    fig.suptitle("표 1 LIBERO 스위트별 성공률과 Wilson 95% 구간 (스위트당 500회 가정)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_libero.png", dpi=170)
    plt.close(fig)


def plot_amb():
    rec = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
    cnt = {}
    for key, v in rec["V7_table5_integer_and_avg"].items():
        m, s = key.split("/")
        cnt.setdefault(m, {})[s] = sum(v["counts"])
    settings = ["ID", "Distractors", "Novel", "Lighting"]
    names = {"pi0-mask": "π0-mask", "pi0-coord": "π0-coord", "FastWAM-coord": "FastWAM-coord", "Ours": "MaskWAM"}
    cols = {"pi0-mask": ORANGE, "pi0-coord": GREY, "FastWAM-coord": "#c9c7bf", "Ours": ACC}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    w = 0.2
    for j, m in enumerate(names):
        for i, s in enumerate(settings):
            k = cnt[m][s]
            lo, hi = wilson(k, 240)
            p = 100 * k / 240
            ax.bar(i + (j - 1.5) * w, p, w * 0.92, color=cols[m], label=names[m] if i == 0 else None)
            ax.errorbar(i + (j - 1.5) * w, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, capsize=1.5, lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["분포 내", "방해물", "새 물체", "조명"])
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.set_title("표 5 설정별 평균 (4과제 x 60회 = 240회)", fontsize=9.5)
    ax = axes[1]
    t4 = [("예측 없음\n(마스크 프롬프트만)", 207, GREY), ("좌표 텍스트\n(+ 마스크 예측)", 175, "#c9c7bf"), ("MaskWAM", 815, ACC)]
    for i, (name, k, c) in enumerate(t4):
        lo, hi = wilson(k, 960)
        p = 100 * k / 960
        ax.bar(i, p, 0.6, color=c)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.text(i, p + 4, f"{p:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([t[0] for t in t4], fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_title("표 4 절제 (16칸 x 60 = 960회, 칸 수는 정수 복원)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_ambiguous.png", dpi=170)
    plt.close(fig)


def plot_fig8():
    series = {
        "Task 5": {"침식": [(0.40, 100), (0.70, 100), (0.82, 100)], "팽창": [(0.35, 98.3), (0.55, 100), (0.84, 100)],
                  "이동": [(0.10, 85.0), (0.32, 91.7), (0.72, 95.0)], "드롭아웃": [(0.23, 96.7), (0.50, 100), (0.70, 100)]},
        "Task 8": {"침식": [(0.40, 80.0), (0.68, 83.3), (0.82, 83.3)], "팽창": [(0.35, 75.0), (0.55, 78.3), (0.84, 83.3)],
                  "이동": [(0.10, 51.7), (0.32, 68.3), (0.72, 75.0)], "드롭아웃": [(0.23, 75.0), (0.50, 80.0), (0.70, 83.3)]},
    }
    online = {"Task 5": 100.0, "Task 8": 83.3}
    cols = {"침식": "#2d6aa8", "팽창": "#d6a62a", "이동": "#555555", "드롭아웃": "#9aa7b8"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for ax, (task, d) in zip(axes, series.items()):
        for name, pts in d.items():
            xs = [x for x, _ in pts]
            ys = np.array([y for _, y in pts])
            ks = np.round(ys * 60 / 100)
            lo = np.array([wilson(k, 60)[0] for k in ks])
            hi = np.array([wilson(k, 60)[1] for k in ks])
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4, lw=1.3, capsize=2, color=cols[name], label=name)
        ax.axhline(online[task], ls="--", color=INK, lw=0.8)
        ax.set_title(f"{task} (그림 판독 값, 60회 가정)", fontsize=9.5)
        ax.set_xlabel("마스크 IoU")
        ax.set_xlim(0, 1.0)
        ax.grid(color="#e5e3dc", lw=0.6)
    axes[0].set_ylabel("성공률 (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "own_fig8.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_libero()
    plot_amb()
    plot_fig8()
    print("wrote", FIG)
