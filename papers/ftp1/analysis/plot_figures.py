"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_ci.png   : 실로봇 seen(표 2)과 비가시(표 3) 과제별 성공률, Wilson 95% 구간 (과제당 20회)
own_sim_ci.png    : UniVTAC 과제별 성공률, Wilson 95% 구간 (과제당 100회)
own_pretrain.png  : 사전학습 대조 (표 4, 표 10, 표 11), UniVTAC 전체 평균과 비-lift 평균
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
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
INK, ACC, GREY, LIGHT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def errs(counts, n):
    pct = np.array([100 * c / n for c in counts])
    lo = np.array([wilson(c, n)[0] for c in counts])
    hi = np.array([wilson(c, n)[1] for c in counts])
    return pct, np.vstack([pct - lo, hi - pct])


def plot_real():
    # 표 2, 표 3 값은 %이며 과제당 20회이므로 성공 수 = % / 5
    seen_tasks = ["Draw\nBalloon", "Fix Hand\n(Tear)", "Fix Hand\n(Finish)", "Twist\nCap", "Flip\nBook", "Wipe\nDish"]
    base_seen = [35, 70, 35, 40, 65, 30]
    ftp_seen = [45, 80, 40, 65, 85, 60]
    unseen_tasks = ["Insert Hanoi\n(Xense)", "Insert USB\n(Xense)", "Wipe Board\n(Contactile)"]
    base_un = [25, 0, 20]  # 표 3 값 (%)
    ftp_un = [55, 30, 55]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, tasks, a, b, title in [
        (axes[0], seen_tasks, base_seen, ftp_seen, "seen 센서 (표 2, 과제당 20회)"),
        (axes[1], unseen_tasks, base_un, ftp_un, "비가시 센서 (표 3, 과제당 20회)"),
    ]:
        x = np.arange(len(tasks))
        ca = [round(v / 5) for v in a]
        cb = [round(v / 5) for v in b] if b is not None else None
        pa, ea = errs(ca, 20)
        pb, eb = errs(cb, 20)
        ax.bar(x - 0.18, pa, width=0.34, color=GREY, label="π0.5")
        ax.errorbar(x - 0.18, pa, yerr=ea, fmt="none", ecolor=INK, capsize=2, lw=0.8)
        ax.bar(x + 0.18, pb, width=0.34, color=ACC, label="FTP-1")
        ax.errorbar(x + 0.18, pb, yerr=eb, fmt="none", ecolor=INK, capsize=2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("성공률 (%)")
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.01, 0.005, "막대는 관측 성공률, 세로선은 Wilson 95% 구간. 과제당 20회라 구간이 넓습니다.",
             fontsize=7.5, color="#5a5852")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_sim():
    tasks = ["Lift\nBottle", "Pull-out\nKey", "Lift\nCan", "Put\nBottle", "Insert\nHole", "Insert\nTube"]
    series = {
        "π0.5": ([97, 38, 72, 16, 31, 41], GREY),
        "FTP-π0.5": ([77, 30, 26, 19, 47, 72], LIGHT),
        "FTP-1": ([97, 48, 65, 47, 64, 79], ACC),
    }
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    x = np.arange(len(tasks))
    w = 0.26
    for i, (name, (vals, col)) in enumerate(series.items()):
        off = (i - 1) * w
        pct, err = errs(vals, 100)
        ax.bar(x + off, pct, width=w, color=col, label=name)
        ax.errorbar(x + off, pct, yerr=err, fmt="none", ecolor=INK, capsize=2, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("성공률 (%)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png", dpi=170)
    plt.close(fig)


def plot_pretrain():
    # 표 4, 표 10, 표 11 (UniVTAC 평균). 구간은 그리지 않는다: 과제별 값이 없는 평균이다
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    labels = ["NTP-1\n(촉각 사전학습 없음)", "FTP-π0.5\n(사전학습 없음)", "π0.5\n(matched VLA)", "FTP-1"]
    overall = [50.0, 45.2, 57.0, 66.7]
    nonlift = [36.5, 42.0, 44.0, 59.5]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, overall, width=0.34, color=ACC, label="전체 6과제 평균")
    ax.bar(x + 0.18, nonlift, width=0.34, color=LIGHT, label="Lift 2과제 제외")
    for i, (a, b) in enumerate(zip(overall, nonlift)):
        ax.text(i - 0.18, a + 1, f"{a:.1f}", ha="center", fontsize=7.5)
        ax.text(i + 0.18, b + 1, f"{b:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 80)
    ax.set_ylabel("성공률 (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    steps = ["0%\n(FTP-π0.5)", "50%\n(25k 스텝)", "100%\n(50k 스텝)"]
    ov = [45.2, 59.0, 66.7]
    nl = [42.0, 46.0, 59.5]
    x = np.arange(3)
    ax.plot(x, ov, marker="o", color=ACC, label="전체 평균")
    ax.plot(x, nl, marker="s", color=GREY, label="Lift 2과제 제외")
    for i in range(3):
        ax.text(i, ov[i] + 2, f"{ov[i]:.1f}", ha="center", fontsize=7.5, color=ACC)
        ax.text(i, nl[i] - 4.5, f"{nl[i]:.1f}", ha="center", fontsize=7.5, color="#5a5852")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=7.5)
    ax.set_ylim(30, 75)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.005, "논문 보고값을 그대로 옮긴 것입니다. 시드 반복이 없어 구간을 그리지 않았습니다.",
             fontsize=7.5, color="#5a5852")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "own_pretrain.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_real()
    plot_sim()
    plot_pretrain()
    print("saved:", sorted(p.name for p in FIG.glob("own_*.png")))
