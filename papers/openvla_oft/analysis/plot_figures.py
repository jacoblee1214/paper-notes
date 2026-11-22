"""우리가 표 값에서 다시 그린 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다: 실사 사진과 로봇 장면이 들어 있음).

own_libero_ci.png   : 표 I의 LIBERO 스위트별 SR과 스위트당 500 시도 가정의 이항 95% 구간 (OFT 대 π0, OpenVLA)
own_efficiency.png  : 표 II의 처리량(로그 축)과 지연, 확산 정책의 추론 단계 수에 따른 변화
own_aloha.png       : 표 X–XIII의 과제별 점수 (부분 점수 평균, 과제당 시도 수 표기)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import math
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
INK, GREY = "#1b1b1a", "#55544f"
BLUE, LIGHT, MID = "#1f5fae", "#c9c6bd", "#6f8fb8"


def wilson(p, n, z=1.959964):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def libero():
    suites = ["Spatial", "Object", "Goal", "Long"]
    rows = [
        ("OpenVLA (미세조정)", [84.7, 88.4, 79.2, 53.7], LIGHT),
        ("π0 (미세조정)", [96.8, 98.8, 95.8, 85.2], MID),
        ("OpenVLA-OFT", [97.6, 98.4, 97.9, 94.5], BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=160)
    w = 0.26
    x = np.arange(len(suites))
    for i, (name, vals, col) in enumerate(rows):
        lo, hi = zip(*[wilson(v / 100, 500) for v in vals])
        err = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=name, edgecolor="white")
        ax.errorbar(x + (i - 1) * w, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_xticks(x, suites)
    ax.set_ylim(40, 105)
    ax.set_ylabel("성공률 (%)", color=INK)
    ax.set_title("LIBERO 스위트별 성공률 (표 I 값, 스위트당 500 시도 가정 95% 구간)", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    style(ax)
    fig.text(0.01, 0.01, "그래프: 논문 표 I 값을 이 저장소가 다시 그린 것. 논문 그림이 아니다. 3시드 평균이면 구간은 더 좁다.",
             fontsize=7.2, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_libero_ci.png")
    plt.close(fig)


def efficiency():
    names = ["OpenVLA", "+PD", "+PD&AC", "+PD&AC\nCont-L1", "확산 T=50", "확산 T=10",
             "확산 T=5", "확산 T=2", "확산 T=1", "+추가 입력"]
    thr = [4.2, 15.9, 108.8, 109.7, 4.2, 19.3, 35.1, 80.3, 109.4, 71.4]
    sr = [53.7, None, 86.5, 90.7, 91.1, 91.0, 90.0, 85.7, 0.0, 94.5]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=160, gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(names))[::-1]
    cols = [LIGHT if "확산" in n else BLUE for n in names]
    a1.barh(y, thr, color=cols)
    a1.set_xscale("log")
    a1.set_yticks(y, names, fontsize=8)
    a1.set_xlabel("처리량 (Hz, 로그 축)", color=INK, fontsize=9)
    a1.set_title("처리량 (A100, 224px, K=8 기준)", color=INK, fontsize=10)
    for yi, t in zip(y, thr):
        a1.text(t * 1.08, yi, f"{t:g}", va="center", fontsize=7.5, color=INK)
    style(a1)
    pts = [(t, s, n) for t, s, n in zip(thr, sr, names) if s is not None]
    for t, s, n in pts:
        a2.scatter(t, s, color=LIGHT if "확산" in n else BLUE, s=36, zorder=3)
        off = (4, -10) if n == "확산 T=2" else (4, 3)
        a2.annotate(n.replace("\n", " "), (t, s), fontsize=6.6, xytext=off, textcoords="offset points", color=INK)
    a2.set_xscale("log")
    a2.set_xlabel("처리량 (Hz, 로그 축)", color=INK, fontsize=9)
    a2.set_ylabel("LIBERO-Long 성공률 (%)", color=INK, fontsize=9)
    a2.set_title("속도와 성공률 (확산은 추론 단계 수)", color=INK, fontsize=10)
    a2.set_ylim(-4, 100)
    style(a2)
    fig.text(0.01, 0.01, "그래프: 표 II 값을 다시 그린 것. 확산 T=1의 성공률 0.0%는 원문 표 그대로 옮겼다.", fontsize=7.2, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_efficiency.png")
    plt.close(fig)


def aloha():
    tasks = ["fold shorts\n(10회)", "fold shirt\n(10회)", "scoop X into bowl\n(12회)", "put X into pot\n(24회)"]
    methods = ["ACT", "Diffusion Policy", "RDT-1B", "pi0", "OpenVLA-OFT+"]
    cols = [LIGHT, "#b5b2a8", "#9a9890", MID, BLUE]
    data = np.array([
        [100, 94, 71.25, 23.75],
        [100, 100, 79.17, 30.83],
        [100, 96, 73.33, 44.17],
        [100, 100, 93.33, 42.08],
        [100, 100, 100, 51.25],
    ])
    fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=160)
    x = np.arange(len(tasks))
    w = 0.16
    for i, (m, c) in enumerate(zip(methods, cols)):
        ax.bar(x + (i - 2) * w, data[i], w, color=c, label=m, edgecolor="white")
    ax.set_xticks(x, tasks, fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("부분 점수 평균 (100점 만점)", color=INK)
    ax.set_title("ALOHA 과제별 점수 (표 X–XIII 값, 점수 평균이며 성공률이 아니다)", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    style(ax)
    fig.text(0.01, 0.01, "그래프: 논문 표 값을 이 저장소가 다시 그린 것. 과제당 시도 수가 적어 차이의 불확실성이 크다.",
             fontsize=7.2, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_aloha.png")
    plt.close(fig)


def main():
    libero()
    efficiency()
    aloha()
    for name in ("own_libero_ci.png", "own_efficiency.png", "own_aloha.png"):
        print("wrote", FIG / name)


if __name__ == "__main__":
    main()
