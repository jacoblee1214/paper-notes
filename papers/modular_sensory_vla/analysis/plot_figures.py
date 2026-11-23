"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_ci.png : 표 1의 방법별 평균 성공률과 Wilson 95% 구간 (4과제 x 24회 = 96회, 그림 판독이 아니라 표 값)
own_synergy.png   : 표 8과 표 1의 같은 방법 단일 모달리티 대 촉각+토크 평균 (기준선은 줄고 MoSS는 오른다는 서술의 구간)
own_forest.png    : 주요 비교의 성공률 차이와 95% 구간 (Newcombe 하이브리드 점수 구간)
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#d98b2b"
Z = 1.959964


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return 100 * d, 100 * (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


# 표 1 합계 성공 횟수 (96회 중), 표의 네 과제 값을 24회 격자로 되돌려 합함
K = {
    "GR00T": {"기본": 20, "Tactile-VLA\n(촉각)": 29, "ForceVLA\n(촉각)": 33, "MoSS\n(촉각)": 41, "TA-VLA\n(토크)": 32, "MoSS\n(토크)": 36, "MoSS\n(촉각+토크)": 47},
    "π0": {"기본": 26, "Tactile-VLA\n(촉각)": 28, "ForceVLA\n(촉각)": 37, "MoSS\n(촉각)": 37, "TA-VLA\n(토크)": 33, "MoSS\n(토크)": 40, "MoSS\n(촉각+토크)": 44},
}


def plot_table1():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, (model, d) in zip(axes, K.items()):
        names = list(d)
        vals = [100 * k / 96 for k in d.values()]
        lo = [100 * wilson(k, 96)[0] for k in d.values()]
        hi = [100 * wilson(k, 96)[1] for k in d.values()]
        cols = [GREY if "기본" in n else (ACC if "MoSS" in n else "#c9c7bf") for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, color=cols, width=0.7)
        ax.errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - np.array(vals)], fmt="none", ecolor=INK, capsize=2, lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5)
        ax.set_title(f"{model} 기반", fontsize=10)
        ax.set_ylim(0, 75)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("평균 성공률 (%), Wilson 95% 구간 (96회)")
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=200)
    plt.close(fig)


def plot_synergy():
    # (방법, 단일 모달리티 96회 중 성공, 촉각+토크 96회 중 성공). 기준선의 단일은 표 1, 둘 다는 표 8의 칸 합
    rows = [("Tactile-VLA", 29, 20), ("ForceVLA", 33, 20), ("TA-VLA", 32, 26), ("MoSS (최고 단일)", 41, 47)]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, (name, ks, kb) in enumerate(rows):
        for k, c, lab in ((ks, GREY, "단일 모달리티 (표 1)"), (kb, ACC if "MoSS" in name else ORG, "촉각+토크 (표 8)")):
            lo, hi = wilson(k, 96)
            y = i + (0.12 if c != GREY else -0.12)
            ax.plot([100 * lo, 100 * hi], [y, y], color=c, lw=2.2, alpha=0.7)
            ax.scatter([100 * k / 96], [y], color=c, s=34, zorder=3, label=lab if (i == 0 or (c == ACC)) else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("GR00T N1.5 기반 평균 성공률 (%), Wilson 95% 구간 (96회)")
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), fontsize=7.5, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_synergy.png", dpi=200)
    plt.close(fig)


def plot_forest():
    items = [
        ("GR00T: MoSS(촉각+토크) 대 기본", 47, 96, 20, 96, False),
        ("π0: MoSS(촉각+토크) 대 기본", 44, 96, 26, 96, False),
        ("GR00T: MoSS(촉각) 대 ForceVLA", 41, 96, 33, 96, False),
        ("π0: MoSS(촉각) 대 ForceVLA", 37, 96, 37, 96, False),
        ("GR00T: MoSS(토크) 대 TA-VLA", 36, 96, 32, 96, False),
        ("GR00T: MoSS 촉각+토크 대 촉각 단독", 47, 96, 41, 96, False),
        ("π0: MoSS 촉각+토크 대 최고 단일", 44, 96, 40, 96, False),
        ("표 2a 분리 스트림 대 단일 스트림 (2과제 48회)", 29, 48, 20, 48, False),
        ("표 2b 2단계 대 1단계 (48회)", 29, 48, 23, 48, False),
        ("표 2c 미래 예측 유 대 무 (48회)", 29, 48, 25, 48, False),
        ("표 4 손 로봇: MoSS 대 GR00T (48회)", 43, 48, 28, 48, False),
        ("표 7 RLBench 접촉 4과제 (시행 수 150 가정)", 229, 600, 164, 600, True),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, (name, k1, n1, k2, n2, assumed) in enumerate(items):
        d, lo, hi = newcombe(k1, n1, k2, n2)
        c = ORG if assumed else ACC
        ax.plot([lo, hi], [i, i], color=c, lw=2.2)
        ax.scatter([d], [i], color=c, s=32, zorder=3)
        ax.text(max(hi, d) + 1.2, i, f"{d:+.1f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("성공률 차이 (%p), 95% 구간 (Newcombe 하이브리드 점수)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_forest.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_synergy()
    plot_forest()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
