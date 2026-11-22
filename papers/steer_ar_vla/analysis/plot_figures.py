"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sir_ci.png   : 표 1과 그림 3b의 SIR을 가정 시도 수(n=8, 16)로 Wilson 95% 구간과 함께
own_rates_ci.png : 언어 따르기 0/10 → 10/10, 드로어 성공률(가정 시도 수)과 Wilson 95% 구간
own_toy_prefix.png: 토이 T2, 관절별 속도 분포와 prefix 방향 정보(코사인)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 python3 analysis/toy_steering_prefix.py 로 결과 JSON을 만들어 둘 것.
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
INK, ACC, GREY, WARN = "#1b1b1a", "#1f5fae", "#9a9890", "#c0392b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def errbar(ax, x, k, n, color, label=None, dx=0.0, marker="o"):
    lo, hi = wilson(k, n)
    p = 100 * k / n
    ax.errorbar([x + dx], [p], yerr=[[max(0, p - lo)], [max(0, hi - p)]], fmt=marker, color=color, capsize=3, label=label, ms=6)


def plot_sir():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ws = [1, 2, 4, 6]
    x = np.arange(4)
    k8 = [3, 8, 8, None]       # w=6의 0.94는 n=8에서 불가능
    k16 = [6, 16, 16, 15]
    for i in range(4):
        if k8[i] is not None:
            errbar(ax, i, k8[i], 8, ACC, "가정 n=8 (공개 노트북은 창당 8회)" if i == 0 else None, dx=-0.08)
        errbar(ax, i, k16[i], 16, WARN, "가정 n=16 (0.94가 가능한 최소 n)" if i == 0 else None, dx=0.08, marker="s")
    ax.axhline(25, ls="--", c=GREY, lw=1)
    ax.text(1.5, 20, "우연 25%", color=GREY, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"w={w}" for w in ws])
    ax.set_ylabel("SIR (%)")
    ax.set_ylim(0, 105)
    ax.set_title("표 1: 주입 창 크기별 SIR, Wilson 95% 구간", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False)

    ax = axes[1]
    bs = [0, 1, 2, 3]
    ks = [(16, 16), (14, 16), (14, 16), (2, 16)]
    for b, (k, n) in zip(bs, ks):
        errbar(ax, b, k, n, ACC)
    ax.axhline(25, ls="--", c=GREY, lw=1)
    ax.set_xticks(bs)
    ax.set_xticklabels([f"b={b}" for b in bs])
    ax.set_ylim(0, 105)
    ax.set_title("그림 3b: 시작 인덱스별 SIR (w=2)", fontsize=10)
    ax.set_xlabel("가정 n=16 (b=3은 2/16=12.5%. 논문의 14%는 n=16에서 불가능)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_sir_ci.png", dpi=170)
    plt.close(fig)


def plot_rates():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    errbar(ax, 0, 0, 10, GREY)
    errbar(ax, 1, 10, 10, ACC)
    errbar(ax, 3, 0, 10, GREY)
    errbar(ax, 4, 10, 10, ACC)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["초록 정육면체\n조향 전", "조향 후", "파란 직육면체\n조향 전", "조향 후"], fontsize=8)
    ax.set_ylim(-3, 105)
    ax.set_ylabel("언어 따르기 성공률 (%)")
    ax.set_title("4.4절: 0/10 → 10/10, Wilson 95% 구간", fontsize=10)
    ax.text(2, 50, "시행 수 10은 부록 A.1(4)의\n40회 ÷ 지시문 4개", fontsize=7.5, color=GREY, ha="center")

    ax = axes[1]
    errbar(ax, 0, 4, 40, GREY, "자율 4/40 가정")
    errbar(ax, 0, 1, 10, WARN, "자율 1/10 가정", dx=0.15, marker="s")
    errbar(ax, 1, 29, 40, ACC, "조향 29/40 (72.5%의 최소 n)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["드로어: 자율", "드로어: 조향"])
    ax.set_ylim(0, 105)
    ax.set_title("표 2 드로어 성공률 (시행 수 미기재, 가정)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "own_rates_ci.png", dpi=170)
    plt.close(fig)


def plot_toy():
    t = json.loads((ANA / "toy_steering_prefix_results.json").read_text(encoding="utf-8"))["T2_prefix_information"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    names = ["forward", "back", "left", "right"]
    cols = {"forward": "#1f5fae", "back": "#7aa6d8", "left": "#c0392b", "right": "#e29a90"}
    width = 0.2
    for i, n in enumerate(names):
        q = np.abs(np.array(t["qdot_per_direction_rad_s"][n]))
        ax.bar(np.arange(7) + (i - 1.5) * width, q, width, color=cols[n], label=n)
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"J{i + 1}" for i in range(7)])
    ax.set_ylabel("|관절 속도| (rad/s, 말단 1 m/s)")
    ax.set_title("T2: 방향별로 쓰는 관절 (DROID 홈 자세, 대략 기구학)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    labels, f_cos, l_cos = [], [], []
    for w in (1, 2, 4, 6):
        c = t["window_sweep_b0"][f"w{w}"]["cos_with_intent"]
        labels.append(f"b=0\nw={w}")
        f_cos.append(c["forward"])
        l_cos.append(c["left"])
    for b in (1, 2, 3):
        c = t["start_sweep_w2"][f"b{b}"]["cos_with_intent"]
        labels.append(f"b={b}\nw=2")
        f_cos.append(c["forward"])
        l_cos.append(c["left"])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, f_cos, 0.36, color="#1f5fae", label="앞(대칭: 뒤)")
    ax.bar(x + 0.18, l_cos, 0.36, color="#c0392b", label="왼쪽(대칭: 오른쪽)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("prefix 방향과 의도의 코사인")
    ax.set_title("T2: 관절 직류 토큰 prefix가 담은 방향", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_prefix.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_sir()
    plot_rates()
    plot_toy()
    print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
