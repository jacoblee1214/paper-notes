"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_ci.png    : 표 1 (칸당 100회) 열별 성공률과 Wilson 95% 구간
own_real_ci.png      : 실로봇(표 2), USB-C(표 4), Wash Crates(표 5) 성공률과 Wilson 95% 구간
own_cycle_time.png   : 표 3 처리량에서 계산한 주기 시간 감소율과 본문의 "최대 57%"
own_accept_rule.png  : 토이 T1, 갱신 채택 확률 (현재 성공률 0.9, 새 100개 instance)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c4691a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def plot_table1():
    cols = ["LIBERO\nobject", "LIBERO-Pro\nswap", "X-Y\n20×20", "basket\nswap", "permu-\ntation",
            "mixed\nall", "pack\nfixed", "pack\nvaried"]
    rows = {
        "π0.5": [.96, .24, .78, .15, .20, .20, .17, .18],
        "MolmoAct2": [.97, .43, .90, .26, .10, .20, .18, .18],
        "TipTop": [.22, .22, .29, .24, .31, .24, .34, .46],
        "GaP": [.95, .95, .95, .97, .93, .97, .99, .98],
    }
    colors = {"π0.5": GREY, "MolmoAct2": "#6c6b66", "TipTop": ORG, "GaP": ACC}
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    x = np.arange(len(cols))
    w = 0.2
    for i, (name, vals) in enumerate(rows.items()):
        pct = np.array(vals) * 100
        lo = np.array([wilson(round(v * 100), 100)[0] for v in vals])
        hi = np.array([wilson(round(v * 100), 100)[1] for v in vals])
        ax.errorbar(x + (i - 1.5) * w, pct, yerr=[pct - lo, hi - pct], fmt="o", ms=4, color=colors[name],
                    ecolor=colors[name], capsize=2, lw=1, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_ylabel("성공률 (%) 와 Wilson 95% 구간")
    ax.set_ylim(0, 102)
    ax.axvspan(1.5, 2.5, color="#f3e4cf", alpha=0.6, lw=0)
    ax.text(2, 5, "GaP 0.95 대 0.90\nFisher p = 0.28", ha="center", fontsize=7.5, color=INK)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.grid(axis="y", color="#e8e6df", lw=0.6)
    ax.set_title("표 1: 열별 성공률 (칸당 100회, 논문 값을 다시 그림)", fontsize=9, pad=32)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=170)
    plt.close(fig)


def plot_real():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1.5, 1.2, 1.2]})
    # 표 2
    ax = axes[0]
    items = [("Fulfill", 8, 25, 25, 25), ("Pack", 10, 30, 28, 30), ("Popcorn", 0, 20, 18, 20)]
    for j, (name, kt, nt, kg, ng) in enumerate(items):
        for off, k, n, c in ((-0.17, kt, nt, ORG), (0.17, kg, ng, ACC)):
            lo, hi = wilson(k, n)
            p = 100 * k / n
            ax.errorbar(j + off, p, yerr=[[p - lo], [hi - p]], fmt="o", color=c, capsize=3, lw=1.2, ms=5)
            ax.text(j + off, hi + 2, f"{k}/{n}", ha="center", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([i[0] for i in items], fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("성공률 (%)")
    ax.set_title("표 2: 실로봇 (주황 TipTop, 파랑 GaP)", fontsize=8.5)
    ax.grid(axis="y", color="#e8e6df", lw=0.6)
    # 표 4
    ax = axes[1]
    items4 = [("삽입 전체\n121/130", 121, 130), ("goal 조건\n103/110", 103, 110), ("위치·각도\n18/20", 18, 20),
              ("시퀀스 완주\n15/20", 15, 20)]
    for j, (name, k, n) in enumerate(items4):
        lo, hi = wilson(k, n)
        p = 100 * k / n
        ax.errorbar(j, p, yerr=[[p - lo], [hi - p]], fmt="o", color=ACC if j < 3 else ORG, capsize=3, lw=1.2, ms=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([i[0] for i in items4], fontsize=7)
    ax.set_ylim(40, 102)
    ax.set_title("표 4: USB-C 케이블", fontsize=8.5)
    ax.grid(axis="y", color="#e8e6df", lw=0.6)
    # 표 5
    ax = axes[2]
    items5 = [("GaP\n143/150", 143, 150, ACC), ("손 설계\n148/150", 148, 150, GREY),
              ("GaP 3h\n55/59", 55, 59, ACC), ("손 설계 3h\n58/59", 58, 59, GREY)]
    for j, (name, k, n, c) in enumerate(items5):
        lo, hi = wilson(k, n)
        p = 100 * k / n
        ax.errorbar(j, p, yerr=[[p - lo], [hi - p]], fmt="o", color=c, capsize=3, lw=1.2, ms=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([i[0] for i in items5], fontsize=7)
    ax.set_ylim(80, 101)
    ax.set_title("표 5: Wash Crates (150회: p = 0.17)", fontsize=8.5)
    ax.grid(axis="y", color="#e8e6df", lw=0.6)
    fig.suptitle("논문 표 2, 4, 5의 성공 횟수를 다시 그림 (Wilson 95% 구간, 독립 시행 가정)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_cycle():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["table3_fig3"]
    names = ["I 식료품 주문", "II 식료품 포장", "III 팝콘"]
    keys = ["I", "II", "III"]
    red = [rec[k]["ct_reduction_pct_from_table3"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar(names, red, color=ACC, width=0.55)
    for i, v in enumerate(red):
        ax.text(i, v - 5, f"{v:.1f}%", ha="center", fontsize=9, color="white")
    ax.axhline(57, color=ORG, ls="--", lw=1.2, label="본문 '최대 57%'")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.2), fontsize=8, frameon=True)
    ax.set_ylim(0, 70)
    ax.set_ylabel("주기 시간 감소율 (%)")
    ax.set_title("표 3 처리량에서 계산한 주기 시간 감소 (3600 / 처리량)", fontsize=9)
    ax.grid(axis="y", color="#e8e6df", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_cycle_time.png", dpi=170)
    plt.close(fig)


def plot_accept():
    d = json.loads((ANA / "toy_accept_rule_results.json").read_text(encoding="utf-8"))["accept_prob"]
    ds = ["-0.10", "-0.05", "-0.03", "-0.02", "-0.01", "+0.00", "+0.02", "+0.05"]
    x = [float(s) * 100 for s in ds]
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for key, c, lab in (("p=0.9,unpaired", ACC, "현재 0.9, 독립 100개씩"),
                        ("p=0.5,unpaired", GREY, "현재 0.5, 독립 100개씩"),
                        ("p=0.9,paired", ORG, "현재 0.9, 쌍체(상한)")):
        ax.plot(x, [100 * d[key][f"d={s}"] for s in ds], marker="o", ms=4, color=c, label=lab, lw=1.3)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("갱신의 실제 성공률 변화 d (%p)")
    ax.set_ylabel("채택 확률 (%)")
    ax.set_title("토이 T1: '하락하면 거부' 규칙에서 채택될 확률", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#e8e6df", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_accept_rule.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_real()
    plot_cycle()
    plot_accept()
    print("wrote 4 figures in", FIG)
