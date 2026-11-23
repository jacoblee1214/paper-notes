"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_ratios.png     : 표 2의 SUC-F 배율과 S 배율 (본문 문구 2.8배, 30배, 40배와 대조)
own_ablation.png   : 표 3의 SUC-L과 내재 S, 표 5와 표 8의 SUC-L
own_real.png       : 실로봇 성공률과 Wilson 95% 구간 (시행 수는 정수 풀이 가정)
own_anchor_toy.png : 토이 T1, 세계 좌표 대 뷰 조건부 대 앵커 상대 이동의 분산
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
"""
import json
import math
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
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#c44e3a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_ratios():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V2_table2_ratios"]
    keys = ["DGN_Float", "DGN_Table", "OBJ_Float", "OBJ_Table"]
    labels = ["DexGraspNet\nFloat", "DexGraspNet\nTableTop", "Objaverse\nFloat", "Objaverse\nTableTop"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    vals = [rec[k]["SUC-F_ratio"] for k in keys]
    ax.bar(range(4), vals, color=[ACC, ACC, GREY, GREY], width=0.6)
    ax.axhline(2.8, color=RED, ls="--", lw=1)
    ax.text(3.45, 2.85, "문구 2.8배", color=RED, fontsize=8, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("SUC-F 배율 (Ours / BimanGrasp)")
    ax.set_ylim(0, 3.4)
    ax.set_title("성공률 배율 (표 2)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    vals = [rec[k]["S_ratio"] for k in keys]
    ax.bar(range(4), vals, color=[ACC, ACC, GREY, GREY], width=0.6)
    ax.axhline(30, color=RED, ls="--", lw=1)
    ax.axhline(40, color=RED, ls=":", lw=1)
    ax.text(3.45, 30.8, "서론 30배", color=RED, fontsize=8, ha="right")
    ax.text(3.45, 40.8, "§6.4 40배", color=RED, fontsize=8, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("S 배율 (Ours / BimanGrasp)")
    ax.set_ylim(0, 50)
    ax.set_title("합성 속도 배율 (표 2의 S)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_ratios.png", dpi=200)
    plt.close(fig)


def plot_ablation():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V3_table3_sgs"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    names = ["Baseline", "+R", "+R+Dc", "+R+Dc+Ps"]
    suc = [28.33, 44.43, 57.66, 80.83]
    sgs = [2.06, 2.80, 3.96, 5.55]
    s_impl = [rec["implied_S_from_SGS_over_SUCL"][n] for n in names]
    x = np.arange(4)
    ax.bar(x, suc, color=ACC, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("SUC-L (%)")
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(x, sgs, "o-", color=RED, ms=4, label="SGS (표 3)")
    ax2.plot(x, s_impl, "s--", color=GREY, ms=4, label="내재 S = SGS / SUC-L")
    ax2.set_ylim(0, 9)
    ax2.set_ylabel("grasp/s")
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("표 3: 성공률은 단계마다 오르고 원시 속도는 거의 일정", fontsize=9)
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax = axes[1]
    rows = ["기준선", "GAF+SAGA", "BCM+GAF", "전체", "직접 쌍 표집\n(표 8)"]
    suc5 = [41.17, 44.68, 52.74, 61.93, 45.49]
    spd = [0.01, 0.01, 0.02, 0.02, 0.01]
    cols = [GREY, "#8fb0d9", "#8fb0d9", ACC, "#d9a441"]
    ax.bar(range(5), suc5, color=cols, width=0.6)
    for i, (v, s) in enumerate(zip(suc5, spd)):
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=8)
        ax.text(i, 3, f"SPD {s:.2f}" if i < 4 else "", ha="center", fontsize=7, color="white")
    ax.set_xticks(range(5))
    ax.set_xticklabels(rows, fontsize=7.5)
    ax.set_ylabel("SUC-L (%)")
    ax.set_ylim(0, 75)
    ax.set_title("표 5와 표 8 (SPD는 0.01에서 0.02로 오름)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=200)
    plt.close(fig)


def plot_real():
    items = [("단일 시점\n(33/50, 가정)", 33, 50, ACC), ("전체 점군\n(161/210, 가정)", 161, 210, ACC),
             ("합계\n(194/260)", 194, 260, "#3b3b3b"), ("v1 표 VII\n(46/60, 메시 입력)", 46, 60, "#d9a441")]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, (lab, k, n, c) in enumerate(items):
        p = 100 * k / n
        lo, hi = wilson(k, n)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color=c, capsize=4, ms=6)
        ax.text(i + 0.12, p, f"{p:.1f}%", fontsize=8, va="center")
    ax.axhline(66.8, color=GREY, ls="--", lw=1)
    ax.text(3.4, 67.4, "시뮬 SUC-L 66.8 (표 4)", fontsize=7.5, color=GREY, ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels([i[0] for i in items], fontsize=8)
    ax.set_xlim(-0.5, 3.7)
    ax.set_ylim(40, 95)
    ax.set_ylabel("성공률 (%), Wilson 95% 구간")
    ax.set_title("실로봇 성공률 (시행 수는 정수 풀이 가정)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real.png", dpi=200)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_anchor_results.json").read_text(encoding="utf-8"))
    conds = [("scale_0.15_0.40", "크기 0.15-0.40 m\n종횡비 0.3-1"), ("scale_fixed_0.275", "크기 고정 0.275 m"),
             ("scale_0.15_0.40_near_spheres", "크기 0.15-0.40 m\n거의 구형")]
    series = [("norm_world", "세계 좌표", GREY), ("norm_world_given_view", "세계 좌표 (뷰 조건부)", "#8fb0d9"),
              ("norm_anchor", "앵커 상대", ACC), ("norm_anchor_given_view", "앵커 상대 (뷰 조건부)", "#16406f")]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    w = 0.2
    x = np.arange(len(conds))
    for j, (key, lab, c) in enumerate(series):
        vals = [res[k][key] for k, _ in conds]
        ax.bar(x + (j - 1.5) * w, vals, w, color=c, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 1.5) * w, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in conds], fontsize=8)
    ax.set_ylabel("손목 이동 표준편차 노름 (cm)")
    ax.set_title("토이 T1: 앵커의 이득은 뷰 정보 효과와 크기 효과로 나뉨", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    ax.set_ylim(0, 33)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_anchor_toy.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_ratios()
    plot_ablation()
    plot_real()
    plot_toy()
    print("wrote figures to", FIG)
