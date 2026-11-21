"""우리가 만든 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 여기서 만들지 않는다).

own_stage_ci.png : 그림 5 단계별 성공률과 Wilson 95% 구간 (시도 20회)
own_diff_forest.png : 비교별 성공률 차이와 Newcombe 95% 구간 (논문 표·본문 수치)
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
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_stage_ci():
    stages = ["Gear", "Switch-Top", "Switch-Side", "Cleaning", "Screw", "Buckle"]
    vals = {  # 그림 5 라벨 (%)
        "ForceVLA": [70, 45, 10, 45, 25, 30],
        "FR-VLA": [90, 60, 35, 70, 40, 55],
    }
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = np.arange(len(stages))
    w = 0.38
    for j, (m, color) in enumerate([("ForceVLA", GREY), ("FR-VLA", ACC)]):
        pct = np.array(vals[m], dtype=float)
        k = np.round(pct * 20 / 100).astype(int)
        lo = np.array([wilson(kk, 20)[0] for kk in k])
        hi = np.array([wilson(kk, 20)[1] for kk in k])
        xs = x + (j - 0.5) * w
        ax.bar(xs, pct, width=w, color=color, label=m)
        ax.errorbar(xs, pct, yerr=np.vstack([pct - lo, hi - pct]), fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("단계별 성공률, 시도 20회 (Wilson 95% 구간, 그림 5 라벨 기준)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_stage_ci.png", dpi=200)
    plt.close(fig)


def plot_forest():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    rows = [
        rec["gear_magnitude_gating"], rec["gear_vs_vla_adapter_wf"],
        rec["table3_gear_unseen"], rec["table3_target_pos"],
        rec["table2_pooled"][0], rec["table2_pooled"][1],
    ]
    names = [
        "Gear: 대 자기 크기 게이팅 (18/20 대 14/20)",
        "Gear: 대 VLA-Adapter w/F (18/20 대 12/20)",
        "미지 기어 (17/20 대 12/20)",
        "미지 목표 위치 (15/20 대 10/20)",
        "어블레이션 합계: 대 w/o admission (120회)",
        "어블레이션 합계: 대 w/o tokenization (120회)",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, r, nm in zip(y, rows, names):
        d, lo, hi = r["diff_pp"], r["newcombe95_pp"][0], r["newcombe95_pp"][1]
        col = ACC if lo > 0 else GREY
        ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ecolor=INK, capsize=3)
        ax.text(hi + 1.5, yi, f"p={r['fisher_p']:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("성공률 차이 (percentage point, Newcombe 95% 구간)")
    ax.set_xlim(-30, 65)
    ax.set_title("비교별 차이: 0을 넘는 구간은 파란색", fontsize=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_diff_forest.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_stage_ci()
    plot_forest()
    print("wrote", FIG / "own_stage_ci.png", FIG / "own_diff_forest.png")
