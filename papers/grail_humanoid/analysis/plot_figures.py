"""우리가 그린 그래프 두 장 (논문 그림 아님). CPU, 수 초.

실행: python3 analysis/plot_figures.py → report/figures/own_*.png
  own_s2r_ci.png : Table 3 실로봇 pick-up 객체별 성공률과 Clopper-Pearson 95% 구간 (객체당 10회)
  own_sr_ci.png  : Table 2 성공률과 Wilson 95% 구간 (분모 124 가정, 논문 본문의 124 벤치마크)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))


def pick_font():
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]:
        if cand in names:
            return cand
    return "DejaVu Sans"


plt.rcParams["font.family"] = pick_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, MUTE = "#1b1b1a", "#1f5fae", "#8a887f"


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return 100 * lo, 100 * hi


def wilson(k, n, z=1.959964):
    """k는 실수여도 된다(보고된 퍼센트 × n)."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return 100 * (c - h), 100 * (c + h)


def fig_s2r():
    names = ["Cube", "Apple", "Tea Box", "Carrot", "Wet Wipes", "Spray Can", "Lint Roller", "Peach",
             "Flashlight", "Medicine Bottle"]
    ks = [10, 6, 10, 7, 9, 10, 5, 9, 8, 8]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    for i, (nm, k) in enumerate(zip(names, ks)):
        lo, hi = cp(k, 10)
        color = ACC if i < 5 else MUTE
        ax.errorbar(i, 100 * k / 10, yerr=[[100 * k / 10 - lo], [hi - 100 * k / 10]], fmt="o",
                    color=color, ecolor=color, capsize=4, ms=7)
        ax.text(i, 3, f"{k}/10", ha="center", fontsize=8, color=INK)
    ax.axvline(4.5, color=MUTE, lw=0.8, ls="--")
    ax.text(2, 108, "Seen (학습 객체)", ha="center", color=INK, fontsize=9)
    ax.text(7, 108, "Unseen", ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("실물 성공률 (%)")
    ax.set_title("Table 3 객체별 성공률과 Clopper-Pearson 95% 구간 (우리가 계산)", fontsize=10, color=INK)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_s2r_ci.png")
    plt.close(fig)


def fig_sr():
    # 논문 보고값(%)을 그대로 쓰고, 분모 124에서 가능한 Wilson 구간을 그린다(정수 개수로 되돌리지 않음).
    rows = [("Ours (Full)", 81.4), ("ResMimic", 49.2), ("HDMI", 48.5), ("Ours w/o π_φ", 39.7),
            ("Ours w/o Rel. Obs.", 57.9), ("Ours w/o SONIC", 45.0)]
    N = 124
    fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=160)
    for i, (nm, p) in enumerate(rows):
        lo, hi = wilson(p * N / 100, N)
        color = ACC if "Full" in nm else MUTE
        ax.barh(i, p, color=color, alpha=0.85, height=0.55)
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.text(hi + 1.5, i, f"{p:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("성공률 (%, 논문 보고값)")
    ax.set_title("Table 2 보고값과 Wilson 95% 구간 (분모 124 가정, 우리가 계산)", fontsize=10, color=INK)
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sr_ci.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_s2r()
    fig_sr()
    print("wrote", FIG / "own_s2r_ci.png", FIG / "own_sr_ci.png", "| font:", plt.rcParams["font.family"])
