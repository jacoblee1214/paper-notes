"""DexUMI 재계산 결과를 그린 자체 그래프 두 장 (논문 그림을 쓰지 않는다).

출력: report/figures/own_table1_ci.png, report/figures/own_efficiency.png
실행: python3 analysis/plot_figures.py   (matplotlib, analysis/recompute_results.json 필요)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())
OUTDIR = ROOT / "report" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다
for cand in ["Noto Sans CJK KR", "Noto Sans CJK JP", "Apple SD Gothic Neo"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

COLS = ["Inspire\nCube", "Inspire\nCarton", "Inspire\nTea tool", "Inspire\nTea leaf",
        "XHand\nTea tool", "XHand\nTea leaf", "XHand\nKnob", "XHand\nPan", "XHand\nSalt"]
REL = [1.00, 0.85, 1.00, 0.85, 1.00, 0.85, 0.95, 0.95, 0.75]
ABS = [0.10, 0.35, 0.80, 0.00, 1.00, 0.25, 0.50, 0.45, 0.00]
N = 20


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def plot_table1():
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    xs = range(len(COLS))
    for series, color, dx, label in [(REL, "#1f5f8b", -0.12, "상대 행동 (Rel)"),
                                     (ABS, "#c0692b", 0.12, "절대 행동 (Abs)")]:
        ks = [round(v * N) for v in series]
        lo = [wilson(k, N)[0] for k in ks]
        hi = [wilson(k, N)[1] for k in ks]
        ax.errorbar([x + dx for x in xs], series,
                    yerr=[[s - l for s, l in zip(series, lo)], [h - s for s, h in zip(series, hi)]],
                    fmt="o", color=color, capsize=3, label=label, ms=5)
    for i in xs:
        p = RES["rel_vs_abs_tactile_Yes"][i]["p_holm"]
        mark = "*" if p < 0.05 else "n.s."
        ax.text(i, 1.08, mark, ha="center", fontsize=9, color="#333")
    ax.axhline(0.86, color="#888", lw=0.8, ls="--")
    ax.set_xticks(list(xs), COLS, fontsize=7.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("단계 누적 성공률 (n = 20)")
    ax.set_title("촉각 입력 있음·Inpaint: 상대 대 절대 행동 (Wilson 95% 구간, Holm 보정 유의 *)", fontsize=9)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "own_table1_ci.png")
    plt.close(fig)


def plot_efficiency():
    ct = RES["collection_efficiency"]["CT"]
    labels = ["사람 손만", "DexUMI", "텔레오퍼레이션"]
    vals = [ct["bare_hand"], ct["DexUMI"], ct["teleop"]]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    bars = ax.bar(labels, vals, color=["#bbbbbb", "#1f5f8b", "#c0692b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontsize=9)
    ax.set_ylabel("15분 세션 성공 시연 수 (그림 7 라벨 값)")
    ax.set_title("수집 처리량: 같은 작업, 같은 운영자 1회 세션", fontsize=9)
    ax.text(0.99, 0.96, f"DexUMI / 텔레오퍼레이션 = {vals[1]/vals[2]:.2f}\n(논문 3.2배, 반복 없음)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "own_efficiency.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_efficiency()
    print("saved:", sorted(p.name for p in OUTDIR.glob("own_*.png")))
