"""우리가 만든 그래프 두 장. 논문 그림이 아니다.
  report/figures/own_table2_heatmap.png : Table II의 24 과제 x 5 rollout Task Progress 히트맵
  report/figures/own_gap_power.png      : toy_gap_power 결과(SD 가정에 따른 유의 비율)
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from recompute import ROWS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def heatmap():
    labels = [f"{c} {i} / {t}" for c, i, t, *_ in ROWS]
    M = np.array([r for *_, r, _ in ROWS])
    fig, ax = plt.subplots(figsize=(7.2, 8.6), dpi=170)
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(5), [f"R{k}" for k in range(1, 6)])
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    for y in range(M.shape[0]):
        for x in range(M.shape[1]):
            ax.text(x, y, f"{M[y, x]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Task Progress (%)", fontsize=8)
    ax.set_title("Table II 재배치: 실세계 Task Progress (24 과제 x 5 rollout)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_table2_heatmap.png")
    plt.close(fig)


def power():
    res = json.loads((Path(__file__).parent / "toy_gap_power_results.json").read_text(encoding="utf-8"))
    sds = sorted(int(k) for k in res["by_sd"])
    rates = [res["by_sd"][str(s)]["welch_p_lt_0.05_rate"] for s in sds]
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=170)
    ax.plot(sds, rates, marker="o", color="#1f5fae")
    ax.axhline(0.8, ls="--", color="#888", lw=1)
    ax.text(sds[-1], 0.82, "검정력 0.8", ha="right", fontsize=8, color="#555")
    ax.set_xlabel("rollout 표준편차 가정 s (pp)", fontsize=9)
    ax.set_ylabel("p < 0.05 비율 (토이)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig. 5 Variation #1 (98.0 대 61.0, 5 rollout): SD 가정에 따른 유의 비율", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_gap_power.png")
    plt.close(fig)


if __name__ == "__main__":
    heatmap()
    power()
    print("wrote", FIG)
