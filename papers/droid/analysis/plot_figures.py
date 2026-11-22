"""우리가 그린 그래프 두 장을 만든다. 입력은 recompute_results.json이다.

  report/figures/own_success_ci.png : 풀링 성공률과 Wilson 95% 구간 (분포 내, 분포 밖, 장면 다양성)
  report/figures/own_task_diff.png  : 과제별 차이 (DROID − 기준선)와 0선

실행: python3 analysis/plot_figures.py  (numpy, matplotlib)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
FIG = ROOT / "report" / "figures"

FONT_CANDIDATES = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"]
available = {f.name for f in font_manager.fontManager.ttflist}
font = next((f for f in FONT_CANDIDATES if f in available), None)
if font:
    plt.rcParams["font.family"] = font
plt.rcParams["axes.unicode_minus"] = False

INK, GREY = "#1b1b1a", "#55544f"
COL = {"noco": "#9bc17a", "oxe": "#5f9e8f", "droid": "#e8833a", "20_scenes": "#2f9c95", "diverse": "#e8833a"}
NAME = {"noco": "도메인 데이터만", "oxe": "OXE 섞기", "droid": "DROID 섞기", "20_scenes": "20개 장면", "diverse": "다양한 장면"}


def success_plot():
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=160)
    groups = [("분포 내 (6 과제)", "fig8_id", ["noco", "oxe", "droid"]),
              ("분포 밖 (7 설정)", "fig8_ood", ["noco", "oxe", "droid"]),
              ("분포 밖 장면 다양성 (3 과제)", "fig10_scene_diversity", ["20_scenes", "diverse"])]
    x0 = 0.0
    ticks, labels = [], []
    for gname, key, methods in groups:
        for i, m in enumerate(methods):
            v = RES[key]["methods"][m]
            k, n = v["pooled_success"], v["pooled_n"]
            lo, hi = v["pooled_wilson95"]
            rate = k / n
            x = x0 + i * 0.9
            ax.bar(x, rate, width=0.8, color=COL[m], edgecolor="none")
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color=INK, capsize=4, lw=1.1)
            ax.text(x, hi + 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=8, color=INK)
            ticks.append(x)
            labels.append(NAME[m])
        ax.text(x0 + (len(methods) - 1) * 0.45, -0.2, gname, ha="center", va="top", fontsize=9, color=GREY,
                transform=ax.get_xaxis_transform())
        x0 += len(methods) * 0.9 + 0.9
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("성공률 (풀링, 셀당 10회)", color=INK, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("논문 그림 8, 10의 풀링 성공률과 Wilson 95% 구간 (우리가 다시 그림)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_success_ci.png")
    plt.close(fig)


def task_diff_plot():
    specs = [
        ("fig8_id", "droid_minus_noco", "분포 내 DROID - 도메인만"),
        ("fig8_ood", "droid_minus_oxe", "분포 밖 DROID - OXE"),
        ("fig10_scene_diversity", "diverse_minus_20_scenes", "장면 다양성 다양 - 20개"),
    ]
    rows = []  # (label, value or None, kind)
    for key, comp, title in specs:
        c = RES[key]["comparisons"][comp]
        for t, d in zip(RES[key]["tasks"], c["per_task_diffs"]):
            rows.append((f"{title} | {t}", d, "task"))
        rows.append((f"{title} | 평균 (95% 구간)", (c["mean_diff"], c["paired_ci95"]), "mean"))
        rows.append(("", None, "gap"))
    fig_h = 0.22 * len(rows) + 0.7
    fig, ax = plt.subplots(figsize=(8.6, fig_h), dpi=160)
    ys, labels = [], []
    y = len(rows)
    for label, val, kind in rows:
        y -= 1
        if kind == "task":
            color = COL["droid"] if val > 0 else (GREY if val == 0 else COL["oxe"])
            ax.scatter(val, y, color=color, s=30, zorder=3)
            ys.append(y); labels.append(label)
        elif kind == "mean":
            mean, ci = val
            ax.errorbar(mean, y, xerr=[[mean - ci[0]], [ci[1] - mean]], color=INK, capsize=4, lw=1.2, zorder=4)
            ax.scatter(mean, y, marker="D", color=INK, s=26, zorder=5)
            ys.append(y); labels.append(label)
    ax.axvline(0, color=GREY, lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_xlabel("성공률 차이 (점: 과제별, 다이아몬드: 과제 평균과 과제 쌍 95% 구간)", fontsize=8.5, color=INK)
    ax.set_xlim(-0.5, 1.0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("과제별 차이 (우리가 다시 그림). 구간이 0을 가로지르면 과제 쌍 검정에서 유의하지 않음", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_task_diff.png")
    plt.close(fig)


if __name__ == "__main__":
    success_plot()
    task_diff_plot()
    print("wrote", FIG / "own_success_ci.png", FIG / "own_task_diff.png", "font:", font)
