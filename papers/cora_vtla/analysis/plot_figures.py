"""재계산 결과를 그래프 세 장으로 그린다 (우리가 그린 그래프, 공개 가능).

실행: python3 analysis/plot_figures.py
  -> report/figures/own_sim_ci.png, own_real_ci.png, own_ablation_ci.png
입력: analysis/recompute_results.json (recompute.py가 만든 파일)
논문 그림은 쓰지 않습니다. 모든 값은 논문 표의 백분율에서 다시 계산한 것입니다.
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def setup_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def wilson_pct(k, n=150, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return 100 * (c - h), 100 * (c + h)


def bar(ax, y, k, color, label):
    mean = 100 * k / 150
    lo, hi = wilson_pct(k)
    ax.errorbar([mean], [y], xerr=[[mean - lo], [hi - mean]], fmt="o", color=color,
                capsize=3, markersize=5)
    ax.annotate(f"{mean:.1f}", (mean, y), textcoords="offset points", xytext=(8, 4), fontsize=8)


def sim_ci():
    rows = [("Base (π0.5)", RES["mujoco_table1_avg"]["Base"]["successes_of_150"]),
            ("Direct", RES["mujoco_table1_avg"]["Direct"]["successes_of_150"]),
            ("FTP-π0.5 (adapted)", RES["mujoco_table1_avg"]["FTP-pi0.5"]["successes_of_150"]),
            ("CoRA-VTLA", RES["mujoco_table1_avg"]["CoRA-VTLA"]["successes_of_150"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (name, k) in enumerate(rows):
        bar(ax, i, k, "#1f5fae" if name == "CoRA-VTLA" else "#777777", name)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("MuJoCo 3과제 합산 성공률 (%), 150회, Wilson 95% 구간")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png")
    plt.close(fig)


def real_ci():
    rows = [("Base (π0.5)", RES["real_table2_avg"]["Base"]["successes_of_150"]),
            ("Direct", RES["real_table2_avg"]["Direct"]["successes_of_150"]),
            ("FTP-π0.5 (adapted)", RES["real_table2_avg"]["FTP-pi0.5"]["successes_of_150"]),
            ("CoRA-VTLA", RES["real_table2_avg"]["CoRA-VTLA"]["successes_of_150"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (name, k) in enumerate(rows):
        bar(ax, i, k, "#1f5fae" if name == "CoRA-VTLA" else "#777777", name)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("실세계 3과제 합산 성공률 (%), 150회, Wilson 95% 구간")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


def ablation_ci():
    abl = RES["ablation_table3_avg"]
    names = list(abl.keys())
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, name in enumerate(names):
        bar(ax, i, abl[name]["successes_of_150"], "#1f5fae" if i == 0 else "#777777", name)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("MuJoCo 절제 성공률 (%), 150회, Wilson 95% 구간")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_ci.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    setup_font()
    sim_ci()
    real_ci()
    ablation_ci()
    print("wrote", FIG)


if __name__ == "__main__":
    main()
