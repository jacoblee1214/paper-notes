"""우리가 만든 그래프 두 장을 report/figures/에 저장한다.

own_closed_loop_ci.png : 폐루프 LIBERO-10 위치 이동 성공률 (100회, Wilson 95% 구간)과 baseline 기준선
own_matched_ctrl.png   : 표 2 매칭 대조의 학습 시드 x 프로브 시드별 R² 점(6개씩)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 입력은 repo/results/ CSV)
한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo" / "results"
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


def plot_closed_loop():
    rows = list(csv.DictReader(open(REPO / "closed_loop_success_by_suite.csv", encoding="utf-8")))
    pos = {r["row"]: (int(r["successes"]), int(r["episodes"])) for r in rows
           if r["eval"] == "libero_pro" and r["suite"] == "libero_10" and r["perturbation"] == "position"}
    order = [("baseline", "π0.5 기준선"), ("clip_diff_id", "DIFF + ID 표적 (시드 0)"),
             ("clip_diff_id_seed1", "DIFF + ID 표적 (시드 1)"), ("token_vjepa2_id_w0.01", "V-JEPA 2 + ID (w 0.01)"),
             ("clip_batch_shuffled", "배치 섞기 대조"), ("clip_random_init", "무작위 초기화 대조"),
             ("token_webdino_w0.01", "Web-DINO (w 0.01)")]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    x = np.arange(len(order))
    vals = [100 * pos[k][0] / pos[k][1] for k, _ in order]
    lo = [wilson(*pos[k])[0] for k, _ in order]
    hi = [wilson(*pos[k])[1] for k, _ in order]
    yerr = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
    colors = [GREY] + [ACC if "DIFF" in n or "V-JEPA" in n else "#b9b6ad" for _, n in order[1:]]
    ax.bar(x, vals, color=colors, width=0.62)
    ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, k in enumerate(k for k, _ in order):
        ax.text(i, 3, f"{pos[k][0]}/{pos[k][1]}", ha="center", va="bottom", color="white", fontsize=7.5)
    ax.axhline(vals[0], color=RED, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in order], rotation=22, ha="right", fontsize=7.5)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 45)
    ax.set_title("LIBERO-10 물체 위치 이동 (과제 10개 x 10회, Wilson 95%)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_closed_loop_ci.png", dpi=200)
    plt.close(fig)


def plot_matched():
    rows = list(csv.DictReader(open(REPO / "matched_controls_per_cell.csv", encoding="utf-8")))
    vals = defaultdict(list)
    for r in rows:
        vals[(r["backbone"], r["condition"])].append(float(r["r2"]))
    backbones = [("vjepa2", "V-JEPA 2 ViT-L"), ("videomae", "VideoMAE V1 ViT-L"), ("webdino", "Web-DINO ViT-L")]
    conds = [("frozen", "고정", "#d9d6cd"), ("true_id", "참 ID", ACC), ("shuffled_id", "섞은 ID", RED),
             ("forward_dynamics", "순방향 동역학", "#e0b36b")]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    w = 0.2
    for bi, (b, bname) in enumerate(backbones):
        for ci, (c, cname, col) in enumerate(conds):
            xs = bi + (ci - 1.5) * w
            v = np.array(vals[(b, c)])
            ax.bar(xs, v.mean(), width=w * 0.9, color=col, alpha=0.6, edgecolor="none")
            jitter = np.linspace(-0.05, 0.05, len(v))
            ax.scatter(np.full(len(v), xs) + jitter * 0.6, v, s=12, color=INK, zorder=3)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(range(len(backbones)))
    ax.set_xticklabels([b[1] for b in backbones], fontsize=8.5)
    ax.set_ylabel("행동 R² (200 에피소드)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.6) for _, _, col in conds]
    ax.legend(handles, [n for _, n, _ in conds], fontsize=7.5, frameon=False, ncol=4, loc="upper right")
    ax.set_title("매칭 대조: 학습 시드 2 x 프로브 시드 3 (점 6개, 고정은 3개)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_matched_ctrl.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_closed_loop()
    plot_matched()
    print("wrote figures to", FIG)
