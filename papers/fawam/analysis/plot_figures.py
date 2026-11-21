"""우리가 만든 그래프 세 장 (report/figures/own_*.png). 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py   (matplotlib, numpy; recompute.py의 결과 사용)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recompute as rc  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

KOREAN_FONTS = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for f in KOREAN_FONTS:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
INK, GREY, BLUE, ORANGE = "#1b1b1a", "#9a988f", "#1f5fae", "#c8702a"

NAMES = {
    "pi05": "π0.5", "ge_act": "GE-Act", "forcevla": "ForceVLA", "ta_vla": "TA-VLA",
    "feam": "FEAM (잔차 없음)", "ge_act_res": "GE-Act + Res.", "ta_vla_res": "TA-VLA + Res.", "fawam": "FAWAM",
}


def fig_success():
    keys = list(NAMES)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    res = json.loads(rc.OUT.read_text(encoding="utf-8"))["table1"]
    for i, k in enumerate(keys):
        d = res[k]
        avg = d["avg_recomputed"]
        lo, hi = d["wilson95_pct"]
        color = BLUE if k == "fawam" else (ORANGE if k in ("ge_act_res", "ta_vla_res") else GREY)
        ax.errorbar(avg, i, xerr=[[avg - lo], [hi - avg]], fmt="o", color=color, ms=5, capsize=3, lw=1.2)
        ax.text(hi + 1.2, i, f"{avg:.2f}% ({d['successes']}/{d['trials']})", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([NAMES[k] for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(20, 100)
    ax.set_xlabel("평균 성공률 (%), 과제 4개 x 20회 풀링, Wilson 95% 구간", fontsize=8)
    ax.axvline(85, color=BLUE, lw=0.6, ls=":")
    ax.set_title("표 1 재계산: 평균 성공률과 95% 구간 (주황색 = 잔차 보정 포함)", fontsize=9, color=INK)
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_success_ci.png")
    plt.close(fig)


def fig_ablation():
    import json
    abl = json.loads(rc.OUT.read_text(encoding="utf-8"))["ablation"]["averages"]
    order = [("base_none", "기본 (힘 없음)"), ("obs", "+ 힘 관측"), ("obs_pre", "+ 힘-행동 공동 예측"),
             ("obs_pre_res", "+ 잔차 보정 (FAWAM)"), ("res_only", "잔차 보정만")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    vals = [abl[k]["reported"] for k, _ in order]
    labels = [lab for _, lab in order]
    colors = [GREY, BLUE, BLUE, BLUE, ORANGE]
    bars = ax.bar(range(len(order)), vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("평균 성공률 (%)", fontsize=8)
    ax.set_title("표 2 재계산: 힘 정보 설계별 평균 성공률 (각 80회)", fontsize=9, color=INK)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_bars.png")
    plt.close(fig)


def fig_perturb():
    import json
    p = json.loads(rc.OUT.read_text(encoding="utf-8"))["perturbation"]
    tasks = list(p["per_task_counts"].keys())
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    x = range(len(tasks))
    no = [p["per_task_counts"][t][0] for t in tasks]
    yes = [p["per_task_counts"][t][1] for t in tasks]
    ax.bar([i - 0.18 for i in x], no, width=0.36, color=GREY, label="보정 없음")
    ax.bar([i + 0.18 for i in x], yes, width=0.36, color=BLUE, label="보정 있음")
    for i in x:
        ax.text(i - 0.18, no[i] + 0.1, f"{no[i]}/5", ha="center", fontsize=7.5)
        ax.text(i + 0.18, yes[i] + 0.1, f"{yes[i]}/5", ha="center", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Erase board", "Peel cucumber", "Slice cucumber*", "Wipe vase"], fontsize=8)
    ax.set_ylim(0, 6)
    ax.set_ylabel("성공 횟수 (시도 5회)", fontsize=8)
    ax.set_title("표 4 재계산: 외란 실험 성공 횟수 (*표 원문 열 이름)", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_perturb_counts.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_success()
    fig_ablation()
    fig_perturb()
    print("wrote figures to", FIG)
