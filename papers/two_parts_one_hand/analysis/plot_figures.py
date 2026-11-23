"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_ci.png      : 표 1 조립 성공 20회와 Wilson 95% 구간, 프로젝트 페이지 영상 카운터 표시
own_ablation_zero.png: 그림 5의 막대 위 숫자를 0 기준 막대로 다시 그림(오차 막대는 그림 판독 값)
own_noise_retained.png: 그림 6의 판독 값을 '시각 정보 이득 유지율'로 환산 (그림 판독 값)
own_reward_shape.png : 식 (3) 한 스텝 보상과 보고된 누적 보상이 암시하는 평균 오차
실행: python3 analysis/plot_figures.py   (numpy, matplotlib, recompute_results.json 필요)
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
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#b4694a"
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_real():
    t1 = REC["V1_table1"]["per_task"]
    page = REC["V2_video_counters"]["project_page_counter"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    tasks = ["Bottle", "Syringe", "Marker"]
    w = 0.34
    for i, t in enumerate(tasks):
        for j, (key, col, lab) in enumerate([("closed_assembly", ACC, "닫힌 루프 (논문 표 1)"),
                                              ("open_assembly", GREY, "열린 루프 재생")]):
            k, n, pct, (lo, hi) = t1[t][key]
            x = i + (j - 0.5) * w
            ax.bar(x, pct, w * 0.92, color=col, label=lab if i == 0 else None)
            ax.errorbar(x, pct, yerr=[[pct - lo], [hi - pct]], fmt="none", ecolor=INK, capsize=3, lw=1)
            ax.text(x + 0.05, hi + 2.5, f"{k}/{n}", ha="center", color=INK, fontsize=8)
        ax.plot([i - 0.5 * w], [100 * page[t] / 20], marker="D", color=WARM, ms=6,
                label="프로젝트 페이지 영상 카운터" if i == 0 else None, zorder=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(tasks)
    ax.set_ylabel("조립 성공률 (%), n = 20")
    ax.set_ylim(0, 122)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    ax.set_title("표 1 조립 기준 성공률과 Wilson 95% 구간", fontsize=10)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    data = {
        "Bottle": [("Ours", 1429.1, 85), ("Proprio.", 805.3, 55), ("History MLP", 1088.4, 95),
                   ("w/o finger", 960.9, 90), ("w/o pose", 1223.5, 110)],
        "Syringe": [("Ours", 921.2, 48), ("Proprio.", 786.9, 88), ("History MLP", 638.5, 50),
                    ("w/o finger", 630.0, 48), ("w/o pose", 790.2, 40)],
    }
    cols = [ACC, WARM, "#6f8299", "#4e7050", "#8a6fa0"]
    for ax, (task, rows) in zip(axes, data.items()):
        x = np.arange(len(rows))
        ax.bar(x, [r[1] for r in rows], color=cols, width=0.62)
        ax.errorbar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], fmt="none", ecolor=INK, capsize=3, lw=1)
        for i, r in enumerate(rows):
            ax.text(i, r[1] + r[2] + 25, f"{r[1]:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, max(r[1] + r[2] for r in rows) * 1.12)
        ax.set_title(f"{task} (0 기준, 오차 막대는 그림 판독 값)", fontsize=9)
        ax.set_ylabel("누적 목표 도달 보상")
        clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_zero.png", dpi=170)
    plt.close(fig)


def plot_noise():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    prop = {"Bottle": 805.3, "Syringe": 786.9, "Marker": 338.0}
    full = {"Bottle": 1429.1, "Syringe": 921.2, "Marker": 1280.1}
    fig6 = {
        "gauss": {"Bottle": {1: 1430, 2: 1340, 3: 1252}, "Syringe": {1: 921, 2: 870, 3: 788}},
        "offset": {"Bottle": {0: 1430, 1: 1403, 2: 1315, 3: 1037}, "Syringe": {0: 920, 1: 868, 2: 763, 3: 543}},
    }
    marker3 = {"gauss": 963.7, "offset": 670.9}
    col = {"Bottle": "#4e7050", "Syringe": WARM, "Marker": ACC}
    for ax, kind, ttl in zip(axes, ("gauss", "offset"), ("가우시안 잡음 (표준편차, cm)", "일정 오프셋 (cm)")):
        for task, d in fig6[kind].items():
            xs = sorted(d)
            ys = [(d[m] - prop[task]) / (full[task] - prop[task]) for m in xs]
            ax.plot(xs, ys, "-o", color=col[task], label=f"{task} (그림 판독)")
        ax.plot([3], [(marker3[kind] - prop["Marker"]) / (full["Marker"] - prop["Marker"])], "s", color=col["Marker"],
                label="Marker (본문 수치, 3 cm)")
        ax.axhline(0, color=GREY, ls="--", lw=1)
        ax.axhline(1, color=GREY, ls=":", lw=1)
        ax.set_xlabel(ttl)
        ax.set_xticks([0, 1, 2, 3] if kind == "offset" else [1, 2, 3])
        clean(ax)
    axes[0].set_ylabel("시각 정보 이득 유지율\n(값 - 고유수용 전용) / (전체 - 고유수용 전용)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("그림 6, 12의 값을 고유수용 전용 정책 기준으로 환산 (0 = 고유수용 전용 수준, 1 = 학습 잡음 1 cm 수준)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_noise_retained.png", dpi=170)
    plt.close(fig)


def goal_reward(exy, ez, eth):
    return 5 * (math.exp(-60 * exy - 60 * ez - 5 * eth) + math.exp(-200 * exy - 120 * ez - 30 * eth))


def plot_reward():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    e = np.linspace(0, 20, 200)
    ax.plot(e, [goal_reward(v / 1000, v / 1000, 0) for v in e], color=ACC, lw=2, label="식 (3), xy = z 오차, 각도 오차 0")
    ax.plot(e, [goal_reward(v / 1000, v / 1000, 1 - math.cos(math.radians(10))) for v in e], color=GREY, lw=1.5,
            ls="--", label="같은 조건 + 축 정렬 오차 10도")
    ax.axvline(10, color=WARM, lw=1, ls=":")
    ax.text(10.2, 8.6, "성공 기준 1 cm (표 1)", color=WARM, fontsize=8)
    imp = REC["V5_reward"]["implied_error_mm"]
    offs = {"Syringe": (14, 46), "Bottle": (40, 22), "Marker": (22, -34)}
    for task, mk in zip(("Syringe", "Bottle", "Marker"), ("o", "s", "D")):
        d = imp[task]
        ax.plot(d["equal_xy_z_error_mm"], d["per_step_mean"], mk, color=INK, ms=6)
        ax.annotate(f"{task}: 스텝당 {d['per_step_mean']:.2f}, 오차 {d['equal_xy_z_error_mm']:.1f} mm",
                    (d["equal_xy_z_error_mm"], d["per_step_mean"]), textcoords="offset points", xytext=offs[task],
                    fontsize=7.5, arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.6})
    ax.set_xlabel("참조점-목표점 오차 (mm), xy 와 z 동일")
    ax.set_ylabel("한 스텝 목표 도달 보상 (최대 10)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("보고된 누적 보상 / 스텝 수 (그림 5, 부록 C)가 암시하는 평균 오차", fontsize=9.5)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_reward_shape.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_real()
    plot_ablation()
    plot_noise()
    plot_reward()
    print("wrote 4 figures to", FIG)
