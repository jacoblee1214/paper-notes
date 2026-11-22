"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_effect_noise.png: 사람 데모의 효과(최대 사람 - 로봇 전용, 진행도)와 시드 잡음 구간
own_sim_real.png    : 시뮬 대 실세계 산점도(Fig. 5 읽은 값)와 사람 데모 효과의 예측 여부
own_sim_values.png  : 시뮬 대 실세계 검증에 쓴 시뮬 값이 Table 3 평균과 얼마나 다른가
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 recompute.py와 digitize_figures.py 결과가 있어야 한다)
한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서로 찾는다.
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
INK, GREY = "#1b1b1a", "#9a9890"
METHODS = ["Phantom", "AMPLIFY", "Point Policy", "GHOST"]
TASKS = ["Pick-and-Place", "Stacking", "Insertion", "Sequential"]
TASK_KO = {"Pick-and-Place": "집기-놓기", "Stacking": "쌓기", "Insertion": "끼우기", "Sequential": "순차 조작"}
MCOL = {"Phantom": "#1f5fae", "AMPLIFY": "#4daa26", "Point Policy": "#e6841a", "GHOST": "#c0392b"}
TMARK = {"Pick-and-Place": "^", "Stacking": "D", "Insertion": "o", "Sequential": "s"}

DIG = json.loads((ANA / "digitized_points.json").read_text(encoding="utf-8"))
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def plot_effect():
    rows = REC["K12_effect_vs_noise"]["rows"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = 0
    ticks, labels = [], []
    for t in TASKS:
        for m in METHODS:
            r = rows[f"{m}|{t}"]
            se = r["seed_sd"] * math.sqrt(2 / 3)
            ax.errorbar(r["delta_prog"], y, xerr=2 * se, fmt="o", color=MCOL[m], ecolor=MCOL[m], capsize=2.5, ms=5)
            ticks.append(y)
            labels.append(f"{TASK_KO[t]} · {m}")
            y += 1
        y += 0.6
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("진행도 변화 (최대 사람 데모 - 로봇 전용, 퍼센트포인트)")
    ax.set_title("사람 데모의 효과와 ±2 표준오차(최대 데모 설정의 시드 SD로 근사)", fontsize=10)
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_effect_noise.png", dpi=170)
    plt.close(fig)


def plot_sim_real():
    f5, f15 = DIG["fig5_max_human"], DIG["fig15a_robot_only"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    ax = axes[0]
    for m in METHODS:
        for t in TASKS:
            x, yv = f5[m][t]
            ax.scatter(x, yv, marker=TMARK[t], color=MCOL[m], s=46, edgecolor="white", lw=0.6)
    ax.plot([0, 100], [0, 100], "--", color=GREY, lw=1)
    w = REC["K7_sim_vs_real"]["max"]
    ax.set_xlabel("시뮬 진행도 (Table 7 값)")
    ax.set_ylabel("실세계 진행도 (Fig. 5에서 읽음)")
    ax.set_title(f"풀링 r = {w['recomputed']['r']:.3f}, 과제 안 Kendall τ 평균 {w['within']['kendall_mean']:.2f}", fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    handles = [plt.Line2D([], [], marker="o", ls="", color=MCOL[m], label=m) for m in METHODS]
    handles += [plt.Line2D([], [], marker=TMARK[t], ls="", color=GREY, label=TASK_KO[t]) for t in TASKS]
    ax.legend(handles=handles, fontsize=7, loc="upper left", ncol=2)
    ax = axes[1]
    k9 = REC["K9_delta_prediction"]["fig5_vs_fig15a"]
    i = 0
    for m in METHODS:
        for t in TASKS:
            ax.scatter(k9["dx"][i], k9["dy"][i], marker=TMARK[t], color=MCOL[m], s=46, edgecolor="white", lw=0.6)
            i += 1
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("시뮬에서 사람 데모의 효과 (진행도 변화)")
    ax.set_ylabel("실세계에서 사람 데모의 효과")
    ax.set_title(f"효과 예측: r = {k9['r']:.2f} (p = {k9['p']:.2f}), 부호 일치 {k9['sign_agree']}/16", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_real.png", dpi=170)
    plt.close(fig)


def plot_sim_values():
    k6 = REC["K6_table7"]["rows"]
    k8 = REC["K8_robot_only_sim_vs_fig4"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    keys = [f"{m}|{t}" for t in TASKS for m in METHODS]
    zs = [k6[k]["z"] for k in keys]
    cols = [MCOL[k.split("|")[0]] for k in keys]
    ax.bar(range(16), zs, color=cols)
    ax.axhline(0, color=INK, lw=0.7)
    for yv in (-2, 2):
        ax.axhline(yv, color=GREY, lw=0.7, ls="--")
    ax.set_xticks([1.5 + 4 * i for i in range(4)])
    ax.set_xticklabels([TASK_KO[t] for t in TASKS], fontsize=8.5)
    ax.set_ylabel("(Table 7 - Table 3 평균) / 시드 SD")
    ax.set_title("검증에 쓴 시뮬 값(Table 7)이 3 시드 평균보다 높은 칸이 많다", fontsize=9)
    ax = axes[1]
    for m in METHODS:
        for t in TASKS:
            r = k8[f"{m}|{t}"]
            ax.scatter(r["fig4_robot_only"], r["fig15a_sim"], marker=TMARK[t], color=MCOL[m], s=42, edgecolor="white", lw=0.6)
    ax.plot([0, 100], [0, 100], "--", color=GREY, lw=1)
    ax.set_xlabel("로봇 전용 진행도, Fig. 4의 3 시드 평균")
    ax.set_ylabel("Fig. 15a의 시뮬 값")
    ax.set_title("로봇 전용 설정도 두 값이 다르다 (최대 20 퍼센트포인트)", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_values.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_effect()
    plot_sim_real()
    plot_sim_values()
    print("saved", [p.name for p in FIG.glob("own_*.png")])
