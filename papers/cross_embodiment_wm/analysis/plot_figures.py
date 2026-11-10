"""직접 만든 그래프 (CPU). 실행: python3 analysis/plot_figures.py → report/figures/*.png

1) table_success_ci.png: Table I 성공 횟수와 Wilson 95% 구간 (논문 값에서 계산, 논문 그림 아님)
2) toy_leave_one_out.png: 장난감 실험 A: 빠진 손 MSE (집계 sum 대 mean, 손별 값)
3) toy_scaling.png: 장난감 실험 A: 학습 손 수 x 대 평균 MSE (부분집합 최대 8개)
4) toy_mpc.png: 장난감 실험 B: 모형 게인 오차 ε와 계획 길이 H에 따른 성공률 (Wilson 95%)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, GRID = "#0b0b0b", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
T = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
P = json.loads((ROOT / "analysis" / "particle_toy_results.json").read_text())
SIZE = (6.5, 3.4)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def table_success():
    hands = list(T["hands"])
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax)
    width = 0.36
    for j, (arm, col) in enumerate([("co_train", BLUE), ("human_only", MUTED)]):
        xs, ys, lo, hi = [], [], [], []
        for i, h in enumerate(hands):
            s = T["hands"][h]["success"]
            k, n = [int(v) for v in (s["co_train"] if arm == "co_train" else s["human_only"]).split("/")]
            w = s["wilson_co"] if arm == "co_train" else s["wilson_ho"]
            xs.append(i + (j - 0.5) * width)
            ys.append(k / n)
            lo.append(k / n - w[0])
            hi.append(w[1] - k / n)
        ax.bar(xs, ys, width=width, color=col, label={"co_train": "공동학습", "human_only": "사람 데이터만"}[arm])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=4, lw=1)
    ax.set_xticks(range(len(hands)))
    ax.set_xticklabels(hands)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("성공률 (20회)")
    ax.set_title("Table I 합계 성공률과 Wilson 95% 구간 (논문 표 값에서 계산)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "table_success_ci.png")


def toy_leave_one_out():
    rows = P["experiment_A_generalization"]["leave_one_out"]
    names = [r["held"].split(" ")[0] for r in rows]
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax)
    xs = np.arange(len(rows))
    ax.semilogy(xs - 0.12, [r["pooled_sum_mse"] for r in rows], "o-", color=ORANGE, label="합(sum) 집계, 5개 손 학습")
    ax.semilogy(xs + 0.12, [r["pooled_mean_mse"] for r in rows], "s-", color=BLUE, label="평균(mean) 집계, 5개 손 학습")
    ax.semilogy(xs, [r["own_sum_mse"] for r in rows], "^--", color=MUTED, label="빠진 손 자신의 절반 데이터(sum)")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_xlabel("빠진 손(장난감 손 H1–H6)")
    ax.set_ylabel("빠진 손에서의 MSE (log)")
    ax.set_title("장난감 실험 A: 빠진 손에서의 오차 (가정 명시, 논문 모델 아님)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "toy_leave_one_out.png")


def toy_scaling():
    cur = P["experiment_A_generalization"]["scaling_curve"]
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax)
    for kind, col, mk, lab in [("sum", ORANGE, "o", "합(sum)"), ("mean", BLUE, "s", "평균(mean)")]:
        xs = [int(x) for x in cur[kind]]
        ys = [cur[kind][str(x)]["avg"] for x in xs]
        sd = [np.std(cur[kind][str(x)]["per_target_mean_mse"]) / np.sqrt(6) for x in xs]
        ax.errorbar(xs, ys, yerr=sd, fmt=mk + "-", color=col, capsize=3, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("학습 손 수 x (빠진 손 제외)")
    ax.set_ylabel("빠진 손 MSE 평균 (log, ±표준오차)")
    ax.set_title("장난감 실험 A: 학습 손 수에 따른 곡선", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "toy_scaling.png")


def toy_mpc():
    res = P["experiment_B_mpc"]["results"]
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax)
    cols = {"1": MUTED, "2": BLUE, "4": ORANGE}
    for H in ["1", "2", "4"]:
        eps = sorted(res[H], key=float)
        ys, lo, hi = [], [], []
        from math import sqrt
        for e in eps:
            k, n = res[H][e]["success"], res[H][e]["n"]
            p = k / n
            z = 1.959964
            den = 1 + z * z / n
            c = (p + z * z / (2 * n)) / den
            h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            ys.append(p)
            lo.append(p - max(0, c - h))
            hi.append(min(1, c + h) - p)
        ax.errorbar([float(e) for e in eps], ys, yerr=[lo, hi], fmt="o-", color=cols[H], capsize=3, label=f"H = {H}")
    ax.set_xlabel("모형 게인 오차 ε (모형 = 참 × (1+ε))")
    ax.set_ylabel("성공률 (시험 %d회)" % P["experiment_B_mpc"]["trials"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("장난감 실험 B: 모형 오차와 계획 길이에 따른 성공률 (기준은 사후 정의)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "toy_mpc.png")


if __name__ == "__main__":
    table_success()
    toy_leave_one_out()
    toy_scaling()
    toy_mpc()
