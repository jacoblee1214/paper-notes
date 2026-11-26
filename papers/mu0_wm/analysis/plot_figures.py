"""우리가 만든 그래프를 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sim_ci.png      : 표 2 RoboCasa365 평균 성공률의 Wilson 95% 구간(방법당 400회)과 과제별 성공률
own_real_ci.png     : 그림 6 실로봇 성공률(그림 판독 값, 방법당 60회 가정)의 Wilson 95% 구간과 과제별 값
own_top5_top1.png   : 표 1 (2D, T=16) top-1 대 top-5 ADE와 top-5/top-1 비율
own_ablation_ci.png : 표 10, 12와 E.4 정책 절제 성공률의 Wilson 95% 구간 (400회 또는 800회 가정)
own_toy_bspline.png : 토이 B-spline 적합 예와 잡음별 재구성 오차

실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
입력: analysis/recompute_results.json, analysis/toy_bspline_results.json (먼저 recompute.py, toy_bspline.py를 실행할 것)
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
INK, ACC, GREY, LIGHT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"
COL = {"DP": "#b8b6ae", "pi0": "#9a9890", "pi05": "#6f6e68", "TraceGen": LIGHT, "mu0": ACC,
       "VLM+AE": "#b8b6ae", "TraceGen+AE": LIGHT, "mu0+AE": ACC}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def ci_bars(ax, labels, ks, n, colors, ylabel):
    pct = [100 * k / n for k in ks]
    ci = [wilson(k, n) for k in ks]
    x = np.arange(len(labels))
    ax.bar(x, pct, color=colors, width=0.62)
    ax.errorbar(x, pct, yerr=[[p - c[0] for p, c in zip(pct, ci)], [c[1] - p for p, c in zip(pct, ci)]],
                fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, (p, k) in enumerate(zip(pct, ks)):
        ax.text(i, 1.5, f"{k}/{n}", ha="center", va="bottom", color="white", fontsize=7.5)
        ax.text(i, ci[i][1] + 1.5, f"{p:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 108)
    ax.spines[["top", "right"]].set_visible(False)


def plot_sim():
    T2 = {"DP": [34, 28, 28, 28, 0, 48, 6, 10], "pi0": [44, 12, 34, 30, 2, 46, 18, 16],
          "pi05": [34, 26, 48, 66, 12, 76, 54, 20], "TraceGen": [38, 36, 42, 30, 0, 28, 0, 10],
          "mu0": [54, 18, 36, 40, 4, 56, 12, 22]}
    names = {"DP": "Diffusion\nPolicy", "pi0": "π0", "pi05": "π0.5", "TraceGen": "TraceGen\n+AE", "mu0": "μ0\n+AE"}
    tasks = ["CloseFridge", "OpenFridge", "CoffeeServeMug", "PnP FridgeShelf\nToDrawer", "TurnOn\nMicrowave",
             "SlideToaster\nOvenRack", "PnP Counter\nToCabinet", "TurnOn\nToasterOven"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), gridspec_kw={"width_ratios": [1, 1.6]})
    ks = [int(sum(T2[m]) / 2) for m in T2]
    ci_bars(axes[0], [names[m] for m in T2], ks, 400, [COL[m] for m in T2], "평균 성공률 (%)")
    axes[0].set_title("표 2 평균 (8과제 × 50회 = 400회, Wilson 95%)", fontsize=9.5)
    ax = axes[1]
    x = np.arange(8)
    w = 0.16
    for j, m in enumerate(T2):
        ax.bar(x + (j - 2) * w, T2[m], width=w, color=COL[m], label=names[m].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=6.8)
    ax.set_ylabel("성공률 (%)")
    ax.set_title("과제별 (50회)", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", frameon=False)
    ax.set_ylim(0, 90)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png", dpi=200)
    plt.close(fig)


def plot_real():
    REAL = {"VLM+AE": [75, 75, 70], "pi0": [85, 80, 50], "pi05": [95, 80, 65], "TraceGen+AE": [80, 80, 85],
            "mu0+AE": [90, 90, 95]}
    names = {"VLM+AE": "VLM\n+AE", "pi0": "π0", "pi05": "π0.5", "TraceGen+AE": "TraceGen\n+AE", "mu0+AE": "μ0\n+AE"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    ks = [int(sum(v) * 20 / 100) for v in REAL.values()]
    ci_bars(axes[0], [names[m] for m in REAL], ks, 60, [COL[m] for m in REAL], "성공률 (%)")
    axes[0].set_title("평균 (3과제 × 20회 = 60회 가정, Wilson 95%)", fontsize=9.5)
    ax = axes[1]
    tasks = ["Pick & Place", "Pour", "Unfold"]
    x = np.arange(3)
    w = 0.16
    for j, m in enumerate(REAL):
        ax.bar(x + (j - 2) * w, REAL[m], width=w, color=COL[m], label=names[m].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_title("과제별 (그림 6 판독 값, 과제당 20회)", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def plot_top5():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    ratio = rec["V7_top5_over_top1"]
    # 표 1 (2D) T=16 ADE: top1, top5
    t1_16 = {"Gemini 3.1 Pro": (0.274, 0.232), "Gemini 3 Flash": (0.271, 0.231), "GPT-5.5": (0.281, 0.249),
             "Track2Act": (0.311, 0.262), "Hamster": (0.276, 0.239), "μ0 (2D)": (0.279, 0.188)}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    names = list(t1_16)
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x - 0.19, [t1_16[n][0] for n in names], 0.36, color=GREY, label="top-1 ADE")
    ax.bar(x + 0.19, [t1_16[n][1] for n in names], 0.36, color=[ACC if n.startswith("μ0") else LIGHT for n in names], label="top-5 ADE")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7.5)
    ax.set_ylabel("ADE (T = 16, 정규화 단위)")
    ax.set_title("표 1 (2D): top-1은 비슷하고 top-5에서 벌어짐", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    order = ["Gemini 3.1 Pro", "Gemini 3 Flash", "GPT-5.5", "Track2Act", "Hamster", "mu0_2D", "3DFlowAction",
             "Dream2Flow", "TraceGen", "mu0_3D"]
    lab = ["Gem3.1Pro", "Gem3Flash", "GPT-5.5", "Track2Act", "Hamster", "μ0 2D", "3DFlowAct", "Dream2Flow", "TraceGen", "μ0 3D"]
    vals = [np.mean(ratio[m]["ADE"]) for m in order]
    ax.bar(np.arange(len(order)), vals, color=[ACC if m.startswith("mu0") else GREY for m in order])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("top-5 ADE / top-1 ADE (T = 8, 16, 32 평균)")
    ax.set_ylim(0, 1.0)
    ax.set_title("표본 5개에서 고르면 줄어드는 비율 (작을수록 표본이 다양)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_top5_top1.png", dpi=200)
    plt.close(fig)


def plot_ablation():
    items = [("μ0 + AE (400M)", 121, 400, ACC), ("완전 디코딩 트레이스", 118, 400, LIGHT),
             ("게이트 고정 σ=0.5", 104, 400, GREY), ("게이트 고정 σ=0.25", 100, 400, GREY),
             ("트레이스 잡음 σ=0.05", 110, 400, LIGHT), ("w/o Trace (400M)", 113, 400, GREY),
             ("μ0 + AE (200M, 800회 가정)", 205, 800, ACC), ("w/o Trace (200M, 800회 가정)", 85, 800, GREY)]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, (lab, k, n, c) in enumerate(items):
        p = 100 * k / n
        lo, hi = wilson(k, n)
        ax.barh(i, p, color=c, height=0.62)
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.text(hi + 0.8, i, f"{p:.2f}% ({k}/{n})", va="center", fontsize=8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("RoboCasa365 평균 성공률 (%)")
    ax.set_xlim(0, 50)
    ax.set_title("표 10, 12, E.4: 400M 머리 절제 대부분은 구간이 겹침 (시행 수는 가정)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_ci.png", dpi=200)
    plt.close(fig)


def plot_toy():
    # 기저 함수는 toy_bspline.py와 같은 식을 짧게 다시 썼다. 오차 곡선은 결과 JSON을 읽는다
    res = json.loads((ANA / "toy_bspline_results.json").read_text(encoding="utf-8"))["results"]
    H, D = 32, 10
    KN = {"bezier3": [0, 0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1, 1, 1, 1],
          "cubic_c2": [0, 0, 0, 0] + [k / 7 for k in range(1, 7)] + [1, 1, 1, 1]}

    def basis(knots, t):
        knots = np.asarray(knots, float)
        N = np.zeros((len(t), D, 4))
        for i in range(D):
            last = i == D - 1
            N[:, i, 0] = np.where((knots[i] <= t) & ((t < knots[i + 1]) | (last & (t <= knots[i + 1]))), 1.0, 0.0)
        for k in range(1, 4):
            for i in range(D):
                d1, d2 = knots[i + k] - knots[i], knots[i + k + 1] - knots[i + 1]
                a = (t - knots[i]) / d1 * N[:, i, k - 1] if d1 > 0 else 0.0
                b = (knots[i + k + 1] - t) / d2 * N[:, i + 1, k - 1] if (d2 > 0 and i + 1 < D) else 0.0
                N[:, i, k] = a + b
        return N[:, :, 3]

    t = np.arange(H + 1) / H
    G = np.zeros((D - 1, D))
    for i in range(D - 1):
        G[i, i], G[i, i + 1] = -1.0, 1.0
    rng = np.random.default_rng(3)
    s = np.where(t < 0.45, 0.5 * (10 * np.clip(t / .45, 0, 1) ** 3 - 15 * np.clip(t / .45, 0, 1) ** 4 + 6 * np.clip(t / .45, 0, 1) ** 5),
                 np.where(t < 0.55, 0.5, 0.5 + 0.5 * (10 * np.clip((t - .55) / .45, 0, 1) ** 3 - 15 * np.clip((t - .55) / .45, 0, 1) ** 4 + 6 * np.clip((t - .55) / .45, 0, 1) ** 5)))
    clean = s
    obs = clean + 0.03 * rng.normal(size=clean.shape)
    obs[0] = 0
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(t, clean, color=INK, lw=1.2, label="원래 궤적 (도달–정지–이동)")
    ax.scatter(t, obs, s=9, color=GREY, label="관측 (잡음 0.03)")
    for name, c, lab in [("bezier3", "#c0392b", "Bezier 3구간 (공개 코드 매듭)"), ("cubic_c2", ACC, "균일 3차 B-spline (C2)")]:
        B = basis(KN[name], t)
        A = np.vstack([B, 0.2 * G])
        rhs = np.concatenate([obs, np.zeros(D - 1)])
        P = np.linalg.lstsq(A, rhs, rcond=None)[0]
        ax.plot(t, B @ P, color=c, lw=1.6, label=lab + ", λ=0.2")
    for tj in (1 / 3, 2 / 3):
        ax.axvline(tj, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("정규화 시간 (구간 경계 1/3, 2/3 점선)")
    ax.set_ylabel("변위 (정규화 단위)")
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    ax.set_title("제어점 10개 적합 예 (한 축)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    noises = [0.0, 0.02, 0.05]
    for basis_name, c, ls in [("bezier3", "#c0392b", "-"), ("cubic_c2", ACC, "-")]:
        y = [np.mean([res[f"{k}|noise{n}|{basis_name}|lam0.2"]["rmse_to_clean"] for k in ["reach", "pick_place", "oscillate"]]) for n in noises]
        ax.plot(noises, y, marker="o", color=c, ls=ls, label=("Bezier 3구간" if basis_name == "bezier3" else "균일 3차 B-spline"))
    ax.plot(noises, noises, color=GREY, ls="--", lw=1, label="관측 잡음 크기")
    ax.set_xlabel("관측 잡음 σ (정규화 단위)")
    ax.set_ylabel("원래 궤적과의 RMSE")
    ax.set_title("재구성 오차 (3종 궤적 평균, λ=0.2)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_bspline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_sim()
    plot_real()
    plot_top5()
    plot_ablation()
    plot_toy()
    print("wrote figures to", FIG)
