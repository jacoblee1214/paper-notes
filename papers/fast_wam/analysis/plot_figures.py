"""그래프 세 장을 report/figures/에 만든다 (matplotlib). 입력은 recompute_results.json과 toy_imagination_results.json.

  own_libero_ci.png       : LIBERO 스위트별 성공률과 Wilson 95% 구간 (500회 가정)
  own_robotwin_paired.png : RoboTwin 과제별 차이 (Fast-WAM 빼기 다른 방법), 평균과 95% 구간
  own_toy_imagination.png : 토이 E1 (상상 잡음 sigma별 오차)과 E2 (N별 오차)

실행: python3 analysis/plot_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_imagination_results.json").read_text(encoding="utf-8"))

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"fw": "#1f5fae", "joint": "#e0741b", "idm": "#7a4fb5", "wo": "#8a8a85"}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def libero():
    suites = ["Spatial", "Object", "Goal", "Long"]
    methods = [("Fast-WAM", "Fast-WAM", C["fw"]), ("Fast-WAM-Joint", "Joint", C["joint"]),
               ("Fast-WAM-IDM", "IDM", C["idm"]), ("w.o.video-co-train", "w.o. video co-train", C["wo"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    width = 0.2
    for j, (key, label, col) in enumerate(methods):
        counts = RES["libero"][key]["suite_counts_of_500"]
        xs = np.arange(len(suites)) + (j - 1.5) * width
        pts = [100 * c / 500 for c in counts]
        lo = [wilson(c, 500)[0] for c in counts]
        hi = [wilson(c, 500)[1] for c in counts]
        err = np.clip(np.array([np.array(pts) - np.array(lo), np.array(hi) - np.array(pts)]), 0, None)
        ax.errorbar(xs, pts, yerr=err, fmt="o", color=col, capsize=2, label=label, ms=4)
    ax.set_xticks(np.arange(len(suites)))
    ax.set_xticklabels(suites)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(80, 102)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=4, loc="lower center", frameon=False)
    ax.set_title("LIBERO 스위트별 성공률과 Wilson 95% 구간 (스위트당 500회 가정)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "own_libero_ci.png")
    plt.close(fig)


def robotwin():
    keys = [("Fast-WAM minus Fast-WAM-Joint", "Joint"), ("Fast-WAM minus Fast-WAM-IDM", "IDM"),
            ("Fast-WAM minus w.o.video-co-train", "w.o. video co-train"),
            ("Fast-WAM minus LingBot-VA", "LingBot-VA")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    for i, (k, label) in enumerate(keys):
        v = RES["robotwin_paired"][k]
        m = v["mean_diff"]
        lo, hi = v["ci95"]
        ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="o", color=C["fw"], capsize=3, ms=5)
        rb = RES["robotwin_robustness"].get(k)
        if rb:
            ax.plot([rb["median_diff"]], [i], marker="|", color="black", ms=12)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([l for _, l in keys])
    ax.invert_yaxis()
    ax.set_xlabel("과제 평균 차이 (Fast-WAM 빼기 상대, 포인트)")
    ax.set_title("RoboTwin 2.0 과제별 쌍체 차이 (50개 과제, 평균과 t 95% 구간, 검은 막대는 중앙값)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "own_robotwin_paired.png")
    plt.close(fig)


def toy():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)
    e1 = TOY["E1"]
    xs = ["direct"] + [f"imagine_sigma_{s}" for s in ["0.0", "0.5", "1.0", "2.0"]]
    labels = ["직접", "σ=0", "σ=0.5", "σ=1", "σ=2"]
    means = [e1[k]["mean"] for k in xs]
    sds = [e1[k]["sd"] for k in xs]
    axes[0].bar(labels, means, yerr=sds, color=[C["fw"]] + [C["idm"]] * 4, capsize=3)
    axes[0].set_ylabel("행동 테스트 MSE")
    axes[0].set_title("E1 상상 잡음별 오차 (5 시드)", fontsize=9)
    axes[0].set_yscale("log")
    e2 = TOY["E2"]
    Ns = [50, 200, 1000, 4000]
    for pref, col, lab in [("raw", C["wo"], "raw x"), ("future_subspace", C["fw"], "미래 예측 부분공간")]:
        m = [e2[f"{pref}_N{n}"]["mean"] for n in Ns]
        s = [e2[f"{pref}_N{n}"]["sd"] for n in Ns]
        axes[1].errorbar(Ns, m, yerr=s, fmt="o-", color=col, capsize=3, label=lab, ms=4)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("학습 표본 수 N")
    axes[1].set_ylabel("행동 테스트 MSE")
    axes[1].set_title("E2 표현별 오차 (5 시드)", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("토이 (합성 데이터, 메커니즘 방향만 확인)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "own_toy_imagination.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    libero()
    robotwin()
    toy()
    for p in sorted(OUT.glob("own_*.png")):
        print("wrote", p.name)
