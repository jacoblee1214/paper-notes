"""AdaVLA 직접 그린 그래프 네 장 (CPU). 실행: python3 analysis/plot_figures.py

1) success_ci.png: Table II LIBERO 평균 성공률과 Wilson 95% 구간 (π0.5, X-VLA, 기준선 포함)
2) nfe_model.png: Eq.8 비용 모형의 구간당 NFE와, 지연 역산 NFE_avg 대 평균 가지치기 비율 p
3) toy_curvature.png: 2봉우리 toy(a=2)와 단일 가우시안에서 첫 반복 곡률 C와 AdaVLA 오차, Cth=0.15 표시
4) toy_pareto.png: 평균 NFE 대 평균 끝점 오차 (Euler, 고정 RK2, AdaVLA 두 변형)
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
F = json.loads((ROOT / "analysis" / "flow_toy_results.json").read_text())
SIZE = (6.5, 3.6)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plainlog(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter, NullLocator
    f = FuncFormatter(lambda v, _: f"{v:g}")
    if axis == "y":
        ax.yaxis.set_major_formatter(f)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    else:
        ax.xaxis.set_major_formatter(f)
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("saved", name)


# 1) 성공률 + Wilson
def fig_success():
    t2 = T["T1_table2"]["counts_and_avg"]
    labels, k, n = [], [], []
    rows = [("π0.5 기준", "pi05_base"), ("π0.5 +SparseVLM", "pi05_SparseVLM"), ("π0.5 +FastV", "pi05_FastV"),
            ("π0.5 +Ours", "pi05_ours"), ("X-VLA 기준", "xvla_base"), ("X-VLA +SparseVLM", "xvla_SparseVLM"),
            ("X-VLA +FastV", "xvla_FastV"), ("X-VLA +Ours", "xvla_ours")]
    fig, ax = plt.subplots(figsize=SIZE)
    for i, (lab, key) in enumerate(rows):
        K = int(t2[key]["pooled"].split("/")[0])
        lo, hi = t2[key]["pooled_wilson95_pct"]
        p = 100 * K / 2000
        col = ORANGE if "Ours" in lab else (MUTED if "기준" in lab else BLUE)
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=col, capsize=3, ms=5)
        ax.text(hi + 0.4, i, f"{p:.2f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(40, 101)
    ax.set_xlabel("LIBERO 평균 성공률 (%), 2000회 Wilson 95% 구간")
    ax.axhline(3.5, color=GRID, lw=1)
    ax.axhline(7.5, color=GRID, lw=1)
    style(ax)
    save(fig, "success_ci.png")


# 2) NFE 모형 + 역산
def fig_nfe():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE)
    r = np.linspace(1, 8, 400)  # C/Cth, 고곡률 분기(C >= Cth)만
    s = 0.1
    shat = np.minimum(0.5, np.maximum(s, 2 * s / r))
    nfe_unit = 2 / shat
    a1.plot(r, nfe_unit, color=BLUE, lw=1.6, label="AdaVLA 고곡률 분기 (τ=0)")
    a1.axhline(10, color=MUTED, ls="--", lw=1, label="Euler 10단계")
    a1.axvline(1, color=ORANGE, lw=0.8)
    a1.set_xlabel("C / Cth")
    a1.set_ylabel("구간 단위 NFE (Euler 대비)")
    a1.set_ylim(0, 24)
    a1.legend(fontsize=7, loc="upper right")
    style(a1)
    ps = np.array([float(k) for k in T["T5_implied_nfe"]["nfe_avg_full_by_p"].keys()])
    nf = np.array(list(T["T5_implied_nfe"]["nfe_avg_full_by_p"].values()))
    a2.plot(ps, nf, "o-", color=AQUA, lw=1.4, ms=4)
    a2.axhline(10, color=MUTED, ls="--", lw=1)
    a2.set_xlabel("평균 MLP 가지치기 비율 p (선형 가정)")
    a2.set_ylabel("역산 NFE_avg (chunk당)")
    a2.set_title("513 ms에서 역산", fontsize=8)
    a2.set_ylim(0, 12)
    style(a2)
    save(fig, "nfe_model.png")


# 3) 곡률 대 오차 (toy)
def fig_curv():
    # 점 산포는 다시 계산 (재현 가능)
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("flow_toy", ROOT / "analysis" / "flow_toy.py")
    ft = importlib.util.module_from_spec(spec)
    sys.modules["flow_toy"] = ft
    spec.loader.exec_module(ft)
    rng = np.random.default_rng(ft.SEED)
    x0 = rng.normal(size=(ft.N, 2))
    fig, axes = plt.subplots(1, 2, figsize=SIZE)
    for ax, (name, g) in zip(axes, [
        ("단일 가우시안 μ=2", ft.GMM([[2.0, 0.0]], [0.3], [1.0])),
        ("두 봉우리 a=2", ft.GMM([[-2.0, 0.0], [2.0, 0.0]], [0.2, 0.2], [0.5, 0.5])),
    ]):
        ref = ft.reference(g, x0)
        e1, _ = ft.euler(g, x0, 1)
        c1 = ft.adavla(g, x0, 0.15)[2]
        err = np.linalg.norm(e1 - ref, axis=1)
        ax.scatter(c1, np.maximum(err, 1e-4), s=3, color=BLUE, alpha=0.4, lw=0)
        ax.axvline(0.15, color=ORANGE, lw=1)
        ax.axhline(0.1, color=MUTED, ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xscale("log")
        plainlog(ax, "y")
        plainlog(ax, "x")
        ax.set_xlabel("첫 반복 곡률 C (τ=0)")
        ax.set_title(name, fontsize=8)
        style(ax)
    axes[0].set_ylabel("Euler 1단계 끝점 오차")
    save(fig, "toy_curvature.png")


# 4) NFE 대 오차 (Pareto)
def fig_pareto():
    fig, axes = plt.subplots(1, 2, figsize=SIZE)
    for ax, key in zip(axes, ["B_two_modes_a2", "C_ring4_r3"]):
        d = F[key]
        e = [(d[f"euler_{n}"]["nfe_mean"], d[f"euler_{n}"]["err_mean"]) for n in [1, 2, 3, 5, 10, 20]]
        ax.plot(*zip(*e), "o-", color=MUTED, ms=3, lw=1, label="Euler n")
        rk = [(d[f"rk2_fixed_{n}"]["nfe_mean"], d[f"rk2_fixed_{n}"]["err_mean"]) for n in [1, 2, 5]]
        ax.plot(*zip(*rk), "s-", color=BLUE, ms=3, lw=1, label="RK2 고정")
        aw = [(d[f"adavla_as_written_{c}"]["nfe_mean"], d[f"adavla_as_written_{c}"]["err_mean"]) for c in [0.01, 0.03, 0.1, 0.15, 0.3, 0.5, 1.0]]
        ax.plot(*zip(*aw), "^-", color=ORANGE, ms=3, lw=1, label="AdaVLA (논문 식)")
        cs = [(d[f"adavla_consistent_{c}"]["nfe_mean"], d[f"adavla_consistent_{c}"]["err_mean"]) for c in [0.01, 0.03, 0.1, 0.15, 0.3, 0.5, 1.0]]
        ax.plot(*zip(*cs), "d-", color=AQUA, ms=3, lw=1, label="AdaVLA 일관 중점 (절제)")
        ax.set_yscale("log")
        plainlog(ax, "y")
        ax.set_xlabel("평균 NFE (샘플당)")
        ax.set_title("두 봉우리 a=2" if key.startswith("B") else "4봉우리 링", fontsize=8)
        style(ax)
    axes[0].set_ylabel("평균 끝점 오차 (log)")
    axes[0].legend(fontsize=6.5)
    save(fig, "toy_pareto.png")


if __name__ == "__main__":
    fig_success()
    fig_nfe()
    fig_curv()
    fig_pareto()
