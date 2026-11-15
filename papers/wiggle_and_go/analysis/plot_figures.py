"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_real_ci.png     표 6 (lobbing·draping) 성공률과 Wilson 95% 구간, 24회 가정 (Φ-NN 대 Φ-CMA-ES)
  own_sim_ci.png      표 3 설정별 평균 추적 오차와 로프 4개 사이 t 95% 구간 (학습 분포 안 4 로프)
  own_toy_fisher.png  토이 W: 잡음 SD에 따른 k, c 상대 CRLB와 모형 오차(N=26 대 20)에 따른 편향

입력: analysis/recompute_results.json, analysis/toy_wiggle_fisher_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "analysis"))
import recompute as rc  # noqa: E402

RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_wiggle_fisher_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

BLUE, GREY, RED = "#1f5fae", "#8a8a85", "#b03a2e"


def real_ci():
    t6 = RC["table6"]
    keys = [("target_NN", "목표 도달 NN"), ("target_CMA", "목표 도달 CMA"),
            ("stay_NN", "체류 NN"), ("stay_CMA", "체류 CMA"),
            ("draping_NN", "드레이핑 성공 NN"), ("draping_CMA", "드레이핑 성공 CMA")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    for i, (k, label) in enumerate(keys):
        d = t6[k]
        color = BLUE if k.endswith("NN") else GREY
        lo, hi = d["wilson95"]
        ax.errorbar(d["pct"], i, xerr=[[d["pct"] - lo], [hi - d["pct"]]], fmt="o", color=color, capsize=3)
        ax.text(hi + 2, i, f"{d['k']}/{d['n']}", va="center", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k[1] for k in keys], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("성공률 (%) · Wilson 95% 구간 · 24회 가정", fontsize=9)
    ax.axvline(50, color="#cccccc", lw=0.8, ls="--")
    ax.set_title("lobbing·draping 성공률 (논문 표 6 재계산)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


def sim_ci():
    NN, CMA, RND = rc.NN, rc.CMA, rc.RND
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    x = np.arange(6)
    for arr, color, off, name in [(NN, BLUE, -0.2, "Φ-NN"), (CMA, GREY, 0.0, "Φ-CMA-ES"), (RND, RED, 0.2, "Φ-Random")]:
        m = arr.mean(axis=0)
        t = stats.t.ppf(0.975, arr.shape[0] - 1)
        ci = t * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ax.errorbar(x + off, m, yerr=ci, fmt="o", color=color, capsize=3, label=name, ms=4)
    ax.set_xticks(x)
    ax.set_xticklabels(rc.CONFIGS, fontsize=8)
    ax.set_ylabel("목표와의 거리 (cm)", fontsize=9)
    ax.set_title("설정별 평균 오차 (학습 분포 안 4 로프, 4 로프 t 95%)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png")
    plt.close(fig)


def toy_fisher():
    grid = TOY["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=160)
    ax = axes[0]
    for T, color in [(2.0, RED), (4.0, GREY), (6.7, BLUE)]:
        xs, yk, yc = [], [], []
        for key in ["1mm", "5mm", "10mm", "20mm", "50mm"]:
            rows = [g for g in grid[key] if g["T"] == T and g["a"] == 0.04 and g["f"] == 0.75]
            if rows:
                xs.append(int(key[:-2]))
                yk.append(rows[0]["relsd_k"] * 100)
                yc.append(rows[0]["relsd_c"] * 100)
        ax.plot(xs, yc, "-o", color=color, ms=3, label=f"c, T={T} s")
        ax.plot(xs, yk, "--", color=color, lw=1, label=f"k, T={T} s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("관측 잡음 SD (mm)", fontsize=9)
    ax.set_ylabel("상대 CRLB (%)", fontsize=9)
    ax.set_title("잡음과 기록 길이 (진폭 4 cm, 0.75 Hz)", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.grid(color="#eeeeee", lw=0.6)
    ax = axes[1]
    mm = TOY["mismatch_test"]
    labels = ["잡음 없음", "잡음 5 mm"]
    kb = [mm["noiseless"]["k_bias_pct"], mm["sig5mm"]["k_bias_pct"]]
    cb = [mm["noiseless"]["c_bias_pct"], mm["sig5mm"]["c_bias_pct"]]
    x = np.arange(2)
    ax.bar(x - 0.18, kb, 0.36, color=BLUE, label="k 편향")
    ax.bar(x + 0.18, cb, 0.36, color=GREY, label="c 편향")
    for xi, v in zip(x, kb):
        ax.text(xi - 0.18, v - 1.2, f"{v:.0f}%", ha="center", va="top", fontsize=7, color="white")
    for xi, v in zip(x, cb):
        ax.text(xi + 0.18, v - 1.2, f"{v:.0f}%", ha="center", va="top", fontsize=7, color="white")
    ax.set_ylim(-26, 4)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("추정 편향 (%)", fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title("모형 오차 (N=26 참값, N=20 추정)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_fisher.png")
    plt.close(fig)


def main():
    real_ci()
    sim_ci()
    toy_fisher()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))


if __name__ == "__main__":
    main()
