"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 여기서 만들지 않는다.

  own_traversal.png : 9회 주행 시간의 평균과 95% 구간 (±를 SD로 읽은 경우와 SE로 읽은 경우), 보고 속도와 L/T
  own_latency.png   : 표 1 질의 지연 대 이력 크기 N (S=1000, CPU와 GPU, 로그-로그)
  own_fig10_M.png   : 그림 10 주석 ρ에서 역산한 지지 질량 M 대 이력 길이 t (구간은 반올림 범위)

실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

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


def fig_traversal():
    n = 9
    tc = stats.t.ppf(0.975, n - 1)
    base, prop = (9.304, 0.316), (8.135, 0.007)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    xs = [0, 1]
    labels = ["prior only", "vision + interaction"]
    for i, (m, s) in enumerate([base, prop]):
        se = s / np.sqrt(n)
        a.errorbar(i - 0.12, m, yerr=tc * se, fmt="o", color="#1f5fae", capsize=4, ms=6,
                   label="±를 SD로 읽음 (95% 구간)" if i == 0 else None)
        a.errorbar(i + 0.12, m, yerr=tc * s, fmt="s", color="#c0504d", capsize=4, ms=5,
                   label="±를 SE로 읽음 (95% 구간)" if i == 0 else None)
    a.set_xticks(xs)
    a.set_xticklabels(labels, fontsize=9)
    a.set_ylabel("주행 시간 (s), 구간 200–300 m", fontsize=9)
    a.set_title("9회 주행 시간 (논문 §4.3 값)", fontsize=10)
    a.legend(fontsize=7.5, loc="upper right")
    a.set_ylim(7.5, 10.5)

    speeds = {"prior only": (10.51, 100 / 9.304), "vision + interaction": (11.78, 100 / 8.135)}
    names = list(speeds)
    x = np.arange(2)
    w = 0.36
    rep = [speeds[k][0] for k in names]
    lt = [speeds[k][1] for k in names]
    b.bar(x - w / 2, rep, w, color="#9bb7d9", label="논문 보고 평균 속도")
    b.bar(x + w / 2, lt, w, color="#e0a36b", label="L / T (100 m ÷ 주행 시간)")
    for xi, r, l in zip(x, rep, lt):
        b.text(xi - w / 2, r + 0.1, f"{r:.2f}", ha="center", fontsize=8)
        b.text(xi + w / 2, l + 0.1, f"{l:.2f}", ha="center", fontsize=8)
    b.set_xticks(x)
    b.set_xticklabels(names, fontsize=8)
    b.set_ylabel("m/s", fontsize=9)
    b.set_ylim(0, 14)
    b.set_title("평균 속도와 L/T의 차이", fontsize=10)
    b.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "own_traversal.png", dpi=200)
    plt.close(fig)


def fig_latency():
    N = np.array([1e4, 1e5, 1e6])
    cpu = np.array([0.81, 4.01, 49.84])
    gpu = np.array([0.74, 2.07, 15.22])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.loglog(N, cpu, "o-", color="#1f5fae", label="CPU (S=1000)")
    ax.loglog(N, gpu, "s-", color="#c0504d", label="GPU (S=1000)")
    xx = np.logspace(4, 6, 20)
    ax.loglog(xx, 50e-6 * xx, "--", color="#999999", lw=0.8,
              label="선형 참고 (50 ns/항목)")
    ax.set_xlabel("이력 크기 N (항목 수)", fontsize=9)
    ax.set_ylabel("질의 시간 (ms)", fontsize=9)
    ax.set_title("표 1 재작도: 질의 시간은 N에 거의 선형", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "own_latency.png", dpi=200)
    plt.close(fig)


def fig_fig10():
    cols = [c for c in RES["fig10_column_consistency"] if c["t"] > 0]
    t = np.array([c["t"] for c in cols], dtype=float)
    lo = np.array([c["M_common"][0] for c in cols])
    hi = np.array([c["M_common"][1] for c in cols])
    mid = (lo + hi) / 2
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.errorbar(t, mid, yerr=[mid - lo, hi - mid], fmt="o", color="#1f5fae", capsize=3, ms=5,
                label="그림 10 주석에서 역산한 M (행 5개 교집합)")
    tt = np.linspace(1, 330, 100)
    ax.plot(tt, 0.95 * tt, ":", color="#999999", lw=0.8, label="M = 0.95 t (기울기 참고)")
    ax.set_xlabel("이력 길이 t (그림 10의 열)", fontsize=9)
    ax.set_ylabel("지지 질량 M = Σ K", fontsize=9)
    ax.set_title("그림 10 주석의 내부 일관성 (행마다 같은 M)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig10_M.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_traversal()
    fig_latency()
    fig_fig10()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
