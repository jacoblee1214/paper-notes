"""리포트 그래프. table_check.py, dust_toy.py를 먼저 실행할 것.

  report/figures/ladders.png        Table 1: 예산별 test loss vs population, 20M power law 극한의 한 칸 빼기 민감도
  report/figures/compute.png        update 한 번의 연산량(forward 환산): Dust vs backprop vs EGGROLL
  report/figures/cosine_16k.png     Table 8 cosine 법칙과 학습 population(16k)에서의 값
  report/figures/toy.png            numpy toy: cosine 법칙과 이론 c, 토큰 간 영향 β에 따른 c_max
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK, INK2, GRID, MUTED = "#1f1e1c", "#52514e", "#e6e5e0", "#a9a8a2"
# dataviz 기준 팔레트 슬롯 1–5 (고정 순서). Dust가 슬롯 1.
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

r = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
toy = json.loads((ROOT / "analysis" / "dust_toy_results.json").read_text())
POPS = [64, 256, 1024, 4096, 16384]
DUST = {"100k": [7.240, 7.165, 7.151, 7.164, 7.170], "1M": [6.290, 6.067, 5.934, 5.911, 5.916],
        "10M": [5.397, 5.231, 5.111, 5.072, 5.049], "20M": [5.265, 5.139, 4.960, 4.912, 4.802]}
EGG = {"100k": [7.961, 7.605, 7.386, 7.302, 7.263], "1M": [7.428, 7.200, 7.046, 6.892, 6.708],
       "10M": [7.056, 6.906, 6.692, 6.352, 6.033], "20M": [6.987, 6.811, 6.517, 6.200, 5.865]}
BP = {"100k": 7.216, "1M": 5.959, "10M": 4.989, "20M": 4.633}
TABLE8 = {"MLP out": (0.8779, 31623), "MLP in": (0.8577, 48417), "Attn out": (0.8729, 21627), "Attn Q": (1.0, 76736),
          "Attn K": (1.0, 616595), "Attn V": (0.8776, 173780), "Value gate": (0.6657, 47863), "Value emb": (1.0, 495450),
          "Token emb": (0.7318, 19498), "LM head": (1.0, 759)}


def ladders():
    fig, axs = plt.subplots(1, 5, figsize=(11, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
    for a, b in zip(axs[:4], ["100k", "1M", "10M", "20M"]):
        a.plot(POPS, EGG[b], "o-", color=MUTED, ms=3.5, lw=1.4, label="EGGROLL")
        a.plot(POPS, DUST[b], "o-", color=C1, ms=3.5, lw=1.8, label="Dust")
        a.axhline(BP[b], color=INK, ls=":", lw=1.2, label="backprop")
        a.set_xscale("log", base=2)
        a.set_xticks(POPS, ["64", "256", "1k", "4k", "16k"])
        steps = r["R1_regime"][b]["steps"]
        a.set_title(f"{b} tokens ({steps:g} step)", fontsize=8.5, color=INK2, loc="left")
        a.grid(True, color=GRID)
        a.set_xlabel("population")
    axs[0].set_ylabel("test loss")
    axs[0].legend(frameon=False, fontsize=7)
    b = axs[4]
    loo = r["V2_refits"]["20M"]["limits_leave_one_out(64,256,1k,4k,16k)"]
    labs = ["64 뺌", "256 뺌", "1k 뺌", "4k 뺌", "16k 뺌"]
    vals = [max(v, 3.5) for v in loo]
    cols = [C2 if v > BP["20M"] else C1 for v in loo]
    b.barh(range(5), [v - 3.5 for v in vals], left=3.5, color=cols, height=0.6, zorder=3)
    b.axvline(BP["20M"], color=INK, ls=":", lw=1.2)
    b.axvline(4.431, color=C1, ls="--", lw=1)
    for i, v in enumerate(loo):
        b.text(max(v, 3.5) + 0.03, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
    b.set_yticks(range(5), labs, fontsize=7.5)
    b.set_xlim(3.5, 5.0)
    b.set_title("20M 극한 재적합(한 칸 빼기)", fontsize=8.5, color=INK2, loc="left")
    b.text(3.53, 4.55, "파란 점선: 5칸 적합 4.43", fontsize=6.8, color=C1)
    b.text(3.53, -0.8, "검은 점선: backprop 4.63", fontsize=6.8, color=INK2)
    b.set_ylim(-1.1, 4.9)
    b.xaxis.grid(True, color=GRID, zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "ladders.png", dpi=200)
    plt.close(fig)


def compute():
    c = r["R2_compute"]["by_population"]
    K = [int(k) for k in c]
    dust = [c[k]["forward_equiv"] for k in c]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(K, dust, "o-", color=C1, lw=1.8, label="Dust (update 1회, forward 환산)")
    ax.plot(K, K, "o--", color=MUTED, lw=1.4, label="EGGROLL (population = forward 수)")
    ax.axhline(3, color=INK, ls=":", lw=1.2, label="backprop (forward 1 + backward ≈ forward 3)")
    for i, (k, v) in enumerate(zip(K, dust)):
        last = i == len(K) - 1
        ax.annotate(f"{v:,}회 = backprop ×{round(v / 3):,}", (k, v), xytext=(-6 if last else 6, -16),
                    textcoords="offset points", fontsize=7, color=INK2, ha="right" if last else "left")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(K, ["256", "1k", "4k", "16k"])
    ax.set_xlabel("population K (손실 보상 draw 수)")
    ax.set_ylabel("forward 환산 연산량 / update")
    ax.set_ylim(1, 1e5)
    ax.grid(True, color=GRID, which="major")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "compute.png", dpi=200)
    plt.close(fig)


def cosine():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    pts = []
    Ks = np.logspace(np.log10(64), np.log10(1.3e5), 200)
    groups = {"자기 토큰 보상(γ=0), 뒤 토큰에도 영향": (["MLP out", "MLP in", "Attn out", "Token emb"], C2),
              "영향이 자기 토큰뿐(head, Q)": (["LM head", "Attn Q"], C1),
              "미래 토큰 credit(γ≈0.98)": (["Attn K", "Attn V", "Value emb", "Value gate"], C3)}
    for g, (names, col) in groups.items():
        for j, n in enumerate(names):
            cm, c = TABLE8[n]
            ax.plot(Ks, cm / np.sqrt(1 + c / Ks), color=col, lw=1.4, label=g if j == 0 else None)
            y16 = cm / np.sqrt(1 + c / 16384)
            ax.plot([16384], [y16], "o", color=col, ms=4)
            pts.append((y16, n, cm, col))
    ax.axvline(16384, color=INK, ls=":", lw=1)
    pts.sort()
    ylab, prev = [], -1
    for y, n, cm, col in pts:                     # 라벨이 겹치지 않게 세로 간격을 둔다
        yl = max(y, prev + 0.062)
        ylab.append(yl)
        prev = yl
    shift = max(0, ylab[-1] - 1.0)
    for (y, n, cm, col), yl in zip(pts, ylab):
        yl -= shift
        ax.annotate(f"{n} {y:.2f} (c_max {cm:.2f})", (16384, y), xytext=(2.2e5, yl), textcoords="data",
                    fontsize=6.6, color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_xscale("log")
    ax.set_xlim(64, 3e6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("population K")
    ax.set_ylabel("cosine(Dust, backprop)")
    ax.set_title("부록 Table 8의 법칙 cos(K) = c_max/√(1 + c/K), 100M 토큰 체크포인트. 점선 = 학습에 쓴 16k",
                 fontsize=8, color=INK2, loc="left")
    ax.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "cosine_16k.png", dpi=200)
    plt.close(fig)


def toy_fig():
    z = np.load(ROOT / "analysis" / "dust_toy_curves.npz")
    K = z["K"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a = axs[0]
    Kf = np.logspace(np.log10(4), np.log10(4096), 100)
    for key, lab, col in [("T1_W1", "hidden층 W1 (d_out 64)", C1), ("T1_W2", "head W2 (d_out 32)", C3)]:
        t = toy["T1"]["W1 (hidden, d_out=64)" if key == "T1_W1" else "W2 (head, d_out=32)"]
        a.plot(K, z[key], "o", color=col, ms=4, label=f"{lab}: c 적합 {t['c']:.0f}, 이론 {t['c_theory']:.0f}")
        a.plot(Kf, 1 / np.sqrt(1 + t["c_theory"] / Kf), color=col, lw=1.2)
    a.plot(K, z["T4_es"], "s--", color=MUTED, ms=3.5, lw=1, label="weight-space ES (같은 forward 수)")
    a.set_xscale("log", base=2)
    a.set_xlabel("population K")
    a.set_ylabel("cosine")
    a.set_title("(a) 실선 = 이론식 1/√(1 + c/K)", fontsize=8, color=INK2, loc="left")
    a.legend(frameon=False, fontsize=6.6, loc="upper left")
    a.grid(True, color=GRID)
    b = axs[1]
    betas = [0, 1, 3]
    rows = [toy["T2"][f"beta={x}"] for x in betas]
    xs = np.arange(3)
    b.bar(xs - 0.2, [rw["gamma=0"]["c_max"] for rw in rows], 0.38, color=C2, label="γ = 0 (자기 토큰만)", zorder=3)
    b.bar(xs + 0.2, [rw["gamma=1"]["c_max"] for rw in rows], 0.38, color=C1, label="γ = 1 (미래 토큰까지)", zorder=3)
    b.plot(xs - 0.2, [rw["cos(local grad, full grad)"] for rw in rows], "k_", ms=18, mew=1.5, label="cos(국소 grad, 전체 grad)", zorder=4)
    for i, rw in enumerate(rows):
        b.text(i + 0.2, 1.02 if i else 1.02, f"c ×{rw['gamma=1']['c'] / rw['gamma=0']['c']:.0f}", ha="center", fontsize=7, color=INK2)
    b.set_xticks(xs, [f"β = {x}" for x in betas])
    b.set_ylim(0, 1.5)
    b.set_ylabel("적합한 c_max")
    b.set_title("(b) 토큰 사이 영향 β와 credit", fontsize=8, color=INK2, loc="left")
    b.legend(frameon=False, fontsize=6.6, loc="upper left", ncol=1)
    b.yaxis.grid(True, color=GRID, zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "toy.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ladders()
    compute()
    cosine()
    toy_fig()
    print("saved:", sorted(p.name for p in FIG.glob("*.png")))
