"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 데이터는 recompute.py의 표 값을 그대로 씀.

1) own_speed_tau.png : 모델별 τ 대 속도 향상, 사이클 비용 c = τ/speedup 등고선
2) own_bf16_fp32.png : 표 B.7 BF16 대 FP32 SR과 짝지은 결과 일치율
3) own_relax_tradeoff.png : 표 4 완화 검증의 SR 하락 대 속도 향상 (Wilson 95% 구간, n = 2000 가정)
실행: python3 analysis/plot_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

from recompute import T1, B7, T4, METHODS, MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

KO_FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
for cand in (KO_FONT, "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, GRID = "#1b1b1a", "#6b6a64", "#dcdad3"
COLORS = {"EAGLE3": "#9a9a92", "P-EAGLE": "#b7b5ad", "DFlash": "#c9b98f", "DFlash2": "#d9c79e",
          "DSpark": "#5c8fc4", "VSpark": "#1f5fae"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)


def fig_speed_tau():
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=False)
    for ax, m in zip(axes, MODELS):
        taus = np.linspace(1.2, 4.6, 50)
        for c in (1.5, 2.0, 2.5):
            ax.plot(taus, taus / c, color=GRID, linewidth=0.9, linestyle="--", zorder=0)
            ax.annotate(f"c={c}", (taus[-1], taus[-1] / c), fontsize=6.5, color=MUTED, ha="right", va="bottom")
        for meth in METHODS:
            tau, spd, ta, sa = T1[m][meth]
            ax.scatter(ta, sa, s=46 if meth == "VSpark" else 26, color=COLORS[meth],
                       edgecolor=INK if meth == "VSpark" else "none", linewidth=0.6, zorder=3)
            ax.annotate(meth, (ta, sa), xytext=(4, 3), textcoords="offset points", fontsize=6.8, color=INK)
        ax.set_title(m, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("평균 수용 길이 τ (LIBERO 평균)", fontsize=7.5, color=INK)
        style(ax)
    axes[0].set_ylabel("속도 향상 (AR 대비)", fontsize=8, color=INK)
    fig.suptitle("속도 = τ ÷ 사이클 비용 c (c는 AR 토큰 단위, 점선은 c가 같은 선)", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "own_speed_tau.png", dpi=170)
    plt.close(fig)


def fig_bf16_fp32():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    labels, x = [], []
    pos = 0
    for m in ("pi0-FAST", "ActionCodec"):
        rows = B7[m]
        for suite_row in rows:
            name = suite_row[0]
            ar, arlo, arhi, vs, vslo, vshi = suite_row[1:7]
            ax.errorbar(pos - 0.13, ar, yerr=[[ar - arlo], [arhi - ar]], fmt="o", color=MUTED, ms=4, capsize=2, lw=1)
            ax.errorbar(pos + 0.13, vs, yerr=[[vs - vslo], [vshi - vs]], fmt="o", color=COLORS["VSpark"], ms=4, capsize=2, lw=1)
            ax.scatter(pos - 0.13 + 0.0, suite_row[8], marker="x", color=MUTED, s=18, zorder=4)
            labels.append(f"{m.split('-')[0] if m=='pi0-FAST' else 'AC'}\n{name}")
            x.append(pos)
            pos += 1
        pos += 0.6
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("성공률 (%) BF16, 5 시드 평균과 95% 구간", fontsize=7.5, color=INK)
    ax.set_title("AR(회색) 대 VSpark(파랑), x는 FP32 AR 단일 실행", fontsize=8.5, color=INK, loc="left")
    style(ax)
    ax = axes[1]
    agree = []
    names = []
    for m in ("pi0-FAST", "ActionCodec"):
        for suite_row in B7[m]:
            agree.append(suite_row[7])
            names.append(f"{'π0' if m=='pi0-FAST' else 'AC'} {suite_row[0]}")
    ypos = np.arange(len(agree))
    ax.barh(ypos, agree, color=COLORS["DSpark"], height=0.6)
    ax.axvline(100, color=INK, linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(70, 101)
    ax.set_xlabel("BF16 짝지은 에피소드 결과 일치율 (%)", fontsize=7.5, color=INK)
    ax.set_title("FP32에서는 모든 스위트 100%", fontsize=8.5, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "own_bf16_fp32.png", dpi=170)
    plt.close(fig)


def fig_relax():
    n = 2000
    z = 1.96

    def wilson(p):
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return 100 * (centre - half), 100 * (centre + half)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    spd = [r[3] for r in T4]
    sr = [r[4] for r in T4]
    lo_hi = [wilson(s / 100) for s in sr]
    err = np.array([[s - lo for s, (lo, _) in zip(sr, lo_hi)], [hi - s for s, (_, hi) in zip(sr, lo_hi)]])
    ax.errorbar(spd, sr, yerr=err, fmt="o-", color=COLORS["DSpark"], ms=4, lw=1, capsize=2, label="완화 검증 (DSpark 초안)")
    ax.axhline(92.22, color=COLORS["VSpark"], linewidth=1, linestyle="--")
    ax.annotate("VSpark 정확 검증 (표 B.7, BF16 5시드 평균 92.2%)", (1.62, 92.22), xytext=(0, 6), textcoords="offset points",
                fontsize=7, color=COLORS["VSpark"])
    ax.annotate("완화 검증 1.93× → SR 70.5%", (1.93, 70.5), xytext=(-150, 14), textcoords="offset points",
                fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("속도 향상 (AR 대비)", fontsize=8, color=INK)
    ax.set_ylabel("LIBERO 평균 성공률 (%)", fontsize=8, color=INK)
    ax.set_title("표 4: 완화 검증은 같은 속도에서 성공률을 떨어뜨림 (π0-FAST)", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_ylim(66, 96)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "own_relax_tradeoff.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_speed_tau()
    fig_bf16_fp32()
    fig_relax()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
