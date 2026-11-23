"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_success_ci.png   : 표 2와 표 4의 성공률을 25회 격자로 되돌려 Wilson 95% 구간과 함께 그린 것 (시행 수 25 가정은 5.2절·8.6절 기재값)
own_admittance_cells.png : 표 4의 admittance 사용 전후 변화(접촉 드리프트, 병진 가속도)를 셀별로 그린 것 (11개 셀, 표 값)
own_toy_admittance.png   : 토이(스프링 접촉)에서 루프 이득에 따른 정상상태 오차: 기억 없는 비례항 대 누적형
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#d98b2b"
Z = 1.959964


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_success():
    # (라벨, 성공 횟수/25)
    rows = [
        ("표 2 T3  Tel.", 15), ("표 2 T3  Comb.", 20),
        ("표 4 FMT T3  w/o", 20), ("표 4 FMT T3  w/", 23),
        ("표 4 DiT T1  w/o", 13), ("표 4 DiT T1  w/", 13),
        ("표 4 π0.5 T2  w/o", 17), ("표 4 π0.5 T2  w/", 19),
        ("표 4 π0.5 T3  w/o", 21), ("표 4 π0.5 T3  w/", 19),
        ("표 4 Ψ0 T2  w/o", 22), ("표 4 Ψ0 T2  w/", 24),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, (lab, k) in enumerate(rows):
        lo, hi = wilson(k, 25)
        y = len(rows) - 1 - i
        col = ACC if lab.endswith(("Comb.", "w/")) else GREY
        ax.plot([lo, hi], [y, y], color=col, lw=2.2)
        ax.plot(100 * k / 25, y, "o", color=col, ms=6)
        ax.text(101.5, y, f"{k}/25", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlim(20, 108)
    ax.set_xlabel("성공률 (%), 점은 보고 값, 선은 Wilson 95% 구간 (시행 수 25)")
    ax.set_title("차이가 보이는 짝의 구간은 서로 크게 겹친다", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_success_ci.png", dpi=170)
    plt.close(fig)


def plot_cells():
    cells = [  # (라벨, drift w/o, w/, trans w/o, w/)
        ("FMT T1", 18.12, 15.67, 3.14, 2.98), ("FMT T2", 21.04, 19.44, 1.87, 1.95), ("FMT T3", 25.00, 22.08, 1.85, 1.29),
        ("DiT T1", 21.79, 18.00, 2.92, 2.54), ("DiT T2", 21.22, 19.61, 4.67, 2.41), ("DiT T3", 26.22, 22.20, 2.33, 2.05),
        ("π0.5 T1", 21.51, 11.78, 4.69, 4.82), ("π0.5 T2", 19.78, 15.97, 5.06, 5.03), ("π0.5 T3", 20.18, 19.40, 4.11, 3.53),
        ("Ψ0 T1", 15.80, 13.56, 4.20, 3.27), ("Ψ0 T2", 20.48, 18.69, 4.94, 5.44),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, idx, ttl in ((axes[0], (1, 2), "접촉 중심 드리프트 변화율 (%)"), (axes[1], (3, 4), "병진 가속도 변화율 (%)")):
        for i, c in enumerate(cells):
            chg = 100 * (c[idx[1]] / c[idx[0]] - 1)
            y = len(cells) - 1 - i
            col = ACC if chg < 0 else ORG
            ax.barh(y, chg, color=col, height=0.62)
            ax.text(chg + (-1.5 if chg < 0 else 1.5), y, f"{chg:+.1f}", va="center", ha="right" if chg < 0 else "left", fontsize=8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlim(-60, 20)
        ax.grid(axis="x", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(cells)))
    axes[0].set_yticklabels([c[0] for c in cells][::-1], fontsize=8.5)
    fig.suptitle("표 4: admittance 사용 후 변화 (파랑 = 감소, 주황 = 증가). 드리프트는 11개 셀 모두 감소, 가속도는 8개 셀", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_admittance_cells.png", dpi=170)
    plt.close(fig)


def plot_toy():
    res = json.loads((ROOT / "analysis" / "toy_admittance_results.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for mode, col, lab in (("A", ORG, "A. 기억 없는 비례항 (식 5를 그대로 읽음)"), ("B", ACC, "B. 누적형 (비례 이득으로 적분)")):
        xs = [r["g"] for r in res[mode] if r.get("stable")]
        ys = [r["err_pct"] for r in res[mode] if r.get("stable")]
        ax.plot(xs, ys, "o-", color=col, label=lab, lw=2)
    ax.axhline(3.5, color=GREY, ls="--", lw=1)
    ax.text(0.052, 5.2, "보고된 정상상태 오차 3.5%", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.3, 0.5, 1.0, 1.5])
    ax.set_xticklabels(["0.05", "0.1", "0.3", "0.5", "1", "1.5"])
    ax.minorticks_off()
    ax.set_xlabel("루프 이득 g = k·K_F (한 틱 지연, 안정한 점만 표시)")
    ax.set_ylabel("정상상태 오차 (%)")
    ax.set_ylim(-3, 80)
    ax.set_title("스프링 접촉 토이: 1 N에서 4 N으로 계단 지령", fontsize=11)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_admittance.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_success()
    plot_cells()
    plot_toy()
    print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
