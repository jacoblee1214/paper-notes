"""Qwen-RobotWorld 논문 표 값으로 그린 그래프 두 장.

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
입력은 recompute.py의 표 값이다. 논문 그림은 쓰지 않는다.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recompute import EWM, PB, WMB  # noqa: E402

OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleSDGothicNeo"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False


def fig_margins():
    """논문이 주장한 비교 대상별로 Ours와 가장 가까운 모델의 점수 차이."""
    ewm = {m: sum(v) for m, (v, _) in EWM.items()}
    dream = {"Ours": 4.952, "LVP": 4.758, "Wow": 4.728, "GigaWorld": 4.216, "Cosmos-sft": 4.129, "Vidar": 3.341}
    pb = {m: (v[1] + v[2]) / 2 for m, v in PB.items()}
    wmb = {m: v[6] for m, v in WMB.items()}
    open_pb = [m for m in PB if m not in ("Veo3", "Wan2.6", "Sora2", "Kling26", "LTX-2")]
    open_wmb = [m for m in WMB if m not in ("Veo3", "Wan2.6", "Sora2", "Kling26", "LTX-2")]
    claims = [
        ("EWMBench (전체 대비 1위)", ewm, list(ewm)),
        ("DreamGen Total (전체 대비 1위)", dream, list(dream)),
        ("PBench Overall (오픈소스 대비)", pb, open_pb),
        ("WorldModelBench Total (오픈소스 대비)", wmb, open_wmb),
    ]
    rows = []
    for label, scores, pool in claims:
        ours = scores["Ours"]
        other = max((scores[m], m) for m in pool if m != "Ours")
        rows.append((label, ours - other[0], other[1]))

    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    labels = [r[0] for r in rows]
    gaps = [r[1] for r in rows]
    colors = ["#1f5fae" if g > 0.05 else "#b45309" for g in gaps]
    ax.barh(labels, gaps, color=colors)
    for i, (r, g) in enumerate(zip(rows, gaps)):
        ax.text(max(g, 0) + 0.012, i, f"{g:+.3f}  (비교: {r[2]})", va="center", ha="left", fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Ours 점수 - 비교 대상 중 최고 (보고값 기준, 샘플 노이즈 미고려)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("논문이 주장한 비교 대상별 차이", fontsize=10)
    ax.invert_yaxis()
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "own_margin_bars.png")
    plt.close(fig)


def fig_pbench():
    """PBench Quality와 Domain 평면 위의 모델 위치와 Overall 등고선."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    xs = np.linspace(0.70, 0.80, 50)
    for o in (0.78, 0.80, 0.82):
        ax.plot(xs, 2 * o - xs, color="#bbb", lw=0.8, ls="--")
        ax.text(0.8, 2 * o - 0.8, f"Overall {o:.2f}", fontsize=7, color="#666", va="center")
    for m, (_, q, d, o) in PB.items():
        hi = m == "Ours"
        opn = m not in ("Veo3", "Wan2.6", "Sora2", "Kling26", "LTX-2")
        ax.scatter(q, d, s=60 if hi else 30,
                   color="#1f5fae" if hi else ("#444" if opn else "#999"),
                   marker="o" if opn else "s", zorder=3)
        ax.annotate(m, (q, d), xytext=(4, 3), textcoords="offset points", fontsize=7,
                    color="#1f5fae" if hi else "#333")
    ax.set_xlabel("Quality Score (VBench 8개 평균)")
    ax.set_ylabel("Domain Score (QA, Qwen2.5-VL 판정)")
    ax.set_title("PBench: 원형은 오픈소스, 사각형은 상용·일반 모델", fontsize=9)
    ax.set_xlim(0.71, 0.80)
    ax.set_ylim(0.78, 0.90)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "own_pbench_plane.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_margins()
    fig_pbench()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
