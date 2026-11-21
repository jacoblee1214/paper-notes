"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 입력은 논문에 보고된 값과 recompute_results.json이다.

실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
BLUE, GREY, RED = "#1f5fae", "#8a8a85", "#c44e3a"


def user_study():
    fig, ax = plt.subplots(figsize=(6.2, 2.9), dpi=160)
    names, rows = [], []
    for name, d in RES["user_study"].items():
        names.append(f"vs {name}")
        rows.append(d)
    ys = list(range(len(rows)))[::-1]
    for y, d in zip(ys, rows):
        p = d["reported_pct"]
        lo, hi = d["wilson95"]
        ax.plot([lo, hi], [y, y], color=BLUE, lw=2)
        ax.plot(p, y, "o", color=BLUE, ms=6)
        ax.text(hi + 0.8, y, f"{p:.1f}%  (n = 1003, p = {d['exact_two_sided_p_vs_50']:.1e})", va="center", fontsize=7.5)
    ax.axvline(50, color=GREY, ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(40, 100)
    ax.set_xlabel("Self Forcing (chunk-wise, DMD) 선호율 (%)", fontsize=8)
    ax.set_title("사용자 선호도: 보고값과 Wilson 95% 구간 (논문 Fig. 4 수치 재계산)", fontsize=9)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "own_user_pref.png")
    plt.close(fig)


def speed_quality():
    # 논문 Table 1: (처리량 FPS, VBench Total)
    pts = {
        "LTX-Video": (8.98, 80.00), "Wan2.1-1.3B": (0.78, 84.26), "SkyReels-V2": (0.49, 82.67),
        "MAGI-1": (0.19, 79.18), "CausVid": (17.0, 81.20), "SF chunk-wise": (17.0, 84.31),
        "NOVA": (0.88, 80.12), "Pyramid Flow": (6.7, 81.72), "SF frame-wise": (8.9, 84.26),
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.3), dpi=160)
    for name, (fps, tot) in pts.items():
        color = BLUE if name.startswith("SF") else GREY
        ax.plot(fps, tot, "o", color=color, ms=6)
        ax.annotate(name, (fps, tot), textcoords="offset points", xytext=(5, 3), fontsize=7, color=color)
    ax.axvline(16, color=RED, ls="--", lw=1)
    ax.text(16.3, 78.3, "재생률 16 FPS", color=RED, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("처리량 (FPS, 로그 축, H100 한 장)", fontsize=8)
    ax.set_ylabel("VBench Total", fontsize=8)
    ax.set_ylim(78, 85.5)
    ax.set_title("속도와 품질 (논문 Table 1 값). 실시간 기준은 재생률을 넘는지", fontsize=9)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "own_speed_quality.png")
    plt.close(fig)


def ablation():
    # 논문 Table 2, chunk-wise. (Total, Semantic)
    rows = [("DF (50×2)", 82.95, 80.09), ("TF (50×2)", 83.58, 80.52), ("DF+DMD", 82.76, 79.85),
            ("TF+DMD", 82.32, 80.67), ("SF DMD", 84.31, 81.28), ("SF SiD", 84.07, 78.24), ("SF GAN", 83.88, 79.16)]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), dpi=160, sharey=True)
    labels = [r[0] for r in rows]
    for ax, idx, title, lim in [(axes[0], 1, "VBench Total", (81.5, 85)), (axes[1], 2, "Semantic", (77, 82))]:
        vals = [r[idx] for r in rows]
        colors = [BLUE if l.startswith("SF") else GREY for l in labels]
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
        for i, v in enumerate(vals[::-1]):
            ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=6.5)
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.suptitle("절제 실험 (논문 Table 2, chunk-wise): Semantic에서 SF SiD와 GAN은 TF보다 낮음", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "own_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    user_study()
    speed_quality()
    ablation()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
