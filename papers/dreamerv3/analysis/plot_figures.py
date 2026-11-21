"""재계산 결과를 그림 두 장으로 그린다 (우리가 만든 그래프만 사용, 논문 그림은 쓰지 않음).

입력: .scratch/plot_inputs.json (recompute.py가 만든다), analysis/recompute_results.json
출력: report/figures/own_atari_minecraft.png, report/figures/own_dmc.png
실행: python3 analysis/recompute.py && python3 analysis/plot_figures.py
한글 폰트는 Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


matplotlib.rcParams["font.family"] = korean_font()
matplotlib.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#8a8880"


def fig_atari_minecraft(plot):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=170)
    # (a) Atari 57: 게임별 HNS, Dreamer 대 PPO (로그 축)
    ax = axes[0]
    d, p = np.array(plot["_atari_hns_for_plot"]["dreamer"]), np.array(plot["_atari_hns_for_plot"]["ppo"])
    ax.scatter(np.clip(p, 0.1, None), np.clip(d, 0.1, None), s=16, color=ACC, alpha=0.8, edgecolor="none")
    lim = [0.1, max(d.max(), p.max()) * 1.5]
    ax.plot(lim, lim, color=GREY, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("PPO 게임별 HNS (%, 로그)")
    ax.set_ylabel("DreamerV3 게임별 HNS (%, 로그)")
    ax.set_title(f"(a) Atari 57: Dreamer가 높은 게임 {int((d > p).sum())} / 57", fontsize=9.5, color=INK)
    ax.grid(alpha=0.25, lw=0.5)
    # (b) Minecraft 실행별 최종 반환
    ax = axes[1]
    mc = plot["_minecraft_plot"]
    order = ["Dreamer", "IMPALA", "Rainbow", "PPO"]
    rng = np.random.default_rng(1)
    for i, k in enumerate(order):
        v = np.array(mc[k])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=ACC if k == "Dreamer" else GREY, alpha=0.85, edgecolor="none")
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, lw=1.4)
    ax.axhline(11.5, color="#b5462f", lw=0.8, ls=":")
    ax.text(3.45, 11.7, "12 = 다이아몬드(가정)", fontsize=7, color="#b5462f", ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{k}\n(n={len(mc[k])})" for k in order], fontsize=8)
    ax.set_ylabel("최종 반환 (획득 아이템 수, 가정)")
    ax.set_ylim(-0.5, 12.8)
    ax.set_title("(b) Minecraft: 실행별 최종 반환과 평균 (공식 저장소)", fontsize=9.5, color=INK)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    for a in axes:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_atari_minecraft.png")
    plt.close(fig)


def fig_dmc(plot):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=170)
    for ax, key, other, title in [
        (axes[0], "_dmc_vision_plot", "drqv2", "(a) DMC 시각 20과제: Dreamer 대 DrQ-v2"),
    ]:
        v = plot[key]
        x = np.array(v["drqv2"])
        y = np.array(v["dreamer"])
        ax.scatter(x, y, s=18, color=ACC, edgecolor="none")
        ax.plot([0, 1000], [0, 1000], color=GREY, lw=0.8, ls="--")
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_xlabel("DrQ-v2 과제별 평균 (0–1000)")
        ax.set_ylabel("DreamerV3 과제별 평균 (0–1000)")
        ax.set_title(title + f"\n이긴 과제 {int((y > x).sum())} / {len(x)}", fontsize=9, color=INK)
        ax.grid(alpha=0.25, lw=0.5)
    # (b) 고유감각 18과제: 과제별 Dreamer 값 막대 (DMPO 비교 대신 Dreamer 분포만)
    ax = axes[1]
    pr = plot["_dmc_proprio_plot"]
    vals = np.array(pr["dreamer"])
    ax.bar(range(len(vals)), np.sort(vals)[::-1], color=ACC, alpha=0.85)
    ax.axhline(vals.mean(), color=INK, lw=1, ls="--")
    ax.text(len(vals) - 0.5, vals.mean() + 25, f"평균 {vals.mean():.0f}", ha="right", fontsize=8, color=INK)
    ax.set_xlabel("과제 (값 순 정렬)")
    ax.set_ylabel("DreamerV3 과제별 평균 (0–1000)")
    ax.set_title("(b) DMC 고유감각 18과제: Dreamer 과제별 값", fontsize=9, color=INK)
    ax.set_ylim(0, 1000)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    for a in axes:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_dmc.png")
    plt.close(fig)


if __name__ == "__main__":
    plot = json.loads((ROOT / ".scratch" / "plot_inputs.json").read_text(encoding="utf-8"))
    fig_atari_minecraft(plot)
    fig_dmc(plot)
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
