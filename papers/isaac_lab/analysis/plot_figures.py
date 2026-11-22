"""재계산 결과로 그래프 두 장을 만든다(우리가 그린 그래프, 논문 그림 아님).

  report/figures/own_scaling.png        : 그림 13 판독값의 다중 GPU 처리량과 이상적 선형 기준선
  report/figures/own_asset_scaling.png  : 그림 18a 판독값의 자산 수 대 처리량 (두 GPU, 이중 축)

실행: python3 analysis/plot_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recompute as rc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"


def pick_font():
    """한글 폰트를 Linux(Noto CJK)와 macOS(Apple SD Gothic Neo) 순서로 찾는다."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = pick_font()
    plt.rcParams["axes.unicode_minus"] = False

    # 그래프 1: 다중 GPU 처리량 (두 과제, 로그-로그)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=False)
    for ax, (title, table) in zip(axes, [("Dexsuite Lift (16384 env 기준)", rc.FIG13A),
                                          ("Open Drawer (16384 env 기준)", rc.FIG13B)]):
        x = rc.N_ENVS
        for key, color in [("RTX PRO 6000 1-GPU", "#c0504d"), ("RTX PRO 6000 2-GPU", "#e08e3c"),
                           ("RTX PRO 6000 4-GPU", "#7fa64e"), ("RTX PRO 6000 8-GPU", "#8064a2")]:
            ax.plot(x, table[key], marker="o", color=color, label=key.replace("RTX PRO 6000 ", ""))
        base = table["RTX PRO 6000 1-GPU"]
        for n, color in [(8, "#8064a2")]:
            ideal = [b * n for b in base]
            ax.plot(x, ideal, linestyle=":", color=color, alpha=0.7, label=f"{n}×1-GPU (이상적)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.set_xlabel("환경 수 (그림 판독)")
        ax.set_ylabel("FPS (×10^6, 판독값)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("논문 그림 13 판독: RTX PRO 6000 다중 GPU 처리량 (근사, ±3%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_scaling.png", dpi=170)
    plt.close(fig)

    # 그래프 2: 자산 수 대 처리량 (두 GPU, 이중 축)
    fig, ax1 = plt.subplots(figsize=(6.6, 4.2))
    ax2 = ax1.twinx()
    l1 = ax1.plot(rc.ASSETS, rc.FIG18A_BLACKWELL_K_FPS, marker="o", color="#2a6fb0", label="Blackwell 6000 (왼쪽, ×10^3 FPS)")
    l2 = ax2.plot(rc.ASSETS, rc.FIG18A_ADA_M_RAYS, marker="s", color="#d4622a", label="RTX 6000 Ada (오른쪽, ×10^6 rays/s)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(rc.ASSETS)
    ax1.set_xticklabels([str(v) for v in rc.ASSETS])
    ax1.set_xlabel("추적 자산 수")
    ax1.set_ylabel("처리량 (×10^3 FPS)", color="#2a6fb0")
    ax2.set_ylabel("처리량 (×10^6 rays/s)", color="#d4622a")
    ax1.set_ylim(25, 48)
    ax2.set_ylim(55, 80)
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="lower left")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.set_title("논문 그림 18a 판독: 자산 수 증가에 따른 감소 (두 GPU, 서로 다른 축)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_asset_scaling.png", dpi=170)
    plt.close(fig)
    print("wrote", FIG / "own_scaling.png", FIG / "own_asset_scaling.png")


if __name__ == "__main__":
    main()
