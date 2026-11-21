"""우리가 그린 그래프 세 장 (공개 가능). 입력은 recompute.py와 같은 집계 값이다.

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
한글 폰트는 Linux(Noto CJK)와 macOS(Apple SD Gothic Neo) 후보를 차례로 찾는다.
"""
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "NanumGothic"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

INK, GREY, BLUE, ORANGE = "#1b1b1a", "#8c8a84", "#1f5fae", "#c8702a"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


# 그래프 1: HiddenPhys 합계와 Wilson 95% 구간 (과제당 20회, 합 140)
def fig_hiddenphys():
    rows = [
        ("전문가 상태 기계", 136),
        ("GPT-6 Astra + ForceHarness", 135),
        ("Claude Fable 5.2 + ForceHarness", 135),
        ("Claude Fable 5.2 무힘", 96),
        ("Diffusion Policy (과제별)", 83),
        ("GPT-6 Astra 무힘", 87),
        ("ForceVLA", 70),
        ("π0.5 + 힘", 67),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    for i, (name, k) in enumerate(rows):
        p, lo, hi = wilson(k, 140)
        color = BLUE if "ForceHarness" in name else (GREY if "무힘" in name else ORANGE if "전문가" not in name else INK)
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=color, ecolor=color, capsize=3, ms=5)
        ax.text(hi + 1, i, f"{k}/140", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(40, 105)
    ax.set_xlabel("성공률 (%), Wilson 95% 구간")
    ax.set_title("HiddenPhys: 테스트 시드 9100–9119, 과제당 20회 (논문 표 5)", fontsize=10, loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_hiddenphys_ci.png")
    plt.close(fig)


# 그래프 2: ForceBench 과제별 Astra, 개발 시드 대 보류 시드 (과제당 10회)
def fig_forcebench_seeds():
    tasks = ["Weight", "CoM", "Faucet", "Faucet dir.", "Stove", "Stove press", "Drawer", "Nut"]
    dev_no = [8, 5, 10, 5, 10, 8, 0, 10]
    dev_fh = [10, 7, 10, 5, 10, 9, 10, 9]
    held_no = [6, 5, 10, 4, 10, 6, 0, 9]
    held_fh = [10, 6, 10, 2, 10, 10, 10, 8]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200, sharey=True)
    for ax, (no, fh, title) in zip(axes, [(dev_no, dev_fh, "개발 시드 300–309 (논문 표 4)"),
                                           (held_no, held_fh, "보류 시드 320–329 (저장소 문서)")]):
        y = range(len(tasks))
        ax.barh([i + 0.2 for i in y], [v * 10 for v in no], height=0.38, color=GREY, label="무힘")
        ax.barh([i - 0.2 for i in y], [v * 10 for v in fh], height=0.38, color=BLUE, label="ForceHarness")
        ax.set_xlim(0, 100)
        ax.set_xlabel("성공률 (%), 과제당 10회")
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        style(ax)
    axes[0].set_yticks(list(range(len(tasks))))
    axes[0].set_yticklabels(tasks)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "own_forcebench_seeds.png")
    plt.close(fig)


# 그래프 3: 분포 이동에서 네 과제(Weight, CoM, Wipe, Peg)의 상대 감소
def fig_shift():
    rows = [
        ("GPT-6 Astra + ForceHarness", 75, 69),
        ("Claude Fable 5.2 + ForceHarness", 75, 71),
        ("전문가 상태 기계", 77, 68),
        ("GPT-6 Astra 무힘", 44, 37),
        ("Claude Fable 5.2 무힘", 48, 39),
        ("Diffusion Policy", 52, 40),
        ("ForceVLA", 48, 36),
        ("DP 이미지만", 44, 42),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
    for i, (name, a, b) in enumerate(rows):
        drop = 100 * (a - b) / a
        color = BLUE if "ForceHarness" in name else (GREY if "무힘" in name else ORANGE if "DP" in name or "ForceVLA" in name else INK)
        ax.barh(i, drop, color=color, height=0.55)
        ax.text(drop + 0.4, i, f"{drop:.1f}%  ({a}→{b}, /80)", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 36)
    ax.set_xlabel("분포 이동에서의 상대 감소 (%)")
    ax.set_title("네 과제 (Weight, CoM, Wipe, Peg) 분포 이동 (논문 표 5, 과제당 20회)", fontsize=9.5, loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_shift_drop.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_hiddenphys()
    fig_forcebench_seeds()
    fig_shift()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
