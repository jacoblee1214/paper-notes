"""재계산 결과(recompute_results.json)로 성공률 그래프 두 장을 그린다.

실행: python3 analysis/plot_figures.py → report/figures/own_flip_ci.png, own_wipe_ci.png
그래프는 우리가 그린 것이며 논문 그림을 쓰지 않는다. 막대는 Wilson 95% 구간이다.
한글 폰트는 Linux(Noto CJK)와 macOS(Apple SD Gothic Neo) 후보를 차례로 찾는다.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for path in FONT_CANDIDATES:
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

LABELS = {"ACP": "ACP", "ACP_woFFT": "ACP w.o. FFT", "Compliant": "Compliant", "Stiff": "Stiff"}
COLORS = {"ACP": "#1f5fae", "ACP_woFFT": "#7aa6d8", "Compliant": "#a0a0a0", "Stiff": "#c96a4a"}


def bar_panel(ax, data, title, n_label):
    keys = list(data.keys())
    vals = [data[k]["pct"] for k in keys]
    lo = [v - data[k]["wilson95"][0] for k, v in zip(keys, vals)]
    hi = [data[k]["wilson95"][1] - v for k, v in zip(keys, vals)]
    xs = range(len(keys))
    ax.bar(xs, vals, color=[COLORS[k] for k in keys], width=0.62)
    ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="#222", capsize=4, lw=1)
    for x, k, v in zip(xs, keys, vals):
        ax.text(x, data[k]["wilson95"][1] + 2, f"{v:g}%\n{data[k]['success']}/{data[k]['n']}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[k] for k in keys], fontsize=9)
    ax.set_ylim(0, 125)
    ax.set_ylabel("성공률 (%)")
    ax.set_title(f"{title} ({n_label})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bar_panel(ax, RES["flip"], "물체 뒤집기", "시도 100회")
    fig.tight_layout()
    fig.savefig(FIG / "own_flip_ci.png", dpi=200)
    plt.close(fig)

    wipe = {k: v for k, v in RES["wipe"].items() if k != "Stiff"}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bar_panel(ax, wipe, "벽 닦기", "시도 16회, Table II")
    fig.tight_layout()
    fig.savefig(FIG / "own_wipe_ci.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG / "own_flip_ci.png", FIG / "own_wipe_ci.png")


if __name__ == "__main__":
    main()
