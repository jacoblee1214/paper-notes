"""우리가 그린 그래프 세 장 (논문 그림은 쓰지 않는다: arXiv 라이선스가 CC BY가 아님).

실행: python3 analysis/plot_figures.py   → report/figures/own_*.png (numpy, matplotlib, scipy)
값은 analysis/recompute.py가 쓰는 표 값과 같다. 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from recompute import wilson, COUNTS, TASKS, N_TRIAL  # noqa: E402  (같은 폴더)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

available = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    if name in available:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

COL = {"Being-H0": "#1f5fae", "GR00T N1.5": "#8a8a85", "InternVL3": "#d08a3a"}


def fig_real_robot():
    """표 7: 과제별 성공률과 Wilson 95% 구간 (20회)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=160)
    methods = ["GR00T N1.5", "InternVL3", "Being-H0"]
    width = 0.27
    x = np.arange(len(TASKS))
    for j, m in enumerate(methods):
        ks = COUNTS[m]
        ps = np.array([k / N_TRIAL for k in ks])
        lo = np.array([wilson(k, N_TRIAL)[0] for k in ks])
        hi = np.array([wilson(k, N_TRIAL)[1] for k in ks])
        xs = x + (j - 1) * width
        ax.bar(xs, ps, width, color=COL[m], label=m, alpha=0.9)
        ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("Pick-Place-Toy ", "PPT\n") for t in TASKS], fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("성공률", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", frameon=False)
    ax.set_title("실로봇 과제 성공률 (표 7, 과제당 20회, Wilson 95% 구간)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", bbox_inches="tight")
    plt.close(fig)


def fig_scale():
    """표 3: 시각 기반 생성의 모델 규모별 값. head와 tail, GR00T는 점선."""
    sizes = ["1B", "8B", "14B"]
    x = np.arange(3)
    series = {
        "MPJPE (cm)": ((9.71, 7.20, 6.87), (17.21, 9.02, 8.11), (9.82, 15.35)),
        "M2T R@3 (%)": ((12.1, 15.9, 17.2), (15.3, 18.7, 20.5), (13.1, 14.8)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=160)
    for ax, (title, (head, tail, g)) in zip(axes, series.items()):
        ax.plot(x, head, "o-", color=COL["Being-H0"], label="Being-H0 (head)")
        ax.plot(x, tail, "s-", color="#6aa3e0", label="Being-H0 (tail)")
        ax.axhline(g[0], color=COL["GR00T N1.5"], ls="--", lw=1, label="GR00T N1.5 head")
        ax.axhline(g[1], color="#b5b5b0", ls=":", lw=1, label="GR00T N1.5 tail")
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=8)
        ax.set_title(f"시각 기반 생성, {title}", fontsize=9)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=6.8, frameon=False, loc="upper right")
    fig.text(0.5, -0.01, "1B·8B·14B는 모델 크기. GR00T는 한 값(표 3). 낮을수록 좋은 MPJPE, 높을수록 좋은 M2T R@3.",
             ha="center", fontsize=7.2, color="#55544f")
    fig.tight_layout()
    fig.savefig(FIG / "own_scale_gen.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    """표 6: 기준(8B) 대비 변화율(%), tail 지표. 막대가 오른쪽(양수)이면 나빠진 것(지표 방향 반영)."""
    base = {"MPJPE": 9.02, "MWTE": 8.11, "PA-MPJPE": 1.32, "M2T R@3": 18.7, "FID": 13.4}
    rows = {
        "w/o Translate": {"MPJPE": 9.11, "MWTE": 8.12, "PA-MPJPE": 1.46, "M2T R@3": 11.6, "FID": 15.7},
        "w/o Context": {"MPJPE": 10.97, "MWTE": 9.03, "PA-MPJPE": 1.52, "M2T R@3": 15.2, "FID": 14.9},
        "w/o Balance": {"MPJPE": 12.13, "MWTE": 10.04, "PA-MPJPE": 1.57, "M2T R@3": 10.3, "FID": 16.3},
        "MANO-D114 + 4-Groups": {"MPJPE": 10.35, "MWTE": 9.11, "PA-MPJPE": 1.35, "M2T R@3": 14.7, "FID": 15.6},
        "MANO-D114 + Part-Level": {"MPJPE": 9.92, "MWTE": 8.83, "PA-MPJPE": 1.41, "M2T R@3": 16.1, "FID": 12.3},
    }
    names = list(rows)
    metrics = list(base)
    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=160)
    h = 0.16
    y = np.arange(len(names))
    palette = ["#1f5fae", "#4f86c6", "#8fb3de", "#c9d9ee", "#e8a25b"]
    for j, mname in enumerate(metrics):
        vals = []
        for n in names:
            v = rows[n][mname]
            b = base[mname]
            pct = 100 * (v - b) / b
            if mname.startswith("M2T"):
                pct = -pct  # 높을수록 좋으므로 나빠지면 양수
            vals.append(pct)
        ax.barh(y + (j - 2) * h, vals, h, color=palette[j], label=mname)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("기준(Being-H0-8B, tail) 대비 나빠진 정도 (%, 양수가 나쁨)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("절제 실험 (표 6, tail 분할, 단일 학습 실행)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_real_robot()
    fig_scale()
    fig_ablation()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
