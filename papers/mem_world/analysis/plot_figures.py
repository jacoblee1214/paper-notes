"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_fig4_reread.png : 그림 4를 판독 값으로 다시 그린 산점도와 회귀선, 과제별 편차
own_table_gains.png : 표 1·2의 상대 개선(%)을 제3자 시점과 손목 시점으로 나란히
own_toy_retrieval.png: 토이 T1, 검색 규칙별 커버리지와 95% 구간
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#d9772b", "#9a9890"


def plot_fig4():
    real = {"pi0": [.70, .65, .40, .40, .20], "pi05": [.90, .80, .60, .70, .45]}
    mem = {"pi0": [.55, .55, .25, .25, .10], "pi05": [.70, .70, .40, .45, .25]}
    ctrl = {"pi0": [.60, .55, .10, .10, .05], "pi05": [.70, .65, .15, .20, .15]}
    # 가려진 두 점(π0 wipe, π0.5 pick&place)은 recompute.py(V5)가 회귀식에서 역산한 값
    X = np.array(real["pi0"] + real["pi05"])
    YM = np.array(mem["pi0"] + mem["pi05"])
    YC = np.array(ctrl["pi0"] + ctrl["pi05"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    xs = np.linspace(0, 1, 50)
    for y, c, lab, mk in [(YM, ACC, "Mem-World", "o"), (YC, ORG, "Ctrl-World", "s")]:
        lr = stats.linregress(X, y)
        ax.scatter(X, y, color=c, s=34, alpha=.8, marker=mk, label=f"{lab}  r = {lr.rvalue:.3f}")
        ax.plot(xs, lr.slope * xs + lr.intercept, color=c, lw=1.4)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("실로봇 성공률 (그림 판독)")
    ax.set_ylabel("세계모델 안 성공률 (그림 판독)")
    ax.set_title("그림 4 다시 그리기 (10점 = 5과제 × 2정책)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax2 = axes[1]
    tasks = ["pick&place", "wipe", "1→2", "2→1", "2→2"]
    gm = [(real["pi0"][i] - mem["pi0"][i] + real["pi05"][i] - mem["pi05"][i]) / 2 for i in range(5)]
    gc = [(real["pi0"][i] - ctrl["pi0"][i] + real["pi05"][i] - ctrl["pi05"][i]) / 2 for i in range(5)]
    w = .38
    xx = np.arange(5)
    ax2.bar(xx - w / 2, gm, w, color=ACC, label="Mem-World")
    ax2.bar(xx + w / 2, gc, w, color=ORG, label="Ctrl-World")
    ax2.set_xticks(xx)
    ax2.set_xticklabels(tasks, fontsize=8)
    ax2.set_ylabel("과소 추정 폭 (실로봇 성공률 빼기 세계모델 성공률, 두 정책 평균)")
    ax2.set_title("과제별 편차: 장기 과제에서 차이가 벌어짐", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig4_reread.png", dpi=170)
    plt.close(fig)


def plot_gains():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    g = rec["V1_table1_gains"]
    h = rec["V2_table2_gains"]
    mets = ["psnr", "ssim", "lpips", "obj"]
    names = ["PSNR", "SSIM", "LPIPS", "Obj.Con."]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (title, a, b) in zip(axes, [("표 1: Mem-World 대 Ctrl-World", g["third_Mem_vs_Ctrl"], g["wrist_Mem_vs_Ctrl"]),
                                        ("표 2: W-VMem 대 Stride", h["third_WVMem_vs_Stride"], h["wrist_WVMem_vs_Stride"])]):
        x = np.arange(4)
        w = .38
        ax.bar(x - w / 2, [a[m]["rel_pct"] for m in mets], w, color=ACC, label="제3자 시점")
        ax.bar(x + w / 2, [b[m]["rel_pct"] for m in mets], w, color=ORG, label="손목 시점")
        for i, m in enumerate(mets):
            ax.text(i - w / 2, a[m]["rel_pct"] + .4, f'{a[m]["rel_pct"]:.1f}', ha="center", fontsize=7.5)
            ax.text(i + w / 2, b[m]["rel_pct"] + .4, f'{b[m]["rel_pct"]:.1f}', ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("상대 개선 (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_table_gains.png", dpi=170)
    plt.close(fig)


def plot_toy():
    d = json.loads((ANA / "toy_retrieval_results.json").read_text(encoding="utf-8"))["K3"]
    keys = ["short", "stride", "pose", "wvmem", "wvmem_nodecay", "oracle_all_history"]
    labels = ["short-term\n(최근 K)", "stride\n(등간격)", "자세 근접", "W-VMem\n(식 2)", "W-VMem\n시간 감쇠 없음", "상한\n(전체 기록)"]
    m = [d["mean_coverage"][k] for k in keys]
    lo = [d["mean_coverage"][k] - d["ci95"][k][0] for k in keys]
    hi = [d["ci95"][k][1] - d["mean_coverage"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    cols = [GREY, GREY, GREY, ACC, ACC, INK]
    ax.bar(range(6), m, yerr=[lo, hi], color=cols, capsize=3)
    for i, v in enumerate(m):
        ax.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(.6, 1.05)
    ax.set_ylabel("목표 시점에서 보일 조각의 커버리지")
    ax.set_title("토이 T1 (정적 2차원 장면, K = 3, 400회): 식 2의 시간 감쇠가 정적 장면에서는 손해", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_retrieval.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_fig4()
    plot_gains()
    plot_toy()
    print("saved", [p.name for p in FIG.glob("own_*.png")])
