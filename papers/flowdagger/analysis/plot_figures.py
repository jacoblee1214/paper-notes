"""우리가 그린 그래프 (논문 그림은 쓰지 않는다: arXiv 라이선스가 비독점 배포(nonexclusive-distrib)라서).

  report/figures/own_table1_means.png  : 표 1 방법별 평균 성공률과 과제 단위 부트스트랩 95% 구간 (12개 과제)
  report/figures/own_table3_heldout.png: 표 3 보류 과제별 성공률 (기본 정책 대비 변화)
  report/figures/own_table4_wilson.png : 표 4 실로봇 성공률과 Wilson 95% 구간 (n = 30)
  report/figures/own_inversion_toy.png : 역변환 toy: 립시츠 상수와 단계 수에 따른 복원 오차

실행: python3 analysis/plot_figures.py   (recompute.py를 불러와 같은 입력값을 쓴다)
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recompute as rc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for path in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

COL = {"Base": "#9a948c", "SFT": "#a9c4de", "LoRA-DAg": "#4f86b8", "Res-DAg": "#b5d9b5", "DSRL": "#c9b3d9",
       "FlowDAgger": "#e07b39", "FD": "#e07b39"}


def table1_means():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=200)
    names = list(rc.T1)
    for i, m in enumerate(names):
        v = np.array(rc.T1[m])
        boot = [rng.choice(v, 12).mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.bar(i, v.mean(), color=COL[m], edgecolor="#333", linewidth=0.5, width=0.65)
        ax.errorbar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor="#222", capsize=3, lw=0.9)
        ax.scatter(np.full(12, i) + rng.uniform(-0.22, 0.22, 12), v, s=7, color="#333", alpha=0.55, zorder=3)
        ax.text(i, 1.04, f"{v.mean():.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("성공률 (12개 과제 평균)", fontsize=9)
    ax.set_title("표 1 방법별 평균과 과제 단위 부트스트랩 95% 구간 (점: 과제별 값)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "own_table1_means.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def table3_heldout():
    cols = rc.T3_COLS
    methods = ["Base", "FlowDAgger", "Res-DAg", "LoRA-DAg", "SFT"]
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
    w = 0.16
    for j, m in enumerate(methods):
        for i, c in enumerate(cols):
            ax.bar(i + (j - 2) * w, rc.T3[m][i], width=w * 0.92, color=COL[m], edgecolor="#333", linewidth=0.4,
                   label=m if i == 0 else None)
    ax.axvline(0.5, color="#888", lw=0.8, ls="--")
    ax.text(0.0, 1.09, "학습 과제", ha="center", fontsize=8, color="#555")
    ax.text(3.0, 1.09, "보류 과제 (기본 정책이 거의 포화)", ha="center", fontsize=8, color="#555")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("성공률 (n = 25 격자)", fontsize=9)
    ax.set_title("표 3 보류 과제 성공률: FlowDAgger의 Push는 0.96에서 0.56 (Fisher p = 0.002)", fontsize=10)
    ax.legend(fontsize=7, ncol=5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "own_table3_heldout.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def table4_wilson():
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
    w = 0.26
    for j, (m, key) in enumerate((("Base", "Base"), ("SFT", "SFT"), ("FlowDAgger", "FD"))):
        for i in range(8):
            k = round(rc.T4[key][i] * 30)
            if key == "FD" and i == 7:
                k = 22  # 0.72는 n = 30에서 불가능한 값. 가장 가까운 22/30 (0.733)로 둔다
            p = k / 30
            lo, hi = rc.wilson(k, 30)
            x = i + (j - 1) * w
            ax.bar(x, p, width=w * 0.92, color=COL[m], edgecolor="#333", linewidth=0.4, label=m if i == 0 else None)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="#222", capsize=1.8, lw=0.7)
    ax.set_xticks(range(8))
    ax.set_xticklabels([t.replace(" ", "\n") for t in rc.T4_TASKS], fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("성공률 (n = 30, Wilson 95%)", fontsize=9)
    ax.set_title("표 4 실로봇 성공률: 구간이 겹치는 과제가 많음 (Plug 0.72는 22/30으로 둠)", fontsize=10)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "own_table4_wilson.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def inversion_toy():
    res = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))["toy_inversion"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), dpi=200)
    ax = axes[0]
    Ls = sorted(res["lipschitz_sweep_K10"], key=float)
    xs = [float(L) * 0.1 for L in Ls]  # K = 10, dt = 0.1
    for key, lab, c in (("euler", "명시적 역 (Euler reverse)", "#9a948c"), ("traj_k5", "궤적 단위 FP (k=5)", "#4f86b8"),
                        ("perstep_code_M5", "단계별 FP (코드식, M=5)", "#e07b39"), ("perstep_paper_M5", "단계별 FP (식 3 시작점, M=5)", "#a0522d")):
        ax.plot(xs, [res["lipschitz_sweep_K10"][L][key] for L in Ls], marker="o", ms=3, lw=1.2, label=lab, color=c)
    ax.axvline(1.0, color="#888", ls="--", lw=0.8)
    ax.text(1.02, 1e-3, "Δt·L = 1", fontsize=7, color="#555")
    ax.set_yscale("log")
    ax.set_xlabel("Δt · L (K = 10)", fontsize=8)
    ax.set_ylabel("복원 오차 (MSE)", fontsize=8)
    ax.set_title("립시츠 상수가 커질 때", fontsize=9)
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    ax = axes[1]
    Ks = sorted(res["step_sweep_L6"], key=int)
    for key, lab, c in (("euler", "명시적 역", "#9a948c"), ("traj_k5", "궤적 단위 FP (k=5)", "#4f86b8"),
                        ("perstep_code_M5", "단계별 FP (M=5)", "#e07b39")):
        ax.plot([int(k) for k in Ks], [res["step_sweep_L6"][k][key] for k in Ks], marker="o", ms=3, lw=1.2, label=lab, color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("적분 단계 수 K (L = 6 고정)", fontsize=8)
    ax.set_title("단계 수가 커질 때 (Δt가 작아질 때)", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    from matplotlib.ticker import FuncFormatter
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=7)
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    fig.suptitle("우리가 만든 무작위 흐름장 toy (D = 32, 30회 평균): 역변환 방식별 복원 오차", fontsize=9)
    fig.tight_layout()
    out = FIG / "own_inversion_toy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (table1_means(), table3_heldout(), table4_wilson(), inversion_toy()):
        print("wrote", f)
