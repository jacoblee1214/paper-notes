"""우리가 그린 그래프 두 장 (논문 그림은 쓰지 않는다).

  report/figures/own_auc_ci.png   : 표 1·2의 AUC, 시드 3의 95% t 구간(평균 ± 4.303 SE)
  report/figures/own_max_bias.png : 최댓값 보고 편향 몬테카를로 (toy_max_selection_results.json)

실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
한글 폰트는 Linux(Noto CJK)와 macOS(Apple SD Gothic Neo) 후보를 차례로 찾는다.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import recompute  # noqa: E402

FONT_CANDIDATES = ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"]
available = {f.name for f in font_manager.fontManager.ttflist}
font = next((f for f in FONT_CANDIDATES if f in available), None)
if font:
    plt.rcParams["font.family"] = font
plt.rcParams["axes.unicode_minus"] = False

T_CRIT = stats.t.ppf(0.975, df=2)
SUITES = ["LONG", "SPATIAL", "OBJECT", "GOAL"]
ALGOS = ["SeqL", "ER", "EWC", "PackNet"]
COLORS = {"SeqL": "#7a7a7a", "ER": "#1f5fae", "EWC": "#c0504d", "PackNet": "#2e8b57"}
ARCH = {"RNN": "ResNet-RNN", "T": "ResNet-T", "ViT": "ViT-T"}


def plot_auc(t8):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    width = 0.2
    for j, algo in enumerate(ALGOS):
        xs, ms, ci = [], [], []
        for i, s in enumerate(SUITES):
            m, se = t8[(algo, "T", s, "AUC")]
            xs.append(i + (j - 1.5) * width)
            ms.append(m)
            ci.append(T_CRIT * se)
        ax.errorbar(xs, ms, yerr=ci, fmt="o", color=COLORS[algo], capsize=3, label=algo, markersize=5)
    ax.set_xticks(range(len(SUITES)))
    ax.set_xticklabels([f"LIBERO-{s}" for s in SUITES], fontsize=9)
    ax.set_ylabel("AUC (성공률)")
    ax.set_title("알고리즘별 AUC (ResNet-T, 표 2)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    ax = axes[1]
    arch_colors = {"RNN": "#999999", "T": "#1f5fae", "ViT": "#ff8c00"}
    for j, arch in enumerate(["RNN", "T", "ViT"]):
        xs, ms, ci = [], [], []
        for i, s in enumerate(SUITES):
            m, se = t8[("ER", arch, s, "AUC")]
            xs.append(i + (j - 1) * 0.25)
            ms.append(m)
            ci.append(T_CRIT * se)
        ax.errorbar(xs, ms, yerr=ci, fmt="s", color=arch_colors[arch], capsize=3, label=f"ER · {ARCH[arch]}", markersize=5)
    ax.set_xticks(range(len(SUITES)))
    ax.set_xticklabels(SUITES, fontsize=9)
    ax.set_title("ER에서 아키텍처별 AUC (표 8)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    fig.text(0.01, 0.01, "오차 막대: 시드 3개의 95% t 구간 (평균 ± 4.303·SE). SE는 논문 표의 값. 출처: arXiv:2306.03310 v2 표 2, 표 8.",
             fontsize=7.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = FIG / "own_auc_ci.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_bias():
    data = json.loads((HERE / "toy_max_selection_results.json").read_text(encoding="utf-8"))
    ps = [0.3, 0.5, 0.7, 0.9]
    mx = [data[str(p)]["E[max 보고값] - p"] for p in ps]
    last = [data[str(p)]["E[마지막 평가] - p"] for p in ps]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ps, mx, "o-", color="#c0504d", label="최댓값 보고 (코드 규칙, c_{k,k})")
    ax.plot(ps, last, "s--", color="#1f5fae", label="마지막 평가 (편향 없음)")
    ax.axhline(0, color="#55544f", lw=0.8)
    ax.set_xlabel("진짜 성공률 p (정책 능력은 학습 내내 고정)")
    ax.set_ylabel("보고값 - p")
    ax.set_title("평가 11회 중 최댓값 보고의 편향 (롤아웃 20회, 몬테카를로)", fontsize=10)
    ax.set_ylim(-0.02, 0.22)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.text(0.01, 0.01, "몬테카를로 20만 회. 정책 변화가 없다고 가정한 토이이며 논문 수치의 보정이 아님.",
             fontsize=7.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = FIG / "own_max_bias.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    lines = recompute.PAPER_TXT.read_text(encoding="utf-8").split("\n")
    _, t8 = recompute.parse_tables(lines)
    print("wrote", plot_auc(t8))
    print("wrote", plot_bias())
    print("font:", font)


if __name__ == "__main__":
    main()
