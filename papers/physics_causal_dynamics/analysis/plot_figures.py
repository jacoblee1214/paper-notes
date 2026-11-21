"""우리가 만든 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_add_bands.png : 표 1 ADD(논문 3 시드 평균)와 저장소 시드 0 값(표 6과 같음)을 나란히 비교
own_film_ci.png   : 표 5 조건별 ADD, 평균 ± 3 시드 t 95% 구간 (두 자유도 t, 검정 아님)
실행: python3 analysis/plot_figures.py   (numpy, scipy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#c4622d"

# 표 1 (논문, 3 시드 평균, ADD cm). 순서: 이름, 0.1-2 kg, 2-4 kg
T1 = [("PICR", 4.93, 6.76), ("RWM", 7.45, 11.15), ("CaDM", 6.06, 11.40),
      ("CaDeLaC", 8.75, 21.41), ("DeLaN", 6.96, 25.78), ("Neural-Fly", 8.05, 22.27)]
# 저장소 README와 프로젝트 페이지, 논문 표 6의 시드 0 값
SEED0 = {"PICR": (4.58, 6.92), "RWM": (7.09, 11.63), "CaDM": (6.79, 10.90),
         "CaDeLaC": (9.07, 24.75), "DeLaN": (6.82, 25.65), "Neural-Fly": (7.66, 24.78)}
# 표 5 (3 시드, 평균 ± SD)
T5 = [("FiLM (PICR)", (4.93, 0.54), (6.76, 1.28)), ("Input concat", (4.45, 0.37), (8.49, 0.69)),
      ("Shift only", (4.59, 0.54), (8.43, 0.80)), ("Low-rank hypernet", (4.26, 0.35), (11.75, 2.63))]


def add_bands():
    names = [r[0] for r in T1]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=False)
    for ax, idx, title in ((axes[0], 1, "0.1–2 kg (학습 범위)"), (axes[1], 2, "2–4 kg (학습 밖)")):
        vals = [r[idx] for r in T1]
        ax.bar(x, vals, color=[ACC if n == "PICR" else GREY for n in names], width=0.6,
               label="표 1 (3 시드 평균)")
        ax.scatter(x, [SEED0[n][idx - 1] for n in names], marker="D", color=WARM, zorder=3, s=26,
                   label="시드 0 (저장소, 표 6)")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(x, names, rotation=20, fontsize=8)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_ylabel("ADD (cm), 낮을수록 좋음" if idx == 1 else "")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("표 1 PICR 행과 저장소 시드 0 값의 차이 (기준선은 시드 SD 없음)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "own_add_bands.png", dpi=200)
    plt.close(fig)


def film_ci():
    names = [r[0] for r in T5]
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    t_crit = stats.t.ppf(0.975, df=2)
    for i, (name, seen, unseen) in enumerate(T5):
        for j, (band, (m, sd)) in enumerate((("0.1–2 kg", seen), ("2–4 kg", unseen))):
            h = t_crit * sd / math.sqrt(3)
            xpos = i + (-0.14 if j == 0 else 0.14)
            ax.errorbar(xpos, m, yerr=h, fmt="o", color=ACC if j == 1 else GREY, capsize=4, ms=5,
                        label=band if i == 0 else None)
            ax.text(xpos + 0.05, m, f"{m:.2f}", fontsize=7.5, va="center", color=INK)
    ax.set_xticks(range(len(names)), names, fontsize=8.5)
    ax.set_ylabel("ADD (cm)", fontsize=9)
    ax.set_title("표 5 조건별 ADD: 3 시드 평균과 t 95% 구간 (2–4 kg, FiLM 대비 Welch p 0.06–0.14)",
                 fontsize=9.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_film_ci.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    add_bands()
    film_ci()
    print("wrote", FIG / "own_add_bands.png", FIG / "own_film_ci.png")
