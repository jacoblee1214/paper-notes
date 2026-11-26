"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_main_ci.png     : 두 데이터셋을 합친 과제별 성공률과 Wilson 95% 구간(시행 수 40)과 전체(200)
own_dataset_gap.png : 같은 모델의 두 데이터셋 성공률(20회씩)과 Fisher p
own_mtd_delta.png   : 멀티 촉각 드롭아웃 전후 성공률 차이와 95% 구간(Newcombe), 데이터셋 2의 같은 모델 값
own_attention.png   : 부록 그림 9를 판독한 모달리티별 평균 어텐션과 토큰 수 비율

실행: python3 analysis/plot_figures.py   (numpy, matplotlib. 먼저 recompute.py와 digitize_attention.py를 실행)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))
A = json.loads((HERE / "digitize_attention_results.json").read_text(encoding="utf-8"))

for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GREY, RED, GOLD, LBLUE = "#1f5fae", "#d9822b", "#8a8984", "#b3372f", "#c9a227", "#6fa8dc"
TASKS = ["Gear", "Board", "Lamp", "Key", "Lightbulb"]
TASKS_KO = ["Gear\nAssembly", "Board\nWiping", "Lamp\nInstall.", "Key\nin Lock", "Light-\nbulb"]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def wilson_pts(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """두 비율 차이(p2 - p1)의 Newcombe 하이브리드 점수 95% 구간 (%p)."""
    l1, u1 = wilson_pts(k1, n1)
    l2, u2 = wilson_pts(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    lo = d - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return 100 * lo, 100 * hi


def main_ci():
    T8 = {"MiTaS (V+G+E)": [34, 38, 33, 30, 25], "Sparsh-X (V+G+E)": [18, 32, 27, 14, 20],
          "ViT-CNN (V)": [15, 28, 13, 7, 6], "ViT (V)": [15, 24, 19, 9, 4]}
    cols = [BLUE, ORANGE, LBLUE, GREY]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(6)
    w = 0.2
    for i, ((name, ks), c) in enumerate(zip(T8.items(), cols)):
        vals = list(ks) + [sum(ks)]
        ns = [40] * 5 + [200]
        for j, (k, n) in enumerate(zip(vals, ns)):
            lo, hi = wilson(k, n)
            xx = x[j] + (i - 1.5) * w
            ax.bar(xx, 100 * k / n, w * 0.9, color=c, alpha=0.9 if j < 5 else 1.0, label=name if j == 0 else None)
            ax.plot([xx, xx], [lo, hi], color="black", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS_KO + ["전체\n(n=200)"], fontsize=9)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 105)
    ax.axvline(4.5, color="#cccccc", lw=0.8)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("두 데이터셋을 합친 성공률과 Wilson 95% 구간 (과제별 n = 40, 표 8)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_main_ci.png", dpi=160)
    plt.close(fig)


def dataset_gap():
    cells = R["V7"]["cells"]
    names = list(cells.keys())
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 3.9), sharey=True)
    cmap = {"ViT-CNN V": LBLUE, "ViT V": GREY, "Sparsh V+G+E": ORANGE, "MiTaS V+G+E": BLUE, "MiTaS V+G MTD": "#4c9a62"}
    for ti, (ax, t) in enumerate(zip(axes, TASKS)):
        for ni, n in enumerate(names):
            c = cells[n][t]
            y1, y2 = 5 * c["D1"], 5 * c["D2"]
            ax.plot([0, 1], [y1, y2], "-o", color=cmap[n], lw=2 if c["p"] < 0.05 else 1.0, alpha=0.95 if c["p"] < 0.05 else 0.6,
                    label=n.replace(" V+G+E", " (V+G+E)").replace(" V+G MTD", " (V+G, MTD)").replace(" V", " (V)") if ti == 0 else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["데이터셋 1", "데이터셋 2"], fontsize=8)
        ax.set_title(["Gear Assembly", "Board Wiping", "Lamp Install.", "Key in Lock", "Lightbulb"][ti], fontsize=9)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    axes[0].set_ylabel("성공률 (%, 20회)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("같은 모델·과제의 두 데이터셋 결과 (굵은 선: Fisher p < 0.05, 25칸 중 5칸)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "own_dataset_gap.png", dpi=160)
    plt.close(fig)


def mtd_delta():
    # 데이터셋 1의 V+G 대 MTD V+G (표 3). 성공 수 = % / 5
    base = {"MiTaS": [45, 70, 65, 55, 20], "Sparsh-X": [25, 50, 55, 20, 15]}
    mtd = {"MiTaS": [55, 85, 60, 20, 45], "Sparsh-X": [15, 85, 70, 5, 10]}
    d2_mtd = [55, 80, 55, 65, 45]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    x = np.arange(5)
    for fi, (fam, c) in enumerate((("MiTaS", BLUE), ("Sparsh-X", ORANGE))):
        for j in range(5):
            k1, k2 = base[fam][j] // 5, mtd[fam][j] // 5
            lo, hi = newcombe(k1, 20, k2, 20)
            xx = x[j] + (fi - 0.5) * 0.28
            ax.plot([xx, xx], [lo, hi], color=c, lw=2, alpha=0.5)
            ax.plot([xx], [5 * (k2 - k1)], "o", color=c, ms=7, label=f"{fam} (표 3, 데이터셋 1)" if j == 0 else None)
    # 데이터셋 2의 MiTaS MTD(V+G)를 데이터셋 1의 V+G와 비교한 차이 (참고)
    for j in range(5):
        diff = d2_mtd[j] - base["MiTaS"][j]
        ax.plot([x[j] + 0.32], [diff], "D", color="#4c9a62", ms=6, label="MiTaS MTD 데이터셋 2 빼기 V+G 데이터셋 1 (참고)" if j == 0 else None)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS_KO, fontsize=9)
    ax.set_ylabel("MTD 빼기 V+G (%p)")
    ax.set_ylim(-80, 80)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("멀티 촉각 드롭아웃의 과제별 변화와 95% 구간 (각 20회)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_mtd_delta.png", dpi=160)
    plt.close(fig)


def attention():
    tasks = list(A.keys())[:5]
    share = A["token_share"]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    x = np.arange(len(tasks))
    w = 0.26
    for i, (k, c, lab) in enumerate((("vision", BLUE, "Vision"), ("gelsight", RED, "GelSight"), ("evetac", GOLD, "Evetac"))):
        ax.bar(x + (i - 1) * w, [A[t]["mean"][k] for t in tasks], w * 0.92, color=c, label=lab)
        if k != "gelsight":   # GelSight와 Evetac는 토큰 수가 같아(192) 점선 하나로 그린다
            ax.hlines(share[k], x[0] - 0.45, x[-1] + 0.45, color=c if k == "vision" else "#555555", ls="--", lw=1.0, alpha=0.9,
                      label="토큰 수 비율: Vision 256/644" if k == "vision" else "토큰 수 비율: GelSight, Evetac 각 192/644")
    ax.set_xticks(x)
    ax.set_xticklabels(["Gear\nAssembly", "Board\nWiping", "Lamp\nInstall.", "Key\nin Lock", "Light-\nbulb"], fontsize=9)
    ax.set_ylabel("에피소드 평균 어텐션 (그림 판독)")
    ax.set_ylim(0, 0.72)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.set_title("모달리티별 평균 어텐션(성공 에피소드 1개, 부록 그림 9 판독)과 토큰 수 비율", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_attention.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main_ci()
    dataset_gap()
    mtd_delta()
    attention()
    print("wrote 4 figures to", FIG)
