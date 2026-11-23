"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_table1_ci.png   표 1 최종 성공률과 Wilson 95% 구간 (과제당 20회) + 기반 정책
  own_forest.png      Ours와 각 비교 대상의 성공률 차이 (Newcombe 95% 구간, 표 1과 표 2, 시행 20회 가정)
  own_base_vs_ours.png 그림 5와 그림 6 판독 값의 전/후 성공률과 Wilson 구간 (시행 20회 가정)

입력: analysis/recompute_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

BLUE, GREY, ORANGE, GREEN, RED = "#1f5fae", "#8a8a85", "#c77a12", "#3b8a5a", "#b04a4a"
TASKS = ["Peg-in-Hole", "Charger Insertion", "Cap Opening", "Box Opening"]
T1 = {"기반 정책": [40, 10, 5, 5], "PLD*": [65, 60, 50, 35], "PLD (시각만)": [60, 30, 40, 20],
      "ViTAL": [50, 50, 45, 30], "Ours": [100, 100, 90, 85]}
COLORS = {"기반 정책": GREY, "PLD*": ORANGE, "PLD (시각만)": GREEN, "ViTAL": RED, "Ours": BLUE}


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def table1_ci():
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=180)
    x = np.arange(len(TASKS))
    w = 0.16
    for i, (name, vals) in enumerate(T1.items()):
        off = (i - 2) * w
        lo = [wilson(v / 5, 20)[0] * 100 for v in vals]
        hi = [wilson(v / 5, 20)[1] * 100 for v in vals]
        yerr = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
        ax.bar(x + off, vals, w, color=COLORS[name], label=name, yerr=yerr, capsize=2, error_kw={"lw": 0.7})
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 3, f"{v}", ha="center", fontsize=6.5, color="#333")
    ax.set_xticks(x, TASKS, fontsize=8.5)
    ax.set_ylabel("성공률 (%)", fontsize=8.5)
    ax.set_ylim(0, 122)
    ax.axhline(100, color="#ccc", lw=0.6)
    ax.set_title("표 1: 최종 성공률(20회로 가정)과 Wilson 95% 구간", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png")
    plt.close(fig)


def forest():
    rows = []
    for c in RC["V4_ours_vs_baselines"]:
        rows.append((f'{c["task"]} vs {c["baseline"].replace("PLD (Visual Only)", "PLD (시각만)")}',
                     c["diff_pp"], c["newcombe_95"], BLUE))
    for c in RC["V10_table2_pairs"]:
        if (c["ours"], c["baseline"]) in (("Ours (ACT)", "ACT + Tactile"), ("Ours (DP)", "RDP"),
                                          ("Ours (pi0.5)", "pi0.5 + Tactile")):
            rows.append((f'Peg-in-Hole 표 2: {c["ours"]} vs {c["baseline"]}', c["diff_pp"], c["newcombe_95"], ORANGE))
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=180)
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d, ci, col) in zip(y, rows):
        ax.plot(ci, [yi, yi], color=col, lw=1.6)
        ax.plot([d], [yi], "o", color=col, ms=4.5)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.8)
    ax.set_xlabel("성공률 차이 (%p), Newcombe 95% 구간, 시행 20회 가정", fontsize=8.5)
    ax.set_title("Ours와 비교 대상의 차이: 파랑은 표 1 (12개), 주황은 표 2의 같은 백본 모방 학습 (3개)", fontsize=8.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_forest.png")
    plt.close(fig)


def base_vs_ours():
    f5 = [("Flow (사람)", 40, 100), ("Flow (원격조작)", 50, 90), ("ACT", 25, 80), ("DP", 30, 90), ("π0.5", 15, 75)]
    f6 = [("Peg: AnyTouch2", 40, 100), ("Peg: Sparsh", 40, 100), ("Peg: T3", 40, 90), ("Peg: Marker", 40, 100),
          ("Charger: AnyTouch2", 10, 100), ("Charger: Sparsh", 10, 80), ("Charger: T3", 10, 70),
          ("Charger: Marker", 10, 100)]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=180, gridspec_kw={"width_ratios": [5, 8]})
    for ax, data, title in ((axes[0], f5, "그림 5: 기반 정책별 전/후 (Peg-in-Hole)"),
                            (axes[1], f6, "그림 6: 촉각 표현별 전/후 (그림 판독 값)")):
        x = np.arange(len(data))
        for off, idx, col, lab in ((-0.19, 1, GREY, "전"), (0.19, 2, BLUE, "후")):
            vals = [d[idx] for d in data]
            lo = [wilson(v / 5, 20)[0] * 100 for v in vals]
            hi = [wilson(v / 5, 20)[1] * 100 for v in vals]
            yerr = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
            ax.bar(x + off, vals, 0.36, color=col, label=lab, yerr=yerr, capsize=2, error_kw={"lw": 0.7})
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 3, f"{v}", ha="center", fontsize=6.5)
        ax.set_xticks(x, [d[0] for d in data], fontsize=7, rotation=25 if len(data) > 5 else 0, ha="right" if len(data) > 5 else "center")
        ax.set_ylim(0, 120)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("성공률 (%), 20회 가정, Wilson 95%", fontsize=8)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "own_base_vs_ours.png")
    plt.close(fig)


if __name__ == "__main__":
    table1_ci()
    forest()
    base_vs_ours()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
