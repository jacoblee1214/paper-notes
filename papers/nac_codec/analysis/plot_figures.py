"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림이 아니다).

own_real_ci.png    : 표 6 실로봇 성공률(방법당 80회)과 Wilson 95% 구간, 과제별 성공 횟수
own_sim_pm.png     : 표 2 시뮬 평균과 ± (±를 표준오차로 읽은 구간), LIBERO-10과 RoboMimic
own_ablation.png   : 표 1 ablation의 재구성 MSE 대 정책 성공률 산점도
own_layout.png     : 공식 설정에서 읽은 토큰 배치 (32 x 7 = 224 샘플 → 잠재 8프레임 → 거친 4 + 세밀 8 = 12 토큰)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

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
INK, ACC, GREY, SOFT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"
METHODS = ["Bin", "Diffusion Policy", "FAST", "VQ-VLA", "OAT", "NAC"]
COLORS = [GREY, GREY, GREY, GREY, GREY, ACC]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_real():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V4_real_world"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    counts = [rec["counts_of_80"][m] for m in METHODS]
    pct = [100 * c / 80 for c in counts]
    lo = [wilson(c, 80)[0] for c in counts]
    hi = [wilson(c, 80)[1] for c in counts]
    x = np.arange(len(METHODS))
    ax.bar(x, pct, color=COLORS, width=0.62)
    ax.errorbar(x, pct, yerr=[np.array(pct) - lo, np.array(hi) - pct], fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, c in enumerate(counts):
        ax.text(i, 2, f"{c}/80", ha="center", va="bottom", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Bin", "DP", "FAST", "VQ-VLA", "OAT", "NAC"], fontsize=9)
    ax.set_ylabel("성공률 (%), Wilson 95% 구간")
    ax.set_ylim(0, 80)
    ax.set_title("8과제 합산 (방법당 80회)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    tasks = ["Weighing", "Grapes", "Marker", "Two Blocks", "Three Blocks", "Chess", "Place Stone", "Fold Towel"]
    T6 = {"FAST": [8, 8, 6, 0, 0, 1, 1, 8], "OAT": [4, 8, 5, 0, 1, 4, 3, 7], "NAC": [9, 10, 5, 3, 0, 1, 4, 8]}
    w = 0.26
    for j, (m, col) in enumerate([("FAST", GREY), ("OAT", SOFT), ("NAC", ACC)]):
        ax.bar(np.arange(8) + (j - 1) * w, T6[m], width=w, color=col, label=m)
    ax.set_xticks(range(8))
    ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("성공 횟수 (과제당 10회)")
    ax.set_ylim(0, 10.5)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax.set_title("과제별 성공 횟수 (논문 표 6)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_sim():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    data = {
        "LIBERO-10": [(3.95, 0.8), (25.48, 1.3), (38.02, 1.3), (10.85, 1.85), (44.17, 1.2), (49.73, 1.0)],
        "RoboMimic": [(7.56, 1.05), (27.25, 1.87), (28.38, 2.37), (21.44, 1.45), (31.94, 2.15), (33.94, 1.86)],
    }
    for ax, (env, rows) in zip(axes, data.items()):
        means = [r[0] for r in rows]
        pms = [r[1] for r in rows]
        x = np.arange(6)
        ax.bar(x, means, color=COLORS, width=0.62)
        ax.errorbar(x, means, yerr=pms, fmt="none", ecolor=INK, capsize=3, lw=1)
        for i, (m, p) in enumerate(rows):
            ax.text(i, m + p + 0.8, f"{m:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Bin", "DP", "FAST", "VQ-VLA", "OAT", "NAC"], fontsize=9)
        ax.set_ylabel("성공률 (%)")
        ax.set_title(f"{env}: 평균 ± (논문 표 2, ± 정의는 본문에 없음)", fontsize=9)
        ax.set_ylim(0, 60 if env == "LIBERO-10" else 42)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_pm.png", dpi=170)
    plt.close(fig)


def plot_ablation():
    pts = {  # 라벨: (MSE, 성공률, 글자 위치 오프셋 dx, dy)
        "L1": (0.002, 44.78, 6, 3), "MSE (1a)": (0.0008, 49.2, -4, -13), "DCT": (0.0007, 47.85, -28, -12),
        "Mel (no log)": (0.0006, 44.6, 6, -12), "Spectrogram (= ISTFT 행)": (0.0002, 48.3, 6, 4),
        "DAC (1b)": (0.0005, 49.45, -26, 7), "MPD": (0.0005, 46.28, -24, -4), "MRD": (0.0004, 45.68, -26, -11),
        "None (MSE)": (0.0010, 50.1, 6, 3), "Linear head": (0.0006, 42.1, 6, -3),
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for k, (m, sc, dx, dy) in pts.items():
        ax.scatter(m, sc, color=ACC, s=26, zorder=3)
        ax.annotate(k, (m, sc), xytext=(dx, dy), textcoords="offset points", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([0.0002, 0.0005, 0.001, 0.002])
    ax.set_xticklabels(["0.0002", "0.0005", "0.001", "0.002"])
    ax.minorticks_off()
    ax.set_xlim(0.00014, 0.0034)
    ax.set_xlabel("재구성 MSE (표 1, 로그 축)")
    ax.set_ylabel("정책 성공률 (%, LIBERO-10)")
    ax.set_ylim(40, 52)
    ax.set_title("표 1 행 10개: 재구성 MSE와 성공률의 관계 (Log-Mel 0.038, 0%는 축 밖)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=170)
    plt.close(fig)


def plot_layout():
    fig, ax = plt.subplots(figsize=(9.4, 3.3))
    ax.set_xlim(0, 224)
    ax.set_ylim(0, 10.8)
    ax.axis("off")
    # 224 샘플 = 32 스텝 x 7 차원 (스텝 우선으로 7개씩 이어 붙임)
    for t in range(32):
        ax.add_patch(Rectangle((t * 7, 8.2), 7, 1.1, facecolor=SOFT if t % 2 == 0 else "#c9d9ee", edgecolor="white", lw=0.4))
    ax.text(0, 9.55, "평탄화한 1차원 신호: 32 스텝 x 7 차원 = 224 샘플 (스텝마다 7개 값이 이어짐)", fontsize=8.5, va="bottom")
    # 잠재 프레임: hop 28 → 8프레임
    for f in range(8):
        ax.add_patch(Rectangle((f * 28, 5.7), 28, 1.1, facecolor="#e7dfc8" if f % 2 == 0 else "#d8cca8", edgecolor="white", lw=0.8))
        ax.text(f * 28 + 14, 6.25, f"{f+1}", ha="center", va="center", fontsize=8)
    ax.text(0, 6.95, "인코더 출력: 잠재 8프레임 (hop = 7 x 4 = 28 샘플, 프레임 하나가 4스텝)", fontsize=8.5, va="bottom")
    # 거친 스케일: 풀링 2 → 4 토큰
    for c in range(4):
        ax.add_patch(Rectangle((c * 56, 3.2), 56, 1.1, facecolor=ACC, edgecolor="white", lw=0.8))
        ax.text(c * 56 + 28, 3.75, f"거친 {c+1}", ha="center", va="center", fontsize=8, color="white")
    ax.text(0, 4.45, "스케일 0 (먼저 예측): 평균 풀링 2 → 4 토큰", fontsize=8.5, va="bottom")
    for c in range(8):
        ax.add_patch(Rectangle((c * 28, 0.7), 28, 1.1, facecolor="#3f7fcf", edgecolor="white", lw=0.8, alpha=0.75))
        ax.text(c * 28 + 14, 1.25, f"세밀 {c+1}", ha="center", va="center", fontsize=7, color="white")
    ax.text(0, 1.95, "스케일 1 (나중에 예측): 풀링 1 → 8 토큰. 각 토큰은 FSQ 4채널 (8 x 8 x 4 x 4 = 1024 코드)", fontsize=8.5, va="bottom")
    fig.savefig(FIG / "own_layout.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_real()
    plot_sim()
    plot_ablation()
    plot_layout()
    print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
