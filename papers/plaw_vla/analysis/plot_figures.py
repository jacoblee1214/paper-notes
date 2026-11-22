"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_robotwin_delta.png : RoboTwin Hard, 과제별 (PLaW-VLA - pi0.5) 성공률 차이와 100회 가정 잡음 띠
own_ckpt_gap.png       : 사전학습 이득의 체크포인트 의존성 (표 8, 표 9, 그림 6 수치)
own_future_mask.png    : 추론 시 미래 마스킹 (표 10) 과 재학습 절제 (표 4)
own_real_world.png     : 실로봇 과제 진행도와 최악 분산 95% 반폭 (그림 4 수치)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from paper_tables import TABLE7  # noqa: E402

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
INK, ACC, GREY, WARN = "#1b1b1a", "#1f5fae", "#9a9890", "#c0502d"


def plot_robotwin():
    names = [r[0] for r in TABLE7]
    hz = np.array([r[1] for r in TABLE7])
    d = np.array([r[2][7] - r[2][5] for r in TABLE7])
    order = np.argsort(-d)
    col = {1: GREY, 2: "#8fb0d9", 3: ACC}
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    y = np.arange(len(d))
    ax.barh(y, d[order], color=[col[hz[i]] for i in order], height=0.78)
    band = 1.96 * np.sqrt(2 * 0.8 * 0.2 / 100) * 100
    ax.axvspan(-band, band, color="#e9e7e0", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("RoboTwin Hard: PLaW-VLA - pi0.5 성공률 차이 (포인트)")
    ax.text(band + 1, 24, f"옅은 띠: 과제당 100회 가정, p=0.8일 때\n두 값 차이의 95% 잡음 범위 (±{band:.0f})", fontsize=7.5, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col[h], label=lab) for h, lab in [(1, "Horizon I"), (2, "Horizon II"), (3, "Horizon III")]],
              loc="center right", fontsize=8, frameon=False)
    ax.set_title("과제별 이득은 고르지 않다 (36 향상, 4 동률, 10 하락)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_robotwin_delta.png", dpi=180)
    plt.close(fig)


def plot_ckpt():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    suites = ["Spatial", "Object", "Goal", "Long", "평균"]
    d30 = [1.90, 0.60, 5.90, 5.00, 3.35]
    d50 = [1.50, 0.10, 3.90, -1.50, 1.00]
    x = np.arange(5)
    ax.bar(x - 0.19, d30, 0.38, color=ACC, label="30k (본문 헤드라인)")
    ax.bar(x + 0.19, d50, 0.38, color=WARN, label="50k (부록 표 8)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(suites, fontsize=8)
    ax.set_ylabel("사전학습(Stage I-II) 이득, LIBERO (포인트)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("LIBERO: 이득이 3.35에서 1.00으로", fontsize=9.5, loc="left")
    ax = axes[1]
    ax.bar([0, 1], [11.90, 2.83], color=[ACC, WARN], width=0.55)
    for i, v in enumerate([11.90, 2.83]):
        ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["30k\n(표 4 d, 30k로 추정)", "50k\n(표 9)"], fontsize=8)
    ax.set_ylabel("사전학습 이득, LIBERO-Plus (포인트)")
    ax.set_title("LIBERO-Plus: 11.90에서 2.83으로", fontsize=9.5, loc="left")
    ax = axes[2]
    ck = [10, 20, 30, 50]
    sr = [78.22, 94.30, 97.40, 96.95]
    ax.plot(ck, sr, marker="o", color=INK)
    ax.axhline(96.85, color=GREY, ls="--", lw=1)
    ax.text(31, 95.2, "pi0.5 96.85 (표 1)", fontsize=7.5, color=GREY)
    ax.set_ylim(76, 99)
    ax.set_xlabel("Stage III 체크포인트 (k 스텝)")
    ax.set_ylabel("LIBERO 평균 (%)")
    ax.set_title("30k가 평가한 체크포인트 중 최댓값", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_ckpt_gap.png", dpi=180)
    plt.close(fig)


def plot_mask():
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    suites = ["Spatial", "Object", "Goal", "Long", "평균"]
    full = [98.90, 99.00, 97.10, 94.60, 97.40]
    rows = {
        "미래 전부 마스킹": [89.60, 95.40, 97.40, 89.80, 93.05],
        "먼 미래 마스킹": [89.60, 97.40, 99.80, 95.00, 95.45],
        "가까운 미래 마스킹": [92.40, 97.00, 99.00, 94.80, 95.80],
    }
    x = np.arange(5)
    for j, (k, v) in enumerate(rows.items()):
        ax.bar(x + (j - 1) * 0.26, np.array(v) - np.array(full), 0.26, label=k, color=[WARN, "#e0a48d", GREY][j])
    ax.axhline(-0.55, color=ACC, ls="--", lw=1.2)
    ax.text(2.0, -1.15, "재학습한 'w/o future conditioning' 평균 -0.55 (점선)", fontsize=7.5, color=ACC, ha="center", va="top")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(suites)
    ax.set_ylabel("전체 모델 대비 성공률 차이 (포인트)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_ylim(-10.5, 3.5)
    ax.set_title("추론 시 마스킹은 평균 -4.35인데 재학습 절제는 -0.55 (Goal은 마스킹하면 오히려 상승)", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_future_mask.png", dpi=180)
    plt.close(fig)


def plot_real():
    tasks = ["Fold\nTowel", "Play\nBasketball", "Pack\nToy", "Organize\nStationery", "Fold\nTowel*", "Organize\nStationery*"]
    vals = np.array([[50.0, 60.0, 80.0], [47.5, 42.5, 50.0], [46.7, 50.0, 66.7],
                     [43.3, 41.7, 55.0], [30.0, 50.0, 80.0], [35.0, 40.0, 48.3]])
    hw = 100 * 1.96 * 0.5 * np.sqrt(2 / 30)  # 최악 분산(점수 SD 0.5), 두 값 차이의 95% 반폭
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    x = np.arange(6)
    labs = ["pi0", "pi0.5", "PLaW-VLA"]
    cols = [GREY, "#8fb0d9", ACC]
    for j in range(3):
        ax.bar(x + (j - 1) * 0.26, vals[:, j], 0.26, color=cols[j], label=labs[j])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("과제 진행도 (%)")
    ax.set_ylim(0, 100)
    ax.text(1.4, 92, f"설정당 30회라 두 방법 차이의 95% 반폭이 최대 약 {hw:.0f}포인트 (점수 SD 0.5 가정)", fontsize=7.5, va="center")
    ax.annotate("47.5, 42.5는 30회 x 최대 2점의\n정수 점수로 만들 수 없음", xy=(0.95, 49), xytext=(1.3, 72),
                fontsize=7.5, arrowprops=dict(arrowstyle="-", color=WARN), color=WARN)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("실로봇: 막대 높이는 20에서 시작하는 논문 그림 4와 달리 0에서 시작", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_real_world.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_robotwin()
    plot_ckpt()
    plot_mask()
    plot_real()
    print("wrote 4 figures in", FIG)
