"""논문 표 값으로 우리가 그린 그래프 두 장을 만든다 (report/figures/own_*.png).

논문 그림은 쓰지 않는다 (원문 그림은 데이터셋 프레임과 실로봇 사진이라 공개본에 넣을 수 없다).
실행: python3 analysis/plot_figures.py
한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

KOREAN_CANDIDATES = ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]
available = {f.name for f in font_manager.fontManager.ttflist}
for name in KOREAN_CANDIDATES:
    if name in available:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACCENT, GREY = "#1b1b1a", "#55544f", "#1f5fae", "#b9b6ad"


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def fig_gain():
    # 표 2: PhysCoRe의 PhysTwin 대비 상대 개선 (%), 양수 = 더 좋음
    labels = ["CD\n(기하)", "Track\n(점 추적)"]
    elastic = [43.72, 15.22]
    plast = [30.46, 8.29]
    x = range(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=150)
    b1 = ax.bar([i - w / 2 for i in x], elastic, w, color=ACCENT, label="탄성 객체 (rope, towel, bear)")
    b2 = ax.bar([i + w / 2 for i in x], plast, w, color=GREY, label="탄소성 객체 (Play-Doh)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylabel("PhysTwin 대비 오차 감소율 (%)", color=INK, fontsize=9)
    ax.set_ylim(0, 52)
    ax.set_title("표 2: PhysCoRe의 PhysTwin 대비 오차 감소 (평균만, 분산 없음)", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.text(0.01, 0.01, "출처: 논문 v3 표 2 값 재계산 (analysis/recompute.py). 에피소드별 값이 없어 구간은 그리지 않았다.",
             fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_gain_bars.png")
    plt.close(fig)


def fig_runtime_planning():
    # 왼쪽: 표 5 식별 시간 (로그 축). 오른쪽: 표 4 계획 Chamfer 거리
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=150, gridspec_kw={"width_ratios": [1, 1.1]})
    names = ["PGND", "PhysTwin", "PhysCoRe"]
    times = [8280.0, 930.0, 11.4]
    cols = [GREY, GREY, ACCENT]
    a1.bar(names, times, color=cols)
    a1.set_yscale("log")
    for i, t in enumerate(times):
        a1.text(i, t * 1.25, f"{t:g} s", ha="center", fontsize=8, color=INK)
    a1.set_ylabel("식별·최적화 시간 (s, 로그)", color=INK, fontsize=9)
    a1.set_title("표 5: 재료 식별 시간", fontsize=9, color=INK, loc="left")
    a1.set_ylim(5, 30000)
    style(a1)

    tasks = ["rope", "towel"]
    data = {"PGND": [48.8, 30.6], "PhysTwin": [17.3, 7.1], "PhysCoRe": [6.2, 5.3]}
    w = 0.26
    for j, (k, v) in enumerate(data.items()):
        bars = a2.bar([i + (j - 1) * w for i in range(2)], v, w, color=[GREY, "#8f8c83", ACCENT][j], label=k)
        for b in bars:
            a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():g}",
                    ha="center", fontsize=7.5, color=INK)
    a2.set_xticks(range(2))
    a2.set_xticklabels(tasks, color=INK)
    a2.set_ylabel("목표 대비 Chamfer (mm, 낮을수록 좋음)", color=INK, fontsize=8)
    a2.set_title("표 4: 계획 결과 (반복 횟수 미기재)", fontsize=9, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=7.5)
    style(a2)
    fig.text(0.01, 0.01, "출처: 논문 v3 표 4, 표 5 값 (우리가 그림). 계획 실험의 반복 횟수는 논문에 없다.",
             fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "own_runtime_planning.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_gain()
    fig_runtime_planning()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
