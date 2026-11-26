"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_pm_ratio.png     : 표의 ± 를 시드당 100회 이항 SD 로 나눈 비율의 분포 (V1)
own_table1_vs_2.png  : 표 1 conveyor 열과 표 2 보간값(속도 5–25 cm/s 균등 가정) (V5)
own_real_ci.png      : 실로봇 성공률 Wilson 95% 구간과 속도 스윕, 시뮬 표 2 와 나란히 (V9)
own_toy_kinematic.png: 토이, 정속 대 등가속 외삽의 위치 오차 (analysis/toy_kinematic.py)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 recompute.py 와 toy_kinematic.py 를 실행)
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

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
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#b0412e"
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_pm_ratio():
    # 셀별 비율은 recompute.py 를 다시 돌려 얻는다(json 에는 요약만 있다)
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", ANA / "recompute.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    ratios = np.array([c["ratio"] for c in rc.CELLS if "ratio" in c])
    rng = np.random.default_rng(1)
    hypA = np.sqrt(rng.chisquare(4, 200000) / 4)
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    bins = np.linspace(0, 2.6, 53)
    ax.hist(ratios, bins=bins, color=ACC, alpha=0.85, density=True, label=f"표의 ± (셀 {len(ratios)}개)")
    ax.hist(hypA, bins=bins, histtype="step", color=RED, lw=1.8, density=True,
            label="가정 A: ± = 5시드 평균의 SD, 시드 변동 = 이항 잡음뿐")
    ax.axvline(1.0, color=GREY, lw=1, ls=":")
    ax.set_xlabel("± / sqrt(p(1-p)/100)   (시드당 100회 이항 SD 대비)")
    ax.set_ylabel("밀도")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("표의 ± 는 이항 잡음 크기 근처에 모여 있다 (시뮬레이션 표 전체)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_pm_ratio.png", dpi=170)
    plt.close(fig)


def plot_t1_t2():
    v5 = REC["V5"]
    names = list(v5.keys())
    t1 = [v5[n]["table1_const_vel_conveyor"] for n in names]
    t2 = [v5[n]["table2_interp_avg_5_25"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.bar(x - 0.19, t1, 0.38, color=ACC, label="표 1 정속 conveyor 열")
    ax.bar(x + 0.19, t2, 0.38, color=GREY, label="표 2를 5–25 cm/s 로 평균 (선형 보간, 가정)")
    for i, (a, b) in enumerate(zip(t1, t2)):
        ax.text(i - 0.19, a + 1.5, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.19, b + 1.5, f"{b:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" VLA", "\nVLA").replace("Streaming DP", "Streaming\nDP").replace("Realtime ACT", "Realtime\nACT") for n in names], fontsize=8.5)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("같은 conveyor + red cup 과제의 두 표 (속도 분포 5–25 cm/s 가정)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_vs_2.png", dpi=170)
    plt.close(fig)


def plot_real():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    tasks = ["컨베이어 A\n(정지 물체→\n움직이는 상자)", "컨베이어 B\n(움직이는 물체→\n정지 상자)", "패들", "공 멈추기", "발사체 잡기"]
    ahead = [30, 29, 23, 30, 19]
    best = [12, 9, 9, 12, 0]
    x = np.arange(5)
    for i, (k, b) in enumerate(zip(ahead, best)):
        lo, hi = wilson(k, 30)
        ax.errorbar(i - 0.12, 100 * k / 30, yerr=[[max(0, 100 * k / 30 - lo)], [max(0, hi - 100 * k / 30)]], fmt="o", color=ACC, capsize=3,
                    label="AHEAD (30회, Wilson 95%)" if i == 0 else None)
        lo, hi = wilson(b, 30)
        ax.errorbar(i + 0.12, 100 * b / 30, yerr=[[max(0, 100 * b / 30 - lo)], [max(0, hi - 100 * b / 30)]], fmt="s", color=GREY, capsize=3,
                    label="가장 높은 기준선 (과제별)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=7.6)
    ax.set_ylim(-3, 105)
    ax.set_ylabel("성공률 (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("실로봇 과제 (표 4 왼쪽)", fontsize=10)
    ax = axes[1]
    speeds = [0, 5, 10, 15, 20, 25]
    sweep = [10, 10, 10, 7, 6, 2]
    for i, (s, k) in enumerate(zip(speeds, sweep)):
        lo, hi = wilson(k, 10)
        ax.errorbar(s, 10 * k, yerr=[[max(0, 10 * k - lo)], [max(0, hi - 10 * k)]], fmt="o", color=ACC, capsize=3,
                    label="실로봇 AHEAD (10회, Wilson 95%)" if i == 0 else None)
    ax.plot([0, 10, 20], [97.6, 97.4, 97.0], "-", color=RED, lw=1.5, label="시뮬 AHEAD (표 2, 500회)")
    ax.set_xlabel("컨베이어 속도 (cm/s)")
    ax.set_ylim(-3, 105)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("속도 스윕: 실로봇 대 시뮬", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_toy():
    toy = json.loads((ANA / "toy_kinematic_results.json").read_text(encoding="utf-8"))
    cases = toy["cases"]
    pick = ["완만한 가속 a=0.02", "큰 가속 a=0.30", "충돌 (속도 점프 SD 1.0)"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=False)
    for ax, name in zip(axes, pick):
        c = cases[name]
        ks = [1, 5, 10]
        ax.plot(ks, c["const_K1_5_10"], "o-", color=GREY, label="정속 외삽")
        ax.plot(ks, c["accel_K1_5_10"], "s-", color=ACC, label="등가속 외삽 (식 3)")
        ax.set_title(f"{name}\n(광류 잡음 σ={c['sigma']} px/프레임)", fontsize=8.8)
        ax.set_xlabel("예측 스텝 K")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("위치 RMS 오차 (px)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_kinematic.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_pm_ratio()
    plot_t1_t2()
    plot_real()
    plot_toy()
    print("wrote 4 figures to", FIG)
