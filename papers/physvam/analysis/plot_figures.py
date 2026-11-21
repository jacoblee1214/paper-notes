"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다: 라이선스가 CC BY가 아니다).

own_sim_scenes.png   : 시뮬 장면별 성공률 (논문 표 1의 평균 ± SD, 시드 3개) 과 원래 기준선 대비 PhysVAM
own_real_ci.png      : 실로봇 장면별 성공률과 Wilson 95% 구간 (논문 표 8, 시행 20회 단위)
own_value_target.png : 식 (3) 가치 목표의 시간 구조 (토이 V3, 성공과 실패, ±16 창)
실행: python3 physvam/analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

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
INK, ACC, GREY, SOFT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


# 논문 표 1 (시뮬, 장면 10개). 값은 평균, 괄호 밖 ±는 시드 3개의 SD (논문 표기 그대로)
SCENES = ["Avocado", "Beer", "Potato", "Apple", "Lemon", "Can", "D-Bottle", "W-Bottle", "Egg", "Cup"]
SIM = {
    "DynamicVLA": ([23.33, 10.00, 5.00, 8.33, 16.67, 1.67, 83.33, 10.00, 6.67, 8.33],
                   [2.36, 7.07, 4.08, 2.36, 2.36, 2.36, 10.27, 7.07, 6.24, 8.50]),
    "PUMA": ([10.00, 65.00, 41.67, 11.67, 85.00, 1.67, 58.33, 10.00, 0.00, 35.00],
             [0, 8.16, 10.27, 8.50, 7.07, 2.36, 2.36, 7.07, 0, 4.08]),
    "PUMA+": ([13.33, 85.00, 51.67, 26.67, 70.00, 0.00, 88.33, 43.33, 5.00, 31.67],
              [4.71, 12.25, 2.36, 6.24, 10.80, 0, 2.36, 2.36, 4.08, 31.71]),
    "PhysVAM": ([60.00, 41.67, 65.00, 45.00, 88.33, 36.67, 86.67, 93.33, 28.33, 43.33],
                [10.80, 10.27, 7.07, 4.08, 6.24, 10.27, 8.50, 4.71, 8.50, 4.71]),
}


def plot_sim():
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    x = np.arange(len(SCENES))
    w = 0.2
    colors = {"DynamicVLA": GREY, "PUMA": "#c9c6bd", "PUMA+": SOFT, "PhysVAM": ACC}
    for i, (name, (mean, sd)) in enumerate(SIM.items()):
        ax.bar(x + (i - 1.5) * w, mean, w, color=colors[name], label=name)
        ax.errorbar(x + (i - 1.5) * w, mean, yerr=sd, fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENES, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("시뮬레이션 장면별 성공률 (논문 표 1, 평균 ± SD, 시드 3)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_scenes.png", dpi=170)
    plt.close(fig)


# 논문 표 8 (실로봇). 성공/시행
REAL = {  # 장면: {방법: (성공, 시행)} (장면 내 객체를 합친 값)
    "컨베이어": {"DynamicVLA": (16, 40), "PUMA": (19, 40), "PhysVAM": (29, 40)},
    "턴테이블": {"DynamicVLA": (9, 40), "PUMA": (13, 40), "PhysVAM": (26, 40)},
    "굴림": {"DynamicVLA": (5, 20), "PUMA": (6, 20), "PhysVAM": (11, 20)},
}


def plot_real():
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    scenes = list(REAL)
    x = np.arange(len(scenes))
    w = 0.25
    cols = {"DynamicVLA": GREY, "PUMA": "#c9c6bd", "PhysVAM": ACC}
    for i, m in enumerate(["DynamicVLA", "PUMA", "PhysVAM"]):
        ks = [REAL[s][m][0] for s in scenes]
        ns = [REAL[s][m][1] for s in scenes]
        pct = [100 * k / n for k, n in zip(ks, ns)]
        lo = [wilson(k, n)[0] for k, n in zip(ks, ns)]
        hi = [wilson(k, n)[1] for k, n in zip(ks, ns)]
        xs = x + (i - 1) * w
        ax.bar(xs, pct, w, color=cols[m], label=m)
        ax.errorbar(xs, pct, yerr=np.array([np.array(pct) - lo, np.array(hi) - np.array(pct)]),
                    fmt="none", ecolor=INK, capsize=2, lw=0.8)
        for xi, k, n, p in zip(xs, ks, ns, pct):
            ax.text(xi, 3, f"{k}/{n}", ha="center", va="bottom", fontsize=7,
                    color="white" if m == "PhysVAM" else INK)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=9)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("실로봇 장면별 성공률과 Wilson 95% 구간 (논문 표 8, 객체 합산)", fontsize=10)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


K, TAU = 4, 16.0


def value_target(Te, c, t):
    # 식 (3)과 같은 목표. 검사 코드는 analysis/toy_value_target.py에 있다
    H = [max(1, math.ceil(k * Te / K)) for k in range(1, K + 1)]
    return c / K * sum(math.exp(-max(Hk - 1 - t, 0) / TAU) for Hk in H)


def plot_value():
    Te = 200
    t = np.arange(Te)
    succ = [value_target(Te, +1, i) for i in t]
    fail = [value_target(Te, -1, i) for i in t]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(t, succ, color=ACC, lw=1.8, label="성공 에피소드 (c = +1)")
    ax.plot(t, fail, color=GREY, lw=1.8, label="실패 에피소드 (c = -1)")
    tc = Te // 2
    ax.axvspan(tc - 16, tc + 16, color=SOFT, alpha=0.35, lw=0, label="±16 스텝 창 (가운데 시점)")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("에피소드 시점 t (스텝)")
    ax.set_ylabel("목표 V_t (식 (3))")
    ax.set_title("식 (3) 가치 목표의 시간 구조 (Te = 200, K = 4, τ = 16)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_value_target.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_sim()
    plot_real()
    plot_value()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
