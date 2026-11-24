"""그래프 두 장을 report/figures/ 에 만든다 (matplotlib).

  own_table1.png    : 표 1 다시 그림 (Erase는 평균 ± SD, 나머지 이진 과제는 60회 풀링 Wilson 95%, 시행 수 가정)
  own_label_toy.png : 토이: 라벨 규칙(K_low, d)과 1축 접촉의 힘 민감도

실행: python3 analysis/plot_figures.py   (recompute.py 와 toy_contact_1d.py 를 먼저 실행)
"""
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

C = {"pi0": "#8a8a85", "ForceVLA": "#c98a2b", "TA-VLA": "#6aa36a", "CompVLA": "#1f5fae"}


def fig_table1():
    R = json.load(open(ROOT / "analysis" / "recompute_results.json"))
    ci = R["V4_wilson_pooled60"]
    erase = {"pi0": (17.8, 7.1), "ForceVLA": (68.1, 4.1), "TA-VLA": (52.8, 9.7), "CompVLA": (92.9, 3.7)}
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    names = ["pi0", "ForceVLA", "TA-VLA", "CompVLA"]
    for ax, task, title in zip(axes, ["Erase", "Pivot", "Cup", "PnP"],
                               ["Erase (평균 ± SD, 3 run)", "Pivot (60회 Wilson)", "Cup Dispense (60회 Wilson)", "Pick and Place (60회 Wilson)"]):
        for i, m in enumerate(names):
            if task == "Erase":
                v, s = erase[m]
                ax.bar(i, v, color=C[m], yerr=s, capsize=3)
            else:
                d = ci[f"{m}/{task}"]
                lo, hi = d["wilson95"]
                ax.bar(i, d["pct"], color=C[m], yerr=[[d["pct"] - lo], [hi - d["pct"]]], capsize=3)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["π0", "Force\nVLA", "TA-\nVLA", "Comp\nVLA"], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("성공률 (%)")
    fig.suptitle("표 1 다시 그림 (우리가 계산: 이진 과제 60회 = 20회 × 3 run 가정)", fontsize=9, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "own_table1.png", dpi=170)
    plt.close(fig)


def fig_toy():
    f = np.linspace(0, 50, 501)
    klow = np.where(f < 1, 500, np.where(f > 15, 200, 500 - 300 * (f - 1) / 14))
    d = np.where(f < 1, 0, f / klow) * 1000
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    ax.plot(f, klow, color="#1f5fae", label="k_low (N/m)")
    ax.set_xlabel("측정 접촉력 |f| (N)")
    ax.set_ylabel("k_low (N/m)", color="#1f5fae")
    ax2 = ax.twinx()
    ax2.plot(f, d, color="#c0392b", label="d (mm)")
    ax2.set_ylabel("가상 변위 d = f / k_low (mm)", color="#c0392b")
    ax.set_title("라벨 규칙 (부록 B.2, 식 18)", fontsize=9)
    ax.spines[["top"]].set_visible(False)
    ke = 5000.0
    e = np.linspace(-0.03, 0.03, 301)

    def F(K, dd):
        return K * ke / (K + ke) * np.maximum(e + dd, 0)

    p = 10 * (500 + ke) / (500 * ke)
    kB = 500 - 300 * (10 - 1) / 14
    ax = axes[1]
    ax.plot(e * 1000, F(500, p), color="#8a8a85", label="A 위치만, K=500, 깊이 명령")
    ax.plot(e * 1000, F(kB, 10 / kB), color="#1f5fae", label="B CompVLA식 (K≈307, d≈33 mm)")
    ax.plot(e * 1000, F(500, 10 / kB), color="#c98a2b", ls="--", label="C K=500 고정 + 같은 d")
    ax.axhline(10, color="k", lw=0.5, ls=":")
    ax.set_xlabel("목표 높이 오차 e (mm, +는 더 깊음)")
    ax.set_ylabel("접촉력 (N)")
    ax.set_title("1축 토이 (ke = 5000 N/m, 목표 10 N)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    kes = np.array([1000, 2000, 5000, 10000, 20000, 50000.0])
    ax.plot(kes, 500 * kes / (500 + kes) / 1000, "o-", color="#8a8a85", label="K=500")
    ax.plot(kes, kB * kes / (kB + kes) / 1000, "o-", color="#1f5fae", label="K=307 (10 N)")
    ax.plot(kes, 200 * kes / (200 + kes) / 1000, "o-", color="#6aa36a", label="K=200 (15 N 이상)")
    ax.set_xscale("log")
    ax.set_xlabel("환경 강성 ke (N/m)")
    ax.set_ylabel("높이 오차 1 mm 당 힘 변화 (N)")
    ax.set_title("힘 민감도 = K·ke / (K+ke)", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "own_label_toy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_table1()
    fig_toy()
    print("wrote", OUT)
