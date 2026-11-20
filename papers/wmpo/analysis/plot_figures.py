"""WMPO 분석의 자체 그래프 두 장을 만든다 (논문 그림은 쓰지 않는다).

입력: analysis/recompute_results.json, analysis/toy_reward_exploit_results.json
출력: report/figures/own_sim_ci.png, report/figures/own_toy_exploit.png
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def sim_ci():
    # 표 1 평균 (4과제 × 128 초기 상태를 512 에피소드로 합산, Wilson 95%)
    means = {"기저": (33.6, None), "GRPO": (33.2, 37.1), "DPO": (37.3, 42.4), "WMPO": (47.1, 57.6)}
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    width = 0.26
    names = list(means.keys())
    for j, P in enumerate((128, 1280)):
        xs, ys, lo, hi = [], [], [], []
        for i, name in enumerate(names):
            if name == "기저":
                if P == 1280:
                    continue
                val = means[name][0]
            else:
                val = means[name][0] if P == 128 else means[name][1]
            lo_, hi_ = wilson(val / 100, 512)
            xs.append(i + (j - 0.5) * width)
            ys.append(val)
            lo.append(val - 100 * lo_)
            hi.append(100 * hi_ - val)
        ax.bar(xs, ys, width=width, yerr=[lo, hi], capsize=3,
               label=f"P = {P}", color=("#9bb7d9" if P == 128 else "#1f5fae"))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("성공률 평균 (%)")
    ax.set_title("시뮬 4과제 평균 (표 1 값, 512 에피소드 Wilson 95%)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png")
    plt.close(fig)


def toy_exploit():
    res = json.loads((ROOT / "analysis" / "toy_reward_exploit_results.json").read_text(encoding="utf-8"))
    labels = ["FPR 0\n(이상적)", "FPR 0.05\n(같음)", "FPR B = 0.40\n(미관측 모드)"]
    keys = ["scenario0_no_fp", "scenario1_equal_fpr", "scenario2_unseen_fpr"]
    pB = [res[k]["final_pB_mean"] for k in keys]
    pB_sd = [res[k]["final_pB_sd"] for k in keys]
    tr = [res[k]["final_true_success_mean"] for k in keys]
    tr_sd = [res[k]["final_true_success_sd"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    x = range(len(keys))
    ax.bar([i - 0.2 for i in x], pB, width=0.36, yerr=pB_sd, capsize=3, color="#c96a3a", label="최종 π(B) (정책이 모드 B를 고를 확률)")
    ax.bar([i + 0.2 for i in x], tr, width=0.36, yerr=tr_sd, capsize=3, color="#1f5fae", label="최종 실제 성공률")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("토이: 보상 판정기 거짓 양성과 GRPO 쏠림 (시드 200개, 우리 모형)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_exploit.png")
    plt.close(fig)


if __name__ == "__main__":
    sim_ci()
    toy_exploit()
    print("wrote", FIG / "own_sim_ci.png", FIG / "own_toy_exploit.png")
