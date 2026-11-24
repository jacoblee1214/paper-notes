"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_ci.png   : 실로봇 성공률(웹 과제 3개의 폐루프 대 개루프, 자체 수집 과제 2개의 손 대 그리퍼)과 Wilson 95% 구간
own_scaling.png   : 그림 6 판독값(성공률, 손실)과 부록 C.3 고정 지수 멱법칙 외삽 (우리가 다시 그림)
own_toy_closed.png: 토이, 폐루프 대 일회 질의 방식의 성공률
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Noto CJK KR → Apple SD Gothic Neo 순서)
입력: analysis/recompute_results.json, analysis/toy_closed_loop_results.json (먼저 recompute.py, toy_closed_loop.py 실행)
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
INK, ACC, GREY, RED, GOLD = "#1b1b1a", "#1f5fae", "#9a9890", "#b5483f", "#c9962b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def bars(ax, xs, ks, n, color, width, label):
    pct = [100 * k / n for k in ks]
    lo = [wilson(k, n)[0] for k in ks]
    hi = [wilson(k, n)[1] for k in ks]
    ax.bar(xs, pct, width=width, color=color, label=label)
    ax.errorbar(xs, pct, yerr=[np.array(pct) - lo, np.array(hi) - pct], fmt="none", ecolor=INK, capsize=3, lw=1)
    for x, k in zip(xs, ks):
        ax.text(x, 2, f"{k}/{n}", ha="center", va="bottom", color="white", fontsize=8)


def plot_real():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    tasks = ["stirring", "wiping", "binning"]
    x = np.arange(3)
    ko = [rec["web"]["per_task"][t]["ours"][0] for t in tasks]
    kp = [rec["web"]["per_task"][t]["open"][0] for t in tasks]
    bars(ax, x - 0.19, ko, 30, ACC, 0.36, "LUCID (폐루프)")
    bars(ax, x + 0.19, kp, 30, GREY, 0.36, "개루프 기준선 (Veo 3.1)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Stirring", "Wiping", "Binning"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("성공률 (%)")
    ax.set_title("웹 영상 과제 (과제당 30회 = 3 시나리오 x 10회)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax = axes[1]
    d = rec["self_collected"]
    ks_h = [d["push_t"]["hand"][0], d["cable"]["hand"][0]]
    ks_g = [d["push_t"]["gripper"][0], d["cable"]["gripper"][0]]
    x = np.arange(2)
    bars(ax, x - 0.19, ks_h, 15, ACC, 0.36, "LEAP 손")
    bars(ax, x + 0.19, ks_g, 15, RED, 0.36, "평행 그리퍼")
    ax.set_xticks(x)
    ax.set_xticklabels(["Push-T", "Cable routing"])
    ax.set_ylim(0, 105)
    ax.set_title("자체 수집 영상 과제 (칸당 15회)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("그림 12·13의 시도 수로 다시 계산한 성공률과 Wilson 95% 구간 (논문 막대 라벨 기준)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def plot_scaling():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["scaling"]
    clips = np.array(rec["clips"], float)
    succ = np.array(rec["succ_k_of_30"])
    loss = np.array(rec["loss_read"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    ax = axes[0]
    pct = 100 * succ / 30
    lo = [wilson(int(k), 30)[0] for k in succ]
    hi = [wilson(int(k), 30)[1] for k in succ]
    ax.errorbar(clips, pct, yerr=[pct - np.array(lo), np.array(hi) - pct], fmt="o-", color=ACC, capsize=3, label="성공률 (30회, Wilson 95%)")
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_ylabel("binning 성공률 (%)", color=ACC)
    ax.set_xlabel("학습 영상 클립 수")
    ax2 = ax.twinx()
    ax2.plot(clips, loss, "s--", color=GOLD, label="held-out 손실 (그림 판독)")
    ax2.set_ylabel("held-out 손실", color=GOLD)
    ax.set_title("그림 6 판독값 (성공률은 k/30 격자로 보정)", fontsize=9)
    ax.spines[["top"]].set_visible(False)
    ax = axes[1]
    xs = np.logspace(3, 7.4, 200)
    ax.plot(clips, loss, "ks", label="관측 (그림 판독)")
    cols = plt.cm.Blues(np.linspace(0.35, 0.95, 5))
    for col, (alpha, f) in zip(cols, rec["powerlaw_fixed_alpha"].items()):
        ax.plot(xs, f["c"] + f["a"] * xs ** (-float(alpha)), color=col, lw=1.3, label=f"α={alpha}, c={f['c']:.2f}")
    ll = rec["loss_loglinear"]
    ax.plot(xs, ll["intercept"] + ll["slope_per_decade"] * np.log10(xs), "--", color=RED, lw=1, label="로그-선형")
    ax.set_xscale("log")
    ax.axvline(2e4, color=GREY, lw=0.8)
    ax.set_xlabel("학습 영상 클립 수 (20k 뒤는 외삽)")
    ax.set_ylabel("held-out 손실")
    ax.set_title("부록 C.3 방식의 고정 지수 멱법칙 (우리가 다시 적합)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_scaling.png", dpi=200)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_closed_loop_results.json").read_text(encoding="utf-8"))
    keys = list(res.keys())
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    x = np.arange(len(keys))
    w = 0.26
    for i, (mode, col, lab) in enumerate([("closed", ACC, "폐루프 재질의"), ("open_abs", GOLD, "일회 질의, 계획 경로 추종"), ("open_rel", GREY, "일회 질의, 변위열 재생")]):
        ax.bar(x + (i - 1) * w, [100 * res[k][mode] for k in keys], width=w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("sigma", "예측잡음 ").replace("_slip", "\n교란확률 ") for k in keys], fontsize=8)
    ax.set_ylabel("성공률 (%) (시행 500회)")
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("2차원 점 물체 토이: 같은 의도 모델, 질의 방식만 다름", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_closed.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_real()
    plot_scaling()
    plot_toy()
    print("wrote figures to", FIG)
