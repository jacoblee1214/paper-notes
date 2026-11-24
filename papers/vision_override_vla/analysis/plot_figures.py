"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림이 아니다).

own_table3.png    : 표 3의 π0.5 접촉률(grounding)과 성공률 — 지시된 과제(F)와 훈련 과제(B), 기준선 대 TF 대 VA.
                    스위트당 에피소드 수 750/500/500/750 가정에서 합산 비율과 Wilson 95% 구간(표의 평균 열이 아니라 합산)
own_realworld.png : 실세계 표 9·10을 다시 계산한 장면별 under-observed 지시 성공률(기준선 대 CAG), 지시당 10회, 장면당 20회, Wilson 95% 구간
own_cag_toy.png   : 토이 — (왼쪽) 한 번의 행동 청크에서 B 쪽으로 간 비율, (오른쪽) 닫힌 루프 B 도달 확률 대 ω
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py, toy_cag_mixture.py를 실행해 json을 만들어 둘 것.
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
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#d98a1a", "#9a9890"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_table3():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    rows = [r for r in rec["V10_pooled_rates"] if r["model"] == "pi0.5"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, metric, title in zip(axes, ["grounding", "success"], ["접촉률 (grounding)", "성공률 (success)"]):
        for j, cag in enumerate(["base", "TF", "VA"]):
            r = [x for x in rows if x["metric"] == metric and x["cag"] == cag][0]
            lo, hi = r["wilson95_%"]
            ax.bar(j, r["pooled_%"], color=[GREY, "#86a9d6", ACC][j], width=0.6)
            ax.errorbar(j, r["pooled_%"], yerr=[[r["pooled_%"] - lo], [hi - r["pooled_%"]]], color=INK, capsize=3, lw=1)
            ax.text(j, hi + 1.5, f"{r['pooled_%']:.1f}", ha="center", fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["기준선", "TF-CAG", "VA-CAG"])
        ax.set_title(f"π0.5 · 지시된 과제(F) {title}", fontsize=10)
        ax.set_ylim(0, 60)
        ax.grid(axis="y", color="#dcdad3", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("합산 비율 (%) · 2,500 에피소드 가정")
    fig.tight_layout()
    fig.savefig(FIG / "own_table3.png", dpi=170)
    plt.close(fig)


def plot_real():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    tests = rec["V16_scene_tests_under_observed"]
    scenes = list(tests)
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(scenes))
    for off, key, color, lab in ((-0.19, 0, GREY, "π0.5"), (0.19, 1, ORG, "π0.5 + CAG")):
        ys, los, his = [], [], []
        for s in scenes:
            k = tests[s]["success"][key]
            lo, hi = wilson(k, 20)
            ys.append(100 * k / 20)
            los.append(100 * k / 20 - lo)
            his.append(hi - 100 * k / 20)
        ax.bar(x + off, ys, width=0.36, color=color, label=lab)
        ax.errorbar(x + off, ys, yerr=[los, his], fmt="none", ecolor=INK, capsize=2.5, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("/", "/\n") for s in scenes], fontsize=7.5)
    ax.set_ylabel("under-observed 지시 성공률 (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#dcdad3", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("실세계 장면별 under-observed 두 지시의 성공률 (장면당 20회, Wilson 95%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_realworld.png", dpi=170)
    plt.close(fig)


def plot_toy():
    toy = json.loads((ANA / "toy_cag_mixture_results.json").read_text(encoding="utf-8"))
    case = toy["case1_w_u=(0.9,0.1)_w_c=(0.6,0.4)"]
    cl = toy["closed_loop_random_walk"]["cases"]["w_u=(0.9,0.1)_w_c=(0.6,0.4)"]
    oms = list(case)
    xs = [float(o) for o in oms]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.plot(xs, [case[o]["exact_tilt"]["P_positive_side"] for o in oms], "o-", color=ACC, label="분포 기울이기 (p_u^(1-ω) p_c^ω)")
    ax.plot(xs, [case[o]["velocity_cfg"]["P_positive_side"] for o in oms], "s--", color="#4a8f5b", label="속도장 CFG")
    ax.plot(xs, [case[o]["action_shared_noise"]["P_positive_side"] for o in oms], "^-", color=ORG, label="행동 외삽 (노이즈 공유)")
    ax.plot(xs, [case[o]["action_shared_noise"]["P_off_mode"] for o in oms], "x:", color=ORG, label="행동 외삽: 두 모드 밖 비율")
    ax.set_xlabel("안내 배율 ω")
    ax.set_ylabel("확률")
    ax.set_title("한 번의 행동 청크: B 쪽(+) 비율", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#dcdad3", lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax = axes[1]
    ax.plot(xs, [cl[o]["exact_tilt"]["P_reach_B"] for o in oms], "o-", color=ACC, label="분포 기울이기")
    ax.plot(xs, [cl[o]["action_shared"]["P_reach_B"] for o in oms], "^-", color=ORG, label="행동 외삽 (노이즈 공유)")
    ax.plot(xs, [cl[o]["action_independent"]["P_reach_B"] for o in oms], "v--", color="#b0702a", label="행동 외삽 (노이즈 따로)")
    ax.set_xlabel("안내 배율 ω")
    ax.set_title("닫힌 루프(매 스텝 재선택): B 도달 확률", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#dcdad3", lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "own_cag_toy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_table3()
    plot_real()
    plot_toy()
    print("wrote", FIG)
