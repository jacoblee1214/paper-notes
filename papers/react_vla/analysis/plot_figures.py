"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_success_ci.png : 합산 성공률과 Wilson 95% 구간 (시뮬 Clean, Rand, 실세계), recompute_results.json 값
own_task_diff.png  : REACT w/o DD 대 π0.5 H50E50, REACT 대 Async+RTC의 과제별 성공률 차 (시뮬 Clean, 실세계)
own_ablation.png   : 그림 11·12 재집계. S=10 열(K와 지평 H가 함께 변함)과 H=50 칸 비교
own_toy_modes.png  : 토이의 반응 지연 대 블록 부호 전환 (σ_o=1)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py와 toy_rolling_modes.py를 실행해 JSON을 만들어 둘 것.
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
INK, ACC, GREY, ORG, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#c4622d", "#3f8f5a"
NAMES = {"H50E50": "H50E50", "H50E10": "H50E10", "H10E10": "H10E10", "AsyncRTC": "Async+RTC",
         "REACT_noDD": "REACT w/o DD", "REACT": "REACT"}
COLORS = {"H50E50": GREY, "H50E10": "#c98f8f", "H10E10": "#a05a5a", "AsyncRTC": "#8a5aa0", "REACT_noDD": ACC, "REACT": GRN}


def success_ci(R):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=False)
    for ax, key, title in zip(axes, ("sim_clean", "sim_rand", "real"),
                              ("RoboTwin Clean (700회)", "RoboTwin Randomized (700회)", "실세계 6개 과제 (180회, 과제당 30회 가정)")):
        for i, m in enumerate(NAMES):
            p, (lo, hi) = R["V10_headline_wilson"][m][key]
            ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=COLORS[m], capsize=3, ms=6)
            ax.text(hi + 0.8, i, f"{p:.1f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(NAMES)))
        ax.set_yticklabels(list(NAMES.values()) if key == "sim_clean" else [""] * len(NAMES), fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("성공률 (%) 와 Wilson 95% 구간", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(0, max(R["V10_headline_wilson"][m][key][1][1] for m in NAMES) + 9)
    fig.suptitle("표 1·2의 성공률을 합산해 다시 그림 (시드 하나, 평가 롤아웃 표집 잡음만 반영)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_success_ci.png", dpi=170)
    plt.close(fig)


def task_diff():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", ANA / "recompute.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    mi = {m: i for i, m in enumerate(rc.METHODS)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [7, 6]})
    tasks = list(rc.T1)
    x = np.arange(len(tasks))
    w = 0.38
    d1 = [rc.T1[t][mi["REACT_noDD"]][0] - rc.T1[t][mi["H50E50"]][0] for t in tasks]
    d2 = [rc.T1[t][mi["REACT"]][0] - rc.T1[t][mi["AsyncRTC"]][0] for t in tasks]
    axes[0].bar(x - w / 2, d1, w, color=ACC, label="REACT w/o DD - H50E50")
    axes[0].bar(x + w / 2, d2, w, color=GRN, label="REACT - Async+RTC")
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([t.replace("_", "\n", 1) for t in tasks], fontsize=7)
    axes[0].set_ylabel("성공률 차 (%p, 100회)", fontsize=9)
    axes[0].set_title("RoboTwin Clean 과제별 차이", fontsize=10)
    axes[0].legend(fontsize=8)
    rt = list(rc.T2)
    labels = [f"{r[0]}\n{r[1].split()[0]}" for r in rt]
    e1 = [rc.T2[k][mi["REACT_noDD"]] - rc.T2[k][mi["H50E50"]] for k in rt]
    e2 = [rc.T2[k][mi["REACT"]] - rc.T2[k][mi["AsyncRTC"]] for k in rt]
    xr = np.arange(len(rt))
    axes[1].bar(xr - w / 2, e1, w, color=ACC)
    axes[1].bar(xr + w / 2, e2, w, color=GRN)
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_xticks(xr)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_title("실세계 과제별 차이 (30회 가정)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_task_diff.png", dpi=170)
    plt.close(fig)


def ablation(R):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    Ks = [1, 2, 3, 4, 5]
    vals = [R["V8d_K_H_confound"]["S10_column"][f"K{K}"]["success"] for K in Ks]
    axes[0].plot(Ks, vals, "o-", color=ACC)
    for K, v in zip(Ks, vals):
        axes[0].text(K, v + 1.5, f"H={10 * K}", ha="center", fontsize=8)
    axes[0].set_xticks(Ks)
    axes[0].set_xlabel("블록 수 K (S=10 고정)", fontsize=9)
    axes[0].set_ylabel("RoboTwin Clean 평균 성공률 (%)", fontsize=9)
    axes[0].set_title("K를 올리면 지평 H도 함께 늘어남", fontsize=10)
    axes[0].set_ylim(0, 60)
    axes[0].grid(alpha=0.25)
    same = R["V8d_K_H_confound"]["same_H50"]
    names = ["K=5, S=10", "K=10, S=5", "K=25, S=2", "K=50, S=1"]
    v = [same["K5_S10"], same["K10_S5"], same["K25_S2"], same["K50_S1"]]
    axes[1].bar(names, v, color=[ACC, GREY, GREY, GREY])
    for i, y in enumerate(v):
        axes[1].text(i, y + 1, f"{y:.1f}", ha="center", fontsize=9)
    axes[1].set_title("지평 H=50 고정, 블록 길이만 바꿈", fontsize=10)
    axes[1].set_ylim(0, 60)
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=170)
    plt.close(fig)


def toy():
    T = json.loads((ANA / "toy_rolling_modes_results.json").read_text(encoding="utf-8"))["sigma_o=1.0"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    pts = [("H50E50", "H50E50", GREY, (10, 14)), ("H50E10", "H50E10", "#c98f8f", (8, 6))]
    for key, lab, c, off in pts:
        v = T[key]
        ax.scatter(v["mean_lag_steps"], v["block_sign_flips_per_episode"], s=200 * v["accuracy"] ** 2, color=c, alpha=0.9)
        ax.annotate(f"{lab}\n정확도 {v['accuracy']:.2f}", (v["mean_lag_steps"], v["block_sign_flips_per_episode"]),
                    textcoords="offset points", xytext=off, fontsize=8)
    OFFS = {"1.0": (6, 6), "0.1": (6, 6), "0.03": (-60, 12), "0.01": (14, -26), "0.0": (-85, 14)}
    for beta in ("1.0", "0.1", "0.03", "0.01", "0.0"):
        v = T[f"REACT|beta={beta}"]
        if v["mean_lag_steps"] in (None, 0.0):
            continue
        ax.scatter(v["mean_lag_steps"], v["block_sign_flips_per_episode"], s=200 * v["accuracy"] ** 2, color=GRN, alpha=0.85)
        ax.annotate(f"REACT β={beta}\n정확도 {v['accuracy']:.2f}", (v["mean_lag_steps"], v["block_sign_flips_per_episode"]),
                    textcoords="offset points", xytext=OFFS[beta], fontsize=8, ha="left")
    ax.set_xlabel("모드 전환 뒤 반응 지연 (스텝, 반응한 경우의 평균)", fontsize=9)
    ax.set_ylabel("에피소드당 블록 부호 전환 횟수 (참 전환 2.65회 포함)", fontsize=9)
    ax.set_title("토이 σ_o=1: 원 크기는 정확도. β는 버퍼 증거를 믿는 정도", fontsize=9)
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 23)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_modes.png", dpi=170)
    plt.close(fig)


def main():
    R = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    success_ci(R)
    task_diff()
    ablation(R)
    toy()
    print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])


if __name__ == "__main__":
    main()
