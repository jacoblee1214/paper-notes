"""우리가 그린 그래프 세 장. 논문 그림은 쓰지 않는다(논문 그림은 사진·렌더 화면이 섞여 있어 이 판에서는 넣지 않았다).
입력: toy_euler_steps_results.json, toy_ratio_vs_likelihood_results.json, table_check_results.json
출력: report/figures/own_*.png
실행: python3 analysis/plot_figures.py  (matplotlib, numpy; 한글 폰트는 Apple SD Gothic Neo → Noto CJK KR 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = False

BLUE, ORANGE, GREY, GREEN = "#1f5fae", "#d9822b", "#7a7a75", "#3d8c5a"


def fig_euler():
    d = json.loads((AN / "toy_euler_steps_results.json").read_text(encoding="utf-8"))
    rows = d["rows"]
    K = [r["K"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax = axes[0]
    ax.plot(K, [abs(r["zero_err"]) for r in rows], "o-", color=BLUE, label="0 초기화 (zero-sampling)")
    ax.plot(K, [abs(r["random_std_err"]) for r in rows], "s-", color=ORANGE, label="무작위 초기화, 표준편차 오차")
    ax.plot(K, [abs(r["random_pos_mass_err"]) for r in rows], "^-", color=GREY, label="무작위 초기화, 양의 쪽 질량 오차")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Euler 단계 K (로그 눈금)")
    ax.set_ylabel("|K 출력 - K=1024 출력|")
    ax.set_title("적분 단계별 오차 (토이, 1차원 혼합)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axhline(d["ref_K1024"]["zero_action"], color=GREY, lw=1, ls="--", label="K=1024 기준 값")
    ax.plot(K, [r["zero_action"] for r in rows], "o-", color=BLUE, label="K단계 0 초기화 출력")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Euler 단계 K (로그 눈금)")
    ax.set_ylabel("평가 행동 값")
    ax.set_title("0 초기화 출력의 수렴", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("토이 1: 설정은 가정이며 논문 값에 맞추지 않았습니다 (numpy, 단일 프로세스)", fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "own_euler_steps.png", dpi=200)
    plt.close(fig)


def fig_ratio():
    d = json.loads((AN / "toy_ratio_vs_likelihood_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

    ax = axes[0]
    sw = d["scale_sweep"]
    s = [r["s"] for r in sw]
    ax.plot(s, [r["slope"] for r in sw], "o-", color=BLUE, label="측정 기울기")
    import numpy as np
    sg = np.linspace(0.08, 1.1, 50)
    ax.plot(sg, np.pi / 2 * sg, "--", color=GREY, lw=1, label="(π/2)·s")
    ax.set_xlabel("행동 척도 s")
    ax.set_ylabel("기울기 (CFM 차이 / 로그 가능도 비율)")
    ax.set_title("대리 비율의 척도 의존", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    bd = d["by_delta"]
    for key, col in zip(["0.05", "0.1", "0.2"], [GREY, ORANGE, BLUE]):
        vr = bd[key]["variance_by_Nmc"]
        nm = [r["Nmc"] for r in vr]
        ax.plot(nm, [r["per_action_logratio_sd_across_a"] for r in vr], "o-", color=col, label=f"δ={key}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N_mc (행동당 샘플 수)")
    ax.set_ylabel("per-action 로그 비율 표준편차")
    ax.set_title("N_mc에 따른 분산", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    vr = bd["0.1"]["variance_by_Nmc"]
    nm = [r["Nmc"] for r in vr]
    ax.plot(nm, [r["per_sample_clip_frac"] for r in vr], "o-", color=BLUE, label="per-sample 비율")
    ax.plot(nm, [r["per_action_clip_frac"] for r in vr], "s--", color=ORANGE, label="per-action 비율")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("N_mc (행동당 샘플 수)")
    ax.set_ylabel("|비율 - 1| > 0.05 인 비중")
    ax.set_title("δ=0.1일 때 클리핑 빈도", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("토이 2: 1차원 가우시안 정확 장, 옛 정책 대비 새 정책 (설정은 가정)", fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "own_ratio_vs_likelihood.png", dpi=200)
    plt.close(fig)


def fig_c7():
    d = json.loads((AN / "table_check_results.json").read_text(encoding="utf-8"))
    rows = d["c7_pooled_wilson"]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels = [r["name"] for r in rows]
    means = [r["p"] for r in rows]
    lo = [r["p"] - r["wilson95"][0] for r in rows]
    hi = [r["wilson95"][1] - r["p"] for r in rows]
    colors = [BLUE if "PPO trust" in r["name"] else ORANGE for r in rows]
    ax.bar(labels, means, color=colors, alpha=0.85)
    ax.errorbar(labels, means, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4, lw=1)
    for i, m in enumerate(means):
        ax.text(i, m + 0.06, f"{m:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("성공률 (논문 보고값)")
    ax.set_ylim(0, 1.05)
    ax.set_title("부록 C.7 보고값과 풀링 Wilson 95% 구간 (10 시드 x 100 에피소드 = 1000)", fontsize=9)
    plt.xticks(fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "시드별 값이 없어 시드 간 변동은 반영하지 않았습니다(03 §4의 민감도 참조).", fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_c7_pooled.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_euler()
    fig_ratio()
    fig_c7()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
