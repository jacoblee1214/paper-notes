"""우리가 그리는 그래프 세 장을 만든다 (report/figures/own_*.png).

  own_success_table1.png : 표 1의 성공률(전문가 수별 학습 집합, 시험 집합 여섯 행)과 부록 D의 시드 3개 평균 ± SD
  own_toy_grad.png       : 토이 T1, 갱신 크기에 따른 MAPO와 PPO 기울기 코사인과 상호작용 항
  own_toy_ppo.png        : 토이 T2, PPO·MAPO 학습 곡선과 전문가 수별 교차항 비율

실행: python3 analysis/plot_figures.py   (matplotlib; 먼저 recompute.py, toy_algebra.py, toy_moe_ppo.py 실행)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"):
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e2dc", "grid.linewidth": 0.6, "font.size": 10})
BLUE, ORANGE, GREY, GREEN = "#2a6fbb", "#d9822b", "#777777", "#3a9a6b"


def fig_success():
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.05, 1]})
    ns = [4, 6, 8, 16]
    ppo = [83.96, 83.38, 82.28, 82.95]
    mapo = [86.87, 86.60, 87.85, 87.73]
    x = range(len(ns))
    w = 0.36
    ax[0].bar([i - w / 2 for i in x], ppo, w, color=ORANGE, label="PPO with MoE")
    ax[0].bar([i + w / 2 for i in x], mapo, w, color=BLUE, label="MAPO")
    for i, (a, b) in enumerate(zip(ppo, mapo)):
        ax[0].text(i - w / 2, a + 0.15, f"{a:.2f}", ha="center", fontsize=8)
        ax[0].text(i + w / 2, b + 0.15, f"{b:.2f}", ha="center", fontsize=8)
    # N = 6의 시드 3개 평균 ± SD
    ax[0].errorbar([1 - w / 2 + 0.55 * w], [82.89], yerr=[0.43], fmt="D", color="black", ms=5, capsize=3)
    ax[0].errorbar([1 + w / 2 + 0.55 * w], [87.03], yerr=[0.64], fmt="D", color="black", ms=5, capsize=3,
                   label="시드 3개 평균 ± SD (N = 6, 부록 D)")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels([f"N = {n}" for n in ns])
    ax[0].set_ylim(78, 90)
    ax[0].set_ylabel("학습 집합 성공률 (%)")
    ax[0].set_title("표 1(a) 전문가 수별 성공률 (한 시드)", fontsize=10.5)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    names = ["Monolithic\nPPO", "PPO with\nMoE", "MAPO\nfixed router", "MAPO", "HOVER", "TWIST 2"]
    vals = [84.36, 82.78, 85.21, 85.45, 83.39, 57.85]
    cols = [GREY, ORANGE, GREEN, BLUE, GREY, GREY]
    ax[1].bar(range(6), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.6, f"{v:.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(range(6))
    ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_ylim(50, 92)
    ax[1].set_ylabel("시험 집합(OMOMO) 성공률 (%)")
    ax[1].set_title("표 1(b) 시험 집합 성공률 (한 시드)", fontsize=10.5)
    fig.text(0.01, 0.005, "논문 표 1과 부록 D의 값을 다시 그린 그래프. 시험 집합 모션 수와 시드별 값은 논문에 없다.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "own_success_table1.png", dpi=180)
    plt.close(fig)


def fig_grad():
    r = json.loads((HERE / "toy_algebra_results.json").read_text(encoding="utf-8"))
    d = r["d_gradient_equivalence"]
    scales = [k for k in d if float(k) > 0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = [float(k) for k in scales]
    ax[0].plot(xs, [d[k]["router_cos"] for k in scales], "o-", color=ORANGE, label="라우터 기울기")
    ax[0].plot(xs, [d[k]["expert_cos"] for k in scales], "s-", color=BLUE, label="전문가 기울기")
    ax[0].fill_between(xs, [d[k]["router_cos_min"] for k in scales], [d[k]["router_cos"] for k in scales], color=ORANGE, alpha=0.15)
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.005, 0.01, 0.02, 0.05, 0.1])
    ax[0].set_xticklabels(["0.005", "0.01", "0.02", "0.05", "0.1"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("갱신 크기 (파라미터 잡음 SD)")
    ax[0].set_ylabel("MAPO와 PPO 기울기의 코사인 유사도")
    ax[0].set_title("T1: 갱신이 커질수록 라우터 기울기가 갈라진다", fontsize=10.5)
    kl = r["abc"]["mean_measured_kl_by_param_scale"]
    for k in ("0.01", "0.05"):
        ax[0].annotate(f"측정 KL ≈ {kl[k]:.3f}", (float(k), d[k]["router_cos"]), textcoords="offset points", xytext=(6, -14), fontsize=8)
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    inter = r["abc"]["interaction_over_total_change_by_param_scale"]
    ks = list(inter)
    ax[1].plot([float(k) for k in ks], [inter[k] for k in ks], "o-", color=GREEN)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([0.01, 0.02, 0.05, 0.1])
    ax[1].set_xticklabels(["0.01", "0.02", "0.05", "0.1"])
    ax[1].set_yticks([0.02, 0.05, 0.1, 0.2])
    ax[1].set_yticklabels(["0.02", "0.05", "0.1", "0.2"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("갱신 크기 (파라미터 잡음 SD)")
    ax[1].set_ylabel("상호작용 항 / 평균 변화 크기")
    ax[1].set_title("MAPO가 따로 세지 않는 Σ Δw·Δμ 항", fontsize=10.5)
    fig.text(0.01, 0.005, "무작위 MoE(전문가 6개, 상태 512개, 시행 50회)의 한 번 갱신. PPO 신뢰 영역의 크기는 KL ≈ ε² = 0.04.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "own_toy_grad.png", dpi=180)
    plt.close(fig)


def fig_ppo():
    r = json.loads((HERE / "toy_moe_ppo_results.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    its = list(range(0, r["config"]["iters"], 10))
    for m, c, lab in (("ppo", ORANGE, "PPO"), ("mapo", BLUE, "MAPO"), ("mapo_avg", GREEN, "MAPO (손실 1/2)")):
        ax[0].plot(its, r["perf_N6"][m]["curve_mean"], color=c, label=f"{lab}  최종 {r['perf_N6'][m]['final_err_mean']:.2f}")
    ax[0].set_yscale("log")
    ax[0].set_yticks([2, 3, 5, 10, 20])
    ax[0].set_yticklabels(["2", "3", "5", "10", "20"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("반복")
    ax[0].set_ylabel("목표 행동과의 제곱 오차 (낮을수록 좋음)")
    ax[0].set_title("T2: 전문가 6개, 시드 4개 평균", fontsize=10.5)
    ax[0].legend(frameon=False, fontsize=8)
    mis = r["misestimation_ppo_by_N"]
    ns = list(mis)
    x = range(len(ns))
    ax[1].bar([i - 0.19 for i in x], [mis[n]["over_mean"] * 100 for n in ns], 0.36, color=ORANGE, label="과대 (d_w·d_μ/σ² > ε²)")
    ax[1].bar([i + 0.19 for i in x], [mis[n]["under_mean"] * 100 for n in ns], 0.36, color=BLUE, label="과소 (< -ε²)")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels([f"N = {n}" for n in ns])
    ax[1].set_ylabel("교차항이 ε²를 넘는 상태 비율 (%)")
    ax[1].set_title("PPO 학습 중 비사소한 오추정 비율", fontsize=10.5)
    ax[1].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "한 단계짜리 문맥형 밴딧 토이(휴머노이드 아님). 시드 4개(왼쪽), 2개(오른쪽). 논문 수치의 재현이 아니다.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "own_toy_ppo.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_success()
    fig_grad()
    fig_ppo()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
