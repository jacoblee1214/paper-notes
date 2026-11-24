"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_success.png     : Fig.5 성공률(100회)을 다시 그리고 Wilson 95% 구간과 GenAN 대비 Fisher 검정(Holm 보정) 표시
own_datasize.png    : Fig.6 그림 판독 값 + Wilson 구간(시행 수 100 가정), 1스텝 오차의 그림 간 대조
own_consistency.png : 같은 설정(GenAN, 전체 데이터, H=3, s=1)의 오차를 그림별로 나열 (그림 판독 값)
own_toy.png         : 장난감 K2(질량행렬 조건수 분포)와 K3(토크 손실 대 위치 손실)
실행: python3 plot_figures.py   (numpy, matplotlib; recompute.py, toy_position_vs_torque_loss.py를 먼저 실행)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#c44e3a"
COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
REC = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))
FV = json.loads((HERE / "figure_values.json").read_text(encoding="utf-8"))
TOY = json.loads((HERE / "toy_position_vs_torque_loss_results.json").read_text(encoding="utf-8"))


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_success():
    tasks = ["reacher", "ball_in_cup", "table_tennis"]
    names = {"reacher": "reacher", "ball_in_cup": "ball-in-a-cup", "table_tennis": "table tennis"}
    cfgs = ["GenAN", "Torque GenAN", "No disag.", "No ensemble", "Low action pen."]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), sharey=True)
    rows = {(r["task"], r["cfg"]): r for r in REC["V4_genan_vs_baselines"]}
    full = ["GenAN", "Torque GenAN", "No disag. penalty", "No ensemble", "Low action penalty"]
    data = {
        "reacher": [90, 97, 86, 93, 70], "ball_in_cup": [75, 76, 56, 74, 55], "table_tennis": [96, 75, 82, 83, 54],
    }
    for ax, t in zip(axes, tasks):
        ks = data[t]
        lo = [wilson(k, 100)[0] for k in ks]
        hi = [wilson(k, 100)[1] for k in ks]
        x = np.arange(5)
        ax.bar(x, ks, color=COLORS, width=0.66)
        ax.errorbar(x, ks, yerr=[np.array(ks) - np.array(lo), np.array(hi) - np.array(ks)], fmt="none", ecolor=INK, capsize=3, lw=1)
        for i, k in enumerate(ks):
            ax.text(i, 3, str(k), ha="center", color="white", fontsize=8, fontweight="bold")
            if i > 0:
                p = rows[(t, full[i])]["holm_p"]
                if p < 0.05:
                    ax.text(i, hi[i] + 2, "*" if p >= 0.01 else ("**" if p >= 0.001 else "***"), ha="center", fontsize=10, color=RED)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(names[t], fontsize=10)
        ax.set_ylim(0, 108)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("성공률 (%) (100회, Wilson 95%)")
    fig.suptitle("Fig. 5를 다시 그림: 별표는 GenAN 대비 Fisher 검정 Holm 보정 p (* <0.05, ** <0.01, *** <0.001; 12개 비교)", fontsize=8.5, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "own_success.png", dpi=200)
    plt.close(fig)


def plot_datasize():
    f6 = FV["fig6"]
    sizes = f6["sizes"]
    ens = [round(x) for x in f6["success_ensemble_pct"]]
    sgl = [round(x) for x in f6["success_no_ensemble_pct"]]
    x = np.arange(len(sizes))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    ax = axes[0]
    for vals, col, lab, off in [(ens, ACC, "앙상블", -0.07), (sgl, RED, "단일 모델", 0.07)]:
        lo = [wilson(v, 100)[0] for v in vals]
        hi = [wilson(v, 100)[1] for v in vals]
        ax.errorbar(x + off, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="o-", color=col, capsize=2.5, ms=4, lw=1, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_xlabel("GenAN 학습 궤적 수")
    ax.set_ylabel("reacher 성공률 (%)")
    ax.set_title("Fig. 6b 판독 값 + Wilson 구간 (n=100 가정)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(x, [v * 1e3 for v in f6["pos_err_1step_deg"]], "o-", color="#e08a2c", label="1 step (×10⁻³ °)")
    ax.axhline(REC["V6_genan_1step_deg"]["fig4a"] * 1e3, color=RED, ls="--", lw=1)
    ax.text(3.0, REC["V6_genan_1step_deg"]["fig4a"] * 1e3 + 0.25, "Fig. 4a·8a·9a·10a의 GenAN: 6.8", color=RED, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_xlabel("GenAN 학습 궤적 수")
    ax.set_ylabel("1 step 위치 오차 (×10⁻³ °)")
    ax.set_title("Fig. 6a 1 step 판독 값", fontsize=9)
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    ax.plot(x, f6["pos_err_500step_deg"], "o-", color="#2c8f2c")
    ax.axhline(REC["V6_genan_500step_deg"]["fig4b"], color=RED, ls="--", lw=1)
    ax.text(3.0, REC["V6_genan_500step_deg"]["fig4b"] - 0.9, "Fig. 4b·8b·9b·10b의 GenAN: 4.4", color=RED, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_xlabel("GenAN 학습 궤적 수")
    ax.set_ylabel("500 step 위치 오차 (°)")
    ax.set_title("Fig. 6a 500 step 판독 값", fontsize=9)
    ax.set_ylim(0, 12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_datasize.png", dpi=200)
    plt.close(fig)


def plot_consistency():
    v1 = REC["V6_genan_1step_deg"]
    v5 = REC["V6_genan_500step_deg"]
    labels = ["Fig.4", "Fig.8\n(H=3)", "Fig.9\n(s=1)", "Fig.10\n(R=1)", "Fig.6a\n(2000)", "Fig.6a\n(2500)"]
    k1 = ["fig4a", "fig8a_H3", "fig9a_s1", "fig10a_R1", "fig6a_2000", "fig6a_2500"]
    k5 = ["fig4b", "fig8b_H3", "fig9b_s1", "fig10b_R1", "fig6a_2000", "fig6a_2500"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, vals, ttl, unit, scale in [
        (axes[0], [v1[k] for k in k1], "1 step (2 ms)", "×10⁻³ °", 1e3),
        (axes[1], [v5[k] for k in k5], "500 steps (1 s)", "°", 1.0),
    ]:
        cols = [ACC] * 4 + [RED, RED]
        ax.bar(range(6), [v * scale for v in vals], color=cols, width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v * scale + 0.05 * max(vals) * scale, f"{v * scale:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(6))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"위치 오차 ({unit})")
        ax.set_title(f"같은 설정의 GenAN, {ttl} (그림 판독 값)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_consistency.png", dpi=200)
    plt.close(fig)


def plot_toy():
    k2 = TOY["K2_anisotropy"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    ax = axes[0]
    ax.hist(np.log10(k2["cond_M_list"]), bins=40, color=ACC)
    ax.set_xlabel("log10 조건수 cond(M(q))")
    ax.set_ylabel("자세 개수 (2000개)")
    ax.set_title(f"K2: 질량행렬 조건수 (중앙값 {k2['cond_M_median']:.0f})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.hist(k2["ratio_list"], bins=40, color="#e08a2c")
    ax.set_xlabel("같은 크기 토크 오차가 만드는 위치 오차의 최대/최소 비")
    ax.set_title(f"K2: 방향에 따른 비 (중앙값 {k2['equal_norm_torque_err_to_position_err_ratio_max_over_min_median']:.0f})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    sweep = TOY["K3_toy_training"]["sweep"]
    labels = [f"잡음 {r['noise_deg']:.4g}°\n은닉 {r['hidden']}" for r in sweep]
    pos = [r["ratio_position_over_torque_pos_err_measured"] for r in sweep]
    tq = [r["ratio_position_over_torque_torque_err"] for r in sweep]
    x = np.arange(len(sweep))
    ax.bar(x - 0.18, pos, width=0.36, color=ACC, label="위치 오차 (위치손실/토크손실)")
    ax.bar(x + 0.18, tq, width=0.36, color=RED, label="토크 오차 (위치손실/토크손실)")
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title("K3: 위치 손실로 학습했을 때의 상대 오차", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_success()
    plot_datasize()
    plot_consistency()
    plot_toy()
    print("wrote 4 figures to", FIG)
