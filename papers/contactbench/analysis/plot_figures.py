"""리포트·게시본 그래프 (CPU, matplotlib). 실행: python3 analysis/plot_figures.py → report/figures/*.png
입력은 table_check_results.json, power_toy_results.json이다. 논문 그림은 그리지 않는다(우리가 그린 그래프만).
팔레트: 시각 정책 #0072B2, 촉각 정책 #A8560B (contact_disassembly와 같은 계열).
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
T = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
P = json.loads((ROOT / "analysis" / "power_toy_results.json").read_text())

for f in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

VIS, TAC = "#0072B2", "#A8560B"
INK, MUTED, GRID = "#1b1b1a", "#55544f", "#dcdad3"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


TASK_KO = {"thread": "너트 나사 체결", "tighten": "육각렌치 조이기", "plug": "충전기 꽂기",
           "door": "문손잡이 돌려 열기", "utensil": "통에서 수저 한 개", "card": "얇은 카드 집기"}
POL_KO = {"DP": "DP (시각)", "PI05": "π0.5 (시각)", "RDP": "RDP (촉각)", "FTP1": "FTP-1 (촉각)"}


# 1) 표 III 셀별 성공률과 Wilson 95% 구간
def fig_table3():
    rows = []
    for task in ["thread", "tighten", "plug", "door", "utensil", "card"]:
        for pol in ["DP", "PI05", "RDP", "FTP1"]:
            key = f"{pol}/{task}"
            if key in T["cells_wilson"]:
                rows.append((task, pol, T["cells_wilson"][key]))
    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=200)
    ylabels = []
    for i, (task, pol, c) in enumerate(rows):
        y = len(rows) - i
        p = 100 * c["p"]
        lo, hi = 100 * c["wilson95"][0], 100 * c["wilson95"][1]
        color = VIS if pol in ("DP", "PI05") else TAC
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="butt")
        ax.plot([p], [y], "o", color=color, ms=5)
        ax.text(101.5, y, f'{c["k"]}/30', va="center", fontsize=7.5, color=MUTED)
        ylabels.append((y, f"{TASK_KO[task]} · {POL_KO[pol]}"))
    ax.set_yticks([y for y, _ in ylabels])
    ax.set_yticklabels([s for _, s in ylabels], fontsize=7.6, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("성공률 (%), 막대는 Wilson 95% 구간 (30회)", color=INK, fontsize=9)
    ax.set_title("표 III 재계산: 칸마다 구간이 넓다. 시각(파랑)과 촉각(주황)의 차이는 과제마다 다르다",
                 fontsize=8.6, color=INK, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "table3_ci.png")
    plt.close(fig)


# 2) 검정력 곡선: 단순 정규 근사 대 Fisher 정확 검정 시뮬레이션
def fig_power():
    n_grid = np.arange(5, 161)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    za = norm.ppf(0.975)
    p1 = 0.5
    cols = {"0.2": "#7B3FA0", "0.3": "#A8560B"}
    sim = P["P1_power"]["simulated_power_p1_0.5_fisher_exact"]
    for d in ("0.2", "0.3"):
        p2 = p1 + float(d)
        pbar = (p1 + p2) / 2
        # 정규 근사 검정력: Φ((√n·Δ − z·√(2 p̄ q̄)) / √(p1q1 + p2q2)), 방법당 n
        power = norm.cdf((np.sqrt(n_grid) * (p2 - p1) - za * np.sqrt(2 * pbar * (1 - pbar)))
                         / np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
        ax.plot(n_grid, power, color=cols[d], lw=1.6, label=f"정규 근사, Δ = {d}")
        xs = [int(k) for k in sim[f"d={d}"]]
        ys = [sim[f"d={d}"][str(k)] for k in xs]
        ax.plot(xs, ys, "o", color=cols[d], ms=4.5, label=f"Fisher 정확 검정 시뮬레이션, Δ = {d}")
    ax.axhline(0.8, color=MUTED, lw=0.8, ls="--")
    ax.axvline(30, color=INK, lw=0.9, ls=":")
    ax.text(31.5, 0.05, "논문: 방법당 30회", fontsize=8, color=INK)
    ax.set_xlabel("방법당 시행 수 n", color=INK, fontsize=9)
    ax.set_ylabel("검정력 (p1 = 0.5, 양측 0.05)", color=INK, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title("30회로는 약 36 pp 이상의 차이만 80% 검정력으로 잡힌다", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "power_curve.png")
    plt.close(fig)


# 3) 초기조건 공유 여부와 학습 실행 분산이 거짓 양성률에 주는 영향
def fig_ic():
    res = P["P2_initial_conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=200, sharey=True)
    for ax, n in zip(axes, (30, 150)):
        labels, vals_r, vals_s = [], [], []
        for s in ("0.0", "0.25", "0.5"):
            labels.append(f"σ={s}")
            vals_r.append(res[f"design=R,sigma_s={s},n={n},delta=0.0"]["fisher_rate"])
            vals_s.append(res[f"design=S,sigma_s={s},n={n},delta=0.0"]["fisher_rate"])
        x = np.arange(len(labels))
        ax.bar(x - 0.18, vals_r, 0.34, color=VIS, label="재추출 초기조건 (Fisher)")
        ax.bar(x + 0.18, vals_s, 0.34, color="#9bbbd9", label="공유 초기조건 (Fisher)")
        ax.axhline(0.05, color=INK, lw=0.9, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(f"방법당 n = {n} (귀무가설: 차이 없음)", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("학습 실행 분산 σ_s (로짓)", color=INK, fontsize=8.5)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("거짓 양성률 (목표 0.05)", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.4, loc="upper left")
    fig.suptitle("초기조건 공유는 거짓 양성을 거의 늘리지 않는다. 학습 실행 분산이 늘린다", fontsize=9,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "ic_fpr.png")
    plt.close(fig)


# 4) 과제 가중치 민감도: 순서가 유지될 확률
def fig_weights():
    W = P["P3_weights"]
    keys = list(W["prob_ordering_holds_dirichlet_weights"].keys())
    ko = {"PI05>DP (6 tasks)": "π0.5 > DP (6과제)", "PI05>DP (4 tasks)": "π0.5 > DP (4과제)",
          "DP>RDP (4 tasks)": "DP > RDP (4과제)", "PI05>FTP1 (4 tasks)": "π0.5 > FTP-1 (4과제)"}
    names = [ko[k] for k in keys]
    d = [W["prob_ordering_holds_dirichlet_weights"][n] for n in keys]
    b = [W["prob_ordering_holds_dirichlet_plus_binomial_noise"][n] for n in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    y = np.arange(len(names))
    ax.barh(y + 0.18, d, 0.34, color=VIS, label="과제 가중치만 무작위 (Dirichlet)")
    ax.barh(y - 0.18, b, 0.34, color="#9bbbd9", label="가중치 + 이항 잡음 (30회)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("순서가 유지될 확률 (2만 번 추출)", color=INK, fontsize=9)
    ax.set_title("평균 순위는 대체로 유지된다. DP 대 RDP는 plug 과제의 가중치에 민감하다",
                 fontsize=8.6, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "weight_sensitivity.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_table3()
    fig_power()
    fig_ic()
    fig_weights()
    print("wrote figures to", FIG)
