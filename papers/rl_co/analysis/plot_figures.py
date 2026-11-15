"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_real_ci.png     표 1 과제별 성공률과 Wilson 95% 구간 (2 세션 합산, OpenVLA와 π0.5)
  own_mix_toy.png     토이 A: 실제 앵커 가중치 β에 따른 실제 오차, β*와 시뮬레이션 격차의 관계
  own_eval_toy.png    토이 B: 세션 "±"는 시드 변동을 못 잡고, 거짓 양성률은 시드 변동에 따라 올라감

입력: analysis/recompute_results.json, analysis/toy_mixing_results.json, analysis/toy_eval_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
MIX = json.loads((ROOT / "analysis" / "toy_mixing_results.json").read_text(encoding="utf-8"))
EVAL = json.loads((ROOT / "analysis" / "toy_eval_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

BLUE, GREY, RED = "#1f5fae", "#8a8a85", "#b03a2e"
COL = {"Real-only": GREY, "SFT": BLUE, "RL-Co": RED}
TASKS = ["PnP", "Push", "OD", "CD"]
TASK_KO = {"PnP": "Pick and Place", "Push": "Push Cube", "OD": "Open Drawer", "CD": "Close Drawer"}


def real_ci():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, (vla, title) in zip(axes, [("OpenVLA", "OpenVLA"), ("pi05", "π0.5")]):
        pc = RC["pooled_counts"]
        for j, method in enumerate(["Real-only", "SFT", "RL-Co"]):
            for i, t in enumerate(TASKS):
                cell = pc[f"{vla}|{t}"][method]
                k, n = cell["k"], cell["n"]
                lo, hi = cell["wilson95"]
                x = i + (j - 1) * 0.22
                ax.errorbar(x, k / n * 100, yerr=[[k / n * 100 - lo * 100], [hi * 100 - k / n * 100]],
                            fmt="o", color=COL[method], ms=5, capsize=2.5, lw=1.3,
                            label=method if i == 0 else None)
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([TASK_KO[t].replace(" ", "\n", 1) for t in TASKS], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-3, 105)
        ax.grid(axis="y", ls="--", alpha=0.35)
    axes[0].set_ylabel("성공률 (%)")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("표 1 재계산: 과제별 2 세션 합산 성공률과 Wilson 95% 구간", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def mix_toy():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax = axes[0]
    cases = [c for c in MIX["cases"] if c["N_real"] == 20]
    colors = ["#333333", BLUE, "#2a9d8f", RED]
    for c, col in zip(cases, colors):
        beta = np.array(c["curve_beta"])
        ax.plot(beta, c["curve_E_real"], color=col, lw=1.6, label=f"시뮬 격차 G={c['gap']}")
        ax.plot([c["beta_star_theory"]], [c["E_star_theory"]], "o", color=col, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("실제 앵커 가중치 β (로그 축)")
    ax.set_ylabel("실제 과제 오차 (차원당, 낮을수록 좋음)")
    ax.set_title("실제 N=20 데모, 점 = 해석식 β*", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(ls="--", alpha=0.35)

    ax = axes[1]
    gaps = np.linspace(0, 2.0, 60)
    for n_r, col in [(20, BLUE), (50, RED)]:
        bs = [(n_r * (g ** 2 / (20 * 1.0) + 1 / 1000)) for g in gaps]
        ax.plot(gaps, bs, color=col, lw=1.6, label=f"해석식, N_real={n_r}")
        mc = [c for c in MIX["cases"] if c["N_real"] == n_r]
        ax.plot([c["gap"] for c in mc], [c["beta_star_mc_argmin"] for c in mc], "o", color=col, ms=5,
                label=f"몬테카를로 argmin, N_real={n_r}")
    ax.set_yscale("log")
    ax.set_xlabel("시뮬레이션-실제 차이 G (벡터 노름)")
    ax.set_ylabel("최적 β*")
    ax.set_title("β*는 격차와 실데이터 수에 따라 커짐", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(ls="--", alpha=0.35)
    fig.suptitle("토이 A: 실제 앵커 가중치가 격차와 분산을 맞바꾼다 (numpy, D=20, N_sim=1000)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_mix_toy.png", dpi=170)
    plt.close(fig)


def eval_toy():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    pm = EVAL["paper_pm_vs_seed"]
    for n_s, col in [(10, "#999999"), (20, BLUE), (32, RED)]:
        rows = [r for r in pm if r["n_per_session"] == n_s]
        ax.plot([r["seed_sd"] * 100 for r in rows], [r["mean_paper_pm"] for r in rows], "o-", color=col,
                lw=1.5, label=f"논문식 ± (세션 {n_s} 시도)")
    ax.plot([0, 10], [0, 10], "k--", lw=1, label="시드 간 참 SD = x축")
    ax.set_xlabel("가정한 학습 시드 간 SD (%p)")
    ax.set_ylabel("%p")
    ax.set_title("세션 ±는 시드 변동을 반영하지 않음", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(ls="--", alpha=0.35)

    ax = axes[1]
    t1 = EVAL["type1"]
    for n_s, col in [(10, "#999999"), (20, BLUE), (32, RED)]:
        rows = [r for r in t1 if r["n_per_session"] == n_s]
        ax.plot([r["seed_sd"] * 100 for r in rows], [r["false_positive_rate"] * 100 for r in rows], "o-",
                color=col, lw=1.5, label=f"세션 {n_s} 시도 (합 {2 * n_s})")
    ax.axhline(5, color="black", ls="--", lw=1)
    ax.set_xlabel("가정한 학습 시드 간 SD (%p)")
    ax.set_ylabel("거짓 양성률 (%) (차이 0, 양측 0.05)")
    ax.set_title("이항 가정의 Fisher 검정은 시드 변동에서 과잉 기각", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(ls="--", alpha=0.35)
    fig.suptitle("토이 B: 1 체크포인트 x 2 세션 설계의 한계 (시뮬레이션 3000회)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_eval_toy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    real_ci()
    mix_toy()
    eval_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
