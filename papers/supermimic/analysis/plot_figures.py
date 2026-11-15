"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다(공개본 포함).

  own_failures.png       Table A.8 실패 수와 Clopper-Pearson 95% 구간 (SONIC 179, SM 1 GPU 25, SM 2 GPU 0)
  own_failure_groups.png  Table A.8 그룹별 실패 수 (SONIC 대 SuperMimic 3 h)
  own_toy_gradient.png   토이 S: 샘플당 분산(로그)과 모델 기울기 상대 편향 대 지평 H
  own_toy_compound.png   토이 S: 모델 롤아웃 오차의 누적 (c = 1, H = 64)

입력: analysis/recompute_results.json, analysis/toy_bptt_vs_pg_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TY = json.loads((ROOT / "analysis" / "toy_bptt_vs_pg_results.json").read_text(encoding="utf-8"))

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
BLUE, RED, GREY, GREEN = "#1f5fae", "#c0392b", "#8a8a85", "#2e8b57"


def failures():
    f = RC["failures"]
    keys = ["sonic_released", "supermimic_1gpu_3h", "supermimic_2gpu_9p8h"]
    labels = ["SONIC\n(공개 체크포인트)", "SuperMimic\n1 GPU 3 h", "SuperMimic\n2 GPU 9.8 h"]
    rates = [f[k]["rate_pct"] for k in keys]
    lo = [r - f[k]["clopper_pearson95_pct"][0] for r, k in zip(rates, keys)]
    hi = [f[k]["clopper_pearson95_pct"][1] - r for r, k in zip(rates, keys)]
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=160)
    ax.bar(labels, rates, color=[GREY, BLUE, BLUE], width=0.55)
    ax.errorbar(range(3), rates, yerr=[lo, hi], fmt="none", ecolor="black", capsize=5)
    counts = [f[k]["failures"] for k in keys]
    for i, (r, c) in enumerate(zip(rates, counts)):
        ax.text(i, r + hi[i] + 0.006, f"{c}건", ha="center", fontsize=9)
    ax.set_ylabel("실패율 (%)")
    ax.set_title("실패 모션 비율 (129,514 모션, Clopper-Pearson 95% 구간)", fontsize=10)
    ax.set_ylim(0, 0.2)
    ax.text(0.99, 0.95, "구간은 모션을 독립 표본으로 볼 때의 값\n시드 변동은 포함하지 않음", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#55544f")
    fig.tight_layout()
    fig.savefig(FIG / "own_failures.png")
    plt.close(fig)


def failure_groups():
    groups = ["측면 구르기\n(22)", "뒤집기\n(44)", "불 속 회전\n(20)", "다리 교차\n(10)",
              "눕기·일어서기\n(99)", "넓이뛰기\n(24)", "춤\n(8)", "그 밖\n(129,287)"]
    sonic = [22, 44, 3, 9, 28, 15, 8, 50]
    sm1 = [2, 1, 3, 0, 8, 0, 0, 11]
    import numpy as np
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    ax.bar(x - 0.2, sonic, 0.4, color=GREY, label="SONIC (공개 체크포인트)")
    ax.bar(x + 0.2, sm1, 0.4, color=BLUE, label="SuperMimic 1 GPU 3 h")
    ax.set_xticks(x, groups, fontsize=7.5)
    ax.set_ylabel("실패 모션 수")
    ax.set_title("그룹별 실패 수 (Table A.8 값; 괄호는 모션 수)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.99, 0.95, "모든 그룹에서 SuperMimic ≤ SONIC\n(불 속 회전은 같음)", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#55544f")
    fig.tight_layout()
    fig.savefig(FIG / "own_failure_groups.png")
    plt.close(fig)


def toy_gradient():
    rows = TY["gradient_rows"]
    Hs = sorted({r["H"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=160)
    ax = axes[0]
    pw = [next(r for r in rows if r["H"] == H and r["c"] == 1.0)["var_trace_pathwise_true"] for H in Hs]
    pg = [next(r for r in rows if r["H"] == H and r["c"] == 1.0)["var_trace_pg"] for H in Hs]
    ax.plot(Hs, pw, "o-", color=BLUE, label="경로 미분 (진짜 시스템)")
    ax.plot(Hs, pg, "s-", color=RED, label="정책 기울기 (REINFORCE)")
    ax.set_yscale("log")
    ax.set_xlabel("지평 H (스텝)")
    ax.set_ylabel("샘플당 분산 합 (log)")
    ax.set_title("(a) 분산", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax = axes[1]
    for c, color in zip([0.5, 1.0, 2.0], [GREEN, BLUE, RED]):
        y = [next(r for r in rows if r["H"] == H and r["c"] == c)["rel_bias_norm"] * 100 for H in Hs]
        ax.plot(Hs, y, "o-", color=color, label=f"모델 오차 c = {c:g}")
    ax.set_xlabel("지평 H (스텝)")
    ax.set_ylabel("기울기 상대 편향 (%)")
    ax.set_title("(b) 모델 기울기의 편향", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("토이 S: 학습된 모델 기울기 대 모델 없는 기울기 (numpy, 단일 프로세스)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_gradient.png")
    plt.close(fig)


def toy_compound():
    cmp = TY["compounding_c1_H64"]
    steps = cmp["steps_index"]
    fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=160)
    ax.plot(steps, cmp["mean_abs_err_p"], "o-", color=BLUE, label="위치 오차 (모델 - 진짜)")
    ax.plot(steps, cmp["mean_abs_err_v"], "s-", color=RED, label="속도 오차 (모델 - 진짜)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.set_xlabel("롤아웃 스텝 t (0.02 s 간격)")
    ax.set_ylabel("표본 평균 절대 오차")
    ax.set_title("모델 롤아웃 오차의 누적 (c = 1, 5,000 표본)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_compound.png")
    plt.close(fig)


if __name__ == "__main__":
    failures()
    failure_groups()
    toy_gradient()
    toy_compound()
    print("wrote figures to", FIG)
