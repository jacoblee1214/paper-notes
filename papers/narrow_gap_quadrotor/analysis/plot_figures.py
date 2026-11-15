"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다(공개본 포함).

  own_toy_horizon.png   토이 G: 샘플당 PW·SF 분산 대 지평 T (θ0, 학습된 정책), 지시함수 기울기 0 비율
  own_toy_beta.png      토이 G: 온도 β에 따른 힌지 근사 오차, 기울기 크기, 충돌 항만 학습한 성공률
  own_trial_ci.png      논문 보고 시행 수의 Wilson 95% 구간 (시뮬 1000, 하이-피델리티 100, 실물 10)
  own_timing_basis.png  Table 3 시간 비율과 공개 로그의 단일 시드 비율 (exec 대 전체 벽시계)

입력: analysis/toy_gap_pathwise_results.json, analysis/recompute_results.json
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
TY = json.loads((ROOT / "analysis" / "toy_gap_pathwise_results.json").read_text(encoding="utf-8"))
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

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


def horizon():
    A = TY["A_horizon_sweep"]
    rows = A["rows"]
    tr = A["trained_policy"]["rows"]
    T = [r["T"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].loglog(T, [r["pw_var_trace"] for r in rows], "o-", color=BLUE, label="PW, θ0 (실패 많음)")
    ax[0].loglog(T, [r["sf_var_trace_single_rollout"] for r in rows], "s-", color=RED, label="SF, θ0")
    ax[0].loglog([r["T"] for r in tr], [r["pw_var_trace"] for r in tr], "o--", color=BLUE, alpha=0.6, label="PW, 학습된 정책")
    ax[0].loglog([r["T"] for r in tr], [r["sf_var_trace_single_rollout"] for r in tr], "s--", color=RED, alpha=0.6, label="SF, 학습된 정책")
    ax[0].set_xlabel("지평 T (스텝, 1/60 s)")
    ax[0].set_ylabel("샘플당 기울기 분산 (합)")
    ax[0].set_title("(a) 한 롤아웃 추정량의 분산")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(T, [100 * r["coll_only_smooth_nonzero_fraction"] for r in rows], "o-", color=BLUE, label="부드러운 충돌 항: 0 아닌 기울기 비율")
    ax[1].plot(T, [100 * r["coll_only_hard_nonzero_fraction"] for r in rows], "s-", color=RED, label="지시함수: 0 아닌 기울기 비율 (FD)")
    ax[1].set_xlabel("지평 T (스텝)")
    ax[1].set_ylabel("샘플 비율 (%)")
    ax[1].set_ylim(-5, 105)
    ax[1].set_title("(b) 충돌 항만 미분했을 때")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle("토이 G: 2차원 이중 적분기 + 움직이는 틈 (우리가 만든 numpy 토이)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_horizon.png", dpi=170)
    plt.close(fig)


def beta():
    rows = [r for r in TY["B_temperature"]["beta_rows"] if r["beta_per_m"] is not None]
    hard = [r for r in TY["B_temperature"]["beta_rows"] if r["beta_per_m"] is None][0]
    b = [r["beta_per_m"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].loglog(b, [r["coll_cost_abs_dev_from_hinge_mean"] for r in rows], "o-", color=BLUE)
    ax[0].set_xlabel("softplus 온도 β (1/m)")
    ax[0].set_ylabel("|부드러운 충돌 비용 - 힌지 극한| 평균")
    ax[0].set_title("(a) 근사 오차는 β와 함께 줄어듦")
    ax[0].grid(alpha=0.3, which="both")
    labels = [f"β={int(x)}" for x in b] + ["지시함수"]
    means = [r["coll_only"]["success_pct_mean"] for r in rows] + [hard["coll_only_zero_grad"]["success_pct_mean"]]
    sds = [r["coll_only"]["success_pct_sd"] for r in rows] + [hard["coll_only_zero_grad"]["success_pct_sd"]]
    fd = [r["coll_only"]["success_pct_mean"] for r in rows] + [hard["coll_only"]["success_pct_mean"]]
    x = range(len(labels))
    ax[1].bar(x, means, yerr=sds, color=[BLUE] * len(rows) + [GREY], capsize=4, label="a.e. 기울기 0 (지시함수)")
    ax[1].scatter([len(rows)], [fd[-1]], color=RED, zorder=5, label="지시함수, 중심차분 (FD)")
    ax[1].axhline(TY["B_temperature"]["initial_theta0_success_pct"], color=GREY, ls="--", lw=1, label="초기 정책 θ0")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("성공률 (%), 3 시드 평균 ± SD")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("(b) 충돌 항만 학습 (150 반복)")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("토이 G: 온도 β와 지시함수 대조군 (우리가 만든 numpy 토이)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_beta.png", dpi=170)
    plt.close(fig)


def trial_ci():
    items = [
        ("시뮬 (학습 분포)\nvision-moving", RC["final_sim_trials"]["vision_moving"]["rate_pct"], RC["final_sim_trials"]["vision_moving"]["wilson95_pct"], "1000 롤아웃"),
        ("하이-피델리티 시뮬\n(Table 2)", RC["hifi_sim"]["success"], RC["hifi_sim"]["wilson95_pct"], "100 롤아웃"),
        ("실물, 채점 10회", RC["real_world"]["scored"]["rate_pct"], RC["real_world"]["scored"]["wilson95_pct"], "10회"),
        ("실물, 12회 중\n2회 제외 시 최저", RC["real_world"]["if_both_excluded_fail"]["rate_pct"], RC["real_world"]["if_both_excluded_fail"]["wilson95_pct"], "12회 가정"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    for i, (lab, rate, ci, note) in enumerate(items):
        y = len(items) - 1 - i
        ax.errorbar(rate, y, xerr=[[rate - ci[0]], [ci[1] - rate]], fmt="o", color=BLUE, capsize=4)
        ax.text(101, y, f"{rate:.1f}% [{ci[0]:.0f}, {ci[1]:.0f}]  {note}", va="center", fontsize=8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items][::-1], fontsize=8)
    ax.set_xlim(30, 100)
    ax.set_xlabel("성공률 (%) 와 Wilson 95% 구간")
    ax.set_title("논문이 보고한 시행 수에서 본 구간 (우리가 계산)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "own_trial_ci.png", dpi=170)
    plt.close(fig)


def timing():
    tb = RC["timing_basis_check_1000env_single_seed"]
    labels = ["정적 과제", "움직이는 과제"]
    t3 = [6.42, 45.53]
    ex = [tb["static"]["ratio_exec_basis"], tb["moving"]["ratio_exec_basis"]]
    gl = [tb["static"]["ratio_global_basis"], tb["moving"]["ratio_global_basis"]]
    x = range(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar([i - w for i in x], t3, w, color=BLUE, label="논문 Table 3 (5시드 평균)")
    ax.bar(list(x), ex, w, color=GREEN, label="공개 로그, exec 시간 기준 (1 시드)")
    ax.bar([i + w for i in x], gl, w, color=GREY, label="공개 로그, 전체 벽시계 기준 (1 시드)")
    for i in x:
        ax.text(i - w, t3[i] + 0.8, f"{t3[i]:.1f}×", ha="center", fontsize=8)
        ax.text(i, ex[i] + 0.8, f"{ex[i]:.1f}×", ha="center", fontsize=8)
        ax.text(i + w, gl[i] + 0.8, f"{gl[i]:.1f}×", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("PPO / BPTT 시간 비율")
    ax.set_title("1000 환경 시간 비율: 논문 값과 공개 로그의 정의별 값 (우리가 계산)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "own_timing_basis.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    horizon()
    beta()
    trial_ci()
    timing()
    print("wrote figures to", FIG)
