"""우리가 그린 그래프를 report/figures/ 에 저장한다 (논문 그림은 쓰지 않는다).

own_ablation_ci.png  : 보고된 성공 횟수와 Wilson 95% 구간, 본 방법 대비 Fisher p (recompute_results.json)
own_throughput.png   : 수집 효율 재계산 (15분 유효 시연 수, 시연 100개에 걸리는 시간)
own_reward_sigma.png : 추종 오차별 보상(σ 선택)과 속도 의존 허용 오차 σ(v)
own_speed_aug.png    : 가변 속도 분포와 재표집 간격(0.02 s 대 0.25 s)에 따른 1초 평균 속도 산포
own_toy_chunk.png    : 토이 V1, 덩어리 기준 자세(예정 목표 대 실행 자세)
실행: python3 analysis/plot_figures.py   (numpy, scipy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

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
INK, ACC, GREY, ORG, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#c8641e", "#2e8b57"
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_ablation():
    cells = REC["K1_cells"]
    abl = {r["label"]: r for r in REC["K2_ablations"]}
    order = [
        ("단일 무릎 줍기: 본 방법", "knee_humi", None),
        ("  인간 개입(IK 미리보기) 없음", "knee_no_hil", "knee: 인간 개입 없음"),
        ("  EE만 지정", "knee_ee_only", "knee: 전신 명세 없음(EE-only)"),
        ("칼 뽑기: 본 방법", "sword_humi", None),
        ("  가변 속도 없음", "sword_no_varspeed", "sword: 가변 속도 없음"),
        ("던지기: 본 방법", "toss_humi", None),
        ("  실행 자세를 기준으로", "toss_executed_ref", "toss: 실행 자세 참조"),
        ("걸으며 닦기: EE+골반(본 방법)", "walk_ee_pelvis", None),
        ("  EE만", "walk_ee_only", "walk: EE-only"),
        ("  EE+골반+발", "walk_ee_pelvis_feet", "walk: EE+pelvis+feet"),
        ("  절대 골반 추적", "walk_absolute_pelvis", "walk: 절대 골반 추적"),
        ("쪼그려 집기: 본 방법(부록 C.1)", "squat_in_domain_C1", None),
        ("  적응형 EE 보상 없음", "squat_no_adaptive_ee", "squat: 적응형 EE 보상 없음"),
        ("쪼그려 집기: 미학습 환경·물체", "squat_gen_unseen", None),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for i, (label, key, abl_key) in enumerate(order):
        c = cells[key]
        y = len(order) - 1 - i
        lo, hi = 100 * c["wilson"][0], 100 * c["wilson"][1]
        p = 100 * c["k"] / c["n"]
        col = ACC if abl_key is None else ORG
        ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o", color=col, capsize=4, ms=6)
        txt = f"{c['k']}/{c['n']}"
        if abl_key:
            pv = abl[abl_key]["fisher_p"]
            txt += f"  p={pv:.3f}" if pv >= 0.001 else ("  p=%.4f" % pv if pv >= 0.0001 else "  p<0.0001")
        ax.text(103, y, txt, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o[0] for o in order][::-1], fontsize=8.8)
    ax.set_xlim(-3, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("성공률 (%), 점은 보고값, 막대는 Wilson 95% 구간")
    ax.axvline(0, color=GREY, lw=0.6)
    clean(ax)
    ax.set_title("보고된 성공 횟수의 불확실성 (파랑: 본 방법, 주황: 절제, p는 본 방법 대비 Fisher 정확 검정)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_ci.png", dpi=170)
    plt.close(fig)


def plot_throughput():
    d = REC["K5_data_efficiency"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    vals = [60, 18]
    ax.bar(["HuMI", "TWIST2"], vals, color=[ACC, GREY], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, str(v), ha="center", fontsize=10)
    ax.set_ylabel("유효 시연 수 (15분 세션, 칼 뽑기)")
    ax.set_title(f"한 세션 대 한 세션 → {d['v3_valid_demos']['ratio']:.2f}배 (정확 구간 1.9–6.0배, 단 세션 1회씩)", fontsize=9)
    clean(ax)
    ax = axes[1]
    labels = ["칼 뽑기\n(세션 속도 환산)", "양손 집어 놓기\n(TWIST2 논문 수치)"]
    humi = [100 / (60 / 15), d["bimanual_pnp_100demos"]["humi_min"]]
    tw = [100 / (18 / 15), d["bimanual_pnp_100demos"]["twist2_min"]]
    x = np.arange(2)
    ax.bar(x - 0.18, humi, 0.34, color=ACC, label="HuMI")
    ax.bar(x + 0.18, tw, 0.34, color=GREY, label="TWIST2")
    for xi, h, t in zip(x, humi, tw):
        ax.text(xi - 0.18, h + 1.5, f"{h:.1f}", ha="center", fontsize=9)
        ax.text(xi + 0.18, t + 1.5, f"{t:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("유효 시연 100개에 걸리는 시간 (분)")
    ax.set_title("과제에 따라 배율이 3.3배(환산)와 2.9배, HuMI 쪽 속도는 4.0 대 15.9개/분", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_throughput.png", dpi=170)
    plt.close(fig)


def plot_reward():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    e = np.linspace(0, 0.10, 400)
    ax = axes[0]
    for sigma, col, lab in [(0.3, GREY, "전신 위치 σ=0.3 m"), (0.1, ACC, "EE σ_max=0.1 m"), (0.01, ORG, "EE σ_min=0.01 m")]:
        ax.plot(100 * e, np.exp(-e ** 2 / sigma ** 2), color=col, lw=2, label=lab)
    ax.set_xlabel("위치 오차 (cm)")
    ax.set_ylabel("보상 exp(-e²/σ²)")
    ax.set_title("σ가 신호를 정한다: 5 cm에서 0.97 / 0.78 / 1e-11", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5)
    clean(ax)
    ax = axes[1]
    v = np.linspace(0, 0.2, 400)
    a = np.clip((v - 0.05) / 0.05, 0, 1)
    ax.plot(v, 100 * (0.01 + 0.09 * a), color=ACC, lw=2, label="σ_p(v) [cm]")
    ax2 = ax.twinx()
    ax2.plot(v, 5 + 15 * a, color=ORG, lw=2, ls="--", label="σ_θ(v) [deg]")
    ax.set_xlabel("기준 EE 속도 (m/s)")
    ax.set_ylabel("위치 허용 오차 σ_p (cm)", color=ACC)
    ax2.set_ylabel("회전 허용 오차 σ_θ (도)", color=ORG)
    ax.set_title("속도 의존 허용 오차: 0.05 m/s 이하에서 최소, 0.10 m/s 이상에서 최대", fontsize=9.5)
    clean(ax)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_reward_sigma.png", dpi=170)
    plt.close(fig)


def plot_speed():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    r = np.linspace(0.25, 1.25, 300)
    for sd, col in [(0.3, GRN), (1.0, ACC)]:
        dist = stats.truncnorm((0.25 - 1) / sd, (1.25 - 1) / sd, loc=1.0, scale=sd)
        ax.plot(r, dist.pdf(r), color=col, lw=2, label=f"σ={sd:g} (평균 {dist.mean():.2f})")
    ax.set_xlabel("재생 속도 배율 s")
    ax.set_ylabel("밀도")
    ax.set_title("σ=1.0이면 속도 1.0 근방 시간이 37%뿐 (s≥0.9)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5)
    clean(ax)
    ax = axes[1]
    v = REC["K9_variable_speed"]
    items = [("0.02 s마다 독립 표집\n(논문 부록 E)", v["window_1s_avg_rate_resample_0.02s"]["sd"], GREY),
             ("0.25 s 격자 보간\n(공식 코드)", v["window_1s_avg_rate_resample_0.25s_interp"]["sd"], ACC)]
    ax.bar([i[0] for i in items], [i[1] for i in items], color=[i[2] for i in items], width=0.5)
    for i, it in enumerate(items):
        ax.text(i, it[1] + 0.004, f"{it[1]:.3f}", ha="center", fontsize=9.5)
    ax.set_ylabel("1초 구간 평균 속도의 표준편차 (σ=1.0)")
    ax.set_title("같은 σ라도 에피소드 안 속도 다양성이 3배 다르다", fontsize=9.5)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_speed_aug.png", dpi=170)
    plt.close(fig)


def plot_toy():
    import sys
    sys.path.insert(0, str(ANA))
    import toy_chunk_anchor as toy  # noqa: E402
    res = json.loads((ANA / "toy_chunk_anchor_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7))
    ax = axes[0]
    tau = 0.05
    t, xa, ua, _ = toy.simulate(tau, "scheduled")
    _, xb, ub, _ = toy.simulate(tau, "executed")
    ax.plot(t, toy.demo(t), color=GREY, lw=2, label="시연 d(t)")
    ax.plot(t, xa, color=ACC, lw=2, label="예정 목표 기준 (HuMI) 실행")
    ax.plot(t, xb, color=ORG, lw=2, label="실행 자세 기준 실행")
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("앞뒤 위치 (m)")
    ax.set_xlim(0, 3)
    ax.set_title(f"추종 지연 τ={tau*1000:.0f} ms, 덩어리 5 Hz", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.3)
    clean(ax)
    ax = axes[1]
    taus = [0.02, 0.05, 0.1, 0.2]
    a = [res[f"tau_{x}"]["scheduled"]["peak_speed_ratio"] for x in taus]
    b = [res[f"tau_{x}"]["executed"]["peak_speed_ratio"] for x in taus]
    xs = np.arange(len(taus))
    ax.bar(xs - 0.18, a, 0.34, color=ACC, label="예정 목표 기준")
    ax.bar(xs + 0.18, b, 0.34, color=ORG, label="실행 자세 기준")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x*1000)} ms" for x in taus])
    ax.set_xlabel("추종 지연 τ")
    ax.set_ylabel("시연 대비 최고 속도")
    ax.set_title("지연이 클수록 실행 자세 기준이 더 느려진다", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.3)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_chunk.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_ablation()
    plot_throughput()
    plot_reward()
    plot_speed()
    plot_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
