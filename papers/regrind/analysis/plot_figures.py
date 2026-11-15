"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_real_ci.png     표 2·3 실로봇 성공률과 Wilson 95% 구간 (10 시도)
  own_sim_ci.png      표 1 시뮬레이션 평균과 3 시드 기준 t 95% 구간 (SR과 추적 오차)
  own_toy_prior.png   토이 B: 리타게팅 오차에 따른 잔차 RL 성공률과 처음부터 RL

입력: analysis/recompute_results.json, analysis/toy_retarget_prior_results.json
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
TOY = json.loads((ROOT / "analysis" / "toy_retarget_prior_results.json").read_text(encoding="utf-8"))

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

BLUE, GREY, RED = "#1f5fae", "#8a8a85", "#b03a2e"


def real_ci():
    rows = [r for r in RC["real_table2"]]
    # 표 3 (무작위 초기 배치) 추가
    t3 = {"LEAP-Scissors": (8, 10), "LEAP-Screwdriver": (10, 10), "WUJI-Screwdriver": (9, 10)}
    labels, ks, ns, colors, lo, hi, marks = [], [], [], [], [], [], []
    for task in ["LEAP-Scissors", "LEAP-Screwdriver", "WUJI-Scissors", "WUJI-Screwdriver"]:
        for r in rows:
            if r["task"] == task and r["method"] in ("REGRIND", "DexMachina", "Mink IK + RL"):
                labels.append(f"{task}\n{r['method']}")
                ks.append(r["sr"] * 100)
                lo.append(r["wilson95"][0] * 100)
                hi.append(r["wilson95"][1] * 100)
                colors.append(BLUE if r["method"] == "REGRIND" else GREY)
                marks.append(f"{r['k']}/{r['n']}")
        if task in t3:
            k, n = t3[task]
            labels.append(f"{task}\nREGRIND (무작위 배치)")
            ks.append(k / n * 100)
            lo.append(0)
            hi.append(0)
            colors.append("#5a8fd6")
            marks.append(f"{k}/{n}")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(labels))[::-1]
    for yi, m, l, h, c, mk, lab in zip(y, ks, lo, hi, colors, marks, labels):
        if l == 0 and h == 0 and "무작위" in lab:
            ax.plot(m, yi, "o", color=c, ms=7)
            ax.text(m + 2, yi, mk, va="center", fontsize=8)
            continue
        ax.plot([l, h], [yi, yi], color=c, lw=2)
        ax.plot(m, yi, "o", color=c, ms=7)
        ax.text(max(h, m) + 2, yi, mk, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-2, 115)
    ax.set_xlabel("성공률 (%)  ·  선: Wilson 95% 구간, 10 시도")
    ax.axvline(0, color="#ddd", lw=0.5)
    ax.set_title("실로봇 성공률 (논문 표 2와 표 3 재계산)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def sim_ci():
    tc = 4.302652729911275  # t_{0.975, df=2}
    tasks = ["LEAP-Scissors", "LEAP-Screwdriver", "WUJI-Scissors", "WUJI-Screwdriver"]
    data = {(e["task"], e["metric"], e["REGRIND"] is not None): e for e in RC["sim_table1"]}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, metric, title in zip(axes, ["sr", "err"], ["성공률 (%)", "객체 추적 오차 (mm, 낮을수록 좋음)"]):
        x = np.arange(len(tasks))
        for off, meth, c in [(-0.13, "REGRIND", BLUE), (0.13, "DexMachina", GREY)]:
            means, hw = [], []
            for t in tasks:
                e = data[(t, metric, True)]
                means.append(e[meth]["mean"])
                hw.append(e[meth]["ci95_halfwidth"])
            ax.errorbar(x + off, means, yerr=hw, fmt="o", color=c, capsize=3, label=meth, ms=5)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("-", "\n") for t in tasks], fontsize=8)
        ax.set_title(title + "  ·  3 시드 평균, t 95% 구간", fontsize=9)
        if metric == "err":
            ax.set_ylim(-5, 100)  # t 구간의 위쪽 끝이 100 mm를 넘는 경우 잘린다(그림 설명에 적음)
        else:
            ax.set_ylim(-5, 105)  # t 구간의 아래쪽 끝이 0 아래로 내려가는 경우가 있다
        ax.grid(axis="y", color="#eee")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png", dpi=170)
    plt.close(fig)


def toy_prior():
    summ = TOY["summary"]
    eps = [float(e) for e in summ["residual_final_success"].keys()]
    keys = list(summ["residual_final_success"].keys())
    succ = [summ["residual_final_success"][k]["mean"] * 100 for k in keys]
    succ_sd = [summ["residual_final_success"][k]["sd"] * 100 for k in keys]
    refe = [summ["residual_ref_keypoint_err"][k]["mean"] for k in keys]
    tail = [summ["residual_final_tail_err"][k]["mean"] for k in keys]
    s_succ = summ["scratch_final_success"]["mean"] * 100
    s_tail = summ["scratch_final_tail_err"]["mean"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    ax.errorbar(refe, succ, yerr=succ_sd, fmt="o-", color=BLUE, capsize=3, label="잔차 RL (리타게팅 기준 사용)")
    ax.axhline(s_succ, color=GREY, ls="--", label="처음부터 RL (기준 없음)")
    ax.axvline(TOY["config"]["SUCCESS_TAU"], color=RED, lw=0.8, ls=":")
    ax.text(TOY["config"]["SUCCESS_TAU"] + 0.01, 60, "성공 임계 τ=0.1", color=RED, fontsize=7)
    ax.set_xlabel("리타게팅 키포인트 오차 (평균 거리, 토이 단위)")
    ax.set_ylabel("성공률 (%) · 10 시드, 평균 ± sd")
    ax.set_ylim(-5, 105)
    ax.set_title("성공률은 기준 오차가 임계 근처에서 떨어짐", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    ax.plot(refe, tail, "o-", color=BLUE, label="잔차 RL 최종 추적 오차 (사람 기준)")
    ax.axhline(s_tail, color=GREY, ls="--", label="처음부터 RL")
    ax.plot([0, max(refe)], [0, max(refe)], color="#bbb", lw=0.8, ls=":", label="y = x")
    ax.set_xlabel("리타게팅 키포인트 오차")
    ax.set_ylabel("마지막 절반의 평균 오차 (사람 시연 기준)")
    ax.set_title("최종 오차는 기준 오차를 따라감", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("토이 B (numpy, 1차 관절 + 2층 비볼록 키포인트 사상, 10 시드)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_prior.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    real_ci()
    sim_ci()
    toy_prior()
    print("wrote figures to", FIG)
