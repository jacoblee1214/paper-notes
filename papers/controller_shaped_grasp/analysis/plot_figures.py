"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_ci.png     : 주요 비교의 성공(안정) 비율과 Wilson 95% 구간 (시행 수는 논문 값)
own_seeds.png  : 시드별 안정 비율(주 실행 시드 42 + 추가 시드 4개)과 시드 평균
own_dyn.png    : 실험 2(동역학 손실): 시드별 ≥10프레임 교사 근접 런 횟수, 그리고 지표에 따라 방향이 갈리는 점유율
own_delay.png  : 토이 T1(가정 기반) 사람 지연 대 반사 지연이 허용 힘 구간을 넘는 강성 경계

실행: python3 analysis/plot_figures.py   (numpy, matplotlib. 먼저 recompute.py, toy_delay_window.py를 실행)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recompute import T9, T10, T11, T12, wilson  # noqa: E402

FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GREY, RED, GREEN = "#1f5fae", "#d9822b", "#8a8984", "#b3372f", "#3b8a4f"


def ci():
    rows = [
        ("ACT 반사 데이터", 19, 20, BLUE), ("ACT 수동 데이터", 3, 20, ORANGE), ("ACT 수동 + 중재기", 10, 20, ORANGE),
        ("ACT 접촉 선별 수동", 6, 20, ORANGE),
        ("π0.5 반사 데이터", 19, 20, BLUE), ("π0.5 수동 데이터", 1, 20, ORANGE), ("π0.5 수동 + 중재기", 11, 20, ORANGE),
        ("종이컵 반사 데이터 (n=10)", 8, 10, BLUE), ("종이컵 수동 데이터 (n=10)", 3, 10, ORANGE),
        ("외란 반사 데이터 policy-only (유지)", 11, 20, GREEN), ("외란 반사 데이터 + 중재기 (유지)", 20, 20, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    y = np.arange(len(rows))[::-1]
    for yi, (name, k, n, c) in zip(y, rows):
        lo, hi = wilson(k, n)
        ax.plot([lo, hi], [yi, yi], color=c, lw=3, alpha=0.55)
        ax.plot([100 * k / n], [yi], "o", color=c, ms=6)
        ax.text(101.5, yi, f"{k}/{n}", va="center", fontsize=8.5, color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("비율 (%), 점 = 보고 값, 선 = Wilson 95% 구간")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("주요 조건의 성공 비율 (파랑 반사 데이터, 주황 수동 데이터, 초록 외란 실험)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_ci.png", dpi=170)
    plt.close(fig)


def seeds():
    main = {"ACT 반사": 19, "ACT 수동": 3, "ACT 중재기": 10, "π0.5 반사": 19, "π0.5 수동": 1, "π0.5 중재기": 11,
            "외란 policy-only": 11, "외란 중재기": 20}
    ser = {
        "ACT 반사": [r[0] for r in T9["반사"]], "ACT 수동": [r[0] for r in T9["수동"]], "ACT 중재기": [r[0] for r in T9["중재기"]],
        "π0.5 반사": [r[0] for r in T10["반사"]], "π0.5 수동": [r[0] for r in T10["수동"]], "π0.5 중재기": [r[0] for r in T10["중재기"]],
        "외란 policy-only": T12["policy-only"], "외란 중재기": T12["policy+arbiter"],
    }
    names = list(main)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, nm in enumerate(names):
        vals = [main[nm]] + ser[nm]
        pct = [100 * v / 20 for v in vals]
        xs = i + np.linspace(-0.18, 0.18, len(pct))
        ax.scatter(xs[1:], pct[1:], color=BLUE, s=26, label="추가 시드 4개" if i == 0 else None, zorder=3)
        ax.scatter(xs[:1], pct[:1], color=RED, s=34, marker="D", label="주 실행(시드 42)" if i == 0 else None, zorder=3)
        ax.hlines(np.mean(pct), i - 0.3, i + 0.3, color=GREY, lw=2, label="시드 5개 평균" if i == 0 else None)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5, rotation=20)
    ax.set_ylabel("안정(또는 유지) 비율 (%), 시드당 20회")
    ax.set_ylim(-4, 106)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5, loc="center right")
    ax.set_title("시드별 결과: 순위는 모든 시드에서 같고, 시드 사이 폭은 ±5%p 안팎", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_seeds.png", dpi=170)
    plt.close(fig)


def dyn():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax = axes[0]
    main = {("ACT", "baseline"): 15, ("ACT", "dyn"): 18, ("π0.5", "baseline"): 12, ("π0.5", "dyn"): 16}
    for j, bb in enumerate(("ACT", "π0.5")):
        for tag, c, dx in (("baseline", ORANGE, -0.12), ("dyn", BLUE, 0.12)):
            vals = [main[(bb, tag)]] + [r[0] for r in T11[(bb, tag)]]
            ax.scatter(np.full(len(vals), j + dx) + np.linspace(-0.04, 0.04, len(vals)), vals, color=c, s=24,
                       label=("기준선" if tag == "baseline" else "동역학 손실") if j == 0 else None)
            ax.hlines(np.mean(vals), j + dx - 0.1, j + dx + 0.1, color=GREY, lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ACT", "π0.5"])
    ax.set_ylabel("≥10프레임 교사 근접 런이 있는 에피소드 수 / 20")
    ax.set_ylim(8, 20.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("시드별 개선(주 실행 + 시드 4개)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    labels = ["ACT", "π0.5"]
    onset = [(72.2, 82.7), (78.5, 85.3)]
    full = [(68.7, 54.8), (70.8, 61.2)]
    for j in range(2):
        ax.annotate("", xy=(j - 0.12, onset[j][1]), xytext=(j - 0.12, onset[j][0]),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
        ax.annotate("", xy=(j + 0.12, full[j][1]), xytext=(j + 0.12, full[j][0]),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.plot([], [], color=BLUE, lw=2, label="CQ-onset 3 s 근접 점유율 (본문 표 4)")
    ax.plot([], [], color=RED, lw=2, label="전 구간 CQ 근접 점유율 (부록 F)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(50, 90)
    ax.set_ylabel("점유율 (%), 기준선에서 동역학 손실로")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("같은 정책, 지표에 따라 방향이 다름", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_dyn.png", dpi=170)
    plt.close(fig)


def delay():
    R = json.loads((HERE / "toy_delay_window_results.json").read_text(encoding="utf-8"))
    ks = np.array(R["k_N_per_mm"])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(ks, [100 * v for v in R["사람_구간내_비율"]], "o-", color=ORANGE, label="사람 루프 (지연 중앙값 0.35 s, 가정)")
    ax.plot(ks, [100 * v for v in R["반사_구간내_비율"]], "s-", color=BLUE, label="25 Hz 반사 (지연 60 ms, 가정)")
    ax.axvline(R["경계_강성_사람_중앙값(N/mm)"], color=ORANGE, ls=":", lw=1)
    ax.axvline(R["경계_강성_반사(N/mm)"], color=BLUE, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k:g}" for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("접촉 강성 k (N/mm, 논문에 없음, 가정 범위)")
    ax.set_ylabel("목표 구간 [0.05, 0.5] N 안에 멈춘 비율 (%)")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("토이: 지연 × 속도 × 강성이 구간 폭을 넘는 경계 (닫힘 속도 3.75 mm/s 가정)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_delay.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    ci()
    seeds()
    dyn()
    delay()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
