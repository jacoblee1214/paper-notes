"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_success_ci.png     그림 4 성공률 라벨 -> 성공 수(n=20)와 Wilson 95% 구간
  own_scaling_fit.png    그림 5 중앙 패널 재판독점, 로그-선형 적합, 논문 식, 대안 곡선과 100k 외삽
  own_toy_scaling.png    토이: 잡음과 점 수에 따른 곡률 검정 검출력과 100k 외삽 오차

입력: analysis/table_check_results.json, analysis/scaling_refit_results.json, analysis/toy_scaling_fit_results.json
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
AN = ROOT / "analysis"
TC = json.loads((AN / "table_check_results.json").read_text(encoding="utf-8"))
SR = json.loads((AN / "scaling_refit_results.json").read_text(encoding="utf-8"))
TOY = json.loads((AN / "toy_scaling_fit_results.json").read_text(encoding="utf-8"))

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

COLORS = {"NoPre": "#b5b3ab", "MidOnly": "#f2c57c", "HumanPre": "#6fb4e6", "HumanPre+Mid": "#d9534f"}
LABELS = {"NoPre": "사전학습 없음", "MidOnly": "중간학습만", "HumanPre": "인간 사전학습", "HumanPre+Mid": "인간 사전학습 + 중간학습"}


def success_ci():
    tasks = ["Shirt", "Card", "Tong", "Bottle", "Syringe"]
    cells = TC["success_cells"]
    ckpts = ["NoPre", "MidOnly", "HumanPre", "HumanPre+Mid"]
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=160)
    w = 0.2
    x = np.arange(len(tasks))
    for j, c in enumerate(ckpts):
        vals, lo, hi = [], [], []
        for t in tasks:
            cell = next(q for q in cells if q["ckpt"] == c and q["task"] == t)
            vals.append(cell["k"] / cell["n"])
            lo.append(cell["k"] / cell["n"] - cell["wilson95"][0])
            hi.append(cell["wilson95"][1] - cell["k"] / cell["n"])
        pos = x + (j - 1.5) * w
        ax.bar(pos, vals, width=w, color=COLORS[c], label=LABELS[c], edgecolor="white")
        ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["셔츠 접기", "카드 분류", "집게 과일", "병뚜껑", "주사기"])
    ax.set_ylabel("성공률 (n=20 가정)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#eee", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    ax.set_title("그림 4(b) 라벨 재구성 성공률과 Wilson 95% 구간 (우리가 그린 그래프)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_success_ci.png")
    plt.close(fig)


def scaling_fit():
    D = np.array(SR["digitized"]["D_thousand_hours"])
    L = np.array(SR["digitized"]["loss"])
    lin = SR["loglinear_fit"]
    grid = np.geomspace(0.8, 150, 200)
    xg = np.log(grid)
    ll = lin["intercept"] + lin["slope"] * xg
    # 적합 신뢰 띠 (평균 응답, 95%)
    from scipy import stats
    x = np.log(D)
    X = np.column_stack([np.ones(5), x])
    beta = np.array([lin["intercept"], lin["slope"]])
    res = L - X @ beta
    s2 = float(res @ res) / 3
    XtXi = np.linalg.inv(X.T @ X)
    Xg = np.column_stack([np.ones(len(xg)), xg])
    se = np.sqrt(s2 * np.einsum("ij,jk,ik->i", Xg, XtXi, Xg))
    t = stats.t.ppf(0.975, 3)
    paper = 0.024 - 0.003 * np.log(grid)
    sat = next(p for p in SR["saturating_profile"] if p["L_inf_fixed"] == 0.0)
    satc = sat["L_inf_fixed"] + sat["A"] * grid ** (-sat["alpha"])

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=160)
    ax.scatter(D, L, s=36, color="#1f5fae", zorder=5, label="그림 5 중앙 판독값 (±0.0001)")
    ax.fill_between(grid, ll - t * se, ll + t * se, color="#1f5fae", alpha=0.12, label="로그-선형 적합 95% 띠")
    ax.plot(grid, ll, color="#1f5fae", lw=1.6, label=f"재적합: L = {lin['intercept']:.4f} {lin['slope']:+.4f} ln D (R²={lin['r2']:.4f})")
    ax.plot(grid, paper, color="#888", lw=1.0, ls="--", label="논문 식 (0.003, 천 시간 단위로 읽음)")
    ax.plot(grid, satc, color="#d9534f", lw=1.0, ls=":", label="포화 곡선 (L∞=0, 가장 나은 비음수 점근)")
    ax.axvspan(20, 150, color="#eee", alpha=0.6, lw=0)
    ax.text(24, 0.0235, "외삽 구간\n(논문 범위 밖)", fontsize=7, color="#555")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 10, 20, 50, 100])
    ax.set_xticklabels(["1k", "2k", "4k", "10k", "20k", "50k", "100k"])
    ax.set_xlabel("인간 데이터 규모 (시간, 로그)")
    ax.set_ylabel("최적 검증 손실 (MSE)")
    ax.set_ylim(0.008, 0.0265)
    ax.grid(color="#eee", lw=0.6)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title("검증 손실 규모 법칙 재적합 (우리가 그린 그래프)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_scaling_fit.png")
    plt.close(fig)


def toy():
    rows = TOY["rows"]
    sigmas = TOY["sigma_values"]
    designs = ["3pts_1_4_20k", "5pts_1_2_4_10_20k", "9pts_log_1_20k"]
    labels = {"3pts_1_4_20k": "3점", "5pts_1_2_4_10_20k": "5점 (현 논문 범위)", "9pts_log_1_20k": "9점"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), dpi=160)
    cols = {"3pts_1_4_20k": "#999", "5pts_1_2_4_10_20k": "#1f5fae", "9pts_log_1_20k": "#2a9d8f"}
    for d in designs:
        y1 = [next(r for r in rows if r["truth"].startswith("T2") and r["design"] == d and r["sigma"] == s)["curvature_test_power_p05"] for s in sigmas]
        y1 = [v if v is not None else np.nan for v in y1]
        if d != "3pts_1_4_20k":  # 3점은 곡률 검정 자유도가 0이라 검출력이 없음
            axes[0].plot(sigmas, y1, marker="o", color=cols[d], label=labels[d])
        y2 = [next(r for r in rows if r["truth"].startswith("T2") and r["design"] == d and r["sigma"] == s)["extrap_100k_rel_err_median"] for s in sigmas]
        axes[1].plot(sigmas, np.array(y2) * 100, marker="o", color=cols[d], label=labels[d])
    axes[0].axhline(0.05, color="#bbb", lw=0.8, ls="--")
    axes[0].set_title("참 곡선이 포화일 때: 곡률 검정 검출력", fontsize=8)
    axes[0].set_ylabel("p<0.05 비율")
    axes[1].axhline(0, color="#bbb", lw=0.8)
    axes[1].set_title("참 곡선이 포화일 때: 로그-선형 100k 외삽 중앙 오차", fontsize=8)
    axes[1].set_ylabel("(예측 - 참)/참, %")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sigmas)
        ax.set_xticklabels(["0.0001", "0.0003", "0.001"])
        ax.set_xlabel("점별 평가 잡음 σ")
        ax.grid(color="#eee", lw=0.6)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle("토이 (우리가 만든 시뮬레이션, 2000회)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_scaling.png")
    plt.close(fig)


if __name__ == "__main__":
    success_ci()
    scaling_fit()
    toy()
    print("wrote figures to", FIG)
