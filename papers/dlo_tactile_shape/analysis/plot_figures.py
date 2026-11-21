"""우리가 만든 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 여기서 그리지 않는다).

own_t1_episode.png : 표 1 모델별 평균 ± SD(프레임 기준)와, 에피소드 4개로 p<0.05가 되는 상한(에피소드별 차이 SD)
own_t2_sensor.png  : 표 2 센서 제거 조건별 평균 ± SD와 전체 모델 기준선
실행: python3 analysis/plot_figures.py   (matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
입력: analysis/recompute_results.json (recompute.py가 만든 값)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

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
INK, ACC, GREY, WARN = "#1b1b1a", "#1f5fae", "#9a9890", "#b5562b"

# 논문 Table I (RMSE 열 평균, SD)
T1 = [("Pose-Only", 55.00, 22.00), ("Pose + Dir.", 50.63, 24.25), ("ResNet-18\n+Cross-Attn", 36.00, 20.00),
      ("w/o B-spline", 35.01, 16.62), ("w/o L_weighted", 37.00, 20.90), ("Proposed", 30.52, 15.22)]


def fig_t1(res):
    rows = [r for r in res["t1_pairwise"] if r["metric"] == "RMSE(mean L2)" and r["a"] == "Proposed"]
    thr = {r["b"]: r["episode_sd_threshold_mm"] for r in rows}
    names = {"Pose-Only (MLP)": "Pose-Only", "Pose + Direction": "Pose + Dir.",
             "ResNet-18 + Cross-Attn": "ResNet-18\n+Cross-Attn", "Proposed w/o B-spline": "w/o B-spline",
             "Proposed w/o L_weighted": "w/o L_weighted"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(T1))
    for i, (lab, m, s) in enumerate(T1):
        c = WARN if lab == "Proposed" else ACC
        ax1.errorbar(m, i, xerr=s, fmt="o", color=c, ecolor=GREY, capsize=3, ms=5)
        ax1.text(m + s + 1.2, i, f"{m:.1f} ± {s:.1f}", va="center", fontsize=7.5, color=INK)
    ax1.set_yticks(y, [t[0] for t in T1], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("RMSE (mm), 평균 ± SD (표본 기준 SD, 시험 프레임 21,853)", fontsize=8)
    ax1.set_xlim(0, 85)
    ax1.set_title("표 1 값 (논문 원문)", fontsize=9, color=INK, loc="left")
    ax1.grid(axis="x", color="#e5e3dc", lw=0.6)
    ax1.spines[["top", "right"]].set_visible(False)

    keys = ["Pose-Only (MLP)", "Pose + Direction", "ResNet-18 + Cross-Attn", "Proposed w/o B-spline", "Proposed w/o L_weighted"]
    vals = [thr[k] for k in keys]
    labs = [names[k] for k in keys]
    ax2.barh(np.arange(len(keys)), vals, color=GREY, height=0.55)
    for i, v in enumerate(vals):
        ax2.text(v + 0.3, i, f"{v:.1f} mm", va="center", fontsize=7.5)
    ax2.set_yticks(np.arange(len(keys)), labs, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("에피소드별 차이의 SD 상한 (mm)\n이보다 작아야 에피소드 4개 짝지은 t검정 p<0.05", fontsize=7.5)
    ax2.set_title("제안 모델 대 각 기준선 (RMSE)", fontsize=9, color=INK, loc="left")
    ax2.grid(axis="x", color="#e5e3dc", lw=0.6)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "own_t1_episode.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_t2(res):
    rows = res["t2_vs_full"]
    labs = ["1개 제거\n재학습", "2개 제거\n재학습", "3개 제거\n재학습", "1개 고장\n그대로", "2개 고장\n그대로", "3개 고장\n그대로"]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x = np.arange(len(rows))
    means = [r["mean"] for r in rows]
    sds = [r["sd"] for r in rows]
    cols = [ACC, ACC, ACC, WARN, WARN, WARN]
    ax.bar(x, means, yerr=sds, color=cols, alpha=0.85, capsize=3, ecolor=GREY, width=0.6)
    ax.axhline(30.52, color=INK, lw=0.9, ls="--")
    ax.text(5.45, 31.5, "전체 모델 30.52", ha="right", fontsize=7.5)
    for i, m in enumerate(means):
        ax.text(i, m + sds[i] + 2, f"{m:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, labs, fontsize=7.5)
    ax.set_ylabel("RMSE (mm), 평균 ± SD", fontsize=8)
    ax.set_ylim(0, 150)
    ax.set_title("표 2 센서 제거와 고장 (논문 원문 수치)", fontsize=9, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "own_t2_sensor.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    res = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    for f in (fig_t1(res), fig_t2(res)):
        print("wrote", f)


if __name__ == "__main__":
    main()
