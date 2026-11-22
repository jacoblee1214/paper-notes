"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_insertion_ci.png : 그림 4c 실로봇 삽입 성공률 판독값과 Wilson 95% 구간, 판독 스텝 (근사)
own_params.png       : 논문 표 1 파라미터 수 대 공식 체크포인트에서 얻은 파라미터 수
own_mask_sens.png    : 그림 3b 마스킹 비율 85% 대 89%의 검증 크기별 차이 구간 (가정 n)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
입력: analysis/recompute_results.json (먼저 analysis/recompute.py를 실행)
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
RES = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, WARN = "#1b1b1a", "#1f5fae", "#9a9890", "#b5561c"
PARTS = ["USB", "Switch", "VGA"]
ARMS = [("none", "촉각 없음"), ("scratch", "처음부터 학습"), ("T3", "T3 인코더")]
COL = {"USB": "#1f5fae", "Switch": "#d6913a", "VGA": "#3c9a72"}


def plot_insertion():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(ARMS))
    w = 0.26
    for i, part in enumerate(PARTS):
        k = [RES["insertion_counts_assumed"][a][part] for a, _ in ARMS]
        n = 15
        p = np.array([kk / n for kk in k])
        lo = np.array([RES["insertion_wilson"][a][part][0] for a, _ in ARMS])
        hi = np.array([RES["insertion_wilson"][a][part][1] for a, _ in ARMS])
        ax[0].bar(x + (i - 1) * w, p, w, color=COL[part], label=part)
        ax[0].errorbar(x + (i - 1) * w, p, yerr=[p - lo, hi - p], fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax[0].set_xticks(x, [b for _, b in ARMS], fontsize=8)
    ax[0].set_ylabel("삽입 성공률 (15회, 판독 근사)", fontsize=8)
    ax[0].set_ylim(0, 1.0)
    ax[0].set_title("성공률과 Wilson 95% 구간", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    steps = RES["steps_read"]
    s_scr = [steps["scratch"][p] for p in PARTS]
    s_t3 = [steps["T3"][p] for p in PARTS]
    xx = np.arange(len(PARTS))
    ax[1].bar(xx - 0.18, s_scr, 0.36, color=GREY, label="처음부터 학습")
    ax[1].bar(xx + 0.18, s_t3, 0.36, color=ACC, label="T3 인코더")
    for j, (a, b) in enumerate(zip(s_scr, s_t3)):
        d = b - a
        ax[1].text(j, max(a, b) + 0.6, f"{d:+.1f}", ha="center", fontsize=7.5, color=WARN if d > 0 else INK)
    ax[1].set_xticks(xx, PARTS, fontsize=8)
    ax[1].set_ylabel("평균 스텝 (성공 에피소드, 판독)", fontsize=8)
    ax[1].set_title("USB에서는 스텝이 늘어남", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False, loc="upper right")
    for a in ax:
        a.tick_params(labelsize=7.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_insertion_ci.png", dpi=170)
    plt.close(fig)


def plot_params():
    sizes = ["tiny", "small", "medium", "large"]
    P = RES["params"]
    x = np.arange(len(sizes))
    w = 0.2
    series = [("논문 표 1", "paper_M", GREY),
              ("4개 인코더+트렁크+포즈 1개", "ckpt_4enc_trunk_1pose_M", ACC),
              ("인코더 8개+트렁크", "ckpt_8enc_trunk_M", WARN),
              ("체크포인트 전체 (fp32 환산)", "ckpt_all_fp32_M", INK)]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, (lab, key, c) in enumerate(series):
        vals = [P[s][key] for s in sizes]
        ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=lab)
    ax.set_xticks(x, [f"T3-{s}" for s in sizes], fontsize=8)
    ax.set_ylabel("파라미터 수 (M)", fontsize=8)
    ax.set_yscale("log")
    ax.set_title("논문 표 1 값은 4개 인코더 조합과 맞는다 (가설)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_params.png", dpi=170)
    plt.close(fig)


def plot_mask():
    S = RES["masking_sensitivity"]
    labels = list(S.keys())
    lo = [S[k]["diff_95CI"][0] * 100 for k in labels]
    hi = [S[k]["diff_95CI"][1] * 100 for k in labels]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    y = np.arange(len(labels))
    ax.hlines(y, lo, hi, color=ACC, lw=3)
    for yi, k in zip(y, labels):
        ax.text(hi[yi] + 0.6, yi, f"p = {S[k]['p']:.3f}", va="center", fontsize=7.5)
    ax.axvline(0, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(y, [f"검증 {k}" for k in labels], fontsize=8)
    ax.set_xlabel("80% 대 30% 마스킹의 정확도 차이 (%p, 95% 구간, 가정 n)", fontsize=8)
    ax.set_title("검증 크기를 보고하지 않아 n은 가정", fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_mask_sens.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_insertion()
    plot_params()
    plot_mask()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
