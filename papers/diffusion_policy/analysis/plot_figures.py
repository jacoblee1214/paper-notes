"""우리가 그린 그래프 두 장을 만든다. 입력: analysis/recompute_results.json.

  report/figures/own_b2_readings.png : 부록 B.2 평균 향상률의 해석별 재계산 대 논문 값 46.9%
  report/figures/own_real_ci.png     : 실로봇 Push-T 성공률과 Wilson 95% 구간 (n = 20)

한글 폰트는 macOS Apple SD Gothic Neo, Linux Noto CJK 순서로 찾는다.
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#8a8880"


def b2_readings():
    labels, vals = [], []
    short = {
        "B2 본문 절차: 최대 성능, mh 제외 (표1 ph 6열 + 표2 ph 6열 + 표4 6열, N=18)": "최대, mh 제외 (N=18)\n본문 절차",
        "최대 성능, mh 포함 (표1 10열 + 표2 10열 + 표4 6열, N=26)": "최대, mh 포함 (N=26)",
        "마지막 10개 평균, mh 포함 (N=26)": "마지막 10개 평균\nmh 포함 (N=26)",
        "마지막 10개 평균, mh 제외 (N=18)": "마지막 10개 평균\nmh 제외 (N=18)",
    }
    for k, v in RES["V1_avg_improvement"].items():
        labels.append(short[k])
        vals.append(100 * v["mean"])
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    bars = ax.bar(range(len(vals)), vals, color=[ACC, GREY, GREY, GREY], width=0.6)
    ax.axhline(46.858, color="#b3261e", ls="--", lw=1.2)
    ax.text(len(vals) - 0.5, 47.8, "논문 주장 46.9%", color="#b3261e", ha="right", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("평균 향상률 (%)", fontsize=9)
    ax.set_ylim(0, 58)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("부록 B.2 평균 향상률: 해석별 재계산", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_b2_readings.png")
    plt.close(fig)


def real_ci():
    rows = RES["V3_real_push_t"]
    names = ["Human", "IBC-pos", "IBC-vel", "LSTM-pos", "LSTM-vel", "DP-T-E2E", "DP-ImgNet", "DP-R3M", "DP-E2E"]
    label = {"Human": "사람 시연", "IBC-pos": "IBC (위치)", "IBC-vel": "IBC (속도)", "LSTM-pos": "LSTM-GMM (위치)",
             "LSTM-vel": "LSTM-GMM (속도)", "DP-T-E2E": "DP Transformer E2E", "DP-ImgNet": "DP CNN ImageNet",
             "DP-R3M": "DP CNN R3M", "DP-E2E": "DP CNN E2E"}
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    for i, n in enumerate(names):
        r = rows[n]
        lo, hi = r["wilson95"]
        rate = r["rate"]
        col = ACC if n == "DP-E2E" else GREY
        ax.errorbar(rate, i, xerr=[[rate - lo], [hi - rate]], fmt="o", color=col, ecolor=col, capsize=3, ms=5)
        ax.text(1.04, i, f"{r['k']}/{r['n']}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([label[n] for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.12)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xlabel("성공률 (Wilson 95% 구간, n = 20)", fontsize=9)
    ax.axvline(0, color="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("실로봇 Push-T 성공률 (논문 표 6 값을 20회로 환산)", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


if __name__ == "__main__":
    b2_readings()
    real_ci()
    print("wrote", FIG / "own_b2_readings.png", FIG / "own_real_ci.png")
