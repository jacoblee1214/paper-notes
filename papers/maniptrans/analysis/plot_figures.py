"""우리가 그린 그래프 두 장을 report/figures/에 만든다(공개 가능, 논문 그림 아님).

  own_sr_ci.png     : 표 2 정책 학습 SR과 Wilson 95% 구간 (분모 320 가정, recompute.py 결과)
  own_time_budget.png: 논문이 말하는 학습 시간 (QuasiSim 대 ManipTrans, 사전학습 포함과 제외)

실행: python3 analysis/plot_figures.py   (matplotlib 필요, 약 2초)
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
for path in FONT_CANDIDATES:
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

RESULTS = json.load(open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8"))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fig_policy_sr():
    k = RESULTS["table2_counts_n320"]
    n = 320
    methods = ["IBC", "BET", "DP-UNet", "DP-Trans"]
    vals = [100.0 * k[m] / n for m in methods]
    lo = [100.0 * wilson(k[m], n)[0] for m in methods]
    hi = [100.0 * wilson(k[m], n)[1] for m in methods]
    err = [[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    colors = ["#9aa5b1", "#9aa5b1", "#1f5fae", "#1f5fae"]
    ax.bar(methods, vals, yerr=err, color=colors, capsize=5, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, hi[i] + 1.2, f"{v:.2f}%\n({k[methods[i]]}/{n})", ha="center", fontsize=8)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 35)
    ax.set_title("표 2 정책 학습 성공률, Wilson 95% 구간 (분모 320 가정)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "우리가 그린 그래프. 분모 320은 표의 네 값을 만드는 유일한 공통 분모 중 최소값입니다.",
             fontsize=6.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "own_sr_ci.png"))
    plt.close(fig)


def fig_time_budget():
    # 단위: 시간(h). 막대 두 줄로 "사전학습 제외"와 "포함"을 나란히 보여 준다.
    labels = ["QuasiSim (본문)", "QuasiSim (부록 C, 3단계 합)", "ManipTrans 잔차만 (본문)", "ManipTrans 사전학습 + 잔차"]
    lo = [40, 30, 0.25, 36.25]
    hi = [40, 60, 0.33, 36.33]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    y = list(range(len(labels)))[::-1]
    for yi, a, b, lab in zip(y, lo, hi, labels):
        mid = (a + b) / 2
        width = max(b - a, 0.6)
        ax.barh(yi, width, left=a if b - a > 0 else a - 0.3, color="#1f5fae" if "ManipTrans" in lab else "#9aa5b1",
                height=0.5)
        txt = f"{a:g} h" if b == a else f"{a:g}–{b:g} h"
        if "잔차만" in lab:
            txt = "약 15–20 분"
        if "사전학습" in lab:
            txt = "약 36.3 h"
        ax.text(max(b, a) + 0.8, yi, txt, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("학습 시간 (h, 같은 과제 기준, 하드웨어 미확인)")
    ax.set_xlim(0, 75)
    ax.set_title("논문의 효율 주장: 사전학습 포함 여부에 따라 달라짐", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "우리가 그린 그래프. 값은 논문 본문(§4.3)과 부록 C를 옮긴 것입니다.",
             fontsize=6.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "own_time_budget.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_policy_sr()
    fig_time_budget()
    print("wrote", OUT)
