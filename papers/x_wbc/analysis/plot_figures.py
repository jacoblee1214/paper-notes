"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다 (논문 그림은 쓰지 않는다).

own_table2.png       : 표 2 (100STYLE 800 클립) 성공 클립 수와 Wilson 95% 구간, 실패 클립의 평균 완료율에서 환산한 실패 시점
own_table1.png       : 표 1 성공률 (G1, H2) 공동 학습 비교와 절제, 평가 표본 잡음 하한 구간
own_robots_readout.png: 그림 3 왼쪽(9개 로봇 성공률·MPKPE)을 막대 높이에서 읽은 근사값
own_toy_alignment.png: 토이 V1, 정렬 손실 유무별 검색 Recall@1과 행동 오차
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
import math
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
INK, ACC, GREY, ORG, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#c8641e", "#2e8b57"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_table2():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["table2"]
    names = ["X-WBC", "SONIC", "TWIST"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    for i, n in enumerate(names):
        k = rec[n]["k"]
        lo, hi = wilson(k, 800)
        ax.errorbar(i, 100 * k / 800, yerr=[[100 * k / 800 - lo], [hi - 100 * k / 800]], fmt="o", color=ACC, capsize=5, ms=7)
        ax.text(i + 0.12, 100 * k / 800, f"{k}/800", va="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_xlim(-0.4, 2.8)
    ax.set_ylabel("성공률 (%)"); ax.set_title("성공 클립과 Wilson 95% 구간", fontsize=10); clean(ax)
    ax = axes[1]
    vals = [rec[n]["seconds_into_clip_at_failure"] for n in names]
    ax.bar(names, vals, color=[ACC, GREY, ORG])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}s", ha="center", fontsize=9)
    ax.set_ylabel("실패 클립이 끝나는 평균 시점 (초)"); ax.set_title("표 2 완료율에서 환산한 실패 시점", fontsize=10); clean(ax)
    ax.set_ylim(0, 1.35)
    ax = axes[2]
    mp = {"X-WBC": 0.0649, "SONIC": 0.0687, "TWIST": 0.0565}
    ve = {"X-WBC": 0.5400, "SONIC": 0.3419, "TWIST": 0.5261}
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, [mp[n] * 1000 for n in names], w, color=ACC, label="MPKPE (mm)")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, [ve[n] for n in names], w, color=GREY, label="속도 오차 (m/s)")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("MPKPE (mm)"); ax2.set_ylabel("속도 오차 (m/s)")
    ax.set_title("오차 (활성 걸음 평균)", fontsize=10); clean(ax)
    ax.set_ylim(0, 90); ax2.set_ylim(0, 0.72)
    fig.tight_layout()
    fig.savefig(FIG / "own_table2.png", dpi=170)
    plt.close(fig)


def plot_table1():
    G1 = {"X-WBC (VR)": 98.60, "MLP (VR)": 97.70, "단일 로봇 (VR)": 97.69, "X-WBC (로봇 명령)": 98.21,
          "X-WBC (사람 명령)": 98.66, "사람·로봇 경로 없음": 95.34, "사람 경로 없음": 95.17, "로봇 경로 없음": 95.33,
          "정렬 손실 없음": 95.41, "로봇별 모듈 없음": 96.54}
    H2 = {"X-WBC (VR)": 93.22, "MLP (VR)": 91.33, "단일 로봇 (VR)": 92.22, "X-WBC (로봇 명령)": 92.08,
          "X-WBC (사람 명령)": 93.02, "사람·로봇 경로 없음": 85.71, "사람 경로 없음": 83.84, "로봇 경로 없음": 85.21,
          "정렬 손실 없음": 85.68, "로봇별 모듈 없음": 88.65}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, T, nmin, title in ((axes[0], G1, 6268, "Unitree G1 (29DoF)"), (axes[1], H2, 5620, "Unitree H2")):
        names = list(T)
        y = np.arange(len(names))[::-1]
        cols = [ACC] * 3 + [GREY] * 2 + [ORG] * 5
        for yi, n, c in zip(y, names, cols):
            v = T[n]
            k = round(v / 100 * nmin)
            lo, hi = wilson(k, nmin)
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=c, capsize=3, ms=5)
            ax.text(v + (0.15 if v < 99 else -0.15), yi + 0.28, f"{v:.2f}", fontsize=8, ha="left" if v < 99 else "right")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.axhline(6.5, color="#dcdad3", lw=0.8); ax.axhline(4.5, color="#dcdad3", lw=0.8)
        ax.set_xlabel("성공률 (%)"); ax.set_title(title, fontsize=10); clean(ax)
    fig.text(0.5, 0.005, "구간은 평가 시도 수를 최소 가능 값(G1 6,268, H2 5,620)으로 가정한 Wilson 95%입니다. 학습 시드 변동은 없습니다. 파랑 구조·공동 학습, 회색 명령 경로, 주황 절제.",
             ha="center", fontsize=7.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "own_table1.png", dpi=170)
    plt.close(fig)


def plot_robots():
    names = ["G1-29", "G1-23", "H1", "H1-2", "R1", "H2", "T1", "Adam", "GR3"]
    sr = [98.6, 98.5, 96.5, 97.6, 97.5, 93.0, 96.6, 92.5, 82.7]
    mp = [41, 60, 68, 51, 53, 63, 33, 113, 67]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(names)); w = 0.4
    ax.bar(x - w / 2, sr, w, color=ACC, label="성공률 (%)")
    ax.set_ylim(70, 100); ax.set_ylabel("성공률 (%) (읽은 근사값)")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, mp, w, color="#b09bd0", label="MPKPE (mm)")
    ax2.set_ylim(0, 120); ax2.set_ylabel("MPKPE (mm) (읽은 근사값)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, frameon=False)
    clean(ax)
    ax.set_title("로봇별 성공률과 MPKPE (막대 높이에서 읽은 근사값, 오차 약 ±1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_robots_readout.png", dpi=170)
    plt.close(fig)


def plot_toy():
    r = json.loads((ANA / "toy_alignment_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    pairs = [("human->robot", "사람→로봇"), ("vr->robot", "VR→로봇"), ("robot0->robot1", "로봇 0→로봇 1")]
    x = np.arange(len(pairs)); w = 0.26
    for i, (key, lab, c) in enumerate([("align_lam1", "정렬 손실 (λ=1)", ACC), ("no_align", "정렬 없음", ORG), ("align_only", "정렬만 (과제 손실 없음)", GREY)]):
        m = [r[key][p]["same_motion"][0] for p, _ in pairs]
        s = [r[key][p]["same_motion"][1] for p, _ in pairs]
        ax.bar(x + (i - 1) * w, m, w, yerr=s, color=c, label=lab, capsize=2)
    ax.axhline(r["chance"]["same_motion"], color="k", lw=0.8, ls="--"); ax.text(2.45, 3, "우연 1.04%", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in pairs]); ax.set_ylabel("같은 모션 Recall@1 (%)")
    ax.set_title("토이 V1: 검색 Recall@1 (576 갤러리, 시드 3)", fontsize=10); ax.legend(fontsize=8, frameon=False); clean(ax)
    ax = axes[1]
    routes = [("human", "사람"), ("vr", "VR(희소)"), ("robot", "로봇")]
    x = np.arange(3)
    for i, (key, lab, c) in enumerate([("align_lam1", "정렬 (λ=1)", ACC), ("no_align", "정렬 없음", ORG)]):
        m = [r[key]["action_mse"][k][0] for k, _ in routes]
        s = [r[key]["action_mse"][k][1] for k, _ in routes]
        ax.bar(x + (i - 0.5) * 0.36, m, 0.36, yerr=s, color=c, label=lab, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in routes]); ax.set_ylabel("행동 MSE (낮을수록 좋음)")
    ax.set_title("같은 토이의 행동 오차", fontsize=10); ax.legend(fontsize=8, frameon=False); clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_alignment.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_table2(); plot_table1(); plot_robots(); plot_toy()
    print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
