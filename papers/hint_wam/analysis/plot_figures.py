"""우리가 그린 그래프 세 장을 report/figures/에 만든다. 입력은 recompute_results.json과 toy_mode_energy_results.json이다.

  own_paired_ci.png     : 쌍체 차이와 95% 구간 (표 6 LIBERO-90, 표 8 RoboTwin 25과제)
  own_cost_accounting.png : 학습 시간(3,000 SFT 스텝 포함)과 LIBERO-90 성공률
  own_mode_toy.png      : 숨은 정보 토이에서 제곱오차와 에너지 점수의 성공률 (3 시드)

실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_mode_energy_results.json").read_text(encoding="utf-8"))

KOREAN_CANDIDATES = ["Noto Sans CJK KR", "Apple SD Gothic Neo"]
for name in KOREAN_CANDIDATES:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACCENT, GREY = "#1b1b1a", "#1f5fae", "#8a8880"


def paired_ci():
    rows = RES["table6_paired"]
    rob = RES["table8_robotwin_paired"]
    labels, means, lo, hi, colors = [], [], [], [], []
    for r in rows:
        labels.append("LIBERO-90: " + r["comparison"])
        means.append(r["mean"]); lo.append(r["95CI"][0]); hi.append(r["95CI"][1]); colors.append(ACCENT)
    from scipy import stats
    t25 = stats.t.ppf(0.975, 24)
    for r in rob:
        labels.append("RoboTwin 25과제: " + r["comparison"])
        means.append(r["mean"]); lo.append(r["mean"] - t25 * r["SE"]); hi.append(r["mean"] + t25 * r["SE"]); colors.append(GREY)
    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=200)
    y = list(range(len(labels)))[::-1]
    for yi, m, l, h, c in zip(y, means, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=2)
        ax.plot(m, yi, "o", color=c, ms=6)
    ax.axvline(0, color=INK, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("쌍체 차이 (성공률 %p), 95% 구간 (t 분포)", fontsize=9)
    ax.set_title("논문 표 6과 표 8 재계산 (차이와 95% 구간)", fontsize=10, color=ACCENT, loc="left")
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_paired_ci.png")
    plt.close(fig)


def cost_accounting():
    # 보충 표 4의 시간(h). HINT-WAM은 3,000스텝 SFT(33.5 h) + 4.4 h + 33.7 h = 71.6 h
    pts = [
        ("SFT 5k (action expert)", 55.8, 78.88, GREY),
        ("SFT 3k + BC 2k", 55.8, 71.56, GREY),
        ("Full SFT 5k", 127.0, 81.24, GREY),
        ("HINT-WAM (3k SFT + A + B)", 71.6, 85.18, ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
    for name, hrs, succ, c in pts:
        ax.scatter(hrs, succ, s=70, color=c, zorder=3)
        ax.annotate(name, (hrs, succ), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    ax.axvline(33.5, color="#c9c6bd", lw=0.8, ls=":")
    ax.text(34.5, 71.0, "3k SFT만: 33.5 h", fontsize=7, color=GREY)
    ax.set_xlabel("학습 시간 (h, 8 GPU, 보충 표 4 기준)", fontsize=9)
    ax.set_ylabel("LIBERO-90 성공률 (%)", fontsize=9)
    ax.set_title("비용 회계 재계산: 본문 1.28배에는 3k SFT 시간이 들어간다", fontsize=10, color=ACCENT, loc="left")
    ax.grid(color="#e5e3dc", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_cost_accounting.png")
    plt.close(fig)


def mode_toy():
    names = ["제곱오차 (MSE)", "에너지 점수 (M=4)"]
    vals = [TOY["squared_error"]["success_per_seed"], TOY["energy_score"]["success_per_seed"]]
    means = [TOY["squared_error"]["mean"], TOY["energy_score"]["mean"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
    ax.bar(names, means, color=[GREY, ACCENT], width=0.5)
    for i, v in enumerate(vals):
        ax.scatter([i] * len(v), v, color=INK, s=14, zorder=3)
    for i, m in enumerate(means):
        ax.text(i, m + 0.03, f"{m:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("성공률 (학생 표본)", fontsize=9)
    ax.set_title("토이: 숨은 정보 아래의 증류 (3 시드, 점은 시드별 값)", fontsize=10, color=ACCENT, loc="left")
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_mode_toy.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    paired_ci()
    cost_accounting()
    mode_toy()
    for n in ["own_paired_ci.png", "own_cost_accounting.png", "own_mode_toy.png"]:
        print("wrote", FIG / n)
