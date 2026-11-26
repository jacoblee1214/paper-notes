"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1.png : 표 1을 과제별로 합산(정책 3개, 과제당 60회 가정)한 성공률과 Wilson 95% 구간 (T, P-, P+)
own_diff.png   : 과제별 P+ - T, P+ - P- 차이와 Newcombe 95% 구간 (60회 합산 가정)
own_toy.png    : 토이: 식 (1)의 보정 후 남는 잔차 비율 beta / (a^2 + beta)와 재생 두 번째 회차
입력: analysis/recompute_results.json, analysis/toy_paired_replay_results.json (먼저 recompute.py, toy_paired_replay.py 실행)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c4651b"
TASKS = ["Screw Driving", "AirPods Case", "Ball Throwing", "Table Cleaning", "Air Freshener"]
SHORT = ["Screw", "AirPods", "Ball", "Table", "Air Fresh."]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_table1(rec):
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    w = 0.26
    x = np.arange(5)
    for i, (cond, col, lab) in enumerate([("T", GREY, "T (텔레오퍼레이션)"), ("P-", "#8fb0d9", "P- (매핑 미세조정 없음)"), ("P+", ACC, "P+ (미세조정)")]):
        ks = [rec["pooled_task"][t][f"k_{cond}"] for t in TASKS]
        pct = [100 * k / 60 for k in ks]
        lo = [wilson(k, 60)[0] for k in ks]
        hi = [wilson(k, 60)[1] for k in ks]
        ax.bar(x + (i - 1) * w, pct, w, color=col, label=lab)
        ax.errorbar(x + (i - 1) * w, pct, yerr=[np.array(pct) - np.array(lo), np.array(hi) - np.array(pct)],
                    fmt="none", ecolor=INK, capsize=2.5, lw=0.9)
        for xi, k in zip(x + (i - 1) * w, ks):
            ax.text(xi, 3, f"{k}/60", ha="center", color="white" if col != "#8fb0d9" else INK, fontsize=7, rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1.png", dpi=200)
    plt.close(fig)


def plot_diff(rec):
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    y = np.arange(5)[::-1]
    for off, key, col, lab in [(0.17, "P+_vs_T", ACC, "P+ - T"), (-0.17, "P+_vs_P-", ORG, "P+ - P-")]:
        d = [rec["pooled_task"][t][key]["diff_pp"] for t in TASKS]
        lo = [rec["pooled_task"][t][key]["ci95_pp"][0] for t in TASKS]
        hi = [rec["pooled_task"][t][key]["ci95_pp"][1] for t in TASKS]
        ax.errorbar(d, y + off, xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)], fmt="o", color=col, capsize=3, label=lab)
    for key, col, off, lab in [("P+_vs_T", ACC, 0.17, "전체"), ("P+_vs_P-", ORG, -0.17, "")]:
        o = rec["pooled_overall"][key]
        ax.errorbar([o["diff_pp"]], [-1 + off], xerr=[[o["diff_pp"] - o["ci95_pp"][0]], [o["ci95_pp"][1] - o["diff_pp"]]], fmt="D", color=col, capsize=3)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(list(y) + [-1])
    ax.set_yticklabels(SHORT + ["전체 (300회)"])
    ax.set_xlabel("성공률 차이 (%p), Newcombe 95% 구간 (정책 3개 합산 60회/조건 가정)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_diff.png", dpi=200)
    plt.close(fig)


def plot_toy(toy):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    beta = np.logspace(-3, 1, 200)
    cols = {0.2: ORG, 0.4: ACC, 0.6: "#2a8a5b", 0.9: GREY}
    for a, col in cols.items():
        ax.plot(beta, beta / (a * a + beta), color=col, lw=1.6, label=f"a = {a}")
        row = toy["exact_g"][f"a={a}"]
        pts = [(b, row[f"beta={b}"]["closed_form_fraction_per_round"]) for b in (0.01, 0.1, 1.0)]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=col, s=22, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.minorticks_off()
    ax.set_xlabel("β (식 (1)의 명령 정규화 가중치)")
    ax.set_ylabel("재생 1회 뒤 남는 잔차 비율")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="lower right", title="접촉 순응 a (작을수록 단단한 물체)", title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy.png", dpi=200)
    plt.close(fig)


def main():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    toy = json.loads((ANA / "toy_paired_replay_results.json").read_text(encoding="utf-8"))
    plot_table1(rec)
    plot_diff(rec)
    plot_toy(toy)
    print("wrote 3 figures to", FIG)


if __name__ == "__main__":
    main()
