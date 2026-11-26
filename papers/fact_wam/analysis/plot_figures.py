"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sim_paired.png  : 표 6 과제별 차이(FACT w/ failure - FACT) 50개와 평균, 과제 부트스트랩 95% 구간
own_real_ci.png     : 표 2, 3 실로봇 성공률과 Wilson 95% 구간 (칸당 20회 합산, 시행 수 가정)
own_value_target.png: 식 (6)의 실패 목표(lambda = 1)와 공개 코드식 목표(시간 잔여 + 1)의 비교 (T = 200, 실패 시작 100)
own_fig6_fig7.png   : 그림 6과 7을 판독한 값(그림 판독 값)과 표 6 같은 과제 값
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; analysis/recompute.py를 먼저 실행)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#b3412a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def table6():
    rows = {}
    for line in (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Z][A-Za-z0-9 ]+?)\s{2,}((?:\d+(?:\.\d+)?\s+){9}\d+(?:\.\d+)?)\s*$", line)
        if m:
            rows[m.group(1).strip()] = [float(x) for x in m.group(2).split()]
    rows.pop("Average")
    return rows


def plot_sim():
    rows = table6()
    names = list(rows)
    M = np.array(list(rows.values()))
    d = (M[:, 8] + M[:, 9]) / 2 - (M[:, 6] + M[:, 7]) / 2
    o = np.argsort(d)
    fig, ax = plt.subplots(figsize=(9.4, 3.7))
    ax.bar(range(50), d[o], color=[ACC if v > 0 else RED for v in d[o]], width=0.75)
    ax.axhline(d.mean(), color=INK, lw=1, ls="--")
    lo, hi = REC["table6_bootstrap_task_cluster_ci"]["FACT w/ failure vs FACT"]
    ax.axhspan(lo, hi, color=GREY, alpha=0.3)
    ax.set_xticks([])
    ax.set_xlabel("RoboTwin 과제 50개 (차이 순)")
    ax.set_ylabel("성공률 차이 (%p, clean·rand 평균)")
    ax.set_title(f"실패 데이터 추가 효과: 평균 {d.mean():+.2f} %p (회색 띠 = 과제 부트스트랩 95% 구간 {lo:.2f}~{hi:.2f}), 증가 31 감소 15 동률 4", fontsize=9)
    ax.set_ylim(-11, 19)
    for i in (0, 1, 49, 48):
        up = d[o][i] > 0
        ax.text(i, d[o][i] + (0.5 if up else -0.5), names[o[i]], fontsize=6.5, rotation=90, ha="center", va="bottom" if up else "top")
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_paired.png", dpi=170)
    plt.close(fig)


def plot_real():
    rw = REC["real_world"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    seen = [("Cosmos", 25), ("pi0", 48), ("pi0.5", 88), ("Motus", 64), ("Ours", 82), ("Ours w/ failure", 89), ("Ours w/ failure + scoring", 92)]
    unseen = [("pi0", 57), ("pi0.5", 85), ("Motus", 62), ("Ours", 67), ("Ours w/ failure", 77), ("Ours w/ failure + scoring", 82)]
    for ax, data, n, title in ((axes[0], seen, 100, "seen 5과제 (n = 100, 칸당 20회 가정)"), (axes[1], unseen, 60, "unseen 3과제 (n = 60)")):
        ks = [round(v / 100 * n) for _, v in data]
        pct = np.array([100 * k / n for k in ks])
        ci = np.array([wilson(k, n) for k in ks])
        x = np.arange(len(data))
        cols = [GREY if not nm.startswith("Ours") else ACC for nm, _ in data]
        ax.bar(x, pct, color=cols, width=0.62)
        ax.errorbar(x, pct, yerr=[pct - ci[:, 0], ci[:, 1] - pct], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([nm.replace("Ours w/ failure", "Ours+F").replace(" + scoring", "+S").replace("Ours", "FACT").replace("+F", " w/ fail") for nm, _ in data], rotation=30, ha="right", fontsize=7.5)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("성공률 (%) 와 Wilson 95% 구간")
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_value():
    T, H, onset = 200, 48, 100
    t = np.arange(0, T - H)
    p = (t + H) / T
    ok = p
    eq6 = np.where(t + H >= onset, np.clip(p - 1.0, 0, 1), p)
    eq6_unclipped = np.where(t + H >= onset, p - 1.0, p)
    # 코드식: 시간 잔여 (L - idx - 1)/(L - 1), 실패 구간 +1 (창 시작 t >= onset), 진행도 눈금으로 환산 = 1 - 값
    fidx = t + H
    raw = (T - fidx - 1) / (T - 1)
    code_goal = np.where(t >= onset, raw + 1.0, raw)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.plot(t, ok, color=GREY, lw=1.4, label="성공 에피소드 (p = (t+H)/T)")
    ax.plot(t, eq6, color=RED, lw=1.8, label="식 (6) 실패, lambda = 1, 0~1로 자름")
    ax.plot(t, eq6_unclipped, color=RED, lw=1.2, ls=":", label="식 (6) 자르지 않은 값")
    ax.plot(t, 1 - code_goal, color=ACC, lw=1.2, ls="--", label="공개 코드 목표(시간 잔여, 실패 +1)를 진행도 눈금으로 뒤집은 값 (1 - 목표)")
    ax.axvline(onset, color=INK, lw=0.8, ls=":")
    ax.text(onset + 1, 0.9, "실패 시작 t = 100", fontsize=8)
    ax.set_xlabel("창 시작 시각 t (T = 200, H = 48 가정)")
    ax.set_ylabel("목표 값")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("식 (6)을 lambda = 1로 쓰면 실패 이후 창의 목표는 0 하나 (시간 진행을 보존하지 않음)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_value_target.png", dpi=170)
    plt.close(fig)


def plot_fig67():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    f6 = REC["fig6_read"]["values_pct_read"]
    names = list(f6) + ["평균"]
    arr = np.array(list(f6.values()) + [REC["fig6_read"]["avg_from_read"]])
    x = np.arange(len(names))
    for j, (lab, c) in enumerate([("p=0%", "#d9c27a"), ("p=50%", "#b3bb7c"), ("p=100%", "#7da66f")]):
        ax.bar(x + (j - 1) * 0.26, arr[:, j], width=0.26, color=c, label=lab)
    t6 = REC["fig6_tasks_in_table6_clean"]
    ax.scatter([3.0], [t6["FACT_mean"]], marker="_", s=300, color=INK, zorder=5)
    ax.scatter([3.0], [t6["FACT w/ failure_mean"]], marker="_", s=300, color=RED, zorder=5)
    ax.text(3.32, t6["FACT w/ failure_mean"], "표 6 FACT w/ failure: 88.3", fontsize=7, color=RED, va="center")
    ax.text(3.32, t6["FACT_mean"], "표 6 FACT: 76.3", fontsize=7, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("성공률 (%) (그림 6 판독 값)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(-0.6, 5.6)
    ax.set_title("그림 6 판독 값과 표 6 값", fontsize=9)
    ax = axes[1]
    f7 = REC["fig7_read"]
    ax.plot(f7["N"], f7["rel_completion_read"], "o-", color="#3a8a4f", label="상대 완료율 (그림 판독 값)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(f7["N"])
    ax.set_xticklabels([str(n) for n in f7["N"]])
    ax.set_xlabel("후보 수 N")
    ax.set_ylabel("상대 과제 완료율", color="#3a8a4f")
    ax2 = ax.twinx()
    ax2.plot(f7["N"], f7["latency_ms_read"], "s--", color=RED, label="지연 (ms, 그림 판독 값)")
    ax2.set_ylabel("지연 (ms)", color=RED)
    ax.set_title("그림 7 판독 값: N = 8에서 완료율이 N = 4보다 낮음", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig6_fig7.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_sim()
    plot_real()
    plot_value()
    plot_fig67()
    print("wrote 4 figures to", FIG)
