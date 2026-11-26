"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_success_ci.png   : 표 2의 과제별 성공률(15회)과 Wilson 95% 구간 (네 구성)
own_clean_partial.png: 구성별 평균 성공률과 clean-only 상한 (그림 4 분해 사용)
own_interaction.png  : 메모리·힘 절제의 가산 예측 대 실제 (합산 105회)
own_toy_gca.png      : 토이 1, 새 모달리티를 넣는 방식별 초기 출력 변화
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 analysis/recompute.py와 analysis/toy_gca_init.py를 실행해 결과 json을 만들어 둘 것.
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#e8923a"
TASKS = ["Scarf", "Whiteboard", "Button", "Handwash", "Lego", "Socket", "Sweep"]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def load():
    return json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def plot_success(r):
    cnt = r["V1_counts"]
    cfgs = [("X-VLA25", GREY), ("X-VLA_AFM", "#8fb0d9"), ("DAM/M", "#4f86c6"), ("DAM-VLA", ACC)]
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    w = 0.2
    for j, (c, col) in enumerate(cfgs):
        xs = np.arange(7) + (j - 1.5) * w
        pct = [100 * k / 15 for k in cnt[c]]
        ci = np.array([wilson(k, 15) for k in cnt[c]])
        ax.errorbar(xs, pct, yerr=[np.array(pct) - ci[:, 0], ci[:, 1] - np.array(pct)], fmt="o", color=col, ms=4, lw=1, capsize=2, label=c.replace("_", "-"))
    ax.set_xticks(range(7))
    ax.set_xticklabels(TASKS, fontsize=8)
    ax.set_ylabel("성공률 (%) 과 Wilson 95% 구간")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", color="#dcdad3", lw=0.6)
    ax.legend(fontsize=8, ncol=4, loc="lower center")
    ax.set_title("과제별 성공률 (과제당 15회, 표 2 재계산)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_success_ci.png", dpi=170)
    plt.close(fig)


def plot_clean(r):
    cfgs = ["X-VLA25", "X-VLA100", "X-VLA_AFM", "DAM/F/M", "DAM/F", "DAM/M", "DAM-VLA"]
    rep = [r["V2_avg"][c]["pooled_pct"] for c in cfgs]
    up = [r["V9_clean_only_upper_bound"][c]["pct_upper_bound"] for c in cfgs]
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.bar(x - 0.19, rep, 0.38, color=ORG, label="보고 평균 (clean + partial)")
    ax.bar(x + 0.19, up, 0.38, color=ACC, label="clean만의 상한 (분해 없는 3과제는 전부 clean 가정)")
    for i in range(len(cfgs)):
        ax.text(x[i] - 0.19, rep[i] + 1, f"{rep[i]:.1f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.19, up[i] + 1, f"{up[i]:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "-") for c in cfgs], fontsize=8)
    ax.set_ylabel("평균 성공률 (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("그림 4 분해로 본 평균 (그림 4는 Button, Handwash, Lego, Socket만)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_clean_partial.png", dpi=170)
    plt.close(fig)


def plot_interaction(r):
    v = r["V8_interaction"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    labels = ["메모리·힘 없음\n(/F/M)", "메모리만\n(/F)", "힘만\n(/M)", "가산 예측", "둘 다\n(전체)"]
    vals = [v["base_FM"], v["base_FM"] + v["memory_effect"], v["base_FM"] + v["force_effect"], v["additive_prediction"], v["base_FM"] + v["both_effect"]]
    cols = [GREY, "#8fb0d9", "#8fb0d9", "#d6d4cc", ACC]
    ax.bar(range(5), vals, color=cols, width=0.62)
    for i, y in enumerate(vals):
        ax.text(i, y + 1.5, f"{y:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("평균 성공률 (%)")
    ax.set_title(f"상호작용 {v['interaction_pp']:+.1f} %p (95% 구간 {v['bootstrap_95'][0]:+.1f} ~ {v['bootstrap_95'][1]:+.1f})", fontsize=9)
    ax = axes[1]
    pt = v["per_task_interaction_pp"]
    ax.barh(range(7), [pt[t] for t in TASKS], color=[ACC if pt[t] > 0 else GREY for t in TASKS])
    ax.set_yticks(range(7))
    ax.set_yticklabels(TASKS, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("과제별 상호작용 (%p)")
    ax.set_title("과제별: Lego와 Socket에 몰림", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_interaction.png", dpi=170)
    plt.close(fig)


def plot_toy():
    t = json.loads((ANA / "toy_gca_init_results.json").read_text(encoding="utf-8"))
    keys = ["concat", "gca_sigmoid_b0", "gca_sigmoid_b-2", "gca_sigmoid_b-5", "gca_tanh0"]
    names = ["토큰 이어 붙이기", "GCA 시그모이드\nb = 0", "GCA 시그모이드\nb = -2", "GCA 시그모이드\nb = -5", "GCA tanh(0)\n(식 1)"]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    vals = [100 * t[k]["mean_rel_change"] for k in keys]
    ax.bar(range(5), vals, color=[ORG, "#8fb0d9", "#8fb0d9", "#8fb0d9", ACC], width=0.6)
    for i, y in enumerate(vals):
        ax.text(i, y + 0.3, f"{y:.2f}%", ha="center", fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("액션 토큰 출력의 상대 변화 (%)")
    ax.set_title("토이 1: 초기화 직후 출력 변화 (무작위 8층 트랜스포머, 50회 평균)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_gca.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    r = load()
    plot_success(r)
    plot_clean(r)
    plot_interaction(r)
    plot_toy()
    print("wrote 4 figures to", FIG)
