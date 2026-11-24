"""W2A 분석의 그래프 네 장을 만든다 (우리가 그린 그래프, 논문 그림 아님).

  report/figures/own_real_ci.png    : 실로봇 유형별 성공률과 Wilson 95% 구간, 소과제별 W2A 대 OpenVLA (표 16)
  report/figures/own_libero_gap.png : LIBERO 스위트별 W2A - 기준선 차이와 근사 95% 구간 (시행 수 가정)
  report/figures/own_ablation.png   : 절제 평균과 시드 변동 띠 (표 4-6, 표 15)
  report/figures/own_toy_pca.png    : 합성 문제의 증류 비교 (toy_pca_distill.py 결과)

입력: analysis/recompute_results.json, analysis/toy_pca_distill_results.json. 실행: python3 analysis/plot_figures.py
한글 폰트: Noto Sans CJK KR -> Apple SD Gothic Neo 순서로 찾는다.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_pca_distill_results.json").read_text(encoding="utf-8"))

for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORANGE, GREEN = "#1b1b1a", "#1f5fae", "#8a8880", "#c0602a", "#2f7d5b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def own_real_ci():
    sub = RES["real_world"]["subtask"]
    types = ["PnP", "Cube", "Long", "Practical"]
    names = {"Practical": "실용 과제 3종"}
    n_by = {"PnP": 120, "Cube": 120, "Long": 120, "Practical": 90}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1.1, 1]})
    cols = {"dp": GREY, "openvla": ORANGE, "w2a": ACC}
    labels = {"dp": "Diffusion Policy", "openvla": "OpenVLA", "w2a": "W2A"}
    w = 0.26
    for j, m in enumerate(["dp", "openvla", "w2a"]):
        for i, t in enumerate(types):
            k = sum(r[m] for r in sub if r["type"] == t)
            n = n_by[t]
            lo, hi = wilson(k, n)
            x = i + (j - 1) * w
            a1.bar(x, 100 * k / n, w, color=cols[m], label=labels[m] if i == 0 else None)
            a1.errorbar(x, 100 * k / n, yerr=[[100 * k / n - lo], [hi - 100 * k / n]], color=INK, capsize=2.5, lw=1)
    a1.set_xticks(range(4), [names.get(t, t) for t in types])
    a1.set_ylabel("성공률 (%) 와 Wilson 95% 구간")
    a1.set_title("유형별 (소과제당 30회, 시행 독립 가정)", fontsize=10)
    a1.legend(frameon=False, fontsize=9)
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(axis="y", color="#e5e3dc", lw=0.6)
    for i, r in enumerate(sub):
        ov, w2 = r["openvla"], r["w2a"]
        c = GREEN if w2 > ov else ORANGE
        a2.plot([ov, w2], [i, i], color=c, lw=2)
        a2.scatter([ov], [i], color=ORANGE, s=22, zorder=3)
        a2.scatter([w2], [i], color=ACC, s=22, zorder=3)
    a2.set_yticks(range(len(sub)), [f"{r['type']}: {r['task']}" for r in sub], fontsize=7.5)
    a2.invert_yaxis()
    a2.set_xlabel("성공 횟수 / 30  (주황 OpenVLA, 파랑 W2A)")
    a2.set_title("소과제별 (13승 2패, 부호 검정 p = 0.007)", fontsize=10)
    a2.spines[["top", "right"]].set_visible(False)
    a2.grid(axis="x", color="#e5e3dc", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def own_libero_gap():
    suites = ["Spatial", "Object", "Goal", "Long", "평균"]
    base = {"π0.5 (미세조정)": ("pi0.5", [98.8, 98.2, 98.0, 92.4]),
            "mimic-video": ("mimic-video", [94.2, 96.8, 90.6, 84.9]),
            "DreamZero-5B": ("DreamZero-5B", [90.4, 94.8, 87.2, 82.2])}
    w = [96.3, 98.8, 95.0, 93.6]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    cols = [ORANGE, ACC, GREEN]
    for j, (label, (_, b)) in enumerate(base.items()):
        for i in range(5):
            wv = w[i] if i < 4 else sum(w) / 4
            bv = b[i] if i < 4 else sum(b) / 4
            n1, n2 = (1500, 500) if i < 4 else (6000, 2000)
            p1, p2 = wv / 100, bv / 100
            se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            d = 100 * (p1 - p2)
            x = i + (j - 1) * 0.22
            ax.errorbar(x, d, yerr=196 * se, fmt="o", color=cols[j], capsize=3, label=label if i == 0 else None)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(5), suites)
    ax.set_ylabel("W2A - 기준선 (%p)")
    ax.set_title("LIBERO: W2A 대 기준선 (W2A 스위트당 1500회, 기준선 500회 가정, 근사 95% 구간)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_libero_gap.png", dpi=170)
    plt.close(fig)


def own_ablation():
    seeds = RES["libero"]["seed_avg_libero"]
    sd2 = RES["ablations"]["detect_limit_note"]["two_single_seed_runs_diff_sd"]
    items = [("W2A 시드 0, 5, 42", seeds), ("w/o distill", [84.4]), ("w/o PCA (K=2048)", [88.6]),
             ("Shuffled Teacher", [78.2]), ("Random Teacher", [69.2]), ("Token0 Teacher", [86.7]),
             ("Current Feature", [82.2]), ("Future Feature", [87.8]), ("Global MLP", [82.7]), ("Grouped MLP", [87.7])]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    mean = sum(seeds) / 3
    ax.axhspan(mean - sd2, mean + sd2, color="#dbe7f5", label=f"W2A 평균 ± {sd2} (시드 변동에서 본 단일 실행 두 개의 차이 SD)")
    ax.axhline(mean, color=ACC, lw=1)
    for i, (name, vals) in enumerate(items):
        ax.scatter([i] * len(vals), vals, color=ACC if i == 0 else ORANGE, s=34, zorder=3)
    ax.set_xticks(range(len(items)), [n for n, _ in items], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("LIBERO 네 스위트 평균 (%)")
    ax.set_title("절제 평균과 시드 변동 (표 4-6, 표 15; 절제는 시드 수 미기재)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=170)
    plt.close(fig)


def own_toy_pca():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, key, title in zip(axes, ["n_train_96", "n_train_1024"], ["학습 표본 96개", "학습 표본 1024개"]):
        r = TOY["results"][key]
        names = list(r.keys())
        means = [r[n]["mean"] for n in names]
        sds = [r[n]["sd"] for n in names]
        cols = [GREY if n == "none" else (ORANGE if n in ("shuffled", "random") else ACC) for n in names]
        ax.bar(range(len(names)), means, yerr=sds, color=cols, capsize=3)
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8.5)
        ax.set_title(title + " (시드 5개 평균, 막대는 SD)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    axes[0].set_ylabel("시험 행동 MSE (낮을수록 좋음)")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_pca.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    own_real_ci()
    own_libero_gap()
    own_ablation()
    own_toy_pca()
    print("wrote 4 figures to", FIG)
