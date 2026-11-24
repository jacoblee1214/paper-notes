"""우리가 그린 그래프 세 장 (논문 그림을 다시 그리지 않음). 입력: recompute_results.json, toy_metadata_results.json.
출력: report/figures/own_*.png
실행: python3 analysis/plot_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
YEL, GRY, BEI, RED, GRN = "#e8b923", "#a9b7bb", "#e9dcb8", "#d98a7a", "#8fb573"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())
FITS = RES["n_fits"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bars(ax, x, vals, ci, color, w, label):
    ax.bar(x, vals, w, color=color, label=label, edgecolor="#333", linewidth=0.6)
    lo = [v - c[0] for v, c in zip(vals, ci)]
    hi = [c[1] - v for v, c in zip(vals, ci)]
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor="#222", capsize=2.5, lw=1)


def fig_specialist():
    tasks = [("laundry_ts", "빨래\n(티셔츠·반바지)"), ("laundry_div", "빨래\n(가장 어려운 옷)"),
             ("espresso", "에스프레소"), ("box", "상자 조립")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    x = np.arange(4)
    for off, key, color, lab in [(-0.2, "pi07", YEL, "π0.7"), (0.2, "spec", GRY, "전문가 정책")]:
        f = [FITS[f"fig6_{t}_{key}"] for t, _ in tasks]
        bars(ax, x + off, [r["k"] / r["n"] * 100 for r in f], [tuple(100 * v for v in wilson(r["k"], r["n"])) for r in f], color, 0.38, lab)
        for xi, r in zip(x + off, f):
            ax.text(xi, 3, f'{r["k"]}/{r["n"]}', ha="center", fontsize=6.5, color="#222", rotation=90, va="bottom")
    ax.set_xticks(x, [t[1] for t in tasks], fontsize=8)
    ax.set_ylim(0, 125); ax.set_ylabel("성공률 (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_title("그림 6 위 줄: 읽은 값과 Wilson 구간 (n은 가정)", fontsize=9.5)
    ax = axes[1]
    for off, key, color, lab in [(-0.27, "pi07", YEL, "π0.7"), (0.0, "nometa", RED, "메타데이터 없음"), (0.27, "noeval", BEI, "평가 데이터 없음")]:
        ks = []
        for t, _ in tasks:
            ks.append(FITS[f"fig6_{t}_pi07"] if key == "pi07" else FITS[f"fig7_{t}_{key}"])
        bars(ax, x + off, [r["k"] / r["n"] * 100 for r in ks], [tuple(100 * v for v in wilson(r["k"], r["n"])) for r in ks], color, 0.26, lab)
    ax.set_xticks(x, [t[1] for t in tasks], fontsize=8)
    ax.set_ylim(0, 125)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    ax.set_title("그림 7: 절제 (π0.7 막대는 그림 6과 같은 값)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_specialist_ablation.png", dpi=170)
    plt.close(fig)


def fig_coaching():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    d = RES["V10_fig16"]
    names = [("scoop_rice", "쌀 뜨기"), ("reverse_fridge", "냉장고→전자레인지\n(역방향)"), ("air_fryer_load", "에어프라이어\n넣기"),
             ("bagel", "베이글 토스트"), ("air_fryer_unload", "에어프라이어\n비우기")]
    x = np.arange(5)
    ax.bar(x - 0.2, [d[k]["coaching"] for k, _ in names], 0.38, color=YEL, edgecolor="#333", linewidth=0.6, label="사람 코칭")
    ax.bar(x + 0.2, [d[k]["autonomous"] for k, _ in names], 0.38, color="#c99a1b", edgecolor="#333", linewidth=0.6, label="자율 (고수준 정책)")
    for xi, k in zip(x, [k for k, _ in names]):
        ax.text(xi, 3, f'−{d[k]["diff"]:.0f}' if d[k]["diff"] else "0", ha="center", fontsize=8)
    ax.set_xticks(x, [n for _, n in names], fontsize=7.5)
    ax.set_ylim(0, 108); ax.set_ylabel("작업 진행도 (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f'그림 16 읽은 값: 코칭 대비 자율 평균 −{d["mean_diff"]:.1f}p', fontsize=9.5)
    ax = axes[1]
    d = RES["V11_fig17"]
    names = [("press_plunger", "프렌치프레스\n플런저"), ("scoop_rice", "쌀 뜨기"), ("wiping_office", "사무용품 닦기")]
    x = np.arange(3)
    for off, key, color, lab in [(-0.2, "pi07", YEL, "π0.7 (언어)"), (0.2, "gc", "#f3d77a", "π0.7 (GC)")]:
        vals, cis = [], []
        for k, _ in names:
            n = d[k]["n"]; kk = d[k]["k"][key]
            vals.append(100 * kk / n); cis.append(tuple(100 * v for v in wilson(kk, n)))
        bars(ax, x + off, vals, cis, color, 0.38, lab)
    ax.set_xticks(x, [n for _, n in names], fontsize=8)
    ax.set_ylim(0, 112); ax.set_ylabel("성공률 (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("그림 17 읽은 값: 언어 조건과 GC 조건 (n 가정)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_coaching_gc.png", dpi=170)
    plt.close(fig)


def fig_metadata():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    v = RES["V13_fig18_left"]
    xs = [30, 50, 80, 100]
    ax.plot(xs, [v["with_meta"][str(x)] for x in xs], "s-", color=YEL, label="메타데이터 있음")
    ax.plot(xs, [v["without_meta"][str(x)] for x in xs], "o-", color="#b8ad8a", label="메타데이터 없음")
    ax.set_xlabel("학습에 쓴 데이터 비율 (%, 품질 순)"); ax.set_ylabel("처리량 (성공/시간)")
    ax.set_ylim(0, 27); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("그림 18 왼쪽 읽은 값 (±0.5 정도)", fontsize=9.5)
    ax = axes[1]
    T = json.loads((ROOT / "analysis" / "toy_metadata_results.json").read_text())
    fr = ["top30", "top50", "top80", "top100"]
    for key, color, lab, ls in [("no_meta", "#b8ad8a", "메타데이터 없음", "o-"), ("meta_u1", YEL, "메타데이터 u=1", "s-"),
                                ("meta_u1_cfg2.2", "#c0392b", "메타데이터 u=1 + CFG β=2.2", "^--")]:
        ax.plot(xs, [T[f][key]["throughput"] for f in fr], ls, color=color, label=lab)
    ax.set_xlabel("학습에 쓴 데이터 비율 (%, 품질 순)"); ax.set_ylabel("토이 처리량 (1/스텝)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("토이: 평균화는 막지만 데이터 이득은 만들지 않음", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_metadata_scaling.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_specialist(); fig_coaching(); fig_metadata()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
