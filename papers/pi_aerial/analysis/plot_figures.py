"""우리가 만든 그래프 세 장을 report/figures/에 저장한다(논문 그림은 쓰지 않는다).

own_headline.png    : 논문이 보고한 조건부 비율과 20회 중 끝까지 성공한 비율(Wilson 95% 구간) 비교
own_nav_synth.png   : 항법 게이트 통과·종단 성공, 합성 데이터 유무(RTC와 naive)
own_toy_guidance.png: toy T2, 가이던스 세기 s에 따른 평균 이동과 식 2의 정확 사후 평균
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Noto Sans CJK KR, Noto Sans KR, Apple SD Gothic Neo 순)
먼저 recompute.py, toy_payload_guidance.py를 실행해 둘 것.
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
INK, ACC, GREY, WARN = "#1b1b1a", "#1f5fae", "#9a9890", "#c0502d"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def fig_headline():
    # (이름, 보고 비율, 보고 분자, 보고 분모, 종단 분자) 종단 분모는 20
    items = [
        ("초록 \"23%→50%\"\n(RTC 대 가이던스, 표 I)", [("RTC", 4, 17, 4), ("가이던스", 10, 20, 10)]),
        ("초록 \"62%\"\n(합성 포함 가이던스, 표 II)", [("가이던스", 10, 16, 10)]),
        ("초록 \"81%→100%\"\n(항법 Hover, 표 I)", [("교시만", 13, 16, 13), ("합성 포함", 19, 19, 19)]),
        ("OOD \"57%\"\n(샌드위치, 표 III)", [("샌드위치", 4, 7, 4)]),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
    xs, labels = [], []
    x = 0
    for title, rows in items:
        for name, kc, nc, ke in rows:
            ne = 10 if "OOD" in title else 20
            pc, pe = 100 * kc / nc, 100 * ke / ne
            lo_c, hi_c = wilson(kc, nc)
            lo_e, hi_e = wilson(ke, ne)
            ax.bar(x - 0.19, pc, 0.36, color=GREY, label="논문 보고(조건부)" if x == 0 else None)
            ax.errorbar(x - 0.19, pc, yerr=[[max(pc - lo_c, 0)], [max(hi_c - pc, 0)]], color=INK, capsize=2, lw=0.9)
            ax.bar(x + 0.19, pe, 0.36, color=ACC, label="끝까지 성공(전체 시도 대비)" if x == 0 else None)
            ax.errorbar(x + 0.19, pe, yerr=[[max(pe - lo_e, 0)], [max(hi_e - pe, 0)]], color=INK, capsize=2, lw=0.9)
            ax.text(x - 0.19, 3, f"{kc}/{nc}", ha="center", color="white", fontsize=7)
            ax.text(x + 0.19, 3, f"{ke}/{ne}", ha="center", color="white", fontsize=7)
            xs.append(x)
            labels.append(f"{name}")
            x += 1
        x += 0.6
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    # 그룹 제목
    gx = 0
    for title, rows in items:
        n = len(rows)
        cx = np.mean([xs[gx + i] for i in range(n)])
        ax.text(cx, 112, title, ha="center", va="bottom", fontsize=7.5, color=INK)
        gx += n
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("성공률 (%)")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 0.86))
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_headline.png")
    plt.close(fig)


def fig_nav():
    cats = ["Gate\n(naive)", "Gate\n(RTC)", "Hover|Gate\n(RTC)", "종단\n(naive)", "종단\n(RTC)"]
    non = [(10, 20), (16, 20), (13, 16), (6, 20), (13, 20)]
    syn = [(9, 20), (19, 20), (19, 19), (9, 20), (19, 20)]
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=200)
    x = np.arange(len(cats))
    for off, data, col, lab in ((-0.19, non, GREY, "교시 데이터만"), (0.19, syn, ACC, "교시 + 합성 50개")):
        p = [100 * k / n for k, n in data]
        lo = [max(100 * k / n - wilson(k, n)[0], 0) for k, n in data]
        hi = [max(wilson(k, n)[1] - 100 * k / n, 0) for k, n in data]
        ax.bar(x + off, p, 0.36, color=col, label=lab)
        ax.errorbar(x + off, p, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=2, lw=0.9)
        for xi, (k, n), pv in zip(x + off, data, p):
            ax.text(xi, 3, f"{k}/{n}", ha="center", color="white", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("성공률 (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.axvline(2.5, color=GREY, lw=0.6, ls=":")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_nav_synth.png")
    plt.close(fig)


def fig_toy():
    res = json.loads((ANA / "toy_payload_guidance_results.json").read_text(encoding="utf-8"))
    rows = res["T2_guidance"]["rows"]
    z = res["T2_guidance"]["설정"]["z_des"]
    sig = [r["sigma(정책 z 표준편차)"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=200)
    xx = np.arange(len(rows))
    keys = [("s=1", "s=1 (상수)", "#7aa6d6"), ("s=1/(1-tau+0.1) 평균", None, None), ("s=10", "s=10 (상수)", ACC)]
    ax.plot(xx, [r["식2 정확 사후 평균"] for r in rows], "o--", color=WARN, label="식 2의 정확한 사후 평균")
    ax.plot(xx, [r["s=1 평균"] for r in rows], "s-", color="#7aa6d6", label="식 3, s = 1")
    ax.plot(xx, [r["s=1/(1-tau+0.1) 평균"] for r in rows], "^-", color="#3d7cc0", label="식 3, s = 1/(1-τ+0.1)")
    ax.plot(xx, [r["s=10 평균"] for r in rows], "D-", color=ACC, label="식 3, s = 10")
    ax.axhline(z, color=INK, lw=0.8, ls=":")
    ax.text(len(rows) - 1, z + 0.004, "z_des = 0.15", ha="right", fontsize=8)
    ax.set_xticks(xx)
    ax.set_xticklabels([f"σ = {s}" for s in sig])
    ax.set_ylabel("가이던스 뒤 z 평균 (사전 평균 0)")
    ax.set_xlabel("정책이 낸 z 불확실성 (정규화 단위 가정)")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_guidance.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_headline()
    fig_nav()
    fig_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
