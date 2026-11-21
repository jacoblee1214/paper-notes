"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 여기서 만들지 않는다).

own_sim_ci.png      : 표 1 시뮬레이션 6과제 평균 성공률과 합산 Wilson 95% 구간 (과제당 100회, 총 600회)
own_ablation_ci.png : 표 2 절제 실험 평균(두 과제, 총 200회)과 TacInno-VLA와의 차이의 Newcombe 95% 구간
own_real_ci.png     : 표 4 실로봇 성공률 평균과 Wilson 95% 구간 (과제당 50회, 총 200회)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def bar_ci(ax, labels, val, lo, hi, colors, xlabel):
    y = range(len(labels))
    ax.barh(list(y), val, color=colors, height=0.6)
    ax.errorbar(val, list(y), xerr=[[v - l for v, l in zip(val, lo)], [h - v for v, h in zip(val, hi)]],
                fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel, color=INK)
    for i, v in enumerate(val):
        ax.text(v + 1.0, i, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
    style(ax)


# 1) 표 1: 시뮬레이션 평균과 합산 Wilson 구간 (600회)
t1 = RES["table1"]
names = sorted(t1, key=lambda n: t1[n]["avg_reported"])
val = [t1[n]["avg_reported"] for n in names]
lo = [t1[n]["pooled_ci95"][0] for n in names]
hi = [t1[n]["pooled_ci95"][1] for n in names]
cols = [ACC if n == "TacInno-VLA" else GREY for n in names]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
bar_ci(ax, names, val, lo, hi, cols, "성공률 (%), 6과제 합산 600회 Wilson 95% 구간")
ax.set_xlim(0, 105)
ax.set_title("표 1 시뮬레이션 평균 성공률 (재계산)", fontsize=11, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_sim_ci.png", dpi=200)
plt.close(fig)

# 2) 표 2: 절제 실험 평균과 차이 구간 (200회)
t2 = RES["table2"]
abl = [n for n in t2 if n != "TacInno-VLA"]
abl = sorted(abl, key=lambda n: t2[n]["avg_recomputed"])
labels = ["TacInno-VLA"] + abl
val = [t2["TacInno-VLA"]["avg_recomputed"]] + [t2[n]["avg_recomputed"] for n in abl]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
cols = [ACC] + [GREY] * len(abl)
bar_ci(ax1, labels, val, val, val, cols, "평균 성공률 (%), 두 과제 200회")
ax1.set_xlim(0, 100)
ax1.set_title("표 2 평균 성공률", fontsize=10.5, color=INK, loc="left")
dl = [-t2[n]["diff_vs_full_pp"] for n in abl]  # TacInno-VLA 빼기 변형 (양수 = 전체 모델이 높음)
ylab = abl
ax2.errorbar(dl, range(len(abl)), xerr=[[d - t2[n]["newcombe_ci_pp"][0] for d, n in zip(dl, abl)],
                                        [t2[n]["newcombe_ci_pp"][1] - d for d, n in zip(dl, abl)]],
             fmt="o", color=ACC, ecolor=INK, capsize=3)
ax2.axvline(0, color=INK, lw=0.8)
ax2.set_yticks(range(len(abl)))
ax2.set_yticklabels(ylab)
ax2.set_xlabel("TacInno-VLA 대비 차이 (pp), Newcombe 95% 구간", color=INK)
ax2.set_title("차이와 구간 (비쌍체 근사)", fontsize=10.5, color=INK, loc="left")
style(ax2)
fig.tight_layout()
fig.savefig(FIG / "own_ablation_ci.png", dpi=200)
plt.close(fig)

# 3) 표 4: 실로봇 평균과 Wilson 구간 (200회), 과제별 값은 표 주석대로 50회
t4 = RES["table4"]
names = sorted(t4, key=lambda n: t4[n]["avg_reported"])
val = [t4[n]["avg_reported"] for n in names]
lo = [t4[n]["pooled_ci95"][0] for n in names]
hi = [t4[n]["pooled_ci95"][1] for n in names]
cols = [ACC if n == "TacInno-VLA" else GREY for n in names]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
bar_ci(ax, names, val, lo, hi, cols, "성공률 (%), 4과제 합산 200회 Wilson 95% 구간")
ax.set_xlim(0, 100)
ax.set_title("표 4 실로봇 평균 성공률 (재계산)", fontsize=11, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
