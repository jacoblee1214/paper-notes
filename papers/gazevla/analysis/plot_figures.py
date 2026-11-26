"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_table1_diff.png   표 1의 셀별 'Ours - π0.5' 차이와 95% 구간(Newcombe, 셀당 100회 가정)
  own_real_diff.png     그림 6(그림 판독 값)의 과제별 'Ours - π0.5' 차이와 95% 구간(셀당 40회 가정), 풀링 포함
  own_ablation_ci.png   표 2 절제 변형의 성공률과 Wilson 95% 구간 (ID 20회, OOD 풀링 30회)
  own_power.png         두 비율 비교에서 80% 검출력으로 구별되는 최소 차이와 이 논문 실험의 시행 수

입력: analysis/recompute_results.json (먼저 python3 analysis/recompute.py)
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
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
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """두 비율 차이 p1 - p2의 Newcombe(Wilson 기반) 95% 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


# ------------------------------------------------ 그림 1: 표 1 셀별 차이
cells = R["V4_table1_cell_tests_assumed_n100"]["cells"]
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
labels, ys = [], []
for i, c in enumerate(cells):
    d, lo, hi = newcombe(c["ours"], 100, c["pi05"], 100)
    y = len(cells) - 1 - i - (i // 6) * 0.6
    color = ACC if c["fisher_p"] >= 0.05 else WARN
    ax.plot([lo * 100, hi * 100], [y, y], color=color, lw=1.6)
    ax.plot(d * 100, y, "o", color=color, ms=4)
    labels.append((y, f"{c['setting']} · {c['task']}"))
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks([y for y, _ in labels])
ax.set_yticklabels([t for _, t in labels], fontsize=7.5)
ax.set_xlabel("Ours - π0.5 (%p), 95% 구간 (셀당 100회 가정, Newcombe)")
ax.set_title("표 1: 18개 셀 중 구간이 0을 벗어나는 셀은 1개(빨강), Holm 보정 후에는 0개", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_table1_diff.png")
plt.close(fig)

# ------------------------------------------------ 그림 2: 그림 6 과제별 차이
pt = R["V5_figure6_real_world"]["per_task_ours_vs_pi05_assumed_n40"]
fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=200)
names = list(pt.keys()) + ["풀링 (6과제, 240회)"]
for i, t in enumerate(pt):
    d, lo, hi = newcombe(pt[t]["ours_k"], 40, pt[t]["pi05_k"], 40)
    y = len(names) - 1 - i
    ax.plot([lo * 100, hi * 100], [y, y], color=ACC, lw=1.8)
    ax.plot(d * 100, y, "o", color=ACC, ms=4.5)
p = R["V5_figure6_real_world"]["pooled_ours_vs_pi05_assumed_n240"]
d, lo, hi = newcombe(p["ours_k"], 240, p["pi05_k"], 240)
ax.plot([lo * 100, hi * 100], [0, 0], color=ACC2, lw=2.4)
ax.plot(d * 100, 0, "s", color=ACC2, ms=6)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names[::-1], fontsize=8)
ax.set_xlabel("Ours - π0.5 (%p), 95% 구간 (그림 판독 값, 칸당 40회 가정)")
ax.set_title("그림 6: 과제별로는 0을 포함하고, 6과제를 풀링하면 구간이 0을 벗어난다", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_real_diff.png")
plt.close(fig)

# ------------------------------------------------ 그림 3: 표 2 절제
rows = R["V6_table2_ablation"]["rows"]
order = ["Ours", "w/o Intention", "w/o Human Finetune", "w/o Human Pretrain"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=200, sharey=True)
id_k = {"Ours": 19, "w/o Intention": 16, "w/o Human Finetune": 17, "w/o Human Pretrain": 13}
for ax, title, getk, n in ((axes[0], "ID (20회)", lambda v: id_k[v], 20),
                           (axes[1], "OOD 풀링 (3종 x 10회 = 30회)", lambda v: rows[v]["ood_k"], 30)):
    for j, v in enumerate(order):
        k = getk(v)
        lo, hi = wilson(k, n)
        y = len(order) - 1 - j
        ax.plot([lo * 100, hi * 100], [y, y], color=ACC if v == "Ours" else MUTED, lw=2)
        ax.plot(100 * k / n, y, "o", color=ACC if v == "Ours" else MUTED, ms=5)
        ax.text(101, y, f"{k}/{n}", va="center", fontsize=7.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("성공률 (%), Wilson 95% 구간")
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order[::-1], fontsize=8)
fig.suptitle("표 2: Ours와 w/o Intention의 구간은 크게 겹친다 (OOD 풀링 Fisher p = 0.43)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_ablation_ci.png")
plt.close(fig)

# ------------------------------------------------ 그림 4: 검출력
from scipy import stats  # noqa: E402


def power_normal(n, p1, p2):
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    dz = abs(p1 - p2) / se
    return float(stats.norm.sf(1.959964 - dz) + stats.norm.cdf(-1.959964 - dz))


ns = np.unique(np.round(np.logspace(math.log10(10), math.log10(800), 60)).astype(int))
mde = []
for n in ns:
    d = next((d for d in np.arange(0.01, 0.6, 0.005) if power_normal(n, 0.5 - d / 2, 0.5 + d / 2) >= 0.8), np.nan)
    mde.append(100 * d)
fig, ax = plt.subplots(figsize=(6.6, 3.7), dpi=200)
ax.plot(ns, mde, color=ACC, lw=2)
marks = [(10, "표 2 OOD 칸 10"), (20, "표 2 ID 20"), (30, "OOD 풀링 30"), (40, "그림 6 칸 40"), (100, "표 1 칸 100"),
         (240, "그림 6 풀링 240"), (600, "표 1 풀링 600")]
for n, lab in marks:
    d = next(d for d in np.arange(0.01, 0.9, 0.005) if power_normal(n, 0.5 - d / 2, 0.5 + d / 2) >= 0.8)
    ax.plot(n, 100 * d, "o", color=ACC2, ms=5)
    ax.annotate(lab, (n, 100 * d), textcoords="offset points", xytext=(5, 4), fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("비교하는 두 쪽의 시행 수 (각각)")
ax.set_ylabel("80% 검출력 최소 차이 (%p)")
ax.set_title("평균 성공률 50% 부근, 양측 5% 정규 근사 (작은 n은 정확 검정이면 더 큼)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_power.png")
plt.close(fig)
print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
