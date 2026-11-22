"""우리가 만든 그래프 두 장을 report/figures/ 에 저장한다. 논문 그림 조각은 만들지 않는다.

  own_task_families.png  표 VI 과제 군집별 성공률 (풀링 Wilson 95% 구간, 과제당 50 시행)
  own_codegen_counts.png 표 I의 값과 공식 노트북 저장 출력의 k/37 (구성별 점, 불일치 표시)

입력: analysis/recompute_results.json (recompute.py가 만듦)
실행: python3 analysis/plot_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# 그림 1: 과제 군집별 성공률. 표 VI의 행에서 합계를 다시 만들고 Wilson 구간을 붙인다.
fams = [("SA_SI_LH", "본 과제·본 속성\n장기 과제 (5)", 250), ("SA_SI_SG", "본 과제·본 속성\n공간 기하 (3)", 150),
        ("UA_SI_LH", "새 속성\n장기 과제 (5)", 250), ("UA_SI_SG", "새 속성\n공간 기하 (3)", 150),
        ("UA_UI_LH", "새 과제·새 속성\n장기 과제 (3)", 150), ("UA_UI_SG", "새 과제·새 속성\n공간 기하 (3)", 150)]
methods = [("CLIPort_oracle", "CLIPort (oracle 종료)", MUTED), ("NL_planner", "언어 계획기", ACC2), ("CaP", "Code as Policies", ACC)]
means = R["T6_recomputed_means"]
fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
width = 0.26
for j, (mk, label, color) in enumerate(methods):
    xs, ys, lo, hi = [], [], [], []
    for i, (fk, _, n) in enumerate(fams):
        m = means.get(fk, {}).get(mk)
        if m is None:
            continue
        k = round(m * n / 100)
        l, h = wilson(k, n)
        x = i + (j - 1) * width
        xs.append(x); ys.append(m)
        lo.append(m - l); hi.append(h - m)
    ax.bar(xs, ys, width=width * 0.95, color=color, label=label, alpha=0.9)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
ax.set_xticks(range(len(fams)))
ax.set_xticklabels([f[1] for f in fams], fontsize=7.5, color=INK)
ax.set_ylabel("성공률 (%)", color=INK)
ax.set_ylim(0, 110)
ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("표 VI 재계산: 과제 군집별 성공률 (막대 = 표 VI 평균, 선 = 풀링 Wilson 95%)", fontsize=9, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_task_families.png")
plt.close(fig)

# 그림 2: 표 I의 보고값과 노트북 출력(k/37)
cfgs = ["GPT-3 6.7B", "GPT-3 175B", "cushman", "code-davinci-002"]
nb_key = {"GPT-3 6.7B": "text-curie-001 (6.7B)", "GPT-3 175B": "text-davinci-002 (175B)",
          "cushman": "code-cushman-001", "code-davinci-002": "code-davinci-002"}
T1 = R["T1_notebook_vs_paper"]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=200, sharey=True)
for ax, arm, title in [(axes[0], "hier", "계층 생성 (Hier CG + Hier prompt)"), (axes[1], "flat", "평면 생성 (Flat CG + Flat prompt)")]:
    for i, c in enumerate(cfgs):
        d = T1[nb_key[c]]
        paper = d["paper_hier"] if arm == "hier" else d["paper_flat"]
        nb = d["notebook_hier_hier_pct"] if arm == "hier" else d["notebook_flat_flat_pct"]
        ax.scatter([i - 0.08], [paper], s=46, color=ACC, zorder=3, label="표 I 보고값" if i == 0 else None)
        ax.scatter([i + 0.08], [nb], s=46, facecolor="white", edgecolor=WARN if abs(paper - round(nb)) > 0 else ACC2,
                   linewidth=1.6, zorder=3, label="노트북 저장 출력 (k/37)" if i == 0 else None)
        ax.text(i + 0.12, nb, f"{nb:.1f}", fontsize=7, va="center", color=INK)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, fontsize=8, color=INK)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.grid(axis="y", color="#e5e3dc", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("성공률 (%, 문제 37개)", color=INK)
axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
fig.suptitle("표 I의 보고값과 공식 노트북의 저장 출력 비교 (빨간 테두리 = 반올림 기준 불일치)", fontsize=9, color=INK, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(FIG / "own_codegen_counts.png")
plt.close(fig)
print("saved", sorted(p.name for p in FIG.glob("own_*.png")))
