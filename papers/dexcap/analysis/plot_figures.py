"""DexCap 논문 표 수치를 우리가 다시 그린 그래프 두 장 (논문 그림은 쓰지 않는다).

출력: report/figures/own_table1_ci.png, report/figures/own_packaging_ci.png
실행: python3 analysis/plot_figures.py  (matplotlib, 수 초)
"""
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 한글 폰트: Linux(Noto CJK) -> macOS(Apple SD Gothic Neo) 순서로 찾는다
for cand in ("Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"):
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------
# 그래프 1: 표 I 전체 성공률 (60회 풀링), Wilson 95% 구간
# ---------------------------------------------------------------
t1 = {  # (sponge, ball, plate) / 20
    "BC-RNN-img-mask": (5, 2, 2),
    "BC-RNN-point": (9, 6, 5),
    "BC-RNN-prec": (10, 6, 7),
    "DP-img-mask": (11, 8, 6),
    "DP-point-raw": (14, 14, 8),
    "DP-point": (15, 13, 10),
    "DP-perc (제안)": (17, 12, 14),
}
rows = sorted(((m, sum(v), 60) for m, v in t1.items()), key=lambda r: r[1] / r[2])
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
for i, (m, k, n) in enumerate(rows):
    p = k / n
    lo, hi = wilson(k, n)
    color = "#1f5fae" if "제안" in m else "#8a8880"
    ax.plot([lo * 100, hi * 100], [i, i], color=color, lw=2.2, alpha=0.9)
    ax.plot(p * 100, i, "o", color=color, ms=6)
    ax.text(hi * 100 + 1.5, i, f"{k}/{n} = {p * 100:.0f}%", va="center", fontsize=8, color="#1b1b1a")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlabel("과제 성공률 (%) · 세 과제 풀링 60회 · Wilson 95% 구간", fontsize=9)
ax.set_xlim(0, 110)
ax.grid(axis="x", color="#dcdad3", lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("표 I 재계산: 세 과제(스펀지, 공, 접시) 풀링 성공률 (과제당 20회)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_table1_ci.png")
plt.close(fig)

# ---------------------------------------------------------------
# 그래프 2: 표 II Packaging, 데이터만 대 교정 30회 후, 세 지표
# ---------------------------------------------------------------
t2 = {  # method: {"data": [(k, n) x 3], "corr": [...]}  지표 순서: subtask(30), all(30), unseen(45)
    "BC-RNN-point": {"data": [(10, 30), (7, 30), (7, 45)], "corr": [(12, 30), (8, 30), (10, 45)]},
    "DP-img-mask": {"data": [(5, 30), (0, 30), (0, 45)], "corr": [(14, 30), (10, 30), (0, 45)]},
    "DP-perc (제안)": {"data": [(21, 30), (14, 30), (18, 45)], "corr": [(25, 30), (17, 30), (19, 45)]},
}
metrics = ["Subtask (학습 객체 30회)", "전체 성공 (학습 객체 30회)", "미학습 객체 (45회)"]
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6), dpi=160, sharey=True)
methods = list(t2.keys())
for j, ax in enumerate(axes):
    for i, m in enumerate(methods):
        for shift, key, col in ((-0.17, "data", "#8a8880"), (0.17, "corr", "#1f5fae")):
            k, n = t2[m][key][j]
            p = k / n
            lo, hi = wilson(k, n)
            y = i + shift
            ax.plot([lo * 100, hi * 100], [y, y], color=col, lw=2)
            ax.plot(p * 100, y, "o", color=col, ms=5)
            if j == 1 or True:
                ax.text(hi * 100 + 2, y, f"{p * 100:.0f}", va="center", fontsize=7, color="#1b1b1a")
    ax.set_title(metrics[j], fontsize=8.5)
    ax.set_xlim(-2, 105)
    ax.grid(axis="x", color="#dcdad3", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("성공률 (%) · Wilson 95%", fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(methods)))
axes[0].set_yticklabels(methods, fontsize=8)
axes[0].invert_yaxis()
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color="#8a8880", lw=2, marker="o", label="데이터만 (DexCap)"),
           Line2D([0], [0], color="#1f5fae", lw=2, marker="o", label="교정 30회 후 (미세조정)")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("표 II 재계산: Packaging (사람 데이터 대 교정 후, 각 점은 Wilson 95% 구간)", fontsize=9.5, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(FIG / "own_packaging_ci.png")
plt.close(fig)

print("wrote", FIG / "own_table1_ci.png", FIG / "own_packaging_ci.png")
