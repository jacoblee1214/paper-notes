"""자체 그래프 두 장 (논문 그림은 쓰지 않음: 라이선스 nonexclusive).

실행: python3 analysis/plot_figures.py → report/figures/own_*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(OUT, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for f in FONT_CANDIDATES:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

BLUE, GREY, RED = "#1f5fae", "#8a8a85", "#b03a2e"

# (1) Table 2 Shift 과 Table 3 Deadline 평균 SR, 평균 ± 표본 SD (3 시드)
methods = ["INTACT", "INTACT Guarded A", "Fast-JEPA"]
shift = [(56.25, 1.64), (35.25, 1.56), (73.83, 4.16)]
deadline = [(75.33, 1.18), (58.75, 0.90), (87.67, 0.38)]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
x = range(len(methods))
w = 0.36
for i, (m, s) in enumerate(shift):
    ax.bar(i - w / 2, m, w, color=BLUE if methods[i] == "Fast-JEPA" else GREY, label="Shift" if i == 0 else None)
    ax.errorbar(i - w / 2, m, yerr=s, color="black", capsize=3, lw=1)
for i, (m, s) in enumerate(deadline):
    ax.bar(i + w / 2, m, w, color=RED if methods[i] == "Fast-JEPA" else "#c9c8c2", alpha=0.9,
           label="Deadline" if i == 0 else None)
    ax.errorbar(i + w / 2, m, yerr=s, color="black", capsize=3, lw=1)
ax.set_xticks(list(x), methods)
ax.set_ylabel("평균 성공률 (%)")
ax.set_ylim(0, 100)
ax.set_title("DynaGoal 평균 성공률 (논문 표 2, 3; 막대 = 평균, 선 = 표본 SD, n = 3 시드)", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_dynagoal_sr.png"))
plt.close(fig)

# (2) 그림 4(0 ms) 근사값 대 표 2 (Shift) 값, 과제별
tasks = ["PushT", "Reacher", "Cube", "TwoRoom"]
table2 = {"INTACT": [50.67, 73.00, 56.00, 45.33], "INTACT Guarded A": [36.33, 48.67, 15.00, 41.00],
          "Fast-JEPA": [57.33, 80.67, 72.00, 85.33]}
fig4_0ms = {"INTACT": [48, 81, 77, 47], "INTACT Guarded A": [30, 84, 96, 42], "Fast-JEPA": [71, 96, 93, 90]}
fig, axes = plt.subplots(1, 4, figsize=(8.4, 3.0), dpi=200, sharey=True)
for j, t in enumerate(tasks):
    ax = axes[j]
    for k, m in enumerate(methods):
        ax.scatter(k, table2[m][j], marker="o", color=BLUE, s=36, zorder=3, label="표 2 (Shift)" if (j == 0 and k == 0) else None)
        ax.scatter(k, fig4_0ms[m][j], marker="x", color=RED, s=40, zorder=3, label="그림 4, 0 ms (읽은 값)" if (j == 0 and k == 0) else None)
        ax.plot([k, k], [table2[m][j], fig4_0ms[m][j]], color="#999", lw=0.8, zorder=2)
    ax.set_title(t, fontsize=9)
    ax.set_xticks(range(3), ["INT", "INT-GA", "FJ"], fontsize=7)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("성공률 (%)")
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.suptitle("같은 0 ms 조건의 두 값: 그림 4 (읽은 값, ±3 pt)과 표 2 (Shift)의 차이", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_fig4_vs_table2.png"))
plt.close(fig)
print("wrote", OUT)
