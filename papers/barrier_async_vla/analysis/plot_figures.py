"""논문 표 값으로 그래프 세 장을 그린다 (CPU, 수 초). 논문 그림은 쓰지 않는다.

출력: report/figures/own_conveyor_ci.png, own_kinetix_ci.png, own_hparam.png
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, LINE = "#1b1b1a", "#8a877f", "#dcdad3"
C = {"naive": "#b9b5ab", "rtc": "#7a9cc6", "freebar": "#1f5fae"}


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=LINE, linewidth=0.6)
    ax.set_axisbelow(True)


# 1) 컨베이어 설정별 성공률과 Wilson 95% 구간 (n=30, 표 3)
hs = [6, 8, 10, 12]
data = {
    "π0": {"naive": [17, 21, 20, 19], "rtc": [22, 25, 24, 23], "freebar": [27, 26, 26, 23]},
    "π0.5": {"naive": [19, 20, 23, 19], "rtc": [23, 24, 25, 24], "freebar": [28, 27, 27, 25]},
}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
for ax, (bb, rows) in zip(axes, data.items()):
    width = 0.26
    for j, (m, label) in enumerate([("naive", "naive"), ("rtc", "RTC"), ("freebar", "FreeBar")]):
        xs = [i + (j - 1) * width for i in range(len(hs))]
        ys = [k / 30 * 100 for k in rows[m]]
        lo = [wilson(k / 30, 30)[0] * 100 for k in rows[m]]
        hi = [wilson(k / 30, 30)[1] * 100 for k in rows[m]]
        err = [[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]]
        ax.bar(xs, ys, width, color=C[m], label=label, edgecolor="none")
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([f"h={h}" for h in hs])
    ax.set_title(f"컨베이어 분류, {bb} (설정당 30회)", color=INK, fontsize=10)
    ax.set_ylim(0, 110)
    style(ax)
axes[0].set_ylabel("성공률 (%)", color=INK)
axes[1].legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
fig.text(0.01, 0.01, "막대: 표 3 값 (30회 중 성공 수). 오차 막대: Wilson 95% 구간. 우리가 그린 그래프.", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "own_conveyor_ci.png", dpi=170)
plt.close(fig)

# 2) Kinetix d=4: 시행 수 정의에 따라 같은 차이가 다르게 보이는 것 (표현: 두 비율의 Wilson 95% 구간)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cases = [("n=100 (설정당)", 100), ("n=1200 (12 과제 × 100)", 1200)]
for i, (label, n) in enumerate(cases):
    for j, (m, p) in enumerate([("RTC", 61.4), ("FreeBar", 68.3)]):
        x = i + (j - 0.5) * 0.3
        lo, hi = wilson(p / 100, n)
        ax.errorbar([x], [p], yerr=[[p - lo * 100], [hi * 100 - p]], fmt="o", color=C["freebar"] if m == "FreeBar" else C["rtc"],
                    ecolor=INK, elinewidth=1, capsize=4, markersize=7)
        ax.text(x + 0.05, p + 0.5, f"{m} {p}%", fontsize=8, color=INK, va="bottom")
ax.set_xticks([0, 1])
ax.set_xticklabels([c[0] for c in cases])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(45, 85)
ax.set_ylabel("해결률 (%), d=4", color=INK)
ax.set_title("Kinetix d=4: 시행 수 정의에 따른 95% 구간", color=INK, fontsize=10)
style(ax)
fig.text(0.01, 0.01, "본문 값 68.3%와 61.4%. 시행 수는 논문에 정의되지 않아 두 가정으로 그렸습니다.", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "own_kinetix_ci.png", dpi=170)
plt.close(fig)

# 3) 하이퍼파라미터 민감도: RoboCasa(표 A5)와 컨베이어 π0(표 A6)의 격자별 값
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
ax = axes[0]
rc = {"RTC": ([0.1, 0.5, 5, 10, 30, 50], [68.0, 66.6, 70.0, 69.6, 68.6, 52.8]),
      "FreeBar": ([0.1, 1, 5, 10, 30, 50], [70.6, 71.4, 70.4, 70.6, 67.6, 67.2])}
for m, (xs, ys) in rc.items():
    ax.plot(range(len(xs)), ys, marker="o", color=C["freebar"] if m == "FreeBar" else C["rtc"], label=m, linewidth=1.4)
ax.annotate("β=50 (RTC 52.8)", xy=(5, 52.8), xytext=(3.2, 56), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xticks(range(6))
ax.set_xticklabels(["0.1", "0.5/1", "5", "10", "30", "50"])
ax.set_title("RoboCasa GR00T (표 A5)", color=INK, fontsize=10)
ax.set_ylabel("평균 성공률 (%)", color=INK)
ax.set_xlabel("강도 (RTC β, FreeBar α)", color=INK, fontsize=8)
ax.legend(frameon=False, fontsize=8)
style(ax)
ax = axes[1]
cv = {"RTC": ([0.1, 0.5, 1, 3, 8], [53.33, 56.67, 67.67, 53.33, 46.67]),
      "FreeBar": ([0.1, 1, 3, 5, 10], [66.70, 73.30, 76.67, 83.33, 73.33])}
for m, (xs, ys) in cv.items():
    ax.plot(range(len(xs)), ys, marker="o", color=C["freebar"] if m == "FreeBar" else C["rtc"], label=m, linewidth=1.4)
ax.set_xticks(range(5))
ax.set_xticklabels(["0.1", "0.5/1", "1/3", "3/5", "8/10"])
ax.set_title("컨베이어 π0 (표 A6), 격자가 다름", color=INK, fontsize=10)
ax.set_xlabel("강도 (격자 순서: RTC 대 FreeBar)", color=INK, fontsize=8)
ax.legend(frameon=False, fontsize=8)
style(ax)
fig.text(0.01, 0.01, "두 방법의 격자가 달라 같은 x 위치가 같은 강도가 아닙니다. 우리가 그린 그래프.", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "own_hparam.png", dpi=170)
plt.close(fig)

print("wrote", [p.name for p in sorted(OUT.glob("own_*.png"))])
