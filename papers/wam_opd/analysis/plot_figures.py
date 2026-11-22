"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다 (matplotlib, CPU, 수 초).

  own_task_ci.png    표 2의 과제별 Student 대 WAM-OPD (Wilson 95% 구간)와 Teacher 기준
  own_gap_gain.png   Teacher-Student 격차 대 WAM-OPD 이득 (표 2), 표 3 실로봇 구간
  own_ablation_ci.png 그림 4와 표 4의 값과 Wilson 구간

입력: analysis/recompute_results.json   실행: python3 analysis/plot_figures.py
"""
import json
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

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


# 1. 과제별
rows = R["table2"]["rows"]
names = list(rows)
order = sorted(names, key=lambda t: rows[t]["kO"] - rows[t]["kS"])
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for i, t in enumerate(order):
    r = rows[t]
    for k, col, dy in ((r["kS"], MUTED, -0.12), (r["kO"], ACC, 0.12)):
        lo, hi = wilson(k, 60)
        ax.plot([lo, hi], [i + dy, i + dy], color=col, lw=2, alpha=0.6)
        ax.plot(100 * k / 60, i + dy, "o", color=col, ms=6)
    ax.plot(100 * r["kT"] / 60, i, "|", color=ACC2, ms=14, mew=2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=9)
ax.set_xlabel("성공률 (%) — 60회, Wilson 95% 구간")
ax.plot([], [], "o", color=MUTED, label="Student")
ax.plot([], [], "o", color=ACC, label="WAM-OPD")
ax.plot([], [], "|", color=ACC2, mew=2, ms=10, label="Teacher")
ax.legend(frameon=False, loc="lower right")
ax.set_title("표 2 재구성: 과제별 구간은 겹치는 곳이 많다", fontsize=11, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_task_ci.png", dpi=170)
plt.close(fig)

# 2. 격차 대 이득 + 실로봇
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
gap = [100 * (rows[t]["kT"] - rows[t]["kS"]) / 60 for t in names]
gain = [100 * (rows[t]["kO"] - rows[t]["kS"]) / 60 for t in names]
a.plot([0, 100], [0, 100], "--", color=MUTED, lw=1)
a.text(30, 62, "이득 = 격차\n(교사 수준)", color=MUTED, fontsize=8)
a.scatter(gap, gain, color=ACC, s=30)
for t, x, y in zip(names, gap, gain):
    if t in ("place_cans_plasticbox", "place_container_plate", "stamp_seal", "put_object_cabinet", "handover_mic"):
        a.annotate(t, (x, y), fontsize=7, xytext=(4, -9), textcoords="offset points", color=INK)
g = R["table2"]["gap_gain"]
a.set_title(f"격차 대 이득 (Spearman {g['spearman']:.2f}, 12점)", fontsize=10)
a.set_xlabel("Teacher 빼기 Student (pp)")
a.set_ylabel("WAM-OPD 빼기 Student (pp)")
style(a)
r3 = R["table3"]["rows"]
tn = list(r3)
for i, t in enumerate(tn):
    r = r3[t]
    for k, col, dy in ((r["kS"], MUTED, -0.12), (r["kO"], ACC, 0.12)):
        lo, hi = wilson(k, 36)
        b.plot([lo, hi], [i + dy, i + dy], color=col, lw=2, alpha=0.6)
        b.plot(100 * k / 36, i + dy, "o", color=col, ms=6)
    b.plot(100 * r["kT"] / 36, i, "|", color=ACC2, ms=14, mew=2)
k0 = R["table3"]["avg"]["counts"]
i = len(tn)
for k, col, dy in ((k0[1], MUTED, -0.12), (k0[2], ACC, 0.12)):
    lo, hi = wilson(k, 144)
    b.plot([lo, hi], [i + dy, i + dy], color=col, lw=2, alpha=0.6)
    b.plot(100 * k / 144, i + dy, "o", color=col, ms=6)
b.set_yticks(range(len(tn) + 1))
b.set_yticklabels(tn + ["합산 (144)"], fontsize=8)
b.set_xlabel("성공률 (%) — 36회 (합산 144), Wilson 95%")
b.set_title("표 3 실로봇", fontsize=10)
style(b)
fig.tight_layout()
fig.savefig(FIG / "own_gap_gain.png", dpi=170)
plt.close(fig)

# 3. 절제
f4 = R["fig4"]
t4 = R["table4"]
labs = list(f4)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
w = 0.2
cols = {"student": MUTED, "action": ACC2, "video": "#59a9dc", "joint": ACC}
for j, key in enumerate(["student", "action", "video", "joint"]):
    for i, t in enumerate(labs):
        k = f4[t]["counts"][key]
        lo, hi = wilson(k, 60)
        x = i + (j - 1.5) * w
        a.bar(x, 100 * k / 60, w * 0.9, color=cols[key], label=key if i == 0 else None)
        a.plot([x, x], [lo, hi], color=INK, lw=1)
a.set_xticks(range(3))
a.set_xticklabels(["a2b_right", "bread_basket", "cans_plasticbox"], fontsize=8)
a.set_ylabel("성공률 (%), 60회, Wilson 95%")
a.set_title("그림 4: Joint와 최고 단일의 차이(5.0, 1.7, 5.0pp)는 구간 안", fontsize=9.5)
a.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
a.set_ylim(0, 112)
style(a)
for j, (key, col, lab) in enumerate((("student", MUTED, "Flash-WAM"), ("teacher_hist", ACC2, "Joint / Teacher 이력"), ("student_hist", ACC, "Joint / Student 이력"))):
    for i, t in enumerate(labs):
        k = t4[t][key]
        lo, hi = wilson(k, 60)
        x = i + (j - 1) * 0.26
        b.bar(x, 100 * k / 60, 0.23, color=col, label=lab if i == 0 else None)
        b.plot([x, x], [lo, hi], color=INK, lw=1)
b.set_xticks(range(3))
b.set_xticklabels(["a2b_right", "bread_basket", "cans_plasticbox"], fontsize=8)
b.set_title("표 4: 이력 출처 차이는 과제마다 다름", fontsize=9.5)
b.legend(frameon=False, fontsize=8, loc="upper left")
b.set_ylim(0, 125)
style(b)
fig.tight_layout()
fig.savefig(FIG / "own_ablation_ci.png", dpi=170)
plt.close(fig)
print("ok")
