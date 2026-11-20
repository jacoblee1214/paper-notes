"""우리가 재계산한 값으로 그린 그래프 두 장 (논문 그림을 쓰지 않음). 입력: recompute_results.json"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "report", "figures")
os.makedirs(OUT, exist_ok=True)
for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

with open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8") as f:
    R = json.load(f)["table3"]

# 그림 1: 표 3 과제별 성공률과 Wilson 95% CI (4DWM 대 UniPi*)
rows = R["per_task"]
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
y = range(len(rows))
for i, r in enumerate(rows):
    for key, color, dx in [("ours", "#1f5fa8", 0.18), ("unipi", "#9aa5b1", -0.18)]:
        v = r[key]
        lo, hi = r[f"{'ours' if key == 'ours' else 'unipi'}_ci95"]
        ax.errorbar(v, i + dx, xerr=[[v - lo], [hi - v]], fmt="o", color=color, capsize=2, ms=5)
ax.set_yticks(list(y))
ax.set_yticklabels([r["task"] for r in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("성공률 (%), 과제당 100 에피소드, 막대는 Wilson 95% CI")
ax.set_xlim(0, 100)
ax.plot([], [], "o", color="#1f5fa8", label="4DWM (논문 표 3)")
ax.plot([], [], "o", color="#9aa5b1", label="UniPi* (논문 표 3)")
ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.set_title("표 3 재계산: 과제별 성공률과 95% CI", fontsize=11)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_table3_ci.png"))
plt.close(fig)

# 그림 2: 표 2 Chamfer 상대 차이 (TesserAct 기준 %)
rel = json.load(open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8"))["table2_chamfer_rel"]
fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=160)
labels, vals, cols = [], [], []
for dom, label in [("real", "실세계"), ("synthetic", "합성(RLBench)")]:
    for m, v in rel[dom].items():
        labels.append(f"{label}\n{m}")
        vals.append(v)
        cols.append("#1f5fa8" if dom == "real" else "#5b8fc9")
ax.bar(range(len(vals)), vals, color=cols)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("TesserAct의 Chamfer 감소율 (%)")
for i, v in enumerate(vals):
    ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
ax.set_title("표 2 재계산: Chamfer를 TesserAct 기준으로 환산 (점추정, 분산 미보고)", fontsize=10)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_table2_chamfer.png"))
plt.close(fig)
print("ok")
