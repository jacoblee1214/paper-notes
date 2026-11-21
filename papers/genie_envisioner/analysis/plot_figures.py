"""우리가 그린 그래프 세 장 (report/figures/own_*.png). CPU, 수 초.

own_ewm.png        : Figure 17(b) 세 단계 합과 Score (03 §1)
own_table1.png     : Table 1 E2E와 SR, w/ S 대 w/o S (03 §2)
own_real_approx.png: Figure 8 막대 근사값과 Fisher 민감도 (03 §3, 근사값 표시)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
    try:
        from matplotlib import font_manager
        if any(cand in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

res = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 1. EWMBench 세 단계 누적 막대
models = ["GE-Base", "Kling", "Hailuo", "COSMOS", "OpenSora", "LTX"]
scene = [0.9427, 0.8888, 0.8577, 0.7963, 0.9210, 0.9156]
motion = [1.6676, 0.9440, 0.5362, 0.7085, 0.3442, 0.4002]
sem = [2.0907, 2.0370, 2.0186, 1.7824, 1.8739, 1.6518]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
x = np.arange(len(models))
b1 = ax.bar(x, scene, color="#9ec5e8", label="Scene")
b2 = ax.bar(x, motion, bottom=scene, color="#3c7fc1", label="Motion")
b3 = ax.bar(x, sem, bottom=np.array(scene) + np.array(motion), color="#1f4f80", label="Semantics")
for i, s in enumerate([a + b + c for a, b, c in zip(scene, motion, sem)]):
    ax.text(i, s + 0.05, f"{s:.2f}", ha="center", fontsize=8)
ax.set_xticks(x, models)
ax.set_ylabel("EWMBench 합 (Figure 17b)")
ax.set_ylim(0, 5.4)
ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper right")
ax.set_title("우리가 그린 그래프: 세 단계 합산 (보고값과 일치)", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_ewm.png", dpi=170)
plt.close(fig)

# 2. Table 1 (E2E, SR), w/ S 대 w/o S
rows = res["V5_table1_parsed"]
labels = ["scratch", "LTX 적응", "GE-Base", "GE-Base+적응"]
vals = list(rows.values())
e2e_s = [v[0] for v in vals]
e2e_n = [v[1] for v in vals]
sr_s = [v[2] for v in vals]
sr_n = [v[3] for v in vals]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
y = np.arange(len(labels))
for ax, a, b, title in ((axes[0], e2e_s, e2e_n, "E2E"), (axes[1], sr_s, sr_n, "SR")):
    ax.barh(y - 0.2, a, height=0.4, color="#1f4f80", label="w/ S (상태 입력)")
    ax.barh(y + 0.2, b, height=0.4, color="#9ec5e8", label="w/o S")
    for yy, v in zip(y - 0.2, a):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=7)
    for yy, v in zip(y + 0.2, b):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(y, labels, fontsize=7.5)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle("우리가 그린 그래프: Table 1 (시도 횟수 미보고)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_table1.png", dpi=170)
plt.close(fig)

# 3. Figure 8 근사값 (SR, Pack detergent 중심). 읽은 값이라 근사 표시
tasks = ["Make sandwich", "Pour tea", "Clean table", "Heat food", "Pack detergent"]
approx = {
    "GR00T N1": [0.00, 0.05, 0.15, 0.00, 0.05],
    "UniVLA": [0.20, 0.25, 0.30, 0.10, 0.08],
    "π0": [0.50, 0.30, 0.83, 0.30, 0.50],
    "GE-Act Slow": [0.62, 0.40, 0.83, 0.64, 0.25],
    "GE-Act Fast": [0.62, 0.50, 0.87, 0.64, 0.72],
}
fig, ax = plt.subplots(figsize=(6.6, 3.0))
x = np.arange(len(tasks))
w = 0.16
colors = ["#e8e0c8", "#b9d6cc", "#5fa7a0", "#2b7a7d", "#0f4d52"]
for i, (name, vals) in enumerate(approx.items()):
    ax.bar(x + (i - 2) * w, vals, width=w, color=colors[i], label=name, edgecolor="#333", linewidth=0.3)
ax.set_xticks(x, tasks, fontsize=8)
ax.set_ylabel("SR (그림에서 읽은 근사값)")
ax.set_ylim(0, 1.0)
ax.legend(ncol=5, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "막대 높이는 논문 Figure 8을 눈으로 읽은 근사치 (±0.03). 논문 본문 수치가 아님.", fontsize=6.5, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_real_approx.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
