"""우리가 그린 그래프 세 장을 report/figures/에 만든다(논문 그림은 쓰지 않는다. 라이선스가 CC BY가 아님).

  own_success_groups.png : Table 2 그룹별 성공률 평균 ± SD (시드 3개), 다섯 방법
  own_latency_tradeoff.png : Table 2 평균 성공률과 Table 3 과제 가중 지연 (로그 축)
  own_real_wilson.png     : Table 4, 5 실로봇 성공률과 Wilson 95% 구간 (20회)

입력은 analysis/recompute.py의 표 값과 같다. 실행: python3 analysis/plot_figures.py
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
COL = {"DP3": "#9aa0a6", "Simple DP3": "#c5c9cc", "FlowPolicy": "#8fb3d9", "MP1": "#4f7fb8", "UCA-Flow": "#c44e3a"}

# Table 2 (v5): 그룹 평균 ± SD. 단일 과제 열은 제외하고 그룹 4개와 전체 평균만 그린다.
G = ["Easy (21)", "Medium (4)", "Hard (4)", "Very Hard (5)", "전체 (37)"]
T2 = {
    "DP3":        [(87.3, 2.2), (44.5, 8.7), (32.7, 7.7), (39.4, 9.0), (68.7, 4.7)],
    "Simple DP3": [(86.8, 2.3), (42.0, 6.5), (38.7, 7.5), (35.0, 11.6), (67.4, 5.0)],
    "FlowPolicy": [(84.8, 2.2), (58.2, 7.9), (40.2, 4.5), (52.2, 5.0), (71.6, 3.5)],
    "MP1":        [(88.2, 1.1), (68.0, 3.1), (58.1, 5.0), (67.2, 2.7), (78.9, 2.1)],
    "UCA-Flow":   [(92.4, 0.7), (79.3, 1.8), (68.0, 1.9), (85.3, 0.7), (86.4, 0.3)],
}
# Table 3 과제 가중 지연 (ms, 재계산 값)과 Table 2 평균
LAT = {"DP3": 132.15, "Simple DP3": 97.02, "FlowPolicy": 12.64, "MP1": 6.76, "UCA-Flow": 3.01}
AVG = {k: v[-1][0] for k, v in T2.items()}

# ---- 그림 1: 그룹별 성공률 ----
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
n = len(T2)
w = 0.16
for i, (name, vals) in enumerate(T2.items()):
    xs = [j + (i - (n - 1) / 2) * w for j in range(len(G))]
    ax.bar(xs, [v[0] for v in vals], width=w * 0.92, color=COL[name], label=name,
           yerr=[v[1] for v in vals], capsize=2, error_kw=dict(lw=0.7, ecolor="#333"))
ax.set_xticks(range(len(G)))
ax.set_xticklabels(G, fontsize=8)
ax.set_ylabel("성공률 (%)", fontsize=8.5)
ax.set_ylim(0, 105)
ax.tick_params(labelsize=7.5)
ax.axvline(3.5, color="#bbb", lw=0.8, ls="--")
ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
ax.set_title("그룹별 성공률 (Table 2 값, 막대는 시드 3개의 SD)", fontsize=9, pad=8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_success_groups.png")
plt.close(fig)

# ---- 그림 2: 성공률과 지연 ----
fig, ax = plt.subplots(figsize=(6.2, 3.9), dpi=200)
for name in T2:
    ax.scatter(LAT[name], AVG[name], s=52, color=COL[name], edgecolor="#333", lw=0.6, zorder=3)
    dy = {"DP3": 2.6, "Simple DP3": -3.4}.get(name, 1.2)
    dx = {"DP3": 1.12, "Simple DP3": 0.7}.get(name, 1.22)
    ax.annotate(f"{name}\n({LAT[name]:.1f} ms, {AVG[name]:.1f}%)", (LAT[name], AVG[name]),
                xytext=(LAT[name] * dx, AVG[name] + dy), fontsize=6.8, va="center")
ax.set_xscale("log")
ax.set_xlim(2, 600)
ax.set_ylim(60, 92)
ax.set_xlabel("과제 가중 추론 지연 (ms, 로그 축)", fontsize=8.5)
ax.set_ylabel("평균 성공률 (%)", fontsize=8.5)
ax.tick_params(labelsize=7.5)
ax.grid(alpha=0.25, lw=0.5)
ax.text(2.2, 61.5, "1 NFE: MP1, FlowPolicy, UCA-Flow / 10 NFE: DP3, Simple DP3", fontsize=6.8, color="#555")
ax.set_title("성공률과 지연의 관계 (Table 2, Table 3)", fontsize=9)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_latency_tradeoff.png")
plt.close(fig)

# ---- 그림 3: 실로봇 Wilson 95% ----
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


REAL = {
    "Pick & Place": [("UCA-Flow", 13), ("MP1", 11), ("DP3", 10), ("AdaLN", 12), ("Cross Attn", 10)],
    "Drawer Close": [("UCA-Flow", 20), ("MP1", 19), ("DP3", 19), ("AdaLN", 18), ("Cross Attn", 19)],
}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=200, sharey=True)
for ax, (task, rows) in zip(axes, REAL.items()):
    for i, (name, k) in enumerate(rows[::-1]):
        lo, hi = wilson(k, 20)
        ax.plot([lo * 100, hi * 100], [i, i], color="#333", lw=1.2, zorder=2)
        ax.scatter(k / 20 * 100, i, s=40, color=COL.get(name, "#c44e3a" if name == "UCA-Flow" else "#8fb3d9"),
                   edgecolor="#333", lw=0.5, zorder=3)
        ax.text(102, i, f"{k}/20", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7.5)
    ax.set_xlim(20, 110)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.tick_params(labelsize=7.5)
    ax.set_title(task + " (20회)", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_xlabel("성공률 (%) · Wilson 95% 구간", fontsize=8)
axes[1].set_xlabel("성공률 (%) · Wilson 95% 구간", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "own_real_wilson.png")
plt.close(fig)
print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
