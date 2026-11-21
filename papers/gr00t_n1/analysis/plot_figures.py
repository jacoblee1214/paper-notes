"""논문 표 숫자로 우리가 그린 그래프 세 장 (matplotlib).

출력: report/figures/own_robocasa_scaling.png, own_sim_pertask.png, own_real_paired.png
논문 그림은 사용하지 않는다 (arXiv 라이선스 CC BY 4.0이지만 PDF에 "All rights reserved" 표기가 있어서).
실행: python3 analysis/plot_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import sys  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tables as T  # noqa: E402

FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

available = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleSDGothicNeo"]:
    if cand in available:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
DP_C, G_C = "#8a8a84", "#1f5fae"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e3dc", lw=0.8)
    ax.set_axisbelow(True)


# 1) 벤치마크별 평균 성공률 vs 데모 수 (Table 4 인쇄 평균)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=False)
for ax, (b, vals) in zip(axes, T.T4_PRINTED_AVG.items()):
    x = [30, 100, 300]
    ax.plot(x, vals[0:3], "o-", color=DP_C, label="Diffusion Policy")
    ax.plot(x, vals[3:6], "o-", color=G_C, label="GR00T N1 (2B)")
    for xi, a, c in zip(x, vals[0:3], vals[3:6]):
        ax.annotate(f"{c - a:+.1f}", (xi, c), textcoords="offset points", xytext=(6, 4), fontsize=8, color=G_C)
    ax.set_xscale("log")
    ax.set_xticks(x, ["30", "100", "300"])
    ax.set_xlabel("과제당 시연 수")
    ax.set_title(f"{b} 평균 (과제 {T.N_TASKS[b]}개)", fontsize=10)
    ax.set_ylim(0, 85)
    style(ax)
axes[0].set_ylabel("성공률 (%)")
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.text(0.01, -0.02, "값: 논문 Table 4 하단 평균(인쇄값). 주석은 GR00T N1에서 DP를 뺀 차이(%p).", fontsize=8, color="#55544f")
fig.tight_layout()
fig.savefig(FIG / "own_robocasa_scaling.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# 2) 시뮬 과제별 차이 (100 demos): GR00T - DP, 57개 과제
rows = [(b, r[0], (r[5] - r[2])) for b in T.TASKS for r in T.TASKS[b]]
rows.sort(key=lambda t: t[2])
colors = {"RoboCasa": "#9bb7d9", "DexMG": "#e0a84f", "GR-1": G_C}
fig, ax = plt.subplots(figsize=(10.5, 4.2))
ax.bar(range(len(rows)), [d for _, _, d in rows], color=[colors[b] for b, _, _ in rows], width=0.8)
ax.axhline(0, color="#1b1b1a", lw=0.8)
ax.set_xticks([])
ax.set_ylabel("GR00T N1 - DP (%p)")
ax.set_xlabel("과제 57개를 차이 순으로 정렬 (100 demos)")
for b, c in colors.items():
    ax.bar([], [], color=c, label=f"{b} ({T.N_TASKS[b]})")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("시뮬레이션 과제별 차이 (논문 Table 4, 100 demos)", fontsize=10)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_sim_pertask.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# 3) 실로봇 과제별 쌍체 그림 (13개 과제, PnP는 seen/unseen 평균)
groups = {}
for g, name, dp10, dpf, g10, gf, n in T.REAL_ROWS:
    key = ("PnP" if g.startswith("PnP") else g, name)
    groups.setdefault(key, []).append((dp10, dpf, g10, gf))
labels, dp_f, g_f, dp_10, g_10 = [], [], [], [], []
for (g, name), vals in groups.items():
    arr = np.mean(np.array(vals), axis=0)
    labels.append(name if g != "PnP" else f"PnP: {name}")
    dp_10.append(arr[0]); dp_f.append(arr[1]); g_10.append(arr[2]); g_f.append(arr[3])
order = np.argsort(dp_f)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)
for ax, (title, a, b_) in zip(axes, [("10% 데이터", dp_10, g_10), ("전체 데이터", dp_f, g_f)]):
    y = np.arange(len(labels))
    for yi, i in zip(y, order):
        ax.plot([a[i], b_[i]], [yi, yi], color="#c9c7bf", lw=2, zorder=1)
    ax.scatter([a[i] for i in order], y, color=DP_C, s=30, zorder=2, label="Diffusion Policy")
    ax.scatter([b_[i] for i in order], y, color=G_C, s=30, zorder=3, label="GR00T N1 (2B)")
    ax.set_yticks(y, [labels[i] for i in order] if ax is axes[0] else [])
    ax.set_xlim(0, 100)
    ax.set_xlabel("과제 평균 성공률 (%)")
    ax.set_title(title, fontsize=10)
    style(ax)
axes[0].legend(fontsize=8, frameon=False, loc="lower right")
fig.text(0.01, -0.02, "값: 논문 Table 5 (과제당 10회, Machinery Packing은 5회, 부분 점수 포함). PnP는 seen과 unseen의 평균.", fontsize=8, color="#55544f")
fig.tight_layout()
fig.savefig(FIG / "own_real_paired.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
