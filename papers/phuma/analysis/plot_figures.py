"""직접 만든 그래프 (논문 그림 아님).

실행: python3 analysis/plot_figures.py  -> report/figures/own_ci_forest.png, own_toy_sensitivity.png
입력: 논문 표 값(4절 Table 4, 0.15 m 임계)과 analysis/sensitivity_toy_results.json
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FONT = next((f for f in ("Noto Sans CJK KR", "Apple SD Gothic Neo")
             if any(f == x.name for x in font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False


def wilson(p, n, z=1.96):
    p = p / 100
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


# ---------- 그래프 1: 데이터셋별 성공률과 이항 95% 구간 (논문 Table 4, G1, 0.15 m) ----------
# 구간은 독립 베르누이 가정(시드 변동 미포함). 판단 근거는 03 문서.
rows = [("LaFAN1 (2.4 h)", 46.1, 28.4), ("AMASS (20.9 h)", 76.2, 70.2),
        ("Humanoid-X (231.4 h)", 50.6, 39.1), ("PHUMA (73.0 h)", 92.7, 82.9)]
N_TEST, N_UNSEEN = 7557, 504

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
for ax, idx, n, title in [(axes[0], 1, N_TEST, f"PHUMA Test (n={N_TEST:,}, 가정: 동수 독립)"),
                          (axes[1], 2, N_UNSEEN, f"Unseen Video (n={N_UNSEEN})")]:
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        p = r[idx]
        lo, hi = wilson(p, n)
        color = "#1f5fae" if "PHUMA" in r[0] else "#8a8880"
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="butt")
        ax.plot(p, y, "o", color=color, ms=6)
        ax.text(hi + 1.2, y, f"{p:.1f}", va="center", fontsize=8.5, color="#1b1b1a")
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlim(20, 100)
    ax.grid(axis="x", color="#e3e1da", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("성공률 (%), 0.15 m 임계", fontsize=8.5)
axes[0].set_yticks(list(range(len(rows)))[::-1])
axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.5)
fig.suptitle("데이터셋별 추적 성공률과 이항 95% 구간 (논문 Table 4 값, 구간은 직접 계산)", fontsize=9.5, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "own_ci_forest.png", dpi=190)
plt.close(fig)

# ---------- 그래프 2: 장난감 민감도 (합성 걸음, 논문 데이터 아님) ----------
res = json.loads((ROOT / "analysis" / "sensitivity_toy_results.json").read_text())
e1 = res["E1_height_threshold_vs_offset"]
e2 = res["E2_skate_threshold_fps_noise"]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
ax = axes[0]
offs = list(e1["noise0mm_thr1.0cm"].keys())
xs = [float(o) * 1000 for o in offs]
for key, color, ls, lab in [("noise0mm_thr1.0cm", "#1f5fae", "-", "잡음 0, 임계 1 cm"),
                            ("noise5mm_thr1.0cm", "#1f5fae", "--", "잡음 5 mm, 임계 1 cm"),
                            ("noise5mm_thr0.5cm", "#c0504d", "--", "잡음 5 mm, 임계 0.5 cm")]:
    nf = [e1[key][o]["nf"] for o in offs]
    ax.plot(xs, nf, color=color, ls=ls, marker="o", ms=3.5, lw=1.6, label=lab)
ax.set_xlabel("입각 중 발 높이 오프셋 (mm, +는 부유)", fontsize=8.5)
ax.set_ylabel("Non-Floating 통과율 (%)", fontsize=8.5)
ax.set_title("(a) 높이 임계와 잡음", fontsize=9.5, loc="left")
ax.set_ylim(-3, 103)
ax.grid(color="#e3e1da", lw=0.8)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
speeds = list(e2["fps30_noise0mm_thr0.10"].keys())
for fps, color in [(15, "#8a8880"), (30, "#1f5fae"), (60, "#c0504d")]:
    key = f"fps{fps}_noise5mm_thr0.10"
    ys = [e2[key][v] for v in speeds]
    ax.plot([float(v) for v in speeds], ys, marker="o", ms=3.5, lw=1.6, color=color,
            label=f"{fps} Hz, 잡음 5 mm")
    key0 = f"fps{fps}_noise0mm_thr0.10"
    ax.plot([float(v) for v in speeds], [e2[key0][v] for v in speeds], ls=":", color=color, lw=1.2,
            label=f"{fps} Hz, 잡음 0")
ax.set_xlabel("실제 입각 미끄럼 속도 (m/s)", fontsize=8.5)
ax.set_ylabel("Non-Skating 통과율 (%, 임계 0.10 m/s)", fontsize=8.5)
ax.set_title("(b) 프레임레이트와 위치 잡음", fontsize=9.5, loc="left")
ax.set_ylim(-3, 103)
ax.grid(color="#e3e1da", lw=0.8)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.suptitle("합성 걸음 장난감: 지표 통과율의 임계·프레임레이트 민감도 (논문 데이터 아님)", fontsize=9.5, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "own_toy_sensitivity.png", dpi=190)
plt.close(fig)
print("wrote", FIG / "own_ci_forest.png", FIG / "own_toy_sensitivity.png", "font:", FONT)
