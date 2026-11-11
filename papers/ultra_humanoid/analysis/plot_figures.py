"""리포트·게시본용 그래프(우리가 그린 것만). 논문 그림은 쓰지 않는다.

출력: report/figures/table3_ci.png, table4_ci.png, power_curve.png, toy_retarget_feasibility.png
실행: python3 analysis/plot_figures.py   (table_check.py, power_toy.py, retarget_toy.py 를 먼저 돌리면 된다)
한글 폰트: 리눅스는 Noto CJK, macOS는 Apple SD Gothic Neo 순서로 찾는다.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".scratch" / "mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
INK, GREY, BLUE, ORANGE = "#1b1b1a", "#8a8880", "#1f5fae", "#c8702a"

T = json.loads((HERE / "table_check_results.json").read_text())


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


# 1) Table III: 칸별 성공률과 Wilson 구간
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
names = ["ID_points", "ID_position", "OOD_points", "OOD_position"]
labels = ["ID · 점군", "ID · 위치", "OOD · 점군", "OOD · 위치"]
x = np.arange(len(names))
for j, (key, col, off) in enumerate([("RL_off", GREY, -0.13), ("RL_on", BLUE, 0.13)]):
    vals = [T["table3"]["cells"][n][key] for n in names]
    p = [v[2] for v in vals]
    lo = [v[2] - v[3][0] for v in vals]
    hi = [v[3][1] - v[2] for v in vals]
    ax.errorbar(x + off, p, yerr=[lo, hi], fmt="o", color=col, capsize=3, ms=5,
                label="RL 미세조정 전" if key == "RL_off" else "RL 미세조정 후")
    for xi, v in zip(x + off, vals):
        ax.text(xi + 0.04, v[2] + 0.02, f"{v[0]}/20", fontsize=7.5, color=col)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("성공률 (MuJoCo, 칸당 n=20)")
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.set_title("Table III 칸별 성공률과 Wilson 95% 구간", fontsize=10, loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "table3_ci.png")
plt.close(fig)

# 2) Table IV: 실물 성공률과 Wilson 구간, 50% 기준선
fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
rows = [("밀집 참조 추적 (19/26)", T["table4"]["dense_19_of_26_wilson"], 19 / 26, BLUE),
        ("희소 목표·MoCap 세로 (8/10)", T["table4"]["cells_wilson"]["MoCap_vertical"]["ci"], 0.8, BLUE),
        ("희소 목표·MoCap 가로 (9/10)", T["table4"]["cells_wilson"]["MoCap_lateral"]["ci"], 0.9, BLUE),
        ("희소 목표·egocentric 세로 (5/10)", T["table4"]["cells_wilson"]["Ego_vertical"]["ci"], 0.5, ORANGE),
        ("희소 목표·egocentric 가로 (6/10)", T["table4"]["cells_wilson"]["Ego_lateral"]["ci"], 0.6, ORANGE)]
for i, (lab, ci, p, col) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([ci[0], ci[1]], [y, y], color=col, lw=2, alpha=0.5)
    ax.plot(p, y, "o", color=col, ms=6)
    ax.text(1.02, y, f"{p*100:.0f}%", va="center", fontsize=8, color=INK)
ax.axvline(0.5, color=GREY, ls=":", lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
ax.set_xlim(0, 1.0)
ax.set_xlabel("성공률 (선은 Wilson 95% 구간, 점선은 50%)")
ax.set_title("Table IV 실물 성공률: 분모가 작아 구간이 넓습니다", fontsize=10, loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "table4_ci.png")
plt.close(fig)

# 3) 검정력 곡선 (Fisher 정확 검정, 정확 계산)
import power_toy as PT

ns = list(range(10, 62, 4))
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
for (p1, p2), col in zip([(0.5, 0.8), (0.5, 0.6)], [BLUE, ORANGE]):
    pw = [PT.power(n, p1, p2, PT.pval_table(n)) for n in ns]
    ax.plot(ns, pw, "-o", color=col, ms=3.5, label=f"{p1:.0%} → {p2:.0%}")
ax.axhline(0.8, color=GREY, ls=":", lw=1)
ax.axvline(10, color=GREY, ls="--", lw=0.8)
ax.text(10.8, 0.05, "현재 실물 칸 n=10", fontsize=7.5, color=GREY)
ax.set_xlabel("칸당 시도 수 n (두 집단 같음)")
ax.set_ylabel("검정력 (α=0.05, Fisher 정확)")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8)
ax.set_title("실물 소표본의 검정력", fontsize=10, loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "power_curve.png")
plt.close(fig)

# 4) 리타게팅 토이: 토크 비율의 시간 변화 (원본과 x1.15 증강, 빠른 경로 T=0.45 s)
import retarget_toy as RT

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), dpi=200, sharey=True)
for ax, (scale, title) in zip(axes, [(1.0, "원본 경로"), (1.15, "x 1.15 기하 증강")]):
    for T_, col, ls in [(0.45, ORANGE, "-"), (1.5, BLUE, "-")]:
        pts = RT.hand_path(T_, scale)
        q = np.array([0.3, 1.2]); qs = []
        for p in pts:
            q, _ = RT.ik(p, q); qs.append(q.copy())
        qs = np.array(qs)
        qd = np.gradient(qs, RT.DT, axis=0); qdd = np.gradient(qd, RT.DT, axis=0)
        tau = np.array([RT.inv_dyn(qs[i], qd[i], qdd[i]) for i in range(len(qs))])
        ratio = np.max(np.abs(tau) / RT.TAU_MAX, axis=1)
        ax.plot(np.arange(len(ratio)) * RT.DT, ratio, ls, color=col, lw=1.6, label=f"{T_} s")
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel("시간 (s)")
    style(ax)
axes[0].set_ylabel("최대 토크 / 한계 (1 초과 = 불가능)")
axes[1].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("토이: 같은 손 경로의 토크 실행 가능성", fontsize=10, x=0.02, ha="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "toy_retarget_feasibility.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("*.png")))
