"""리포트·게시본 그래프 네 장 (직접 그린 것, 논문 그림 아님). 실행: cd analysis && python3 plot_figures.py

1) table1_ci.png       Table 1을 두 인스턴스 합산(n = 20)으로 다시 그림, Wilson 95% 구간
2) chain_vs_table2.png 단계별 성공을 곱한 예측 두 가지와 Table 2 비교
3) fig7_ci.png         Fig. 7 판독값과 n = 10 Wilson 구간
4) physics.png         손가락 패드 방향 최대 힘 분포, 적분기 스펙트럼 반경
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import physics_toy as pt
import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NOTE = "직접 그린 그래프(논문 수치 재계산). 출처: Mana, arXiv:2606.13677"


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def errbar(ax, x, k, n, color, label=None, marker="o", hollow=False):
    lo, hi = tc.wilson(k, n)
    p = k / n
    ax.plot([x, x], [lo, hi], color=color, lw=1.6, solid_capstyle="round", zorder=2)
    ax.scatter([x], [p], s=34, zorder=3, label=label, marker=marker,
               facecolor="white" if hollow else color, edgecolor=color, linewidth=1.6)


# 1) Table 1
fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200)
labels, xs = [], []
x = 0
methods = [("ours", "Mana(정책)", BLUE), ("openloop", "Mana 궤적 open-loop", ORANGE), ("teleop", "원격조작(GeoRT)", AQUA)]
names = {"tongs": "집게", "pliers": "펜치", "clothespins": "빨래집게", "syringes": "주사기"}
pk = {"grasp": "잡기", "open": "열기", "close": "닫기", "use": "사용"}
for tool, phases in tc.PHASES.items():
    for i, ph in enumerate(phases):
        for j, (m, lab, col) in enumerate(methods):
            k = round(sum(tc.T1[m][tool][i]) * tc.N)
            errbar(ax, x + (j - 1) * 0.22, k, 2 * tc.N, col, label=lab if x == 0 else None)
        labels.append(f"{names[tool]}\n{pk[ph]}")
        xs.append(x)
        x += 1
    x += 0.4
ax.set_xticks(xs, labels, fontsize=7.5)
ax.set_ylim(-0.03, 1.0)
ax.set_ylabel("성공 비율 (두 인스턴스 합, n = 20)", color=INK2)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
style(ax)
fig.text(0.01, 0.01, "Table 1 재계산. 세로선: Wilson 95% 구간. 칸 하나(n = 10)로는 22칸 중 3칸만 open-loop와 구분된다(Fisher p < 0.05). " + NOTE,
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "table1_ci.png")
plt.close(fig)

# 2) 단계 연쇄
fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
for i, tool in enumerate(tc.PHASES):
    m = [float(np.mean(p)) for p in tc.T1["ours"][tool]]
    full, cond = float(np.prod(m)), float(np.prod(m[1:]))
    k, n = tc.T2[tool]
    errbar(ax, i, k, n, BLUE, label="Table 2 관측 (n = 10)" if i == 0 else None)
    ax.scatter([i - 0.25], [full], marker="v", s=40, color=ORANGE, label="모든 단계 곱(잡기 포함)" if i == 0 else None, zorder=3)
    ax.scatter([i + 0.25], [cond], marker="^", s=40, color=AQUA, label="잡기 성공 이후 단계만 곱" if i == 0 else None, zorder=3)
ax.set_xticks(range(4), [names[t] for t in tc.PHASES])
ax.set_ylim(0, 1.0)
ax.set_ylabel("연쇄 과제 성공 비율", color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
style(ax)
fig.text(0.01, 0.01, "네 과제 합 23/40: 모든 단계 곱 기대 15.2(p = 0.0095), 잡기 이후 곱 기대 21.8(p = 0.41). " + NOTE,
         fontsize=6.3, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "chain_vs_table2.png")
plt.close(fig)

# 3) Fig. 7
fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), dpi=200, sharey=True)
titles = {"tongs_grasp_vs_trajectories": "집게 잡기 — 궤적 수",
          "syringe_use_vs_keyframes": "주사기 사용 — keyframe 수",
          "pliers_mani_vs_force_rand": "펜치 조작 — 힘 랜덤화 강도"}
for ax, (key, d) in zip(axes, tc.FIG7.items()):
    for i, v in enumerate(d["sr"]):
        errbar(ax, i, round(v * tc.N), tc.N, BLUE)
    ax.plot(range(4), d["sr"], color=BLUE, lw=1.2, alpha=0.6)
    ax.set_xticks(range(4), d["x"])
    ax.set_title(titles[key], fontsize=9.5, color=INK)
    ax.set_ylim(-0.03, 1.0)
    style(ax)
axes[0].set_ylabel("실물 성공 비율 (n = 10)", color=INK2)
fig.text(0.01, 0.01, "Fig. 7 판독값(0.1 단위, 격자와 일치)에 Wilson 95% 구간. 세 곡선 모두 경향은 유의(p ≈ 0.003–0.006)하지만 마지막 두 점은 구분되지 않는다. " + NOTE,
         fontsize=6.3, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "fig7_ci.png")
plt.close(fig)

# 4) 물리 toy
L, tau = pt.PARAMS["links_m"], pt.PARAMS["tau_max_Nm"]
grid = np.radians(np.linspace(0, 90, 31))
f_pad = []
for q1 in grid:
    for q2 in grid:
        for q3 in grid:
            q = np.array([q1, q2, q3])
            c = q.sum()
            f_pad.append(pt.max_force(pt.planar_jacobian(q, L), np.array([-math.sin(c), math.cos(c)]), tau))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), dpi=200)
ax = axes[0]
ax.hist(f_pad, bins=40, color=BLUE, edgecolor="white", linewidth=0.6)
for t, lab in ((3, "3 N"), (7, "7 N"), (10, "10 N")):
    ax.axvline(t, color=INK2, lw=1, ls="--")
    ax.text(t + 0.2, ax.get_ylim()[1] * 0.92, lab, fontsize=7.5, color=INK2)
ax.set_xlabel("패드 방향 최대 손끝 힘 (N), 관절 토크 0.7 Nm", color=INK2)
ax.set_ylabel("자세 수 (31³ 격자)", color=INK2)
ax.set_title("(a) 손가락 힘 용량은 자세에 따라 5–17 N", fontsize=9.5, color=INK)
style(ax)
ax = axes[1]
odt = np.linspace(0.01, 2.2, 200)
for meth, col, lab in (("explicit", ORANGE, "명시적 Euler"), ("semi_implicit", AQUA, "반암시적 Euler"), ("implicit", BLUE, "암시적 Euler")):
    rho = [pt.amplification(o / pt.PARAMS["dt_s"], 0.2, pt.PARAMS["dt_s"], meth) for o in odt]
    ax.plot(odt, rho, color=col, lw=2, label=lab)
ax.axhline(1, color=INK2, lw=1, ls="--")
cases = pt.p2()["cases"]
for v in cases.values():
    ax.axvline(v["omega_dt"], color=GRID, lw=0.8, zorder=0)
ax.set_xlabel("ω·dt (dt = 1/200 s)", color=INK2)
ax.set_ylabel("한 스텝 증폭률 (ζ = 0.2)", color=INK2)
ax.set_ylim(0, 2.3)
ax.set_title("(b) 1보다 크면 발산. 회색 세로선 = 가정한 도구 8가지", fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
style(ax)
fig.text(0.01, 0.01, "가정: 링크 54/38/40 mm 평면 3링크 손가락, 도구 행정 1–2 cm·유효 질량 5–50 g·필요 힘 3–7 N. analysis/physics_toy.py",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "physics.png")
plt.close(fig)
print("wrote", [p.name for p in OUT.glob("*.png") if not p.name.startswith("paper_")])
