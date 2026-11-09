"""리포트용 그래프 네 장 (직접 계산). 실행: python3 analysis/plot_figures.py

1) table4_ci.png: Table IV 성공률과 Wilson 95% 구간 (Bottle 두 모드, Plug, Wipe)
2) pairwise_tests.png: 변형 쌍 비교의 과제별 Fisher p와 과제 합산 MH p
3) actuation_toy.png: 위치 서보에서 leader 목표 vs follower 목표의 접촉력(T1)
4) sensing_toy.png: 한 프레임 qpos vs 토크의 강성 판별(T2), 마찰 속 검출률(T3), 불일치-힘 관계(T4)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import force_proxy_toy as toy  # noqa: E402

FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
T = json.loads((ROOT / "analysis" / "force_proxy_toy_results.json").read_text())
V = ["Base ACT", "ACT + Torque", "ACT-o", "ACT-o + Torque"]
VC = {"Base ACT": BLUE, "ACT + Torque": AQUA, "ACT-o": MUTED, "ACT-o + Torque": ORANGE}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# 1) Table IV 성공률 + Wilson 구간 --------------------------------------------------
from table_check import BOTTLE, PLUG, WIPE  # noqa: E402

panels = [("Soft Bottle · 무른 병 (n=10)", {v: (BOTTLE[v][0] // 10, 10) for v in V}),
          ("Soft Bottle · 단단한 병 (n=10)", {v: (BOTTLE[v][1] // 10, 10) for v in V}),
          ("Plug Insert · 1 or 2회 (n=10)", {v: (PLUG[v][1] // 10, 10) for v in V}),
          ("Board Wipe (n=20)", {v: (round(WIPE[v] * 20 / 100), 20) for v in V})]
fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.2), dpi=200, sharey=True)
for ax, (title, d) in zip(axes, panels):
    for i, v in enumerate(V):
        k, n = d[v]
        lo, hi = wilson(k, n)
        ax.plot([i, i], [lo * 100, hi * 100], color=VC[v], lw=2.2, alpha=0.45, solid_capstyle="round")
        ax.scatter([i], [k / n * 100], color=VC[v], s=38, zorder=3)
        ax.text(i, k / n * 100 + 5, f"{k}/{n}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    if title.startswith("Plug"):
        for i, v in enumerate(V):
            ax.scatter([i], [PLUG[v][0]], facecolor="white", edgecolor=VC[v], s=26, zorder=3, lw=1.2)
        ax.text(0.02, 0.97, "속 빈 점 = 1회 시도만", transform=ax.transAxes, fontsize=7, color=INK2, va="top")
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks(range(4), ["Base", "ACT+τ", "ACT-o", "ACT-o+τ"], fontsize=8)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-5, 112)
    style(ax)
axes[0].set_ylabel("성공률 (%) · 막대 = Wilson 95%")
fig.suptitle("Table IV 다시 그리기: 칸마다 시행 10–20회라 구간이 넓다", fontsize=10.5, x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "table4_ci.png", bbox_inches="tight")
plt.close(fig)

# 2) 쌍별 검정 ------------------------------------------------------------------
C2 = R["C2_pairwise_tests"]
comps = [("remove_cue (Base vs ACT-o)", "단서 제거\nBase vs ACT-o"),
         ("recover (ACT-o+τ vs ACT-o)", "토크로 회복\nACT-o+τ vs ACT-o"),
         ("improve_base (ACT+τ vs Base)", "Base 개선\nACT+τ vs Base"),
         ("explicit_only_vs_base (ACT-o+τ vs Base)", "명시적만 vs Base\nACT-o+τ vs Base"),
         ("hybrid_vs_explicit (ACT+τ vs ACT-o+τ)", "혼합 vs 명시적\nACT+τ vs ACT-o+τ")]
tmark = {"bottle": ("o", "Bottle (두 모드 합산)"), "plug": ("s", "Plug (1 or 2회)"), "wipe": ("^", "Wipe")}
fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
for r, (key, lab) in enumerate(comps):
    y = len(comps) - 1 - r
    for t, (mk, tl) in tmark.items():
        p = C2[key][t]["fisher_p"]
        ax.scatter(-math.log10(p), y + {"bottle": 0.18, "plug": 0.0, "wipe": -0.18}[t], marker=mk, s=30,
                   color=BLUE, alpha=0.75, label=tl if r == 0 else None, zorder=3)
    pm = C2[key]["MH_pooled_3_tasks"]["p"]
    ax.scatter(-math.log10(pm), y, marker="D", s=46, color=ORANGE, label="3과제 합산 (MH)" if r == 0 else None, zorder=4)
ax.axvline(-math.log10(0.05), color=INK2, lw=1, ls="--")
ax.text(-math.log10(0.05) + 0.05, len(comps) - 0.45, "p = 0.05", fontsize=7.5, color=INK2)
ax.axvline(-math.log10(0.05 / 3), color=MUTED, lw=1, ls=":")
ax.text(-math.log10(0.05 / 3) + 0.05, len(comps) - 0.75, "Bonferroni 3과제", fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(comps)), [c[1] for c in comps][::-1], fontsize=8)
ax.set_xlabel("-log10 p  (오른쪽일수록 강한 증거; Fisher 정확검정)")
ax.set_ylim(-0.6, len(comps) - 0.2)
style(ax)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.set_title("단서 제거와 토크 회복은 뚜렷하고, Base 개선은 과제 하나로는 유의하지 않다", fontsize=10, loc="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "pairwise_tests.png", bbox_inches="tight")
plt.close(fig)

# 3) 구동 toy (T1) ----------------------------------------------------------------
k_e = 2e4
t1 = T["T1_actuation"]["k_e=20000"]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200)
ax = axes[0]
zc = np.linspace(-4e-3, 8e-3, 400)
ax.plot(zc * 1e3, toy.contact_force(zc, k_e), color=INK2, lw=1.6)
ax.axhline(toy.F_DEMO, color=MUTED, lw=1, ls="--")
ax.text(-3.9, toy.F_DEMO + 0.25, f"시연 접촉력 {toy.F_DEMO:.0f} N", fontsize=7.5, color=INK2)
for name, z, c in (("follower 목표 (ACT-o)", t1["z_follower_mm"], ORANGE), ("leader 목표 (Base ACT)", t1["z_leader_mm"], BLUE)):
    f = float(toy.contact_force(z * 1e-3, k_e))
    ax.scatter([z], [f], color=c, s=42, zorder=3)
    ax.annotate(f"{name}\n{z:.1f} mm → {f:.1f} N", (z, f), textcoords="offset points",
                xytext=(-80 if c == BLUE else 14, 10 if c == BLUE else -30), fontsize=7.5, color=c)
ax.annotate("", xy=(t1["z_leader_mm"], 0.3), xytext=(t1["z_follower_mm"], 0.3),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
ax.text((t1["z_leader_mm"] + t1["z_follower_mm"]) / 2, 0.55, f"δ ≈ {t1['margin_mm']:.1f} mm", ha="center", fontsize=7.5, color=INK2)
ax.set_xlabel("명령 깊이 (표면 아래 +, mm)")
ax.set_ylabel("정상상태 접촉력 (N)")
ax.set_title("P 서보는 명령과 실제의 차이만큼 민다", fontsize=9.5, loc="left", color=INK)
style(ax)
ax = axes[1]
sig = t1["sigma_mm"]
for name, c, lab in (("leader_target", BLUE, "leader 목표"), ("follower_target", ORANGE, "follower 목표")):
    ax.plot(sig, np.array(t1["P(F>=F_min)"][name]) * 100, color=c, lw=1.8, marker="o", ms=4, label=f"{lab}: 힘 ≥ {toy.F_MIN:.0f} N")
    ax.plot(sig, np.array(t1["P(contact)"][name]) * 100, color=c, lw=1.2, ls="--", label=f"{lab}: 접촉만")
ax.set_xscale("log", base=2)
ax.set_xticks(sig, [f"{s:g}" for s in sig])
ax.set_xlabel("정책 위치 예측 오차 σ (mm)")
ax.set_ylabel("확률 (%)")
ax.set_ylim(-3, 105)
ax.set_title("예측 오차가 있어도 leader 목표는 여유 δ 덕에 힘을 유지", fontsize=9.5, loc="left", color=INK)
ax.legend(fontsize=7, frameon=False, loc="lower left")
style(ax)
fig.text(0.01, -0.03, f"가정: 말단 등가 강성 K_p = {toy.KP:.0f} N/m, 유효 중력 {toy.MG} N(I 게인 0이라 처짐 {toy.MG / toy.KP * 1e3:.1f} mm), 표면 강성 {k_e:.0e} N/m. "
         "follower 목표 쪽 힘은 거의 중력 처짐에서 나온다.", fontsize=7, color=INK2)
fig.tight_layout()
fig.savefig(OUT / "actuation_toy.png", bbox_inches="tight")
plt.close(fig)

# 4) 감지 toy (T2, T3, T4) ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3), dpi=200)
ax = axes[0]
t2 = T["T2_sensing"]
bars = [("qpos\n한 프레임", t2["balanced_acc_single_frame_qpos"], MUTED),
        ("qpos\n5프레임 차이", t2["balanced_acc_qpos_history_5frames"], BLUE),
        ("토크\n한 프레임", t2["balanced_acc_single_frame_tau"], ORANGE)]
for i, (lab, v, c) in enumerate(bars):
    ax.bar(i, v * 100, color=c, width=0.6)
    ax.text(i, v * 100 + 1.5, f"{v * 100:.0f}%", ha="center", fontsize=8, color=INK)
ax.axhline(50, color=INK2, lw=1, ls="--")
ax.set_xticks(range(3), [b[0] for b in bars], fontsize=8)
ax.set_ylim(0, 110)
ax.set_ylabel("균형 정확도 (%)")
ax.set_title("T2 단단한 병 vs 무른 병\n(같은 높이 구간)", fontsize=9, loc="left", color=INK)
style(ax)
ax = axes[1]
t3 = T["T3_proxy_quality"]
Fs = [float(k[2:-1]) for k in t3["detect_rate_at_5pct_false_alarm"]]
rates = list(t3["detect_rate_at_5pct_false_alarm"].values())
ax.plot(Fs, np.array(rates) * 100, color=ORANGE, marker="o", ms=4, lw=1.8)
ax.axvline(t3["min_force_band_N"], color=INK2, lw=1, ls="--")
ax.text(t3["min_force_band_N"] * 1.08, 12, f"정지 마찰 띠\n≈ {t3['min_force_band_N']:.2f} N", fontsize=7.5, color=INK2)
ax.set_xscale("log", base=2)
ax.set_xticks(Fs, [f"{f:g}" for f in Fs])
ax.set_xlabel("접촉력 (N)")
ax.set_ylabel("검출률 (%) · 오경보 5%")
ax.set_ylim(-3, 105)
ax.set_title("T3 전류 토크로 잡히는 최소 힘\n(기어 정지 마찰 0.25 N·m 가정)", fontsize=9, loc="left", color=INK)
style(ax)
ax = axes[2]
t, zL, zf, F = toy.t4_series()
d = (zL - zf) * 1e3
cont = F > 0.1
ax.scatter(d[~cont], F[~cont], s=8, color=MUTED, label="자유 이동", alpha=0.8)
ax.scatter(d[cont], F[cont], s=8, color=BLUE, label="접촉", alpha=0.8)
ax.set_xlabel("δ = q_leader - q_follower (mm)")
ax.set_ylabel("접촉력 (N)")
t4 = T["T4_discrepancy_vs_force"]
ax.set_title(f"T4 δ와 힘: 접촉 R² {t4['R2_delta_vs_force_contact_only']:.2f},\n전체 구간 R² {t4['R2_delta_vs_force_whole_episode']:.2f}",
             fontsize=9, loc="left", color=INK)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "sensing_toy.png", bbox_inches="tight")
plt.close(fig)
print("wrote", [p.name for p in sorted(OUT.glob("*.png"))])
