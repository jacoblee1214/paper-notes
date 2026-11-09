"""리포트용 그래프 네 장 (직접 그림: 논문 수치와 toy 결과만 사용, 논문 그림 없음). 실행: python3 analysis/plot_figures.py

1) table1_pos_ang.png : Table I 폐색 수준별 위치·각도 오차 (평균 ± 표준편차, 3 seeds)
2) table2_success.png : Table II 조작 성공 횟수 (평균 ± 표준편차) 와 GT 상한
3) norm_toy.png       : norm_dominance_toy 결과 (A: attention 질량, B: ridge 촉각 몫)
4) drift_toy.png      : drift_toy 결과 (A: frame-level 비와 chain 비, C: 이득 비에 따른 chain 증폭)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import table_check as tc  # noqa: E402

FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f", "#6b8e23", "#8e44ad"
C = {"V-only": BLUE, "Naive V+H": MUTED, "FingerMLP": AQUA, "16-token": PURPLE, "KineFuse": ORANGE}
MK = {"V-only": "s", "Naive V+H": "^", "FingerMLP": "D", "16-token": "v", "KineFuse": "o"}
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NT = json.loads((ROOT / "analysis" / "norm_dominance_toy_results.json").read_text())
DT = json.loads((ROOT / "analysis" / "drift_toy_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) Table I -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200)
for ax, key, sdk, ylab, title in [(axes[0], "pos", "pos_sd", "위치 오차 (cm)", "위치 오차 (폐색별)"),
                                  (axes[1], "ang", "ang_sd", "각도 오차 (도)", "각도 오차 (폐색별)")]:
    for a in tc.ARCH:
        m, s = np.array(tc.T1[a][key]), np.array(tc.T1[a][sdk])
        ax.errorbar(tc.OCC, m, yerr=s, color=C[a], marker=MK[a], ms=4.5, lw=1.4, capsize=2, label=a)
    ax.set_xlabel("폐색 비율 (%)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10, color=INK)
    style(ax)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, 0.01, "논문 Table I (3 seeds 평균 ± 표준편차). 개방 루프 추적: 이전 프레임 추정이 다음 초기값.", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "table1_pos_ang.png")
plt.close(fig)

# 2) Table II ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
x = np.arange(len(tc.OCC))
w = 0.16
for i, a in enumerate(tc.ARCH):
    m, s = np.array(tc.T2[a]["m"]), np.array(tc.T2[a]["sd"])
    ax.bar(x + (i - 2) * w, m, w, yerr=s, color=C[a], label=a, capsize=1.5, error_kw={"lw": 0.8})
ax.axhline(tc.GT, color=INK2, ls="--", lw=1)
ax.text(5.45, tc.GT + 0.3, "GT 상한 21.25", ha="right", fontsize=8, color=INK2)
ax.set_xticks(x, [f"{o}%" for o in tc.OCC])
ax.set_xlabel("폐색 비율")
ax.set_ylabel("성공 횟수 / 에피소드")
ax.set_title("Table II: 하위 과제 성공 (3 seeds 평균 ± 표준편차)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "table2_success.png")
plt.close(fig)

# 3) norm dominance toy --------------------------------------------------------
A = NT["A_attention_mass_vs_scale"]
s_axis = A["scales"]
Bres = NT["B_ridge_share_vs_scale"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200)
ax = axes[0]
ax.plot(s_axis, A["sharp_aligned_K16_raw"], color=ORANGE, marker="o", ms=3.5, label="촉각 16토큰, 정렬, 원시 노름")
ax.plot(s_axis, A["sharp_rand_K16_raw"], color=PURPLE, marker="v", ms=3.5, label="촉각 16토큰, 무작위, 원시 노름")
ax.plot(s_axis, A["sharp_aligned_K16_LN"], color=ORANGE, ls="--", label="같은 것, 노름 고정(LN 근사)")
ax.plot(s_axis, A["sharp_aligned_K1_raw"], color=BLUE, marker="s", ms=3.5, label="촉각 1토큰, 정렬, 원시 노름")
ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
ax.set_xscale("log", base=2)
ax.minorticks_off()
ax.set_xlabel("촉각 토큰 노름 / 시각 토큰 노름 (s)")
ax.set_xticks(s_axis, [str(v) for v in s_axis], fontsize=7)
ax.set_ylabel("촉각 토큰의 attention 질량")
ax.set_title("A. 노름이 attention을 가르는가 (toy)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=6.6, loc="upper left")
style(ax)
ax = axes[1]
ss = [1, 3, 10, 30, 100]
for lam, col, ls in [("lam0.1", AQUA, "-"), ("lam1.0", ORANGE, "-")]:
    raw = [Bres[f"s{v}"][lam]["share_h_raw"] for v in ss]
    std = [Bres[f"s{v}"][lam]["share_h_std"] for v in ss]
    ax.plot(ss, raw, color=col, ls=ls, marker="o", ms=3.5, label=f"원시 특징, ridge {lam[3:]}")
    ax.plot(ss, std, color=col, ls=":", marker="x", ms=4, label=f"표준화, ridge {lam[3:]}")
ax.plot(ss, [Bres[f"s{v}"]["lam0.0"]["share_h_raw"] for v in ss], color=MUTED, marker=".", label="OLS (벌점 없음)")
ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel("촉각 특징 스케일 s (두 신호의 실제 기여는 같음)")
ax.set_ylabel("예측 분산 중 촉각 몫")
ax.set_ylim(0.4, 0.85)
ax.set_title("B. ridge 벌점과 스케일의 상호작용 (toy)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=6.6, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "norm_toy.png")
plt.close(fig)

# 4) drift toy ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200)
ax = axes[0]
for key, col, lab in [("kappa_out_B_0.55", AQUA, "B: 학습 범위 밖에서도 교정"),
                      ("kappa_out_B_0.0", ORANGE, "B: 학습 범위 밖에서 교정 없음")]:
    rows = DT["B_open_loop_chain"][key]
    ax.plot([r["sigma_u_deg_per_frame"] for r in rows], [r["ratio_B_over_A"] for r in rows],
            color=col, marker="o", ms=3.5, label=lab)
ax.axhline(DT["A_frame_level"]["frame_ratio_B_over_A"], color=INK2, ls=":", lw=1)
ax.text(0.12, DT["A_frame_level"]["frame_ratio_B_over_A"] + 0.12, "frame-level 비 1.12", fontsize=7.5, color=INK2)
ax.set_xscale("log")
ax.set_xticks([0.1, 0.5, 1, 2, 3], ["0.1", "0.5", "1", "2", "3"])
ax.minorticks_off()
ax.set_xlabel("물체 회전 드리프트 σ_u (도/프레임)")
ax.set_ylabel("chain RMS 오차 비 (B/A, T=149)")
ax.set_title("A. 같은 frame-level 격차의 chain 증폭 (toy)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=7.2, loc="upper left")
style(ax)
ax = axes[1]
rows = [r for r in DT["C_analytic_gain_amplification"] if isinstance(r["chain_std_ratio_B_over_A"], float)]
ax.plot([r["kappa_A"] for r in rows], [r["chain_std_ratio_B_over_A"] for r in rows], color=BLUE, marker="o", ms=4)
ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("A의 교정 이득 κ_A (작을수록 약한 교정)")
ax.set_ylabel("chain 정상 표준편차 비 (B/A)")
ax.set_title("B. 해석해: 1.12 격차가 chain에서 커지는 조건", fontsize=10, color=INK)
ax.text(0.42, 1.27, "κ_A ≤ 0.1 이면 B의 ρ ≥ 1 이라 발산 (그림 생략)", fontsize=7.5, color=INK2)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "drift_toy.png")
plt.close(fig)
print("wrote figures to", OUT)
