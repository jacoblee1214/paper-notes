"""리포트용 그래프 네 장 (직접 계산). 실행: python3 analysis/plot_figures.py

논문 그림은 쓰지 않는다(CC BY-SA 4.0이라 게시본 제외). 아래 그림은 논문 수치와 toy 결과로 직접 그렸다.

1) f1_sim_real.png: Table I, 시뮬·실물 F1 (Full, 기준선 최댓값, 교란 변형) 칸별 mean ± sd
2) table3_cra.png:  Table III, 실물 CRA (Oracle, Ours, Proprio-only, 자세 query-asym, MLP) mean ± sd
3) sensing_toy.png: 감각 toy, 가려진 프레임 AUC와 δ=0 대조(속도 결합)에서 생기는 Spearman
4) control_toy.png: 제어 toy, FPR에 따른 정지 성공률과 F1 → FPR 변환
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, ACC, GREY, RED = "#1b1b1a", "#55544f", "#1f5fae", "#9b9a93", "#b5462f"

import importlib.util
spec = importlib.util.spec_from_file_location("tc", ROOT / "analysis" / "table_check.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

COLS = ["Cuboid", "Pent.", "Dodec.", "Star", "Stairs", "Hex.*", "R*"]

# 1) Table I
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200, sharey=True)
x = np.arange(7)
for ax, setting, title in zip(axes, ("sim", "real"), ("시뮬레이션", "실물")):
    M = {k: np.array(v[0]) for k, v in tc.T1[setting].items()}
    S = {k: np.array(v[1]) for k, v in tc.T1[setting].items()}
    best = np.max(np.vstack([M[k] for k in tc.BASE]), axis=0)
    best_sd = np.array([S[max(tc.BASE, key=lambda k: M[k][j])][j] for j in range(7)])
    ax.axvspan(4.5, 6.5, color="#eef3fa", zorder=0)
    ax.errorbar(x - 0.12, M["full"], yerr=S["full"], fmt="o", color=ACC, ms=5, capsize=2, label="Full (제안)")
    ax.errorbar(x + 0.0, best, yerr=best_sd, fmt="s", color=GREY, ms=4, capsize=2, label="기준선 최댓값 (a–g)")
    ax.errorbar(x + 0.12, M["mism"], yerr=S["mism"], fmt="^", color=RED, ms=4, capsize=2, label="Full, 교란된 qcom")
    ax.set_xticks(x)
    ax.set_xticklabels(COLS, fontsize=7.5)
    ax.set_title(f"{title} F1 (Table I)", fontsize=9, color=INK, loc="left")
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", color="#dcdad3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("F1 (mean ± sd, n = 10)")
axes[0].text(5.5, 0.33, "* 학습에 없던 물체", fontsize=7, color=INK2, ha="center")
axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.savefig(OUT / "f1_sim_real.png", bbox_inches="tight")
plt.close(fig)

# 2) Table III, CRA
T3 = tc.T3
fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=200)
series = [("oracle", "Oracle (실물 접촉 정답)", INK2, "D"), ("ours", "Ours (제안)", ACC, "o"),
          ("proprio", "Proprio only (접촉 없음)", "#6b8e23", "s"), ("a", "(a) 자세 query-asym", GREY, "^"),
          ("c", "(c) 자세 MLP", RED, "v")]
for i, (k, lab, col, mk) in enumerate(series):
    m = np.array([T3[k][2 * j + 1][0] for j in range(7)])
    s = np.array([T3[k][2 * j + 1][1] for j in range(7)])
    ax.errorbar(x + (i - 2) * 0.13, m, yerr=s, fmt=mk, color=col, ms=4.5, capsize=2, label=lab)
ax.set_xticks(x)
ax.set_xticklabels(tc.OBJ3, fontsize=7.5)
ax.set_ylabel("CRA (rad, mean ± sd, n = 10)")
ax.set_title("Table III 실물 CRA: Proprio only가 Ours와 구분되지 않는 물체가 많다", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
ax.grid(axis="y", color="#dcdad3", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "table3_cra.png", bbox_inches="tight")
plt.close(fig)

# 3) sensing toy
sr = json.loads((ROOT / "analysis" / "sensing_toy_results.json").read_text())
s0 = next(s for s in sr["scenarios"] if s["sigma_e_rad"] == 0.005 and s["kappa_speed_in_contact"] == 0.0)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), dpi=200, gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
names = [("vision", "시각(가림 인식)"), ("proprio", "관절 |e|"), ("fusion", "융합")]
w = 0.36
for i, (k, lab) in enumerate(names):
    vo, vv = s0[f"{k}_occluded_auc"] if k != "vision" else s0["vision_occluded_auc"], None
    occ = s0["vision_occluded_auc"] if k == "vision" else s0[f"{k}_occluded_auc"]
    vis = s0["vision_visible_auc"] if k == "vision" else s0[f"{k}_visible_auc"]
    ax.bar(i - w / 2, vis, w, color=ACC, label="보이는 프레임" if i == 0 else None)
    ax.bar(i + w / 2, occ, w, color=GREY, label="가려진 프레임" if i == 0 else None)
    ax.text(i - w / 2, vis + 0.01, f"{vis:.2f}", ha="center", fontsize=7)
    ax.text(i + w / 2, occ + 0.01, f"{occ:.2f}", ha="center", fontsize=7)
ax.set_xticks(range(3))
ax.set_xticklabels([n[1] for n in names], fontsize=8)
ax.set_ylim(0.5, 0.9)
ax.set_ylabel("AUC (테스트 절반)")
ax.set_title("감각 toy: 프레임 종류별 AUC", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
nd = sr["null_delta0"]
ks = list(nd.keys())
rho = [nd[k]["spearman_absErr_vs_contact"] for k in ks]
ax.bar(range(len(ks)), rho, color=[RED if r < 0 else "#6b8e23" for r in rho])
for i, r in enumerate(rho):
    ax.text(i, r + (0.01 if r >= 0 else -0.03), f"{r:+.2f}", ha="center", fontsize=7.5)
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([k.replace("kappa=", "κ=") for k in ks], fontsize=7.5)
ax.set_ylabel("Spearman ρ(|e|, 접촉)")
ax.set_title("δ = 0 (접촉이 편차를 만들지 않음), 속도만 결합", fontsize=8.5, color=INK, loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "sensing_toy.png", bbox_inches="tight")
plt.close(fig)

# 4) control toy
cr = json.loads((ROOT / "analysis" / "control_toy_results.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), dpi=200)
ax = axes[0]
fprs = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
for K, col in ((1, RED), (2, GREY), (3, ACC)):
    y = [cr["T3_grid"][f"K={K},fnr=0.1,fpr={f}"]["success"] for f in fprs]
    ax.plot(range(len(fprs)), y, "o-", color=col, ms=4, lw=1.2, label=f"디바운스 K={K}")
ax.set_xticks(range(len(fprs)))
ax.set_xticklabels([str(f) for f in fprs], fontsize=7.5)
ax.set_xlabel("프레임당 위양성률 FPR (FNR = 0.1 고정)")
ax.set_ylabel("정지 성공률 (100 ms 안)")
ax.set_ylim(0, 1.02)
ax.set_title("제어 toy: 위양성과 디바운스", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
for row in cr["T4_f1_mapping"]:
    if row["pi"] != 0.4:
        continue
    R = [m["recall"] for m in row["R_grid"]]
    f = [m["fpr"] for m in row["R_grid"]]
    ax.plot(R, f, "o-", ms=3.5, lw=1, label=f"F1 = {row['F1']:.2f}")
ax.set_xlabel("재현율 R")
ax.set_ylabel("함의되는 FPR (π = 0.4)")
ax.set_title("논문 F1이 함의하는 위양성률", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
ax.grid(color="#dcdad3", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "control_toy.png", bbox_inches="tight")
plt.close(fig)
print("wrote figures to", OUT)
