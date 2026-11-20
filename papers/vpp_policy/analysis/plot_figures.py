"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py   (recompute.py, toy_future_error.py 결과 JSON을 읽는다)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


plt.rcParams.update({"font.family": korean_font(), "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
PAL = {"Diffusion Policy": "#9a9a95", "Susie": "#7fa6d6", "GR-1": "#4f7fbf", "VPP": "#1f5fae"}
rc = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 1. CALVIN 평균 완료 과제 수 (표 1 재계산). 오차막대는 채점 1000 체인의 표집 오차 1.96 SE.
methods = ["RT-1", "Diffusion Policy", "RoboFlamingo", "UniPi", "MDT", "Susie", "GR-1", "Vidman", "RoboUniview", "VPP"]
vals = [rc["calvin_table1"][m]["avg_len_recomputed"] for m in methods]
ses = [1.96 * rc["calvin_table1"][m]["se_chains"] for m in methods]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
cols = ["#1f5fae" if m == "VPP" else "#b9c6d8" for m in methods]
ax.bar(range(len(methods)), vals, yerr=ses, color=cols, capsize=3)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=30, ha="right")
ax.set_ylabel("Avg. Len (CALVIN ABC→D)")
ax.set_ylim(0, 5)
for i, v in enumerate(vals):
    ax.text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=8)
ax.set_title("표 1 재계산: 평균 완료 과제 수와 채점 표집 오차 (1.96 SE, 학습 시드 변동 미포함)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_calvin_avglen.png", dpi=170)
plt.close(fig)

# 2. 실로봇 성공률과 Wilson 95% 구간 (표 6, 7의 범주 값을 시도 수로 합산)
settings = ["Panda seen", "Panda unseen", "Dex seen", "Dex unseen", "Dex tool"]
models = ["Diffusion Policy", "Susie", "GR-1", "VPP"]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
width = 0.2
for j, m in enumerate(models):
    xs, ys, lo, hi = [], [], [], []
    for i, s in enumerate(settings):
        r = rc["real_world"][s][m]
        xs.append(i + (j - 1.5) * width)
        ys.append(r["rate_recomputed"])
        lo.append(r["rate_recomputed"] - r["wilson95"][0])
        hi.append(r["wilson95"][1] - r["rate_recomputed"])
    ax.bar(xs, ys, width=width, color=PAL[m], label=m, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
ax.set_xticks(range(len(settings)))
ax.set_xticklabels([f"{s}\n(n={rc['real_world'][s]['VPP']['trials']})" for s in settings], fontsize=8)
ax.set_ylabel("성공률 (재계산)")
ax.set_ylim(0, 1)
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
fig.savefig(FIG / "own_real_wilson.png", dpi=170)
plt.close(fig)

# 3. 토이: 학습(σ=0.1)과 시험의 예측 오차가 다를 때 MSE 대 기준선 (ρ=0.9, 관측 B)
toy = json.loads((ROOT / "analysis" / "toy_future_error_results.json").read_text(encoding="utf-8"))
row = next(r for r in toy["rows"] if r["rho"] == 0.9 and r["setting"] == "B_current_plus_prev")
sig = [m["sigma_test"] for m in row["mismatch"]]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(sig, [m["mse"] for m in row["mismatch"]], "o-", color="#1f5fae", label="학습 σ=0.1, 시험 σ 변화")
ax.plot([m["sigma"] for m in row["matched"]], [m["mse"] for m in row["matched"]], "s--", color="#9a9a95",
        label="학습과 시험 모두 σ (대조)")
ax.axhline(row["baseline_mse"], color="#c0504d", lw=1, label="예측 입력 없는 기준선")
if row["crossover_sigma_test"]:
    ax.axvline(row["crossover_sigma_test"], color="#c0504d", lw=0.8, ls=":")
    ax.text(row["crossover_sigma_test"] + 0.03, 2.6,f"교차 σ≈{row['crossover_sigma_test']:.2f}", color="#c0504d", fontsize=8)
ax.set_xlabel("시험 때 예측 오차 σ (상태 단위)")
ax.set_ylabel("행동 예측 MSE (토이)")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("토이: 예측 오차가 커지면 이득이 사라지는 지점 (ρ=0.9, 2-차원, 1 시드)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_toy_future_error.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
