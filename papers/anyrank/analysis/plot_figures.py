"""자체 그래프 3장. 실행: python3 analysis/plot_figures.py → report/figures/own_*.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# 1. 정책별 평균 + 95% 구간
d = pd.read_csv(ROOT / "repo/data/roboarena/data2/per_trial_progress_data.csv")
names = {"pi0_fast_droid": "A π0-FAST", "paligemma_fast_droid": "B PG-FAST", "paligemma_fast_specialist_droid": "C PG-FAST-Spec",
         "paligemma_vq_droid": "D PG-FSQ", "paligemma_diffusion_droid": "E PG-Diff", "pi0_droid": "F π0", "paligemma_binning_droid": "G PG-Bin"}
m, ci, n = [], [], []
for k in names:
    x = d[k].dropna().values
    m.append(x.mean()); ci.append(1.96 * x.std(ddof=1) / np.sqrt(len(x))); n.append(len(x))
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.errorbar(range(7), m, yerr=ci, fmt="o", capsize=4, color="#1f5fae")
ax.set_xticks(range(7)); ax.set_xticklabels([f"{v}\nn={k}" for v, k in zip(names.values(), n)], fontsize=7)
ax.set_ylabel("평균 진행 점수 (95% 구간)"); ax.set_title("RoboArena 7정책: 평균과 구간 (저장소 per-trial 데이터)", fontsize=10)
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "own_means_ci.png", dpi=170); plt.close(fig)

# 2. gamma별 위험 배분
R = json.loads((ROOT / "analysis/recompute_results.json").read_text())["V3_alpha_allocation_7pol_wm"]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
gs = list(R)
ax.bar(gs, [R[g]["ratio"] for g in gs], color="#1f5fae")
for i, g in enumerate(gs):
    ax.text(i, R[g]["ratio"], f'{R[g]["ratio"]:.2f}', ha="center", va="bottom", fontsize=8)
ax.set_yscale("log"); ax.set_xlabel("γ"); ax.set_ylabel("초기 α 최대/최소 비 (21가설)")
ax.set_title("식 (5)의 γ별 위험 배분 비 (7정책, 세계모델 사전)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_alpha_gamma.png", dpi=170); plt.close(fig)

# 3. 토이 총 시행
T = json.loads((ROOT / "analysis/toy_anyrank_results.json").read_text())
keys = ["bonferroni_uniform", "graphical_uniform", "graphical_paper_eq4", "graphical_code_rule"]
lab = ["Bonferroni\n균등", "그래프 γ=0\n균등", "그래프\n논문 식 (4)", "그래프\n코드 규칙"]
fig, ax = plt.subplots(figsize=(6, 3.3))
ax.bar(lab, [T[k]["mean_total_samples"] for k in keys], color=["#d9822b", "#1f5fae", "#7c6fd6", "#2a9d8f"])
for i, k in enumerate(keys):
    ax.text(i, T[k]["mean_total_samples"], f'{T[k]["mean_total_samples"]:.0f}', ha="center", va="bottom", fontsize=8)
ax.set_ylabel("6쌍 판정까지 총 시행 (평균, 20회)"); ax.set_title("자체 e-process 토이 (RoboArena 4정책 데이터, α=0.1)", fontsize=10)
ax.tick_params(axis="x", labelsize=8); fig.tight_layout(); fig.savefig(FIG / "own_toy_samples.png", dpi=170); plt.close(fig)
