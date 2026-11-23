"""그래프 세 장을 report/figures/own_*.png 로 저장한다 (논문 표 값을 다시 그린 것, 우리가 그린 그래프).

  own_real_ci.png   : 실로봇 범주별 성공률과 grasp별 성공 시도 수 분포 (표 3, 표 4)
  own_ablation.png  : 표 2 (Knife 훈련 인스턴스 수 절제)
  own_sim_real.png  : 시뮬레이션(표 1)과 실로봇(표 3) 범주별 SR, CSC_mean 비교

실행: python3 analysis/plot_figures.py   (numpy, matplotlib; recompute_results.json 필요)
"""
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())

for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        name = font_manager.FontProperties(fname=cand).get_name()
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GRAY, RED = "#1f5fae", "#d9822b", "#8a8a85", "#b03a2e"
CATS = ["Knife", "Lighter", "Stapler", "Tong"]

# ---- 그림 1: 실로봇 ----
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
real = R["V2_table3_from_table4"]
x = range(4)
sr = [real[c]["recomputed"]["SR_pooled"] for c in CATS]
lo = [real[c]["wilson_SR_trial_level"][0] for c in CATS]
hi = [real[c]["wilson_SR_trial_level"][1] for c in CATS]
ax[0].bar(x, sr, color=BLUE, width=0.55)
ax[0].errorbar(x, sr, yerr=[[s - l for s, l in zip(sr, lo)], [h - s for s, h in zip(sr, hi)]], fmt="none", ecolor="black", capsize=4)
for i, c in enumerate(CATS):
    ax[0].text(i, 3, f"{real[c]['recomputed']['k']:.0f}/75", ha="center", color="white", fontsize=9)
ov = R["V4_real_SR_intervals"]
ax[0].set_xticks(list(x))
ax[0].set_xticklabels(CATS)
ax[0].set_ylim(0, 105)
ax[0].set_ylabel("실로봇 SR (%)")
ax[0].set_title(f"범주별 SR와 Wilson 95% (시도 75회 단위)\n전체 257/300 = {ov['pooled_SR']}%: Wilson {ov['wilson_trial_level_n300']}, 군집 부트스트랩 {ov['cluster_bootstrap_by_grasp_60']}", fontsize=8.6)
h = ov["grasp_trial_success_hist"]
ks = list(range(6))
ax[1].bar(ks, [h[str(k)] for k in ks], color=[RED if k == 0 else BLUE for k in ks], width=0.6)
for k in ks:
    ax[1].text(k, h[str(k)] + 0.8, str(h[str(k)]), ha="center", fontsize=9)
ax[1].set_xlabel("grasp 하나의 5회 중 성공 횟수")
ax[1].set_ylabel("grasp 수 (총 60)")
ax[1].set_title(f"grasp별로 갈린다 (1–2회는 0개)\nICC {ov['icc_within_grasp']}, 유효 시도 수 약 {ov['effective_n']:.0f}", fontsize=8.6)
fig.tight_layout()
fig.savefig(OUT / "own_real_ci.png", dpi=170)
plt.close(fig)

# ---- 그림 2: 절제 ----
T2 = {1: (99.2, 29.4, 30.8, 20.7), 5: (87.0, 64.6, 32.3, 51.5), 10: (88.8, 70.5, 35.7, 59.6), 30: (93.9, 85.0, 55.5, 74.9)}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
labels = ["Anchor (훈련 인스턴스)", "Unseen (미지 인스턴스)", "Geo-OOD", "Dyn-OOD"]
cols = [GRAY, BLUE, ORANGE, RED]
ns = [1, 5, 10, 30]
for j, (lab, col) in enumerate(zip(labels, cols)):
    ax.plot(range(4), [T2[n][j] for n in ns], "-o", color=col, label=lab)
    for i, n in enumerate(ns):
        ax.text(i, T2[n][j] + 2.2, f"{T2[n][j]:.1f}", ha="center", fontsize=7.5, color=col)
ax.set_xticks(range(4))
ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("훈련 인스턴스 수 (Knife)")
ax.set_ylabel("실행 단위 SR (%)")
ax.set_ylim(0, 112)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("표 2 다시 그림: 인스턴스 수가 늘면 Unseen/OOD는 오르고 Anchor는 오르내림", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "own_ablation.png", dpi=170)
plt.close(fig)

# ---- 그림 3: 시뮬 대 실로봇 ----
v9 = R["V9_sim_vs_real"]
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
w = 0.36
xs = range(4)
ax[0].bar([i - w / 2 for i in xs], [v9[c]["sim_SR"] for c in CATS], w, color=GRAY, label="시뮬 (표 1, 전체 grasp)")
ax[0].bar([i + w / 2 for i in xs], [v9[c]["real_SR"] for c in CATS], w, color=BLUE, label="실로봇 (표 3, 시뮬 상위 5 grasp)")
ax[0].set_xticks(list(xs))
ax[0].set_xticklabels(CATS)
ax[0].set_ylim(0, 128)
ax[0].set_ylabel("SR (%)")
ax[0].legend(fontsize=7.5, loc="upper left")
ax[0].set_title("SR: 선택된 grasp와 전체 grasp는 모집단이 다르다", fontsize=9)
ax[1].bar([i - w / 2 for i in xs], [v9[c]["sim_CSCmean"] for c in CATS], w, color=GRAY, label="시뮬 CSC_mean")
ax[1].bar([i + w / 2 for i in xs], [v9[c]["real_CSCmean"] for c in CATS], w, color=BLUE, label="실로봇 CSC_mean")
ax[1].set_xticks(list(xs))
ax[1].set_xticklabels(CATS)
ax[1].set_ylabel("평균 사이클 수")
ax[1].legend(fontsize=7.5, loc="upper left")
ax[1].set_title("CSC_mean: 목표 전환 방식과 시간 절단이 서로 다름", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "own_sim_real.png", dpi=170)
plt.close(fig)
print("wrote 3 figures")
