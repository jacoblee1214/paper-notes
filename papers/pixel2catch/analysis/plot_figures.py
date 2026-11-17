"""우리가 그린 그래프 세 장 (논문 그림 아님).

own_real_ci.png    : 실로봇 Table III의 합산 비율(30x3 = 90회)과 Wilson 95% 구간
own_sim_ci.png     : 시뮬 Table II의 평균 ± SD (3 seed), 보기용
own_toy_features.png : 토이 T1의 조건별 RMSE (깊이, 접근 시간)
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_pixel_features_results.json").read_text(encoding="utf-8"))

for cand in ["Apple SD Gothic Neo", "AppleSDGothicNeo", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

METHODS = ["only-Center", "S-A RL", "Proposed"]
COL = {"only-Center": "#9aa4b2", "S-A RL": "#c9a25b", "Proposed": "#1f5fae"}

# 1. 실로봇 합산 S.R., 30x3 = 90회, Wilson 구간 (recompute.py가 계산한 값을 그대로 씀)
ci = RES["real_pooled_ci_pct_k"]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
for i, m in enumerate(METHODS):
    pt, lo, hi, k = ci[f"{m}|SR"]
    ax.bar(i, pt, color=COL[m], width=0.6)
    ax.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], color="#222", capsize=4, lw=1)
    ax.text(i, pt + 3, f"{pt:.1f}% ({k}/90)", ha="center", fontsize=8)
ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels(METHODS)
ax.set_ylabel("실로봇 성공률 S.R. (%)")
ax.set_ylim(0, 80)
ax.set_title("실로봇 성공률: 물체 3개 합산 90회, Wilson 95% 구간", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png")
plt.close(fig)

# 2. 시뮬 seen 객체, 평균 ± SD (3 seed)
seen = {"only-WH": (5.53, 0.51), "w/o-PF": (8.93, 1.04), "only-Center": (81.27, 0.76),
        "S-A RL": (63.5, 0.85), "Proposed": (84.13, 0.50)}
unseen = {"only-WH": (4.0, 0.17), "w/o-PF": (7.89, 1.46), "only-Center": (80.72, 0.38),
          "S-A RL": (65.44, 2.25), "Proposed": (84.83, 1.17)}
names = list(seen)
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
x = range(len(names))
w = 0.38
ax.bar([i - w / 2 for i in x], [seen[n][0] for n in names], w, yerr=[seen[n][1] for n in names],
       color="#1f5fae", capsize=3, label="seen")
ax.bar([i + w / 2 for i in x], [unseen[n][0] for n in names], w, yerr=[unseen[n][1] for n in names],
       color="#9bb8e0", capsize=3, label="unseen")
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylabel("시뮬 S.R. (%), 평균 ± SD")
ax.set_title("시뮬레이션 성공률 (논문 Table II 값, 3 seed)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "own_sim_ci.png")
plt.close(fig)

# 3. 토이 T1: 조건별 RMSE
conds = ["A_fixed_size", "B_size_varies", "C_size_noise_lag"]
labels = ["A 크기 고정", "B 크기 변동", "C 변동+잡음+지연"]
depth = [TOY[c]["depth_m"]["RMSE"] * 100 for c in conds]
tau = [TOY[c]["tau_s"]["RMSE"] * 1000 for c in conds]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=160)
axes[0].bar(labels, depth, color="#1f5fae")
axes[0].set_ylabel("깊이 RMSE (cm)")
axes[0].set_title("카메라 깊이 추정", fontsize=9)
axes[1].bar(labels, tau, color="#c9a25b")
axes[1].axhline(TOY["oracle_3d_state"]["tau_s"]["RMSE"] * 1000, color="#444", ls="--", lw=1)
axes[1].text(2.4, TOY["oracle_3d_state"]["tau_s"]["RMSE"] * 1000 + 0.8, "3D 오라클", fontsize=7, ha="right")
axes[1].set_ylabel("접근 시간 τ RMSE (ms)")
axes[1].set_title("z_catch 도달까지 남은 시간", fontsize=9)
for ax in axes:
    ax.tick_params(axis="x", labelsize=7)
fig.suptitle("토이 T1: 픽셀 특징만으로 얻는 정보 (구 물체, numpy, 테스트 1만여 샘플)", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "own_toy_features.png")
plt.close(fig)
print("wrote 3 figures to", FIG)
