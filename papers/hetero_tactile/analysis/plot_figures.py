"""우리가 그리는 그래프 네 장 (report/figures/own_*.png). recompute_results.json을 읽는다.

실행: python3 analysis/plot_figures.py   (matplotlib)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))

for cand in ("Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"Scratch": "#8c8c8c", "SPH": "#c98b3b", "MAE": "#4b7ba7", "HTT": "#1f5fae"}

# 1. 전체 평균 분해
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
ov = R["V3_ablation_decomposition"]["overall"]
for a, (task, title, unit) in zip(ax, (("classification", "물체 분류 정확도 (%, 높을수록 좋음)", ""), ("force", "힘 추정 3D MAE (N, 낮을수록 좋음)", ""), ("slip", "슬립 macro-F1 (%, 높을수록 좋음)", ""))):
    vals = [ov[task][m] for m in ("Scratch", "SPH", "MAE", "HTT")]
    bars = a.bar(["Scratch", "SPH", "MAE", "HTT"], vals, color=[C[m] for m in ("Scratch", "SPH", "MAE", "HTT")])
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
    a.set_title(title, fontsize=9)
    a.spines[["top", "right"]].set_visible(False)
    a.set_ylim(0, max(vals) * 1.15)
fig.suptitle("표 1·2의 센서 평균: Scratch → SPH(구조만) → MAE(+사전학습) → HTT(+정렬)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "own_decomposition.png", dpi=170)
plt.close(fig)

# 2. 실로봇 Wilson
v5 = R["V5_real_robot_table3"]
conds = ["qpos", "wrench", "T3", "SITR", "HTT"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for a, task, title in zip(ax, ("Toy screw", "Grasp tofu"), ("toy screw", "grasp tofu")):
    for i, c in enumerate(conds):
        lo, hi = v5["wilson95"][f"{task}|{c}"]
        n = {"Toy screw": {"qpos": 1, "wrench": 18, "T3": 0, "SITR": 4, "HTT": 36}, "Grasp tofu": {"qpos": 2, "wrench": 14, "T3": 2, "SITR": 7, "HTT": 22}}[task][c]
        p = 100 * n / 40
        a.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color="#1f5fae" if c == "HTT" else "#555", capsize=4)
        a.text(i + 0.12, p, f"{n}/40", fontsize=8, va="center")
    a.set_xticks(range(len(conds)))
    a.set_xticklabels(["qpos", "+wrench", "+T3", "+SITR", "+HTT"])
    a.set_title(title)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("성공률 (%) 과 Wilson 95% 구간")
fig.suptitle("표 3 (조건당 40회): 나사는 HTT가 확실히 높고, 두부는 HTT 대 wrench 구간이 겹칩니다 (p = 0.115)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "own_realrobot.png", dpi=170)
plt.close(fig)

# 3. HTT 대 MAE 시드 Welch p
rows = R["V4_seed_welch"]["htt_vs_mae_holm"]
fig, ax = plt.subplots(figsize=(7.5, 4))
import math
for i, r in enumerate(rows):
    better = (r["diff"] > 0) if r["task"] != "force" else (r["diff"] < 0)
    ax.scatter(-math.log10(r["p"]), i, color="#1f5fae" if better else "#b5483a", s=45)
ax.axvline(-math.log10(0.05), color="#999", ls="--")
ax.text(-math.log10(0.05), len(rows) - 0.3, " p = 0.05", fontsize=8, color="#666")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['task']} {r['sensor']}" for r in rows], fontsize=8)
ax.set_xlabel("-log10(p), 시드 3개 Welch (파랑: HTT가 낫다, 빨강: MAE가 낫다)")
ax.set_title("정렬 손실의 효과 (HTT 대 MAE): 12칸 중 보정 전 p < 0.05는 5칸", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "own_align_pvalues.png", dpi=170)
plt.close(fig)

# 4. 데이터셋 규모
v8 = R["V8_dataset_counts"]["headline_1.59M"]
parts = [("사전학습\n(쌍)", 710633, "#1f5fae"), ("분류\n(쌍)", 239184, "#4b7ba7"), ("힘\n(정적 210k+슬라이딩 213k)", 423284, "#c98b3b"), ("슬립\n(같은 슬라이딩)", 213415, "#b5483a")]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
left = 0
for name, v, col in parts:
    ax.barh(0, v / 1e3, left=left, color=col, edgecolor="white")
    ax.text(left + v / 2e3, 0, f"{name}\n{v/1e3:.0f}k", ha="center", va="center", color="white", fontsize=8)
    left += v / 1e3
ax.axvline(v8["distinct_frames"] / 1e3, color="k", ls="--")
ax.text(v8["distinct_frames"] / 1e3, 0.5, f" 고유 프레임 {v8['distinct_frames']/1e3:.0f}k", fontsize=8, va="bottom")
ax.set_ylim(-0.9, 0.8)
ax.text(v8["sum_frame_task_instances"] / 1e3, -0.52, f"합계 {v8['sum_frame_task_instances']/1e3:.0f}k (표제 1.59M)", fontsize=8, ha="right", va="top")
ax.set_yticks([])
ax.set_xlabel("프레임 수 (천, 센서 기본 속도 기준)")
ax.set_title("'1.6M 쌍 프레임'의 구성: 슬라이딩 213k가 두 번 셈, 쌍으로 묶인 건 950k, 사전학습은 711k", fontsize=10)
ax.set_xlim(0, 1700)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "own_dataset.png", dpi=170)
plt.close(fig)
print("ok")
