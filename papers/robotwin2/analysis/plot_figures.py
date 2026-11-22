"""Own figures for RoboTwin 2.0 report (not paper figures).
Reads .scratch/paper_v2.txt and analysis/recompute_results.json.
Writes report/figures/own_*.png
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)
for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
BLUE, ORANGE, GREY, GREEN = "#2b6cb0", "#dd6b20", "#718096", "#2f855a"

res = json.load(open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8"))

# Fig 1: code-generation ASR (10-task subset) with 50-task mean
t8 = {"beat_block_hammer":[16,48,56,23,34,53], "handover_block":[2,41,45,17,50,27],
      "pick_diverse_bottles":[65,65,64,60,60,62], "pick_dual_bottles":[99,99,100,100,100,100],
      "place_container_plate":[66,79,91,84,84,82], "place_dual_shoes":[19,22,25,0,2,22],
      "place_empty_cup":[90,90,100,61,61,85], "place_shoe":[72,90,90,100,100,100],
      "stack_blocks_three":[1,2,4,76,76,82], "stack_blocks_two":[44,68,64,100,100,100]}
names = ["R1.0\nVanilla","R1.0\n+FB","R1.0\n+MM FB","R2.0\nVanilla","R2.0\n+FB","R2.0\n+MM FB"]
arr = np.array(list(t8.values()))
fig, ax = plt.subplots(figsize=(8, 4.2))
means = arr.mean(axis=0)
cols = [GREY]*3 + [BLUE]*3
ax.bar(range(6), means, color=cols, alpha=0.85, width=0.6)
for i in range(6):
    ax.scatter(np.full(arr.shape[0], i) + np.linspace(-0.18, 0.18, arr.shape[0]), arr[:, i], s=12, color="black", alpha=0.6, zorder=3)
    ax.text(i, means[i] + 2, f"{means[i]:.1f}", ha="center", fontsize=9)
ax.axhline(43.34, color=ORANGE, ls="--", lw=1.2)
ax.text(5.45, 44.6, "50과제 평균 43.3 (R2.0 MM FB, 표 9)", ha="right", fontsize=8, color=ORANGE)
ax.set_xticks(range(6)); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("코드 생성 성공률 (%)")
ax.set_ylim(0, 105)
ax.set_title("그림 1. 표 8의 10과제 부분집합 평균 (막대)과 과제별 값 (점)", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_codegen_asr.png")); plt.close(fig)

# Fig 2: Easy vs Hard 50-task means per policy, with per-task dots (parsed from the paper text)
lines = open(os.path.join(ROOT, ".scratch", "paper_v2.txt"), encoding="utf-8").read().split("\n")
s = next(i for i,l in enumerate(lines) if "Table 10: RoboTwin 2.0 Simulation" in l)
e = next(i for i in range(s, len(lines)) if "Average (%)" in lines[i])
row = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ]+?)\s+((?:\d+(?:\.\d+)?%\s*){10})$")
rows = [[float(x.rstrip("%")) for x in re.findall(r"\d+(?:\.\d+)?%", row.match(l).group(2))]
        for l in lines[s:e] if row.match(l)]
A = np.array(rows)
pols = ["RDT","Pi0","ACT","DP","DP3"]
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, p in enumerate(pols):
    ev, hv = A[:, 2*i], A[:, 2*i+1]
    ax.plot([i-0.12, i+0.12], [ev.mean(), hv.mean()], color=GREY, lw=1.5)
    ax.scatter(np.full(len(ev), i-0.12) + np.random.default_rng(0).uniform(-0.04, 0.04, len(ev)), ev, s=8, color=BLUE, alpha=0.35)
    ax.scatter(np.full(len(hv), i+0.12) + np.random.default_rng(1).uniform(-0.04, 0.04, len(hv)), hv, s=8, color=ORANGE, alpha=0.35)
    ax.scatter([i-0.12], [ev.mean()], s=60, color=BLUE, zorder=4, label="Easy 평균" if i == 0 else None)
    ax.scatter([i+0.12], [hv.mean()], s=60, color=ORANGE, zorder=4, label="Hard 평균" if i == 0 else None)
    ax.text(i, 104, f"{ev.mean():.1f} → {hv.mean():.1f}", ha="center", fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(pols)
ax.set_ylabel("성공률 (%)"); ax.set_ylim(-3, 112)
ax.legend(loc="center right", fontsize=8, frameon=False)
ax.set_title("그림 2. 50과제 Easy와 Hard (점은 과제, 굵은 점은 평균; 부록 K 표 10 원자료)", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_bench_easy_hard.png")); plt.close(fig)

# Fig 3: real-world Unseen-Cluttered cell with Wilson CI and cluster-bootstrap ratio
u = res["table4_pooled"]["Unseen-T"]
labels = ["10 real", "10 real\n+ 1k 합성", "합성만\n(zero-shot)"]
vals = [u["10real"]["pct"], u["real1k"]["pct"], u["zs"]["pct"]]
lo = [u["10real"]["wilson95"][0], u["real1k"]["wilson95"][0], u["zs"]["wilson95"][0]]
hi = [u["10real"]["wilson95"][1], u["real1k"]["wilson95"][1], u["zs"]["wilson95"][1]]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.bar(range(3), vals, color=[GREY, BLUE, GREEN], width=0.55, alpha=0.85)
ax.errorbar(range(3), vals, yerr=[np.array(vals)-np.array(lo), np.array(hi)-np.array(vals)], fmt="none", ecolor="black", capsize=4, lw=1)
for i, v in enumerate(vals):
    ax.text(i + 0.32, v + 1.5, f"{v:.1f}%", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("성공률 (%)"); ax.set_ylim(0, 60)
r1 = res["cluster_bootstrap_ratio_1k_vs_10real_CI95"]
r2 = res["cluster_bootstrap_ratio_zs_vs_10real_CI95"]
ax.set_title("그림 3. 미지 배경 + 정돈 셀 (4과제 × 시행 50회 추정), Wilson 95% 구간", fontsize=10)
ax.text(0.02, 0.97, f"초록 367%: 과제 군집 95% 구간 +{(r1[0]-1)*100:.0f}% ~ +{(r1[1]-1)*100:,.0f}%\n"
        f"초록 228%: 과제 군집 95% 구간 +{(r2[0]-1)*100:.0f}% ~ +{(r2[1]-1)*100:,.0f}%",
        transform=ax.transAxes, va="top", fontsize=8, color="black")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_realworld_cell.png")); plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))
