"""재계산 결과를 그래프 세 장으로 그린다 (모두 직접 그린 그래프; 논문 그림을 쓰지 않는다).

출력: report/figures/own_robotwin_pertask.png, own_real_trials.png, own_toy_unidiffuser.png
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다
for name in ["Noto Sans CJK KR", "Noto Sans CJK JP", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

rows = list(csv.DictReader(open(HERE / "table14_robotwin.csv", encoding="utf-8")))
rw = json.load(open(HERE / "recompute_results.json", encoding="utf-8"))["real_world_trials"]
toy = json.load(open(HERE / "toy_unidiffuser_results.json", encoding="utf-8"))

# 1. RoboTwin 과제별 Motus 대 X-VLA (randomized)
motus = np.array([float(r["motus_rand"]) for r in rows])
xvla = np.array([float(r["xvla_rand"]) for r in rows])
order = np.argsort(motus - xvla)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
y = np.arange(len(order))
ax.hlines(y, xvla[order], motus[order], color="#bbbbbb", lw=2)
ax.scatter(xvla[order], y, color="#4c72b0", s=18, label="X-VLA (randomized)")
ax.scatter(motus[order], y, color="#dd8452", s=18, label="Motus (randomized)")
ax.set_yticks(y)
ax.set_yticklabels([rows[i]["task"] for i in order], fontsize=5)
ax.set_xlabel("성공률 (%) · 과제당 100회")
ax.set_title("RoboTwin 2.0 randomized: 과제별 Motus 대 X-VLA (논문 Table 14 값)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "own_robotwin_pertask.png", dpi=160)
plt.close(fig)

# 2. 실세계 부분 점수 분포 (Table 4·5를 시행 10회로 복원)
src = {
    "put_bread_ac_one": ("AC-One: Put Bread into Oven", {"pi05": [(0.0, 6), (0.2, 3), (0.4, 0), (0.6, 1), (0.8, 0), (1.0, 0)],
                                                       "wo_pt": [(0.0, 4), (0.2, 0), (0.4, 2), (0.6, 1), (0.8, 2), (1.0, 1)],
                                                       "motus": [(0.0, 5), (0.2, 0), (0.4, 1), (0.6, 0), (0.8, 1), (1.0, 3)]}),
    "get_water_agilex": ("Agilex: Get Water from Dispenser", {"pi05": [(0.0, 0), (0.4, 5), (0.8, 4), (1.0, 1)],
                                                             "wo_pt": [(0.0, 8), (0.4, 2), (0.8, 0), (1.0, 0)],
                                                             "motus": [(0.0, 0), (0.4, 0), (0.8, 2), (1.0, 8)]}),
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, (key, (title, d)) in zip(axes, src.items()):
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    width = 0.27
    for j, (k, lab, c) in enumerate([("pi05", "π0.5", "#4c72b0"), ("wo_pt", "w/o pretrain", "#999999"), ("motus", "Motus", "#dd8452")]):
        counts = {s: n for s, n in d[k]}
        ax.bar(np.array(levels) + (j - 1) * width * 0.9, [counts.get(s, 0) for s in levels], width=width * 0.9, color=c, label=lab)
    ax.set_xticks(levels)
    ax.set_xlabel("부분 점수 (subtask 합)")
    ax.set_ylabel("시행 수 (합계 10)")
    ax.set_title(title + "\n" + f"평균 π0.5 {rw[key]['pi05']['mean_pct']:.0f}% / Motus {rw[key]['motus']['mean_pct']:.0f}%", fontsize=9)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "own_real_trials.png", dpi=160)
plt.close(fig)

# 3. 토이: 조건부 잔차 분산 (독립 타임스텝 대 공유 타임스텝), 참값 1-rho^2
fig, ax = plt.subplots(figsize=(7, 3.8))
labels = ["IDM\n(a|o) 잔차 분산", "WM\n(o|a) 잔차 분산", "marginal\n(o) 분산"]
ind = [toy["indep"]["idm_resid_var"]["mean"], toy["indep"]["wm_resid_var"]["mean"], toy["indep"]["marg_o_var"]["mean"]]
sha = [toy["shared"]["idm_resid_var"]["mean"], toy["shared"]["wm_resid_var"]["mean"], toy["shared"]["marg_o_var"]["mean"]]
tru = [1 - toy["rho"] ** 2, 1 - toy["rho"] ** 2, 1.0]
x = np.arange(3)
ax.bar(x - 0.27, ind, 0.27, color="#dd8452", label="독립 τ (논문 방식)")
ax.bar(x, sha, 0.27, color="#999999", label="공유 τ (대조군)")
ax.bar(x + 0.27, tru, 0.27, color="white", edgecolor="black", label="해석적 정답")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_title(f"토이 (ρ={toy['rho']}, 시드 {len(toy['seeds'])}개 평균)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "own_toy_unidiffuser.png", dpi=160)
plt.close(fig)
print("saved", sorted(p.name for p in OUT.glob("own_*.png")))
