"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다 (논문 그림은 쓰지 않는다).

  own_ablation_ci.png   RoboCasa 절제 차이와 95% 구간 (n = 1,200 가정)
  own_real_robot.png    그림 5(a) 판독 값 + 과제별 Wilson 95% 구간 (과제당 30회)
  own_robotwin_diff.png RoboTwin 과제별 ACE-Ego-0 - JoyAI-RA 차이 (표 10)
  own_hull_toy.png      토이: 같은 분포에서 볼록 껍질 면적비 분포

입력: analysis/recompute_results.json, analysis/toy_hull.py 를 다시 돌려 얻는 표본 (여기서 직접 재계산)
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1 절제
A = R["V5_ablation_pooled"]
labels = ["로봇 데이터 추가\n(Qwen 대비)", "사람 영상 추가\n(로봇 only 대비)", "시간 정렬 청킹 제거", "URDF 토큰 제거", "신뢰도 가중 제거"]
keys = ["robot_only_vs_from_qwen(68.3-65.4)", "full_vs_robot_only(72.8-68.3)", "full_vs_wo_chunking(72.8-71.7)",
        "full_vs_wo_urdf(72.8-70.9)", "full_vs_wo_human_loss(72.8-69.2)"]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
for i, k in enumerate(keys):
    d = A[k]
    lo, hi = d["ci95_diff_pp"]
    ax.errorbar(d["diff_pp"], i, xerr=[[d["diff_pp"] - lo], [hi - d["diff_pp"]]], fmt="o", color=ACC if lo > 0 else MUTED, capsize=3)
    ax.text(hi + 0.3, i, f'{d["diff_pp"]:+.1f}%p, p={d["fisher_p"]:.2f}', va="center", fontsize=8, color=INK)
ax.axvline(0, color=WARN, lw=0.8, ls="--")
ax.set_yticks(range(len(keys)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(-4, 13)
ax.set_xlabel("성공률 차이 (%p), 95% 구간 (24과제 x 50회 = 1,200회 가정, 독립 이항)")
fig.tight_layout()
fig.savefig(FIG / "own_ablation_ci.png")
plt.close(fig)

# 2 실로봇
T5 = ["Pick Tea", "Scoop\nCoffee", "Category\nSorting", "Sweep\nCubes", "Stack\nBowls", "Pack\nShoes"]
C = R["V9_real_robot"]["counts_from_figure_values"]
fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=200)
w = 0.26
for j, (m, col) in enumerate([("ACE-Ego", ACC), ("pi0.5", ACC2), ("GR00T-N1.7", "#8a8a85")]):
    for i, k in enumerate(C[m]):
        lo, hi = wilson(k, 30)
        ax.errorbar(i + (j - 1) * w, 100 * k / 30, yerr=[[100 * (k / 30 - lo)], [100 * (hi - k / 30)]], fmt="o", color=col,
                    capsize=2, ms=4, label=m if i == 0 else None)
ax.set_xticks(range(6))
ax.set_xticklabels(T5, fontsize=8)
ax.set_ylabel("성공률 (%) (그림 판독 값, 30회)")
ax.set_ylim(0, 105)
ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "own_real_robot.png")
plt.close(fig)

# 3 RoboTwin 차이 (recompute.py의 T10 표를 읽어 온다)
src = (ROOT / "analysis" / "recompute.py").read_text(encoding="utf-8")
import re  # noqa: E402
t10 = {}
for name, vals in re.findall(r'^\s+"([^"]+)": \[((?:\d+|None)(?:, (?:\d+|None)){13})\],$', src, flags=re.M):
    t10[name] = [None if v == "None" else int(v) for v in vals.split(", ")]
assert len(t10) == 50, len(t10)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=200, sharey=True)
for ax, (s, off) in zip(axs, [("Easy", 0), ("Hard", 1)]):
    d = sorted(((v[12 + off] - v[10 + off], k) for k, v in t10.items()))
    ax.bar(range(50), [x for x, _ in d], color=[ACC if x > 0 else (WARN if x < 0 else "#cfcfcb") for x, _ in d])
    ax.set_title(f"{s}: 평균 {np.mean([x for x, _ in d]):+.2f}%p", fontsize=9)
    ax.set_xlabel("과제 (차이 순)")
axs[0].set_ylabel("ACE-Ego-0 - JoyAI-RA (%p)")
fig.tight_layout()
fig.savefig(FIG / "own_robotwin_diff.png")
plt.close(fig)

# 4 볼록 껍질 토이
H = json.loads((ROOT / "analysis" / "toy_hull_results.json").read_text(encoding="utf-8"))
labs = ["정규", "정규 + 지터", "t(3)", "t(3) + 지터"]
ks = ["normal_humanjitter0.0", "normal_humanjitter0.03", "t3_humanjitter0.0", "t3_humanjitter0.03"]
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
for i, k in enumerate(ks):
    h = H[k]
    ax.plot([h["p5"], h["p95"]], [i, i], color=ACC, lw=3, solid_capstyle="round")
    ax.plot(h["median"], i, "o", color=INK)
ax.axvline(4.8, color=WARN, ls="--", lw=0.9)
ax.text(4.95, -0.35, "논문 4.8", color=WARN, fontsize=8)
ax.set_yticks(range(4))
ax.set_yticklabels(labs)
ax.invert_yaxis()
ax.set_xlabel("면적비 (419 에피소드 / 34 에피소드), 같은 분포, 5-95% 구간과 중앙값")
fig.tight_layout()
fig.savefig(FIG / "own_hull_toy.png")
plt.close(fig)
print("ok")
