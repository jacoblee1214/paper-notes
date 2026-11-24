"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_table1_diff.png   표 1: 과제별 BoxWAM - Flow 차이와 근사 95% 구간 (시행 수 300 가정)
  own_table2_pertask.png 표 8·9: 50과제별 pi0.5 + BoxWAM - pi0.5 차이 (과제당 100회, 한 시드)
  own_ablation.png      표 4·5: 상태 표현과 상태-행동 결합 절제, 근사 95% 구간 (시행 수 2,700 가정)
  own_realworld.png     표 3: 실로봇 성공 수와 Wilson 95% 구간 (과제당 50회)

입력: analysis/recompute_results.json   (먼저 python3 analysis/recompute.py)
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
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

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)


def diff_ci(p1, p2, n):
    a, b = p1 / 100, p2 / 100
    se = math.sqrt(a * (1 - a) / n + b * (1 - b) / n)
    return 100 * 1.96 * se


# ------------------------------------------------------------ 표 1
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, setting, title in zip(axes, ("clean", "randomized"), ("청정", "무작위")):
    rows = R["V3_box_vs_flow_table1"][setting]["per_task"]
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        err = diff_ci(r["flow"], r["box"], 300)
        c = ACC if r["diff"] > 0 else WARN
        ax.errorbar(r["diff"], y, xerr=err, fmt="o", color=c, ecolor=c, capsize=2, ms=5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["task"] for r in rows], fontsize=8.5)
    m = R["V3_box_vs_flow_table1"][setting]["paired"]["mean_diff"]
    ax.set_title(f"{title}: 평균 차이 {m:+.1f}p (과제 9개)", fontsize=10, color=INK)
    ax.set_xlabel("BoxWAM - Flow (성공률 포인트)", fontsize=9)
    style(ax)
fig.suptitle("표 1: 과제별 차이와 근사 95% 구간 (시행 수 300 가정, 시드 변동 제외)", fontsize=10.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_table1_diff.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------ 표 8·9
fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.2), sharey=False)
for ax, key, title in zip(axes, ("clean", "randomized"), ("청정 (표 8)", "무작위 (표 9)")):
    per = R["V4_table2_multitask"][key]["per_task"]
    items = sorted(((r["task"], r["diff"], r["base"], r["box"]) for r in per), key=lambda x: x[1])
    ys = np.arange(len(items))
    colors = [ACC if d > 0 else (WARN if d < 0 else MUTED) for _, d, _, _ in items]
    ax.barh(ys, [d for _, d, _, _ in items], color=colors, height=0.72)
    ax.set_yticks(ys)
    ax.set_yticklabels([t for t, _, _, _ in items], fontsize=6.3)
    ax.axvline(0, color=INK, lw=0.8)
    mean = np.mean([d for _, d, _, _ in items])
    ax.set_title(f"{title}: 평균 차이 {mean:+.1f}p", fontsize=10, color=INK)
    ax.set_xlabel("π0.5 + BoxWAM - π0.5 (성공률 포인트, 과제당 100회)", fontsize=8.5)
    style(ax)
fig.suptitle("표 8·9: 다과제 50개의 과제별 차이", fontsize=10.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_table2_pertask.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------ 표 4·5
names4 = [("Flow", "행동만"), ("VAE", "VAE 토큰"), ("DINO", "DINO 토큰"), ("pose6d", "6D 포즈"), ("obb2d", "2D OBB"),
          ("keypoints", "기능 키포인트"), ("ee_traj", "EE 궤적"), ("Box", "3D OBB (BoxWAM)")]
T4 = {"Flow": (73.2, 32.6), "VAE": (80.1, 34.5), "DINO": (69.9, 29.0), "pose6d": (73.2, 32.1), "obb2d": (74.2, 30.7),
      "keypoints": (75.3, 27.8), "ee_traj": (74.0, 32.6), "Box": (78.3, 37.8)}
names5 = [("Flow", "행동만"), ("separate", "완전 분리"), ("joint", "완전 결합"), ("action_then_state", "행동 → 상태"),
          ("state_then_action", "상태 → 행동 (BoxWAM)")]
T5 = {"Flow": (73.2, 32.6), "separate": (76.6, 37.4), "joint": (76.7, 32.9), "action_then_state": (76.1, 35.5),
      "state_then_action": (78.3, 37.8)}
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
for row, (names, tab, title) in enumerate(((names4, T4, "표 4: 미래 상태 표현"), (names5, T5, "표 5: 상태-행동 결합"))):
    for col, (setting, lab) in enumerate((("0", "청정"), ("1", "무작위"))):
        ax = axes[row][col]
        vals = [tab[k][col] for k, _ in names]
        se = [196 * math.sqrt(v / 100 * (1 - v / 100) / 2700) for v in vals]
        ys = np.arange(len(names))[::-1]
        colors = [ACC if k in ("Box", "state_then_action") else "#9a99a3" for k, _ in names]
        ax.barh(ys, vals, xerr=se, color=colors, height=0.62, error_kw={"ecolor": INK, "lw": 0.9, "capsize": 2})
        for y, v in zip(ys, vals):
            ax.text(v + 2.2, y, f"{v:.1f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([n for _, n in names], fontsize=8.5)
        ax.set_xlim(0, 100)
        ax.set_title(f"{title} · {lab}", fontsize=9.5, color=INK)
        ax.set_xlabel("성공률 % (막대 끝의 선: 근사 95%, 시행 수 2,700 가정)", fontsize=8)
        style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_ablation.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------ 표 3
cells = R["V5_table3_realworld"]["cells"]
tasks = ["Make Coffee", "Water Plant", "Place Pokemon"]
methods = [("DP", "Diffusion Policy", "#9a99a3"), ("Flow", "Flow (행동만)", ACC2), ("BoxWAM", "BoxWAM", ACC)]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
w = 0.26
for j, (m, lab, c) in enumerate(methods):
    xs = np.arange(3) + (j - 1) * w
    sr = [cells[f"{m}/{t}"]["SR"] for t in tasks]
    lo = [cells[f"{m}/{t}"]["SR_wilson95"][0] for t in tasks]
    hi = [cells[f"{m}/{t}"]["SR_wilson95"][1] for t in tasks]
    ax.bar(xs, sr, width=w * 0.92, color=c, label=lab)
    ax.errorbar(xs, sr, yerr=[np.array(sr) - np.array(lo), np.array(hi) - np.array(sr)], fmt="none", ecolor=INK, lw=0.9,
                capsize=2)
    for x, v in zip(xs, sr):
        ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels(["커피 만들기", "식물에 물 주기", "포켓몬 놓기"], fontsize=9.5)
ax.set_ylabel("성공률 % (과제당 50회, Wilson 95%)", fontsize=9)
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=9)
style(ax)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
ax.grid(axis="x", visible=False)
ax.set_title("표 3: 실로봇 성공 횟수 (진행률은 표에 있음)", fontsize=10.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_realworld.png", dpi=160)
plt.close(fig)
print("wrote 4 figures")
