"""OmniH2O 분석의 그래프 세 장을 만든다(우리가 그린 그림, 논문 그림은 쓰지 않는다).

입력: analysis/recompute_results.json (recompute.py가 만든 것), 표 값은 recompute.py와 같다.
출력: report/figures/own_sim_success.png, own_lfd_tasks.png, own_real_sensitivity.png
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((Path(__file__).resolve().parent / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if font_path:
    font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#8a8880"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


# 1) 시뮬 성공률 (표 1, N = 14000 가정)
rows = [("교사 (privileged)", 94.77), ("H2O [3]", 87.52), ("OmniH2O", 94.10),
        ("RL만 (w/o DAgger)", 47.11), ("History0", 93.80), ("GRU", 92.85), ("LSTM", 91.03),
        ("22점 추적", 94.72), ("8점 추적", 94.31), ("선속도 입력", 93.80)]
N = 14000
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=160)
for i, (name, s) in enumerate(rows):
    lo, hi = wilson(round(s / 100 * N), N)
    color = ACC if name == "OmniH2O" else (GREY if name != "H2O [3]" else "#b5651d")
    ax.barh(i, s, color=color, alpha=0.85, height=0.6)
    ax.errorbar(s, i, xerr=[[s - lo], [hi - s]], fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.text(s + 0.8, i, f"{s:.2f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
ax.invert_yaxis()
ax.set_xlim(40, 102)
ax.set_xlabel("시뮬레이션 성공률 (%), 오차 막대는 Wilson 95% (N = 14000 가정)", fontsize=9, color=INK)
ax.set_title("표 1 재표시: 시뮬 성공률과 구간 (논문 값, 시퀀스 수 가정)", fontsize=10, color=INK, loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_sim_success.png")
plt.close(fig)

# 2) LfD 작업별 성공 (표 17) 과 풀링 평균 (표 3 재계산)
TASKS = ["Catch-Release", "Squat", "Hammer-Catch", "RPS"]
arch = {"BC": [0, 3, 0, 1], "DP-DDIM": [6, 10, 5, 10], "DP-DDPM": [6, 10, 6, 10]}
data = {"25%": [1, 9, 3, 3], "50%": [3, 10, 6, 9], "100%": [6, 10, 6, 10]}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160, gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
w = 0.26
colors = {"BC": GREY, "DP-DDIM": "#6fa0d6", "DP-DDPM": ACC}
for j, (name, vals) in enumerate(arch.items()):
    xs = [i + (j - 1) * w for i in range(len(TASKS))]
    ax.bar(xs, vals, width=w, color=colors[name], label=name)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.2, str(v), ha="center", fontsize=7, color=INK)
ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels(TASKS, fontsize=8, color=INK)
ax.set_ylim(0, 11.5)
ax.set_ylabel("성공 횟수 (10회 중)", fontsize=9, color=INK)
ax.set_title("작업별 성공 (표 17 (c)): BC 대 확산 정책", fontsize=9.5, color=INK, loc="left")
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
pooled = {k: sum(v) / 40 * 100 for k, v in {**{x: arch[x] for x in arch}, **{}}.items()}
labels = ["BC", "DP-DDIM", "DP-DDPM", "데이터 25%", "데이터 50%", "데이터 100%"]
vals = [sum(arch["BC"]) / 40, sum(arch["DP-DDIM"]) / 40, sum(arch["DP-DDPM"]) / 40,
        sum(data["25%"]) / 40, sum(data["50%"]) / 40, sum(data["100%"]) / 40]
vals = [100 * v for v in vals]
cols = [GREY, "#6fa0d6", ACC, "#c9c4b6", "#b0aa98", "#8f8a7a"]
ax.barh(range(len(labels)), vals, color=cols, height=0.6)
for i, (v, lab) in enumerate(zip(vals, labels)):
    k = round(v / 100 * 40)
    lo, hi = wilson(k, 40)
    ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax.text(hi + 2, i, f"{k}/40", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 110)
ax.set_xlabel("풀링 성공률 (%), Wilson 95%", fontsize=8.5, color=INK)
ax.set_title("4작업 × 10회 풀링 (표 3 재계산)", fontsize=9.5, color=INK, loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_lfd_tasks.png")
plt.close(fig)

# 3) 실로봇 Eg-mpjpe 차이의 p값이 시퀀스별 변동(CV)에 따라 어떻게 변하는지 (표 2, n = 20)
sens = {r["comparison"]: r for r in RES["T2_real_sensitivity"]["rows"]}
cvs = [10, 20, 30, 50]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=160, gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]
names = ["H2O", "MLP+linvel", "GRU+linvel", "History0", "LSTM"]
vals = {"OmniH2O": 47.94, "H2O": 87.33, "MLP+linvel": 50.93, "GRU+linvel": 49.75, "History0": 83.26, "LSTM": 87.00}
order = ["OmniH2O", "MLP+linvel", "GRU+linvel", "H2O", "LSTM", "History0"]
ax.barh(range(len(order)), [vals[o] for o in order], color=[ACC if o == "OmniH2O" else GREY for o in order], height=0.6)
for i, o in enumerate(order):
    ax.text(vals[o] + 1, i, f"{vals[o]:.2f}", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8.5, color=INK)
ax.invert_yaxis()
ax.set_xlabel("Eg-mpjpe (mm), 실로봇 standing 20 시퀀스 (표 2)", fontsize=8.5, color=INK)
ax.set_title("평균만 보고됨 (SD 없음)", fontsize=9.5, color=INK, loc="left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
for comp, color in [("OmniH2O vs H2O (Eg-mpjpe)", GREY), ("OmniH2O vs MLP+linvel (Eg-mpjpe)", ACC),
                    ("OmniH2O vs History0 (Eg-mpjpe)", "#b5651d")]:
    ps = [sens[comp][f"welch_p_cv{c}"] for c in cvs]
    ax.plot(cvs, ps, marker="o", color=color, label=comp.replace(" (Eg-mpjpe)", ""))
ax.axhline(0.05, color=INK, lw=0.8, ls="--")
ax.text(cvs[-1], 0.055, "p = 0.05", fontsize=7.5, color=INK, ha="right")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}"))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("가정한 시퀀스별 SD (평균의 %)", fontsize=8.5, color=INK)
ax.set_ylabel("Welch 양측 p (log)", fontsize=8.5, color=INK)
ax.set_title("n = 20 가정: 차이의 유의성은 SD에 민감", fontsize=9.5, color=INK, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_real_sensitivity.png")
plt.close(fig)
print("wrote", [p.name for p in FIG.glob("own_*.png")])
