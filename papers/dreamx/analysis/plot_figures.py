"""리포트·블로그 그래프 3장 (직접 계산한 값으로 그림). 실행: python3 analysis/plot_figures.py

1) table1_gains.png      Table 1 회귀 평균 MSE·분류 평균 정확도 (방법별, DreamX·FAE-32 강조)
2) policy_push_ci.png    실물 서랍 닫기 reach/push 성공 수와 Wilson 95% 구간 (Table 5)
3) alignment_toy.png     정렬 toy: 시점 회전 θ에 따른 사람→로봇 행동 전이 오차, probe R²
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# 1) Table 1
names = list(tc.T1)
reg = [tc.T1[m][0][4] for m in names]
cls = [tc.T1[m][1][3] for m in names]
colors = [BLUE if m == "DreamX" else (AQUA if m == "DreamX+FAE-32" else MUTED) for m in names]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), dpi=200)
for ax, vals, title in ((axes[0], reg, "회귀 평균 MSE (낮을수록 좋음)"), (axes[1], cls, "분류 평균 정확도 % (높을수록 좋음)")):
    ys = range(len(names))
    ax.barh(list(ys), vals, color=colors, height=0.62)
    for y, v in zip(ys, vals):
        ax.text(v, y, f" {v:g}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(list(ys), names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, color=INK)
    style(ax)
axes[0].axvline(0.38, color=ORANGE, lw=1, ls="--")
axes[0].text(0.385, len(names) - 0.6, "DiLA 0.38", fontsize=7, color=ORANGE)
fig.text(0.01, 0.01, "출처: Table 1 (LARYBench). 회색 = 기준선, 파랑 = DreamX, 초록 = FAE-32 압축 코드(토큰당 32차원). 탐침은 벤치마크마다 따로 학습.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "table1_gains.png")
print("wrote", OUT / "table1_gains.png")

# 2) 실물 정책 reach/push
fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
cats = ["Reach", "Push"]
data = {"Mimic-Video": (28, 2), "DreamX": (30, 28)}
xs = [0, 1]
for i, (m, ks) in enumerate(data.items()):
    c = MUTED if m == "Mimic-Video" else BLUE
    for x, k in zip(xs, ks):
        p = k / 30
        lo, hi = wilson(k, 30)
        xx = x + (i - 0.5) * 0.3
        ax.bar(xx, p, width=0.28, color=c, label=m if x == 0 else None)
        ax.errorbar(xx, p, yerr=[[p - lo], [hi - p]], color=INK2, capsize=3, lw=0.9)
        ax.text(xx, hi + 0.03, f"{k}/30", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(xs, cats)
ax.set_ylim(0, 1.12)
ax.set_ylabel("성공 비율 (3개 설정 × 10회)", color=INK2)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
style(ax)
fig.text(0.01, 0.01, "출처: Table 5. 오차막대 Wilson 95%. Push 차이 Fisher p = 3×10⁻¹², Reach 차이 p = 0.49. 시연 45개, 체크포인트 1개.",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "policy_push_ci.png")
print("wrote", OUT / "policy_push_ci.png")

# 3) 정렬 toy
d = json.loads((ROOT / "analysis" / "alignment_toy_results.json").read_text())
ths = [int(t) for t in d["results"]["S"]]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=200)
lab = {"S": ("공유 인코더 (가중치 공유만)", BLUE), "I": ("embodiment별 인코더", ORANGE), "P": ("embodiment별 + 짝 클립 5% 정렬 손실", AQUA)}
for mode, (name, c) in lab.items():
    r = d["results"][mode]
    mean = [r[str(t)]["transfer_action_err"]["mean"] for t in ths]
    lo = [r[str(t)]["transfer_action_err"]["min"] for t in ths]
    hi = [r[str(t)]["transfer_action_err"]["max"] for t in ths]
    axes[0].plot(ths, mean, color=c, lw=2, marker="o", ms=4, label=name)
    axes[0].fill_between(ths, lo, hi, color=c, alpha=0.12, lw=0)
theory = [d["theory_shared_encoder_rotation_err"][str(t)] for t in ths]
axes[0].plot(ths, theory, color=INK2, lw=1, ls="--", label="이론: 2(1 - cos θ)")
axes[0].set_xlabel("사람 영상에서 움직임이 돌아 보이는 각도 θ (도)", color=INK2)
axes[0].set_ylabel("사람→로봇 행동 전이 오차 ‖Hz_h - a‖²/‖a‖²", color=INK2)
axes[0].set_xticks(ths)
axes[0].legend(frameon=False, fontsize=7.2, loc="upper left")
style(axes[0])
for mode, (name, c) in lab.items():
    r = d["results"][mode]
    axes[1].plot(ths, [r[str(t)]["probe_r2_human"]["mean"] for t in ths], color=c, lw=2, marker="o", ms=4, label=name)
axes[1].set_ylim(0.9, 1.01)
axes[1].set_xticks(ths)
axes[1].set_xlabel("θ (도)", color=INK2)
axes[1].set_ylabel("사람 쪽 probe R² (LARYBench식)", color=INK2)
axes[1].set_title("탐침 점수는 정렬과 무관하게 거의 1 (세 선이 겹침)", fontsize=9.5, color=INK)
style(axes[1])
fig.text(0.01, 0.01, "선형 toy(행동 2차원, 영상 움직임 8차원), 시드 5개 평균과 최소–최대 범위. 로봇 클립에만 행동 보조 손실(DreamX 식 4). analysis/alignment_toy.py",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "alignment_toy.png")
print("wrote", OUT / "alignment_toy.png")
