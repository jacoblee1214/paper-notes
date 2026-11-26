"""우리가 만든 그래프를 report/figures/ 에 저장한다(논문 그림은 여기서 만들지 않는다).

  own_is_heatmap.png     표 1 의도 점수(IS) 전체 열 장(모델 x 장면·수준)
  own_language_free.png  언어를 쓰지 않는 정책의 L0 상한(장면 엔트로피에서)과 빈 언어 π0.5의 L0 완료율
  own_scene3.png         Scene 3의 수준별 IS와 CR (L2의 CR이 모든 모델에서 바닥)
  own_cr_grid.png        Scene 4 L1의 CR 보고값이 놓이는 격자(과제 14개 대 15개, 과제당 10회 가정)
  own_real_robot.png     실로봇 파일럿: 지시문별 성공률, 시행 단위와 지시문 단위 불확실성

입력: analysis/recompute.py 의 표 값과 analysis/recompute_results.json (먼저 python3 analysis/recompute.py)
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recompute import EVAL_TASKS, MODEL_LABEL, ROUNDS, TABLE1, wilson  # noqa: E402

ROOT = HERE.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
LEVELS = ["L0", "L1", "L2", "L3"]

# ---------------------------------------------------------------- 그래프 1: IS 열지도
order = ["PI05", "ACoT", "PI0", "FASTWAM", "MOTUS", "Lingbot", "XVLA", "GROOT", "OFT", "PI05_empty"]
mat = np.array([[TABLE1[m]["scenes"][s][3 * li] for s in range(4) for li in range(4)] for m in order])
fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                color="white" if mat[i, j] > 55 else INK)
ax.set_xticks(range(16))
ax.set_xticklabels([f"S{s+1}\n{L}" for s in range(4) for L in LEVELS], fontsize=7)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([MODEL_LABEL[m] for m in order], fontsize=8)
for x in (3.5, 7.5, 11.5):
    ax.axvline(x, color="white", lw=2.5)
ax.axhline(len(order) - 1.5, color="white", lw=2.5)
ax.spines[:].set_visible(False)
ax.set_title("표 1의 의도 점수(IS, %). 아래 줄은 빈 언어 지시를 준 π0.5(대조)", fontsize=9, loc="left")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
fig.tight_layout()
fig.savefig(FIG / "own_is_heatmap.png")
plt.close(fig)

# ---------------------------------------------------------------- 그래프 2: 언어 없는 정책의 상한
LF = RES["V4_entropy_language_free"]
bound = [LF["language_free_bound_per_scene"][f"S{s}"] * 100 for s in range(1, 5)]
ctrl = [TABLE1["PI05_empty"]["scenes"][s][2] for s in range(4)]
full = [TABLE1["PI05"]["scenes"][s][2] for s in range(4)]
ns = [EVAL_TASKS[f"S{s}"][0] * ROUNDS for s in range(1, 5)]
fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=200)
x = np.arange(4)
w = 0.26
ax.bar(x - w, bound, w, color="#c9d3e3", label="언어 없는 정책의 상한(1/그룹 크기의 가중 평균)")
err = [[c - 100 * wilson(round(c / 100 * n), n)[0] for c, n in zip(ctrl, ns)],
       [100 * wilson(round(c / 100 * n), n)[1] - c for c, n in zip(ctrl, ns)]]
ax.bar(x, ctrl, w, yerr=err, color=ACC2, capsize=2, label="빈 언어 π0.5의 L0 CR (시행 수 가정: 과제 x 10회)")
ax.bar(x + w, full, w, color=ACC, label="언어를 준 π0.5의 L0 CR")
for xi, v, hi_e in zip(x, ctrl, err[1]):
    ax.text(xi, v + hi_e + 1.5, f"{v:.1f}", ha="center", fontsize=7, color=ACC2)
for xi, v in zip(x + w, full):
    ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7, color=ACC)
ax.set_xticks(x)
ax.set_xticklabels(["S1", "S2", "S3", "S4"])
ax.set_ylabel("%")
ax.set_ylim(0, 112)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
ax.set_title("언어를 쓰지 않으면 L0에서 얼마까지 맞을 수 있나", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_language_free.png")
plt.close(fig)

# ---------------------------------------------------------------- 그래프 3: Scene 3 수준별
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), dpi=200, sharey=True)
main = [m for m in TABLE1 if m != "PI05_empty"]
cmap = plt.get_cmap("tab10")
for k, m in enumerate(main):
    row = TABLE1[m]["scenes"][2]
    for ax, off, ttl in ((axes[0], 0, "의도 점수 IS"), (axes[1], 2, "완료율 CR")):
        ax.plot(range(4), [row[3 * li + off] for li in range(4)], marker="o", ms=3, lw=1.2, color=cmap(k),
                label=MODEL_LABEL[m])
        ax.set_title(f"Scene 3 {ttl}", fontsize=9, loc="left")
for ax in axes:
    ax.set_xticks(range(4))
    ax.set_xticklabels(LEVELS)
    ax.axvspan(1.6, 2.4, color="#f1ece1", zorder=0)
axes[0].axhline(30, color=MUTED, lw=0.8, ls="--")
axes[0].text(-0.25, 24, "30 = 코드의 단계 1 가중치 0.3", fontsize=6.5, color=MUTED, ha="left")
axes[0].set_ylabel("%")
axes[1].legend(fontsize=6.5, ncol=2, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "own_scene3.png")
plt.close(fig)

# ---------------------------------------------------------------- 그래프 4: Scene 4 L1의 CR 격자
vals = [v["scenes"][3][5] for v in TABLE1.values()]
fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=200)
g14, g15 = 100 / 140, 100 / 150
for k in range(0, 61):
    ax.vlines(k * g14, 0.62, 1.0, color="#c9d3e3", lw=0.9)
    ax.vlines(k * g15, 0.0, 0.38, color="#f0d7bf", lw=0.9)
ax.scatter(vals, [0.5] * len(vals), s=22, color=INK, zorder=3)
for v in vals:
    ax.vlines(v, 0.38, 0.62, color=INK, lw=0.5, alpha=0.4)
ax.text(34.5, 0.85, "과제 14개 x 10회의 격자(0.714%)", fontsize=7, color=ACC, ha="right")
ax.text(34.5, 0.14, "과제 15개 x 10회의 격자(0.667%)", fontsize=7, color=ACC2, ha="right")
ax.set_xlim(-0.5, 35)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("Scene 4 L1의 CR 보고값 (%, 10개 행, 빈 언어 대조 포함)")
ax.spines["left"].set_visible(False)
ax.set_title("보고값 10개가 모두 15개 과제의 격자 위에 있고, 14개 과제의 격자 위에는 1개만 있습니다", fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_cr_grid.png")
plt.close(fig)

# ---------------------------------------------------------------- 그래프 5: 실로봇 파일럿
rb = RES["V8_real_robot"]["instruction_level"]
tr, he = np.array(rb["train_rates"]) * 100, np.array(rb["held_rates"]) * 100
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=200)
for k, (arr, c, name) in enumerate(((tr, ACC, "학습 지시문 8개"), (he, ACC2, "보류 지시문 8개"))):
    ax.scatter(k + rng.uniform(-0.12, 0.12, len(arr)), arr, s=26, color=c, alpha=0.85, zorder=3)
    ax.hlines(arr.mean(), k - 0.3, k + 0.3, color=c, lw=2)
    boots = [rng.choice(arr, len(arr)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ax.vlines(k + 0.35, lo, hi, color=c, lw=2.2)
    kk = round(arr.mean() / 100 * 40)
    wl, wh = wilson(kk, 40)
    ax.vlines(k - 0.35, 100 * wl, 100 * wh, color=c, lw=2.2, alpha=0.45)
ax.set_xticks([0, 1])
ax.set_xticklabels(["학습 지시문 8개 (20/40)", "보류 지시문 8개 (6/40)"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("지시문별 성공률 (%, 5회 시도)")
ax.set_title("실로봇: 굵은 막대 = 평균, 왼쪽 연한 선 = 시행 단위 Wilson, 오른쪽 선 = 지시문 부트스트랩", fontsize=7.2, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_real_robot.png")
plt.close(fig)
print("wrote figures in", FIG)
