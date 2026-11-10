"""리포트·게시본용 그래프 (직접 계산한 결과만 사용). 실행: python3 analysis/plot_figures.py

논문 그림은 쓰지 않습니다(arXiv 비독점 배포 라이선스). 아래 네 장은 모두 이 저장소의 결과 JSON에서 그립니다.
1) joint_accuracy.png: Fig. 9 본문 수치의 MAE, RMSE, MaxAE와 RMSE/MAE 비율
2) taxel_loc.png: 토이 A, 간격과 잡음에 따른 접촉 위치 오차 중앙값
3) hysteresis.png: 토이 B, 히스테리시스 h와 드리프트 k에 따른 힘 오차 RMSE(% Fmax)
4) retarget.png: 토이 C, 접촉 가중치 lambda에 따른 접촉 높이와 관절 편차
"""
import json
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK2, GRID = "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
AN = ROOT / "analysis"
TC = json.loads((AN / "table_check_results.json").read_text())
TL = json.loads((AN / "taxel_loc_results.json").read_text())
HY = json.loads((AN / "hysteresis_results.json").read_text())
RT = json.loads((AN / "retarget_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# 1) 관절 정확도
acc = TC["joint_accuracy"]["per_mechanism"]
names = list(acc)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
x = np.arange(len(names))
w = 0.26
for i, (k, col) in enumerate([("MAE", BLUE), ("RMSE", ORANGE), ("MaxAE", AQUA)]):
    ax[0].bar(x + (i - 1) * w, [acc[n][k] for n in names], w, color=col, label=k)
ax[0].set_xticks(x, ["Type I", "Type II", "Type III"])
ax[0].set_ylabel("deg (Dynamixel XC330 기준)")
ax[0].legend(frameon=False, fontsize=7.5)
style(ax[0])
ax[0].set_title("본문 수치(Fig. 9)", fontsize=9)
ratio = [acc[n]["RMSE_over_MAE"] for n in names]
ax[1].bar(x, ratio, 0.5, color=MUTED)
ax[1].axhline(TC["joint_accuracy"]["gaussian_RMS_over_MAE"], color=INK2, ls="--", lw=0.8)
ax[1].text(2.4, 1.26, "정규분포 1.253", ha="right", va="bottom", fontsize=7, color=INK2)
ax[1].set_xticks(x, ["Type I", "Type II", "Type III"])
ax[1].set_ylim(1.0, 1.4)
ax[1].set_ylabel("RMSE / MAE")
style(ax[1])
ax[1].set_title("꼬리 두께 지표(샘플 없음)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "joint_accuracy.png", dpi=200)
plt.close(fig)

# 2) 접촉 위치 오차 vs 간격
pitches = TL["pitch_mm"]
noise = TL["noise_frac"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
cols = [BLUE, AQUA, ORANGE, "#7a4fd1"]
for j, sn in enumerate(noise):
    ax[0].plot(pitches, [TL["median_mm"][str(p)][j] for p in pitches], marker="o", color=cols[j], label=f"잡음 {sn:.2f}")
    ax[1].plot(pitches, [TL["p90_mm"][str(p)][j] for p in pitches], marker="o", color=cols[j], label=f"잡음 {sn:.2f}")
for a, t in zip(ax, ["중앙값 오차 (mm)", "90분위수 오차 (mm)"]):
    a.set_xlabel("taxel 간격 (mm, 가정)")
    a.set_ylabel(t)
    style(a)
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "taxel_loc.png", dpi=200)
plt.close(fig)

# 3) 히스테리시스와 드리프트
hs = [0.0, 0.05, 0.10, 0.20]
ks = [0.0, 0.005, 0.01, 0.02]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for k, col in zip(ks, [MUTED, BLUE, AQUA, ORANGE]):
    ax.plot(hs, [HY["rmse_pct_Fmax"][f"h={h:.2f},k={k:.3f}"] for h in hs], marker="o", color=col,
            label=f"드리프트 {k*100:.1f}%/min")
ax.set_xlabel("히스테리시스 h (언로딩 출력 감소 비율)")
ax.set_ylabel("힘 오차 RMSE (% Fmax)")
ax.legend(frameon=False, fontsize=7)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "hysteresis.png", dpi=200)
plt.close(fig)

# 4) 접촉 라벨 가중치
keys = [k for k in RT["runs"] if k.endswith("eps=0")]
lams = [0.0, 1.0, 10.0, 100.0]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for eps, col, lab in [(0.0, BLUE, "라벨 정확"), (0.1, ORANGE, "라벨 10% 오류")]:
    rows = [RT["runs"][f"lam={l:g},eps={eps:g}"] for l in lams]
    ax[0].plot(lams, [r["contact_height_mm_median"] for r in rows], marker="o", color=col, label=lab)
    ax[1].plot(lams, [r["joint_dev_free_deg_p90"] for r in rows], marker="o", color=col, label=lab)
ax[0].set_xscale("symlog", linthresh=1)
ax[1].set_xscale("symlog", linthresh=1)
ax[0].set_xlabel("접촉 가중치 lambda")
ax[0].set_ylabel("접촉 프레임 손끝 높이 중앙값 (mm)")
ax[1].set_xlabel("접촉 가중치 lambda")
ax[1].set_ylabel("비접촉 프레임 관절 편차 90분위 (도)")
ax[0].legend(frameon=False, fontsize=7)
for a in ax:
    style(a)
fig.tight_layout()
fig.savefig(OUT / "retarget.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
