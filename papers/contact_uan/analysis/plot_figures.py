#!/usr/bin/env python3
"""우리가 그린 그래프 네 장 (report/figures/own_*.png). 입력: figure_values.json, recompute_results.json,
toy_contact_sensitivity_results.json (모두 같은 폴더).

  own_force_levels.png    : 그림 7C 판독으로 본 힘 수준별 오차와 감소율 (42.8 ± 12.1 재현)
  own_torque_speed.png    : 그림 7A와 7B 판독으로 본 속도 비와 토크 감소율
  own_reset_survival.png  : 그림 8A와 13의 20 s 생존율 (그림 판독 값, Wilson 구간은 n 가정)
  own_toy_contact.png     : 토이 T1, 높이 오프셋 대 접촉 법선력 (논문 그림 3 판독 값과 함께)
실행: python3 analysis/plot_figures.py
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(FIG, exist_ok=True)

for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/Library/Fonts/AppleGothic.ttf"):
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

FV = json.load(open(os.path.join(HERE, "figure_values.json")))
RR = json.load(open(os.path.join(HERE, "recompute_results.json")))
TOY = json.load(open(os.path.join(HERE, "toy_contact_sensitivity_results.json")))
BLUE, GREY, ORANGE = "#0072b2", "#707070", "#d55e00"


def wilson(p, n, z=1.959964):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1. 힘 수준별
v5 = RR["V5_fig7C"]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
x = range(4)
ax[0].bar([i - 0.2 for i in x], v5["baseline_abs_err_N"], 0.4, color=GREY, label="기준 정책")
ax[0].bar([i + 0.2 for i in x], v5["uan_abs_err_N"], 0.4, color=BLUE, label="UAN 정책")
ax[0].set_xticks(list(x), [f"{c} N" for c in v5["command_N"]])
ax[0].set_ylabel("|측정 힘 - 명령 힘| (N)")
ax[0].set_title("수준별 절대 오차 (그림 판독 값, 평균)")
ax[0].legend(frameon=False)
ax[1].bar(list(x), v5["reduction_pct_per_level"], 0.55, color=BLUE)
m, s = v5["mean_of_levels"], v5["sem_of_levels"]
ax[1].axhline(m, color=ORANGE, ls="--")
ax[1].axhspan(m - s, m + s, color=ORANGE, alpha=0.15)
ax[1].text(3.45, m + 2, f"평균 {m:.1f} ± {s:.1f} (표준오차, 4수준)", ha="right", color=ORANGE, fontsize=8.5)
ax[1].set_xticks(list(x), [f"{c} N" for c in v5["command_N"]])
ax[1].set_ylabel("오차 감소율 (%)")
ax[1].set_title("수준별 감소율과 평균")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_force_levels.png"), dpi=170)
plt.close(fig)

# 2. 속도와 토크 감소
a7, b7 = FV["fig7A_achieved_speed_mps"], FV["fig7B_torque_reduction_pct"]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
cm = a7["command"][:3]
ax[0].plot([0.8, 2.7], [0.8, 2.7], "k--", lw=0.8, label="명령")
ax[0].plot(cm, a7["baseline"][:3], "s-", color=GREY, label="기준 정책")
ax[0].plot(cm, a7["uan"][:3], "o-", color=BLUE, label="UAN 정책")
ax[0].set_xlabel("명령 속도 (m/s)")
ax[0].set_ylabel("달성 속도 (m/s, 그림 판독 값)")
ax[0].legend(frameon=False)
ax[0].set_title("달성 속도")
ks = ("1.0", "2.0", "2.5")
ax[1].bar(range(3), [b7[k]["mean"] for k in ks], 0.5, color=BLUE,
          yerr=[b7[k]["halfwidth"] for k in ks], capsize=4)
ratio = RR["V7_fig7A"]["uan_speed_ratio_to_baseline"]
for i, r in enumerate(ratio):
    ax[1].text(i, 1.0, f"속도비 {r:.2f}", ha="center", fontsize=8.5, color="white")
ax[1].set_xticks(range(3), ["1.0 m/s", "2.0 m/s", "2.5 m/s"])
ax[1].set_ylabel("토크 감소율 (%, 그림 판독 값)")
ax[1].set_title("토크 감소율과 UAN/기준 속도비")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_torque_speed.png"), dpi=170)
plt.close(fig)

# 3. 생존율
s8 = FV["fig8A_survival_pct"]
f13 = FV["fig13_survival_pct"]
labels = ["직접", "추정", "속도\n정답", "접촉\n정답", "자세\n정답", "전부\n정답"]
vals = [s8[k] for k in ("direct", "estimated", "base_oracle", "contact_oracle", "pose_oracle", "all_oracle")]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
cols = [ORANGE, BLUE, "#999", "#999", "#999", "#666"]
ax[0].bar(range(6), vals, 0.6, color=cols)
for i, v in enumerate(vals):
    lo, hi = wilson(v / 100, 4096)
    ax[0].errorbar(i, v, yerr=[[v - 100 * lo], [100 * hi - v]], color="k", lw=0.8, capsize=2)
    ax[0].text(i, v + 3, f"{v:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(range(6), labels, fontsize=8)
ax[0].set_ylim(0, 112)
ax[0].set_ylabel("20 s 생존율 (%, 그림 판독 값)")
ax[0].set_title("그림 8A (세로선: n = 4096 가정 Wilson 95%)")
names = ["Clean", "IMU\n0.5°", "IMU\n1°", "IMU\n2°", "LPF\n20 Hz", "LPF\n10 Hz", "LPF\n5 Hz"]
v13 = [f13["clean"]] + [f13["imu_bias_deg"][k] for k in ("0.5", "1", "2")] + \
      [f13["vel_lpf_hz"][k] for k in ("20", "10", "5")]
ax[1].bar(range(7), v13, 0.6, color=BLUE)
ax[1].axhline(s8["estimated"], color=ORANGE, ls="--", lw=1)
ax[1].text(6.4, s8["estimated"] - 1.6, f"그림 8A 추정 {s8['estimated']:.1f}", color=ORANGE, ha="right", fontsize=8)
ax[1].set_xticks(range(7), names, fontsize=7.5)
ax[1].set_ylim(80, 102)
ax[1].set_title("그림 13 (재학습 대조, 그림 판독 값)")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_reset_survival.png"), dpi=170)
plt.close(fig)

# 4. 토이 접촉 감도
sw = TOY["height_offset_sweep"]
dz = [-2, -1, 0, 1, 2]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.plot(dz, [sw[str(d)]["change_pct"] for d in dz], "o-", color=BLUE, label="토이 (강체 상자 17.5 kg)")
ax.plot(dz, [TOY["paper_fig3_height_read_pct"][str(d)] for d in dz], "s--", color=GREY, label="논문 그림 3 (그림 판독 값)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("루트 높이 오프셋 (mm, +는 위)")
ax.set_ylabel("수직 접촉력 변화 (%)")
ax.set_title("리셋 높이 오차와 접촉력")
ax.legend(frameon=False, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_toy_contact.png"), dpi=170)
plt.close(fig)
print("wrote 4 figures to", os.path.abspath(FIG))
