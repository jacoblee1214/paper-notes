"""직접 만든 그래프 (CPU). 실행: python3 analysis/plot_figures.py → report/figures/*.png

논문 그림·표가 아니라 논문 표 값과 장난감 결과로 그렸습니다. 공개본에도 들어갑니다.
1) table1_success_ci.png: Table 1 성공률. 정책은 n=50 가정의 Wilson 95% 구간, 재생은 n=200 가정
2) subset_check.png: 514개 부분집합과 7,623개 부분집합에서 JL-off 네 방법의 값 (로그 축)
3) toy_leg_tradeoff.png: 장난감 A: 접촉 가중치에 따른 추종 오차와 발 미끄럼·뜸, 정규화 세 방식
4) toy_arm_posture.png: 장난감 B: 자세 가중치에 따른 손바닥 오차, 팔꿈치 오차, seed별 분산
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

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
INK, GRID = "#0b0b0b", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TC = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
RT = json.loads((ROOT / "analysis" / "retarget_toy_results.json").read_text())
SIZE = (6.5, 3.4)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


# 1) Table 1
tasks = ["can_sort", "pouring", "coffee"]
mink_p, warp_p = [94, 74, 8], [100, 78, 34]
mink_r, warp_r = [99.5, 88.5, 50.5], [98.5, 90.5, 51.0]
fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
for ax, (title, m, w, n) in zip(axes, [("정책 성공률 (n=50 가정)", mink_p, warp_p, 50),
                                        ("재생 성공률 (n=200 가정)", mink_r, warp_r, 200)]):
    x = np.arange(len(tasks))
    for off, vals, col, lab in [(-0.2, m, MUTED, "MINK"), (0.2, w, BLUE, "WARP")]:
        ci = [wilson(round(v / 100 * n), n) for v in vals]
        lo = [max(0.0, v - 100 * c[0]) for v, c in zip(vals, ci)]
        hi = [max(0.0, 100 * c[1] - v) for v, c in zip(vals, ci)]
        ax.bar(x + off, vals, 0.38, color=col, label=lab)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=INK, lw=0.8, capsize=2)
    ax.set_xticks(x, tasks)
    ax.set_title(title, fontsize=9)
    style(ax)
axes[0].set_ylabel("성공률 (%)")
axes[0].set_ylim(0, 115)
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
save(fig, "table1_success_ci.png")

# 2) subset check (JL off, 514 vs 7.6k)
meth = ["SEW-M", "MINK-EF", "MINK-TE", "WARP"]
palm_clean_514 = [None, 0.609, 1.360, 0.0005]
palm_clean_76 = [None, 0.7856, 1.586, 1.51]
coll_514 = [0.243, 0.977, 0.640, 0.163]
coll_76 = [0.1028, 0.9203, 0.4639, 0.1112]
nnad_514 = [0.488, 0.368, 1.026, 0.289]
nnad_76 = [1.125, 0.7478, 3.370, 0.7921]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1))
panels = [("동작 충돌 비율 (낮을수록 좋음)", coll_514, coll_76, False),
          ("몸통 NNAD (낮을수록 좋음)", nnad_514, nnad_76, False),
          ("손바닥 오차, 정상 프레임 (mm, 로그)", palm_clean_514, palm_clean_76, True)]
for ax, (title, a, b, logy) in zip(axes, panels):
    x = np.arange(len(meth))
    a = [np.nan if v is None else v for v in a]
    b = [np.nan if v is None else v for v in b]
    ax.bar(x - 0.2, a, 0.38, color=BLUE, label="514 부분집합")
    ax.bar(x + 0.2, b, 0.38, color=ORANGE, label="7,623 부분집합")
    ax.set_xticks(x, meth, rotation=20, fontsize=7.5)
    ax.set_title(title, fontsize=8)
    if logy:
        ax.set_yscale("log")
    style(ax)
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
save(fig, "subset_check.png")

# 3) toy A
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
cols = {"N0_raw": MUTED, "N1_global": BLUE, "N2_segment": ORANGE}
names = {"N0_raw": "N0 정규화 없음", "N1_global": "N1 전체 높이 비", "N2_segment": "N2 링크별 비"}
for mode, d in RT["exp_A_leg"].items():
    wc = [float(k) for k in d]
    ax = axes[0]
    ax.plot([d[k]["ankle_err_mm"] for k in d], [d[k]["stance_skate_mm_mean"] + d[k]["stance_float_mm_mean"] for k in d],
            "o-", color=cols[mode], label=names[mode], ms=4)
    for k, v in zip(d, wc):
        if v in (0.0, 10.0):
            ax.annotate(f"w={v:g}", (d[k]["ankle_err_mm"], d[k]["stance_skate_mm_mean"] + d[k]["stance_float_mm_mean"]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    axes[1].plot(wc[1:], [d[k]["penetration_mm_max"] for k in d][1:], "o-", color=cols[mode], ms=4, label=names[mode])
axes[0].set_xlabel("발목 위치 오차 평균 (mm)")
axes[0].set_ylabel("입양 발 미끄럼+뜸 평균 (mm)")
axes[0].set_title("추종 오차 대 접촉 위반 (w = 접촉 가중치)", fontsize=8)
axes[1].set_xscale("log")
axes[1].set_xlabel("접촉 가중치 w (0.1 이상)")
axes[1].set_ylabel("최대 관통 깊이 (mm)")
axes[1].set_title("접촉 가중치와 관통: 입각만 막음", fontsize=8)
axes[0].legend(frameon=False, fontsize=7)
for ax in axes:
    style(ax)
save(fig, "toy_leg_tradeoff.png")

# 4) toy B
d = RT["exp_B_arm_posture_sweep"]
wp = [float(k) for k in d]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax = axes[0]
ax.plot(wp, [d[k]["palm_err_mm_mean"] for k in d], "o-", color=BLUE, ms=4, label="손바닥 오차 (mm)")
ax.plot(wp, [d[k]["elbow_err_deg_mean"] for k in d], "s-", color=ORANGE, ms=4, label="팔꿈치 자세 오차 (도)")
ax.set_xscale("symlog", linthresh=0.01)
ax.set_xlabel("자세 가중치 w_p")
ax.set_title("가중 IK: 추종 대 자세", fontsize=8)
ax.legend(frameon=False, fontsize=7)
style(ax)
ax = axes[1]
ax.plot(wp, [d[k]["seed_std_deg_mean"] for k in d], "o-", color=AQUA, ms=4)
ax.set_xscale("symlog", linthresh=0.01)
ax.set_yscale("symlog", linthresh=0.01)
ax.set_xlabel("자세 가중치 w_p")
ax.set_ylabel("seed별 해의 표준편차 (도)")
ax.set_title("초기값에 따른 해의 분산 (12 seed)", fontsize=8)
style(ax)
save(fig, "toy_arm_posture.png")
