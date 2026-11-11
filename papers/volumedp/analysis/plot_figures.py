"""직접 만든 그래프 (논문 그림을 쓰지 않음). 결과: report/figures/own_*.png

실행: python3 analysis/plot_figures.py
입력: 원문 표 값(하드코드, 출처 Table I, III, V)과 analysis/volumedp_toy_results.json
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(FIG, exist_ok=True)
from matplotlib import font_manager  # noqa: E402

# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾습니다.
KO_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
ko_font = next((p for p in KO_FONT_CANDIDATES if os.path.exists(p)), None)
if ko_font:
    font_manager.fontManager.addfont(ko_font)
    plt.rcParams["font.family"] = [font_manager.FontProperties(fname=ko_font).get_name(), "DejaVu Sans"]
else:
    plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C_DP, C_OURS, C_DIT = "#8c8c8c", "#1f6fb4", "#d9822b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return c - h, c + h


# 1. Table V: 과제별 성공률과 Wilson 95% 구간
tasks = ["Placing Bowl", "Microwave", "Door Opening", "Nut onto Screw"]
dp = [16, 12, 13, 5]
ours = [17, 15, 17, 12]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
for i, (d, o) in enumerate(zip(dp, ours)):
    for j, (k, col, lab) in enumerate([(d, C_DP, "DP"), (o, C_OURS, "제안")]):
        x = i + (j - 0.5) * 0.36
        lo, hi = wilson(k, 20)
        ax.errorbar(x, k / 20, yerr=[[k / 20 - lo], [hi - k / 20]], fmt="o", color=col, capsize=4,
                    label=lab if i == 0 else None, ms=7)
        ax.text(x + 0.07, k / 20, f"{k}/20", va="center", fontsize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(tasks, fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_ylabel("성공률 (Wilson 95%)")
ax.set_title("실물 Table V 과제별 (n = 20)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_real_per_task.png"))
plt.close(fig)

# 2. LIBERO suite별 막대 (Table I, 인용값 DP 표시)
suites = ["Spatial", "Object", "Goal", "Long", "평균"]
vals = {"DP (표 I, 출처 불명)": [78.3, 92.5, 68.3, 50.5, 72.4],
        "DiT-Block (자체 실행)": [81.1, 89.8, 67.6, 57.6, 74.0],
        "VolumeDP": [90.7, 95.9, 89.9, 78.7, 88.8]}
cols = [C_DP, C_DIT, C_OURS]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
w = 0.26
for i, ((name, v), c) in enumerate(zip(vals.items(), cols)):
    xs = [j + (i - 1) * w for j in range(len(suites))]
    ax.bar(xs, v, width=w, color=c, label=name)
ax.set_xticks(range(len(suites)))
ax.set_xticklabels(suites)
ax.set_ylim(0, 105)
ax.set_ylabel("성공률 (%)")
ax.set_title("LIBERO 표 I 값 (원문 보고, 재계산 아님)", fontsize=10)
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_libero_bars.png"))
plt.close(fig)

# 3. 토이: 관측 표현별 z 예측 RMSE (가림 확률별)
toy = json.load(open(os.path.join(HERE, "volumedp_toy_results.json"), encoding="utf-8"))
settings = ["no_occlusion_p0", "occlusion_p0.5", "occlusion_p1.0"]
labels = ["가림 없음", "가림 0.5", "가림 1.0"]
reps = [("2d_projection_linear", "2D 투영"), ("voxel_occupancy_linear", "복셀 점유"),
        ("point_set_summary_linear", "점 요약")]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
w = 0.26
for i, (key, lab) in enumerate(reps):
    xs = [j + (i - 1) * w for j in range(len(settings))]
    ax.bar(xs, [toy[s][key] for s in settings], width=w, label=lab, color=[C_DP, C_OURS, C_DIT][i])
ax.axhline(toy["no_occlusion_p0"]["mean_baseline"], color="k", lw=0.8, ls="--")
ax.text(2.45, toy["no_occlusion_p0"]["mean_baseline"] + 0.006, "평균 예측", fontsize=7, ha="right")
ax.set_xticks(range(len(settings)))
ax.set_xticklabels(labels)
ax.set_ylabel("테스트 RMSE (선형)")
ax.set_title("토이: 관측 표현별 z 예측 오차 (CPU, 단순화)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_toy_rmse.png"))
plt.close(fig)
print("saved", sorted(os.listdir(FIG)))
