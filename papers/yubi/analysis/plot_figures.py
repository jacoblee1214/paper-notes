"""그래프 세 장을 만든다 (report/figures/own_*.png). recompute_results.json을 읽는다.

실행: python3 analysis/plot_figures.py   (numpy, matplotlib)
논문 그림이 아니라 논문 수치를 다시 그린 것이다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((HERE / "recompute_results.json").read_text())

for cand in ("Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
C_HAND, C_UMI, C_YUBI = "#8a8a85", "#e08a1e", "#1f5fae"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e3dc", lw=0.7)
    ax.set_axisbelow(True)


# 1. 그림 5 다시 그리기: 성공률과 Wilson 95% 구간
f5, f2 = R["V1_fig5"], R["V2_fig5_tests"]
fig, ax = plt.subplots(figsize=(7.6, 3.9))
x = range(len(f5["sizes"]))
for off, name, col, k, w in ((-0.09, "UMI", C_UMI, f2["umi_k"], f2["wilson_umi"]),
                             (0.09, "YUBI", C_YUBI, f2["yubi_k"], f2["wilson_yubi"])):
    ys = [100 * v / 50 for v in k]
    lo = [y - 100 * a for y, (a, b) in zip(ys, w)]
    hi = [100 * b - y for y, (a, b) in zip(ys, w)]
    ax.errorbar([i + off for i in x], ys, yerr=[lo, hi], fmt="o-", color=col, capsize=3, lw=1.4, ms=5, label=name)
for i, p in enumerate(f2["holm_p"]):
    ax.text(i, 108, f"Holm p={p:.2f}" if p >= 0.05 else f"Holm p={p:.3f}", ha="center", fontsize=8, color="#55544f")
ax.set_xticks(list(x))
ax.set_xticklabels(f5["sizes"])
ax.set_ylim(0, 116)
ax.set_ylabel("성공률 (%)")
ax.set_xlabel("육각너트 호칭 지름 (큰 쪽이 오른쪽)")
ax.set_title("너트 집기: 크기별 성공률과 Wilson 95% 구간 (칸당 50회)", fontsize=11)
ax.legend(frameon=False, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "own_dex_ci.png", dpi=170)
plt.close(fig)

# 2. 그림 6 다시 그리기: 완료 시간과 UMI 대비 배속
f6 = R["V3_fig6"]
hand = [13.7, 20.1, 18.1, 12.1, 7.3]
umi = [36.4, 65.4, 105.6, 70.0, 99.8]
yubi = [26.5, 36.6, 70.65, 41.5, 23.8]
labels = ["도미노 3개\n정렬", "연필 통에\n넣고 꺼내기", "티슈 상자\n조립·분해", "화이트보드\n'ABCDE'", "보조배터리로\n휴대폰 충전"]
fig, ax = plt.subplots(figsize=(7.8, 3.9))
w = 0.26
for j, (vals, name, col) in enumerate(((hand, "손", C_HAND), (umi, "UMI", C_UMI), (yubi, "YUBI", C_YUBI))):
    ax.bar([i + (j - 1) * w for i in range(5)], vals, w, color=col, label=name)
for i, s in enumerate(f6["speedup_umi_over_yubi"]):
    ax.text(i, umi[i] + 3, f"{s:.2f}배", ha="center", fontsize=9, color="#1b1b1a")
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("평균 완료 시간 (초, 낮을수록 좋음)")
ax.set_ylim(0, 125)
ax.set_title("작업 완료 시간 (칸당 작업자 약 4명 x 5회, 분산과 검정은 논문에 없음)", fontsize=10.5)
ax.legend(frameon=False, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "own_eff.png", dpi=170)
plt.close(fig)

# 3. 표 3과 표 5 다시 그리기: 로봇 20회 성공과 Wilson 구간
t3, t5 = R["V4_table3"], R["V4_table5"]
names = ["공→바구니\n(UR)", "컵 피라미드\n(UR)", "안경 펴기\n(UR)", "양말\n(Franka)", "테이프→상자\n(Franka)", "컵 놓기\n(ELEY)"]
keys = list(t3["k_of_20"])
fig, ax = plt.subplots(figsize=(7.8, 3.9))
for i, k in enumerate(keys):
    v = t3["k_of_20"][k]
    lo, hi = t3["wilson"][k]
    ax.errorbar(i, 100 * v / 20, yerr=[[100 * v / 20 - 100 * lo], [100 * hi - 100 * v / 20]], fmt="s", color=C_YUBI, capsize=3, ms=6,
                label="pi0.5 (표 3)" if i == 0 else None)
dpk = [20, 9, 0]
from scipy import stats  # noqa: E402
for i, v in enumerate(dpk):
    # Wilson 구간 (recompute.py와 같은 식)
    import math
    z, n = 1.959964, 20
    p = v / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    ax.errorbar(i + 0.2, 100 * p, yerr=[[100 * (p - (c - h))], [100 * ((c + h) - p)]], fmt="o", color=C_UMI, capsize=3, ms=5,
                label="확산 정책 DP (표 5)" if i == 0 else None)
ax.set_xticks(range(6))
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(-5, 108)
ax.set_ylabel("성공률 (%, 20회)")
ax.set_title("로봇 배치: 과제별 성공률과 Wilson 95% 구간", fontsize=11)
ax.legend(frameon=False, loc="lower left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "own_deploy.png", dpi=170)
plt.close(fig)
print("wrote 3 figures")
