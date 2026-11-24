"""우리가 그리는 그래프 네 장 (논문 그림은 쓰지 않는다). 값은 recompute_results.json 에서 읽는다.

  own_scaling.png   : 사전학습 문맥 길이와 평균 점수 (그림 8 판독 값) + 가정한 표준오차 띠, 기준선
  own_forest.png    : 표 1·3의 성공 횟수와 Wilson 95% 구간
  own_budget.png    : 문맥 길이별 한 스텝의 타임스텝 수(전역 배치 x 길이)와 사전학습 궤적 길이 분포
  own_pup.png       : Pup Go Car 점수를 그림 7·10·8에서 역산한 값 비교
실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((HERE / "recompute_results.json").read_text())

for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
C_ROBO, C_GDN, C_GR, C_HIST, INK = "#b8864b", "#6f7a75", "#2f6f9f", "#8aa4bd", "#222222"

# 1. 스케일링
fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=160)
L = [128, 256, 512, 1024, 2048, 4096, 8192]
f8 = R["F8_read_values"]
se = R["V7_se_assumed_sd"]["0.3"]["se_avg"]
ax.plot(L, f8["RoboTTT"], "o-", color=C_ROBO, label="RoboTTT (그림 판독)")
ax.fill_between(L, [v - 1.96 * se for v in f8["RoboTTT"]], [v + 1.96 * se for v in f8["RoboTTT"]], color=C_ROBO, alpha=0.18,
                label="±1.96 SE (시행당 SD 0.3 가정)")
ax.plot(L, f8["GDN"], "s-", color=C_GDN, label="GDN (그림 판독)")
ax.axhline(f8["single_step_ref"], color=C_GR, ls="--", lw=1, label="단일 스텝 기준 0.375")
ax.axhline(f8["short_ctx_ref"], color=C_HIST, ls=":", lw=1.4, label="짧은 문맥 기준 0.456")
ax.set_xscale("log", base=2)
ax.set_xticks(L)
ax.set_xticklabels(["128", "256", "512", "1K", "2K", "4K", "8K"])
ax.set_xlabel("사전학습 문맥 길이 (타임스텝)")
ax.set_ylabel("평균 과제 완료 점수 (3개 과제)")
ax.set_ylim(0, 0.9)
ax.grid(alpha=0.25)
ax.legend(fontsize=7.5, loc="upper left", frameon=False)
ax.set_title("1K에서는 짧은 문맥 기준 아래, 2K부터 위 (논문 그림 8의 판독 값)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(OUT / "own_scaling.png")
plt.close(fig)

# 2. 포레스트
rows = [
    ("Pup Go Car 완전 성공", {"RoboTTT": (9, 20), "GR00T N1.7": (3, 20), "GR00T N1.7 Hist.": (0, 20), "GDN": (3, 20)}),
    ("Circuit 완전 성공", {"RoboTTT": (13, 20), "GR00T N1.7": (3, 20), "GR00T N1.7 Hist.": (8, 20), "GDN": (8, 20)}),
    ("Gear Bot 완전 성공", {"RoboTTT": (2, 10), "GR00T N1.7": (0, 10), "GR00T N1.7 Hist.": (0, 10), "GDN": (0, 10)}),
    ("교란 복구 (지붕+타이어 40회)", {"RoboTTT": (33, 40), "GR00T N1.7": (21, 40), "GR00T N1.7 Hist.": (8, 40), "GDN": (31, 40)}),
]
cols = {"RoboTTT": C_ROBO, "GR00T N1.7": C_GR, "GR00T N1.7 Hist.": C_HIST, "GDN": C_GDN}


def wil(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.4), dpi=160, sharex=True)
for ax, (title, d) in zip(axes, rows):
    for i, (m, (k, n)) in enumerate(d.items()):
        lo, hi = wil(k, n)
        ax.plot([lo, hi], [i, i], color=cols[m], lw=2.2)
        ax.plot([k / n], [i], "o", color=cols[m])
        ax.text(1.02, i, f"{k}/{n}", va="center", fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(list(d.keys()) if ax is axes[0] else [""] * 4, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_title(title, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("비율", fontsize=7)
fig.suptitle("표 1·3의 성공 횟수와 Wilson 95% 구간", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "own_forest.png")
plt.close(fig)

# 3. 예산
bs = R["V10_pretrain_batches"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=160)
xs = list(bs.keys())
a1.bar(range(len(xs)), [bs[x]["timesteps_per_update"] / 1e3 for x in xs],
       color=[C_ROBO if int(x) != 8192 else "#9c5a1c" for x in xs])
a1.set_xticks(range(len(xs)))
a1.set_xticklabels(["128", "256", "512", "1K", "2K", "4K", "8K"], fontsize=8)
a1.set_ylabel("한 갱신당 타임스텝 수 (천)")
a1.set_xlabel("사전학습 문맥 길이")
a1.set_title("전역 배치 64(≤4K) 대 16(8K)이라 8K는 4K의 절반", fontsize=9)
for i, x in enumerate(xs):
    a1.text(i, bs[x]["timesteps_per_update"] / 1e3 + 4, f"배치 {bs[x]['global_batch']}", ha="center", fontsize=7)
a1.grid(axis="y", alpha=0.25)
fa = R["V10_figA1"]
a2.bar(range(5), [c / 1e3 for c in fa["count_est"]], color=C_ROBO)
a2.set_xticks(range(5))
a2.set_xticklabels(fa["bins"], fontsize=8)
for i, (c, s) in enumerate(zip(fa["count_est"], fa["share_pct"])):
    a2.text(i, c / 1e3 + 1.5, f"{s}%", ha="center", fontsize=8)
a2.set_ylabel("궤적 수 추정 (천, 막대 높이 판독)")
a2.set_xlabel("에피소드 길이 (단위 논문 미기재)")
a2.set_title("사전학습 혼합의 궤적 길이 (그림 A.1 판독)", fontsize=9)
a2.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "own_budget.png")
plt.close(fig)

# 4. Pup Go Car 역산
c = R["V7_F8_vs_F7_F10_consistency"]
names = list(c.keys())
fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=160)
w = 0.36
ax.bar([i - w / 2 for i in range(4)], [c[n]["F10_before_pup"] for n in names], w, color=C_ROBO, label="그림 10 'Before DAgger'")
ax.bar([i + w / 2 for i in range(4)], [c[n]["implied_pup_from_F8"] for n in names], w, color=C_GDN,
       label="그림 8 평균에서 역산 (3x평균 - Circuit - Gear Bot, 그림 7)")
ax.set_xticks(range(4))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("Pup Go Car 점수 (판독 값)")
ax.set_ylim(0, 0.85)
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.25)
ax.set_title("DAgger 이전 Pup Go Car 점수: 두 그림에서 같은 값이 나오는지", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "own_pup.png")
plt.close(fig)
print("wrote 4 figures to", OUT)
