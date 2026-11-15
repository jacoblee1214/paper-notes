"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 여기서 만들지 않는다.

  own_rank_flip.png   토이 1: 시도 수, 짝지음, 과제 부분집합에 따른 순위 뒤집힘 확률
  own_recount.png     논문 v1 그림 3 값 대 공개 릴리스 재집계 (LIBERO에서만 차이, Wilson 95% 구간)
  own_drops.png       재계산한 성공률·ATC 하락량과 95% 구간 (표 5, 표 6, 표 8, 표 7)

입력: analysis/recompute_results.json, analysis/toy_rank_flip_results.json
실행: python3 -I analysis/plot_figures.py   (matplotlib, CPU, 수 초)
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
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_rank_flip_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#8a8880", "#d9822b"


def wilson(x, n, z=1.959964):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


# ---------------------------------------------------------------- 1. 순위 뒤집힘
grid = TOY["grid"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
for S, col in [(50, ACC), (100, GREY), (500, ORG)]:
    rows = [r for r in grid if r["S"] == S]
    x = [r["delta_pp"] for r in rows]
    ax.plot(x, [r["flip_independent"] for r in rows], "o-", color=col, label=f"독립, 과제당 {S}회")
ax.axvline(0.8, color="#bbbbbb", ls="--", lw=1)
ax.text(0.9, 0.93, "LIBERO 최고 보고치 차이 약 0.7–0.8pp", fontsize=8, color="#666666")
ax.set_xlabel("참 격차 δ (pp)")
ax.set_ylabel("순위 뒤집힘 확률")
ax.set_title("(a) 시도 수에 따른 뒤집힘", fontsize=10, loc="left")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
rows = [r for r in grid if r["S"] == 50]
x = [r["delta_pp"] for r in rows]
ax.plot(x, [r["flip_independent"] for r in rows], "o-", color=ACC, label="독립 평가")
ax.plot(x, [r["flip_paired_discordance_0.2"] for r in rows], "s-", color=ORG, label="짝지음 (불일치 20% 가정)")
ax.plot(x, [r["flip_paired_monotone_upper_bound"] for r in rows], "^:", color=GREY, label="짝지음 상한 (단조 결합)")
ax.plot(x, [r["flip_task_subset_5of10"] for r in rows], "d-", color="#6b3fa0", label="독립 + 과제 5/10 부분집합")
ax.set_xlabel("참 격차 δ (pp)")
ax.set_title("(b) 과제당 50회에서 설계별 비교", fontsize=10, loc="left")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8, frameon=False)
fig.text(0.01, 0.01, "δ=0일 때 값은 '차이가 있다고 보이는' 비율입니다. 우리가 만든 모형(T=10, Beta 이질성)의 결과입니다.",
         fontsize=7.5, color="#55544f")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_rank_flip.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 2. 그림 3 재집계
fig, ax = plt.subplots(figsize=(8.6, 3.9))
bench = ["LIBERO\n(3 suites, n=789)", "CALVIN\nABC→D (n=107)", "SimplerEnv\nWidowX (n=122)", "RoboCasa\nRSS24 (n=30)", "RoboTwin 2.0\n(n=19)"]
paper_k = [156, 47, 24, 16, 14]     # 논문 v1 그림 3 (frozen paper_inputs와 같음)
paper_n = [789, 107, 122, 30, 19]
F3 = RC["figure3_recount_current_release"]
cur_keys = ["LIBERO spatial+object+goal (pooled)", "CALVIN/abc_d_atc", "SimplerEnv/widowx_bridge",
            "RoboCasa/rss24", "RoboTwin2/hard_randomized"]
cur_k = [F3[k]["provably_significant"] for k in cur_keys]  # 공개 릴리스 현재 분류 (all_comparisons.csv 재집계)
cur_n = [F3[k]["n"] for k in cur_keys]
xs = np.arange(len(bench))
for off, k, n, col, lab in [(-0.18, paper_k, paper_n, GREY, "논문 v1 (frozen paper_inputs)"),
                            (0.18, cur_k, cur_n, ACC, "공개 릴리스 재집계 (현재 분류)")]:
    p = [100 * a / b for a, b in zip(k, n)]
    lo = [wilson(a, b)[0] for a, b in zip(k, n)]
    hi = [wilson(a, b)[1] for a, b in zip(k, n)]
    ax.bar(xs + off, p, width=0.34, color=col, label=lab)
    ax.errorbar(xs + off, p, yerr=[np.array(p) - lo, np.array(hi) - p], fmt="none", ecolor="#333333", capsize=3, lw=1)
    for xi, pi in zip(xs + off, p):
        ax.text(xi, pi + 4, f"{pi:.1f}%", ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(bench, fontsize=8)
ax.set_ylabel("'명백히 유의' 비율 (%)")
ax.set_ylim(0, 100)
ax.set_title("공개 점수만으로 판정한 '명백히 유의' 비율: 논문 v1 대 현재 분류 (Wilson 95%)", fontsize=10, loc="left")
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "own_recount.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 3. 하락량 구간 (단위가 다르므로 두 패널)
pp_items = []  # (라벨, 값, 하한, 상한, 그룹)
for r in RC["table5_simplerenv_newcombe"]:
    pp_items.append((f'SimplerEnv {r["policy"]} / {r["change"]}', r["drop_pp"], r["ci"][0], r["ci"][1], "SimplerEnv 표 5"))
for r in RC["table7_libero_newcombe"]:
    pp_items.append((f'LIBERO {r["policy"]} 새 init', r["drop_pp"], r["ci"][0], r["ci"][1], "LIBERO 표 7"))
atc_items = []
for r in RC["table6_calvin_resampled"]:
    atc_items.append((f'{r["policy"]} 재표본 (표 6)', r["drop"], r["ci"][0], r["ci"][1], "CALVIN 표 6"))
for r in RC["table8_calvin_fresh"]:
    atc_items.append((f'{r["policy"]} 새 시퀀스 (표 8)', r["drop"], r["ci"][0], r["ci"][1], "CALVIN 표 8"))
colors = {"SimplerEnv 표 5": ACC, "LIBERO 표 7": GREY, "CALVIN 표 6": ORG, "CALVIN 표 8": "#6b3fa0"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.34 * len(pp_items) + 1.4),
                               gridspec_kw={"width_ratios": [1.9, 1]})
for ax, items, unit in [(ax1, pp_items, "pp (재표본·새 표본이 낮을수록 +)"), (ax2, atc_items, "ATC (과제 수/5)")]:
    ys = np.arange(len(items))[::-1]
    for y, (lab, v, lo, hi, grp) in zip(ys, items):
        ax.plot([lo, hi], [y, y], color=colors[grp], lw=2)
        ax.plot(v, y, "o", color=colors[grp], ms=4)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=7.5)
    ax.set_xlabel(unit, fontsize=8.5)
    ax.set_ylim(-0.8, len(items) - 0.2)
ax1.set_title("(a) 성공률 하락 (pp), 95% 구간", fontsize=10, loc="left")
ax2.set_title("(b) CALVIN ATC 하락, 95%", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_drops.png", dpi=170)
plt.close(fig)
print("wrote figures to", FIG)
