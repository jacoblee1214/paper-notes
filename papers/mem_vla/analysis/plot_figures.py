"""우리가 그리는 그래프 네 장 (report/figures/own_*.png). 입력은 recompute_results.json, fig_readings.json, toy_memory_results.json.
실행: python3 analysis/plot_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for c in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
          "/Library/Fonts/AppleGothic.ttf"):
    if Path(c).exists():
        font_manager.fontManager.addfont(c)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=c).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

R = json.load(open(HERE / "recompute_results.json"))
RD = json.load(open(HERE / "fig_readings.json"))
TOY = json.load(open(HERE / "toy_memory_results.json"))
BLUE, ORANGE, GRAY, RED = "#1f5fae", "#d9822b", "#8a8a85", "#b5483a"

# 1. 지연 곡선
L = R["V1_fig3_latency"]
fr = L["frames"]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(fr, L["without_s"], "o-", color=RED, label="인코더 없음 (그림 판독 값)")
ax.plot(fr, L["with_s"], "o-", color=BLUE, label="영상 인코더 (그림 판독 값)")
xs = [1, 19]
ax.plot(xs, [L["with_linear_fit_intercept_ms"] / 1000 + L["with_linear_fit_ms_per_frame"] / 1000 * x for x in xs], "--", color=BLUE, alpha=.5, lw=1,
        label=f"선형 맞춤 {L['with_linear_fit_ms_per_frame']} ms/프레임 (R²={L['with_linear_r2']})")
ax.axhline(0.3, color=GRAY, ls=":", label="실시간 기준 300 ms (논문 표기)")
ax.scatter([18, 19], [L["with_extrapolated_18frames_s"], L["with_extrapolated_19frames_s"]], marker="x", color=BLUE, zorder=5, label="외삽 (18, 19 프레임)")
ax.set_ylim(0, 0.6)
ax.set_xlabel("총 프레임 수 (그림 3 가로축)")
ax.set_ylabel("추론 시간 (s), 세로축 0.6까지만 표시")
ax.set_title("그림 3 판독: 지연과 외삽 (3.5 s 지점은 잘림)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "own_latency.png", dpi=170)
plt.close(fig)

# 2. 그림 8: MEM 대 각 기준선 차이 (95% z 구간)
tasks = ["swap_3_mugs", "find_object", "unpack_groceries", "scoop_coffee", "grilled_cheese", "window_cleaning", "average"]
names = ["머그 3개 교체", "물체 찾기", "식료품 꺼내기", "커피 스쿱", "그릴드 치즈", "창 닦기", "평균"]
V = R["V6_fig8_mem_vs_baselines"]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
cols = {"no_memory": RED, "pool": GRAY, "proprio": ORANGE}
labs = {"no_memory": "기억 없음", "pool": "풀링 기억", "proprio": "고유수용 기억"}
for j, k in enumerate(("no_memory", "pool", "proprio")):
    for i, t in enumerate(tasks):
        v = V[t][k]
        z = v["z"]
        se = abs(v["diff_pp"] / z)
        ax.errorbar(i + (j - 1) * 0.22, v["diff_pp"], yerr=1.96 * se, fmt="o", color=cols[k], capsize=2, ms=4, label=labs[k] if i == 0 else None)
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(names, rotation=20, fontsize=8)
ax.set_ylabel("MEM 마이너스 기준선 (%p)")
ax.set_title("그림 8 판독: 과제 진행도 차이와 95% 구간 (표준오차만 사용)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "own_fig8_diff.png", dpi=170)
plt.close(fig)

# 3. 그림 10: MEM 마이너스 π0.6
V = R["V12_fig10_mem_vs_pi06"]
t10 = ["table_bussing", "shirt_folding", "clean_up_counter", "make_bed", "dishes_to_sink", "batch_folding", "box_building", "average"]
n10 = ["테이블 치우기", "셔츠 접기", "카운터 정리", "침대 정리", "접시 싱크대", "빨래 일괄 접기", "상자 만들기", "평균"]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.axhspan(-10, 10, color="#eef3fa", label="±10%p (예시 마진, 논문 값 아님)")
for i, t in enumerate(t10):
    v = V[t]
    if v["ci95_pp"] is None:
        ax.plot(i, v["diff_pp"], "o", color=BLUE, ms=5)
        ax.annotate("오차 0", (i, v["diff_pp"]), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    else:
        ax.errorbar(i, v["diff_pp"], yerr=[[v["diff_pp"] - v["ci95_pp"][0]], [v["ci95_pp"][1] - v["diff_pp"]]], fmt="o", color=BLUE, capsize=3, ms=5)
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(range(len(t10)))
ax.set_xticklabels(n10, rotation=20, fontsize=8)
ax.set_ylabel("MEM 마이너스 π0.6 (%p)")
ax.set_title("그림 10 판독: 메모리가 필요 없는 과제의 차이와 95% 구간", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "own_fig10_diff.png", dpi=170)
plt.close(fig)

# 4. 토이 T2
rows = TOY["T2_naive_language_history_shift"]["rows"]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
f = [r["f"] for r in rows]
ax.plot(f, [r["p_any_retry_closed_form"] for r in rows], "o-", color=BLUE, label="재시도가 한 번이라도 있을 확률")
ax.plot(f, [r["p_history_len_over_1.5S"] for r in rows], "s-", color=ORANGE, label="이력 길이가 학습 스타일의 1.5배 초과")
ax.plot(f, [r["p_same_instruction_3plus_times"] for r in rows], "^-", color=RED, label="같은 지시가 3번 이상 반복")
ax.set_xlabel("하위 과제당 실패 확률 f (가정)")
ax.set_ylabel("확률")
ax.set_title("토이: 압축하지 않은 지시문 이력이 학습에 없던 모양이 되는 비율 (S=8)", fontsize=9.5)
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "own_naive_history.png", dpi=170)
plt.close(fig)
print("figures written")
