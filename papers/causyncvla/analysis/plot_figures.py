"""우리가 그린 그래프 세 장을 만든다(논문 그림은 쓰지 않는다: 라이선스가 CC BY가 아님).

출력: report/figures/own_native_physical_ci.png, own_latency_decomp.png, own_pairs_forest.png
입력: 논문 표 값(analysis/recompute.py와 같은 값), 구간은 recompute_results.json에서 읽는다.
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleSDGothicNeo"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORANGE = "#1b1b1a", "#1f5fae", "#8a8880", "#c0622b"


def ci_bar(ax, x, k, n, lo, hi, color, label):
    p = 100 * k / n
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", color=color, capsize=5, lw=2, ms=7)
    ax.text(x + 0.08, p, f"{k}/{n}\n{p:.1f}%", va="center", fontsize=8.5, color=INK)


# 1. 성공률과 Wilson 95% 구간 (왼쪽: LIBERO native와 실로봇 합계, 오른쪽: 실로봇 과제별 개수)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
ln = RES["libero_native"]
rr = RES["real_robot"]
ci_bar(a1, 0, 456, 500, *ln["wilson_base_pct"], GREY, "")
ci_bar(a1, 1.1, 469, 500, *ln["wilson_cauSync_pct"], ACC, "")
ci_bar(a1, 3.0, 50, 69, *rr["wilson_pi05_pct"], GREY, "")
ci_bar(a1, 4.1, 60, 69, *rr["wilson_cauSync_pct"], ACC, "")
a1.set_xticks([0, 1.1, 3.0, 4.1], ["π0.5\n10-step", "CauSync", "π0.5", "CauSync"])
a1.text(0.55, 99.0, "LIBERO-10 native (500)", ha="center", fontsize=9, color=INK)
a1.text(3.55, 99.0, "실로봇 합계 (69회)", ha="center", fontsize=9, color=INK)
a1.set_xlim(-0.7, 5.6)
a1.set_ylim(55, 102)
a1.set_yticks([60, 70, 80, 90, 100])
a1.set_ylabel("성공률 (%)")
a1.set_title("성공률과 Wilson 95% 구간", fontsize=10.5, color=INK)
a1.grid(axis="y", alpha=0.3)

tasks = ["쌓기", "놓기", "분류"]
b_counts = [17, 16, 17]
c_counts = [19, 22, 19]
xs = range(3)
a2.bar([x - 0.19 for x in xs], [100 * k / 23 for k in b_counts], width=0.36, color=GREY, label="π0.5 10-step")
a2.bar([x + 0.19 for x in xs], [100 * k / 23 for k in c_counts], width=0.36, color=ACC, label="CauSyncVLA")
for x, k in zip(xs, b_counts):
    a2.text(x - 0.19, 100 * k / 23 + 1.5, f"{k}/23", ha="center", fontsize=8.5)
for x, k in zip(xs, c_counts):
    a2.text(x + 0.19, 100 * k / 23 + 1.5, f"{k}/23", ha="center", fontsize=8.5)
a2.set_xticks(list(xs), tasks)
a2.set_ylim(0, 135)
a2.set_ylabel("성공률 (%)")
a2.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
a2.set_title("실로봇 과제별 (과제당 23회, 짝지어지지 않음)", fontsize=10.5, color=INK)
a2.grid(axis="y", alpha=0.3)
fig.text(0.01, 0.01, "점과 오차 막대: 논문 표의 개수와 Wilson 95% 구간. 실로봇 시행은 장면을 다시 놓은 비쌍 시행이다. "
         "과제별 p 값은 recompute_results.json 참조.", fontsize=7.4, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_native_physical_ci.png", dpi=170)
plt.close(fig)

# 2. 지연 분해 (Table 16, model-only)
fig, ax = plt.subplots(figsize=(8.6, 4.0))
labels = ["π0.5\n10-step", "π0.5\n1-step", "CauSyncVLA\n(공유 GPU, 빠른 경로)", "CauSyncVLA\n(분리, 빠른 경로)"]
vals = [60.01, 36.00, 11.91, 4.79]
cols = [GREY, GREY, ACC, ACC]
bars = ax.bar(range(4), vals, color=cols, alpha=0.85, width=0.6)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 1.2, f"{v:.2f} ms", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(4), labels, fontsize=8.5)
ax.set_ylabel("평균 지연 (ms), batch 1, BF16")
ax.set_ylim(0, 70)
ax.annotate("", xy=(1, 40), xytext=(0, 62), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
ax.text(0.62, 52, "1.67×\n(흐름 단계 10→1)", ha="left", fontsize=8.5, color=ORANGE)
ax.annotate("", xy=(2, 15), xytext=(1, 36), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
ax.text(1.28, 34, "3.02×\n(비전 인코딩 제거)", ha="left", fontsize=8.5, color=ORANGE)
ax.set_title("Table 16 재계산: 5.04× = 1.67× × 3.02×", fontsize=10.5, color=INK)
ax.grid(axis="y", alpha=0.3)
fig.text(0.01, 0.01, "논문 Table 16 값. 공유 GPU 값은 비전 인코딩을 다른 CUDA 스트림에서 계속 돌린 상태의 '따뜻한' 추론이다.",
         fontsize=7.2, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_latency_decomp.png", dpi=170)
plt.close(fig)

# 3. 쌍 비교 (Table 10): 이득(pp)과 쌍체 구간
pairs = RES["ablation_pairs"]
names = list(pairs.keys())
short = ["잠재 세계 모델\n(w/ vs w/o)", "PFWS, 전체\n(w/ vs w/o)", "행동 증류\n(w/ vs w/o)", "PFWS, 동결 정책\n(vs cold)"]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
for i, nm in enumerate(names):
    r = pairs[nm]
    g = r["gain_pp"]
    lo, hi = r["gain_ci95_pp_clopper_pearson_on_discordant"]
    col = ACC if r["bonferroni_4_significant"] else GREY
    ax.errorbar(g, i, xerr=[[g - lo], [hi - g]], fmt="o", color=col, capsize=4, lw=2, ms=7)
    ax.text(hi + 0.4, i, f"w/l = {r['wins']}/{r['losses']}, p = {r['exact_mcnemar_p']:.2g}", va="center", fontsize=8.2, color=INK)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(4), short, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(-4, 26)
ax.set_xlabel("성공률 차이 (pp), 쌍체 근사 95% 구간")
ax.set_title("Table 10 쌍 비교: 파란 점은 Bonferroni(4검정, 0.0125) 통과", fontsize=10.5, color=INK)
ax.grid(axis="x", alpha=0.3)
fig.text(0.01, 0.01, "구간: 불일치 쌍 w/(w+l)에 Clopper-Pearson을 쓰고 (w+l)/500을 곱한 근사. 논문 식 (24)의 정확 McNemar p.",
         fontsize=7.2, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_pairs_forest.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
