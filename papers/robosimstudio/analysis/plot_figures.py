"""리포트용 그래프 세 장. 실행: cd analysis && python3 plot_figures.py

1) identifiability.png: 영상(압축량 판독) vs 힘 관측으로 E를 맞출 때의 상대 표준오차 (identifiability_toy.py)
2) latch_effect.png: 공식 wipe_the_bar 데모, latch 유무별 닦인 비율과 스펀지 미끄럼 (table_check.py C4)
3) table1_tally.png: Table 1의 기호 집계, RoboSimStudio만 ✓인 행 표시 (table_check.py C1)
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) 식별성
res = json.loads((ROOT / "analysis" / "identifiability_toy_results.json").read_text())
cv = res["_curves"]
E = cv["E_kPa"]
fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=200)
series = [("1080p_sigma0.5px", "영상 1080p, 판독 잡음 0.5 px", BLUE, "-"),
          ("1080p_sigma2.0px", "영상 1080p, 판독 잡음 2 px", BLUE, "--"),
          ("4K_sigma0.5px", "영상 4K, 판독 잡음 0.5 px", AQUA, "-"),
          ("force", "힘 관측 (σ_F 0.1 N, 압입 2 mm)", ORANGE, "-")]
for key, label, c, ls in series:
    ax.plot(E, [100 * v for v in cv[key]], color=c, lw=2, ls=ls, label=label)
for e, name in ((6, "6"), (20, "20"), (80, "80 (twin)"), (100, "100")):
    ax.axvline(e, color=GRID, lw=1, zorder=0)
    ax.text(e, 0.12, f"{name} kPa", rotation=90, va="bottom", ha="right", fontsize=7, color=INK2)
ax.axhline(50, color=INK2, lw=1, ls=":")
ax.text(2.2, 55, "상대 SE 50%: 사실상 식별 불가", fontsize=7.5, color=INK2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.1, 250)
ax.set_xlim(2, 300)
ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax.set_xticks([3, 10, 30, 100, 300], ["3", "10", "30", "100", "300"])
ax.minorticks_off()
ax.set_xlabel("스펀지 영률 E (kPa, 로그축)", color=INK2)
ax.set_ylabel("E 추정의 상대 표준오차 (%)", color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
style(ax)
fig.text(0.01, 0.01, "toy: 50×75×30 mm 스펀지를 4 N으로 누를 때 선형 압축 δ = F h /(E A), 화면 폭 0.6 m 가정. analysis/identifiability_toy.py",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "identifiability.png")
print("wrote", OUT / "identifiability.png")

# 2) latch 효과
d = tc.DEMO
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)
ax = axes[0]
xs = [0, 1, 2]
ax.bar([x - 0.18 for x in xs], d["latch"]["cleaned"], width=0.34, color=BLUE, label="latch 있음 (공개 twin)")
ax.bar([x + 0.18 for x in xs], d["no_latch"]["cleaned"], width=0.34, color=MUTED, label="마찰만")
for x, v in zip(xs, d["latch"]["cleaned"]):
    ax.text(x - 0.18, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, color=INK2)
for x, v in zip(xs, d["no_latch"]["cleaned"]):
    ax.text(x + 0.18, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, color=INK2)
ax.set_xticks(xs, ["take 1", "take 2", "take 3"])
ax.set_ylim(0, 1.08)
ax.set_ylabel("닦인 비율 (cleaned fraction)", color=INK2)
ax.set_title("마커 줄 닦인 비율", fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=7, loc="lower right")
style(ax)
ax = axes[1]
ax.bar(xs, d["no_latch"]["slide_mm"], width=0.5, color=MUTED)
for x, v in zip(xs, d["no_latch"]["slide_mm"]):
    ax.text(x, v + 4, f"{v} mm", ha="center", fontsize=7, color=INK2)
ax.axhline(10, color=ORANGE, lw=1.2, ls="--")
ax.text(2.35, 14, "10 mm", fontsize=7, color=ORANGE, ha="right")
ax.set_xticks(xs, ["take 1", "take 2", "take 3"])
ax.set_ylabel("손 안에서 폼이 미끄러진 거리 (mm)", color=INK2)
ax.set_title("마찰만일 때 스펀지 미끄럼", fontsize=9.5, color=INK)
style(ax)
fig.text(0.01, 0.01, "출처: 공식 저장소 examples/08_demos/wipe_the_bar/README.md (커밋 81f57c9). 마찰만으로 30회 더 시도해도 미끄럼 10 mm 미만인 경우 없음.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "latch_effect.png")
print("wrote", OUT / "latch_effect.png")

# 3) Table 1 집계
order = ("Y", "P", "C", "N")
labels = {"Y": "✓ 지원", "P": "∼ 일부", "C": "Custom", "N": "– 없음"}
colors = {"Y": BLUE, "P": AQUA, "C": "#c8c8cc", "N": "#ececee"}
fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
left = [0] * len(tc.FRAMEWORKS)
counts = {fw: {c: 0 for c in order} for fw in tc.FRAMEWORKS}
for row in tc.TABLE1.values():
    for fw, c in zip(tc.FRAMEWORKS, row):
        counts[fw][c] += 1
for c in order:
    vals = [counts[fw][c] for fw in tc.FRAMEWORKS]
    ax.barh(range(len(tc.FRAMEWORKS)), vals, left=left, color=colors[c], edgecolor="white", linewidth=1.5, label=labels[c])
    for i, (lft, v) in enumerate(zip(left, vals)):
        if v:
            ax.text(lft + v / 2, i, str(v), ha="center", va="center", fontsize=7.5, color="white" if c == "Y" else INK2)
    left = [a + b for a, b in zip(left, vals)]
ax.set_yticks(range(len(tc.FRAMEWORKS)), tc.FRAMEWORKS)
ax.invert_yaxis()
ax.set_xlim(0, 10)
ax.set_xlabel("Table 1의 10개 행", color=INK2)
ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
style(ax)
ax.grid(False)
fig.text(0.01, 0.01, "RoboSimStudio만 ✓인 행 5개는 모두 이 논문이 정의한 워크플로(구조화 피드백, 공동 정제, 영상·힘·실패 기반 Real2Sim). 출처: Table 1.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "table1_tally.png")
print("wrote", OUT / "table1_tally.png")
