"""Own plots (not paper figures) for the report and public post.

Inputs : analysis/survey_taxonomy_results.json, analysis/recompute_survey_results.json, analysis/recompute_survey.py data (via import-free copy below)
Output : report/figures/own_coverage.png, own_long_vs_spatial.png, own_wilson_top.png
Run    : python3 -I analysis/plot_figures.py
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

tax = json.load(open(os.path.join(HERE, "survey_taxonomy_results.json"), encoding="utf-8"))
rec = json.load(open(os.path.join(HERE, "recompute_survey_results.json"), encoding="utf-8"))

# 1. coverage per subsection (unique citation keys)
subs = [(k, v) for k, v in tax["per_subsection"].items() if k.startswith(("2.", "3.", "4.", "5.", "6.", "7.", "8."))]
subs = [(k, v) for k, v in subs if "." in k and k.count(".") == 1]
labels = [f"{k} {v['title'][:18]}" for k, v in subs]
vals = [v["unique_refs_matched"] for k, v in subs]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.barh(range(len(vals)), vals, color="#1f5fae")
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("본문에서 인용된 고유 참고문헌 수 (저자·연도 기준, 근사)", fontsize=8)
ax.set_title("서베이 절별 인용 분포", fontsize=10)
for i, v in enumerate(vals):
    ax.text(v + 0.3, i, str(v), va="center", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_coverage.png"), dpi=160)
plt.close(fig)

# 2. Long vs Spatial per LIBERO row
T5 = {
 "Say-Dream-ACT": (99.4, 99.2, 98.6, 95.4), "Cosmos Policy": (98.1, 100.0, 98.2, 97.6),
 "UD-VLA": (94.1, 95.7, 91.2, 89.6), "Motus": (96.8, 99.8, 96.6, 97.6),
 "LingBot-VA": (98.5, 99.6, 97.2, 98.5), "RynnVLA-002": (99.0, 99.8, 96.4, 94.4),
 "DreamVLA": (97.5, 94.0, 89.5, 89.5), "UniVLA (Bu)": (96.5, 96.8, 95.6, 92.0),
 "Unified VLA (Wang)": (95.4, 98.8, 93.6, 94.0), "CoWVLA": (97.2, 97.8, 94.6, 92.8),
 "F1": (98.2, 97.8, 95.4, 91.3), "TriVLA": (91.2, 93.8, 89.8, 73.2),
 "VLA-JEPA": (96.2, 99.6, 97.2, 95.8), "JEPA-VLA": (97.2, 98.0, 95.6, 94.8),
}
fig, ax = plt.subplots(figsize=(6.4, 4.4))
names = list(T5.keys())
for i, n in enumerate(names):
    s, o, g, l = T5[n]
    ax.plot([s, l], [i, i], color="#bbbbbb", lw=1.2, zorder=1)
    ax.scatter([s], [i], color="#1f5fae", s=18, zorder=2)
    ax.scatter([l], [i], color="#c0392b", s=18, zorder=2)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("성공률 (%)  파란 점 = Spatial, 빨간 점 = Long", fontsize=8)
ax.set_title("LIBERO Spatial 대 Long (표 5 보고값, 14행)", fontsize=10)
ax.grid(axis="x", color="#eeeeee")
lv = rec["long_vs_spatial"]
ax.text(0.02, 0.02, f"평균 Long - Spatial = {lv['mean_long_minus_spatial']} %p ({lv['rows_long_below_spatial']}/{lv['n_rows_long']}행 하락)",
        transform=ax.transAxes, fontsize=7, color="#333333")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_long_vs_spatial.png"), dpi=160)
plt.close(fig)

# 3. Wilson 95% intervals for LIBERO suite-level cells at n = 500 (assumed) for the top rows
def wilson(p, n=500, z=1.96):
    phat = p / 100
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return 100 * (centre - half), 100 * (centre + half)

top = [("Cosmos Policy", 98.5), ("LingBot-VA", 98.5), ("VLA-JEPA", 97.2), ("Motus", 97.7), ("F1", 95.7), ("JEPA-VLA", 96.4)]
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for i, (n, v) in enumerate(top):
    lo, hi = wilson(v)
    ax.errorbar([v], [i], xerr=[[v - lo], [hi - v]], fmt="o", color="#1f5fae", capsize=3, lw=1.2)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([t[0] for t in top], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(93, 100)
ax.set_xlabel("LIBERO 평균 성공률 (%), Wilson 95% (n = 500 가정)", fontsize=8)
ax.set_title("표 5 평균의 구간 (위 식은 각 수치를 500회 시행으로 본 근사)", fontsize=9)
ax.grid(axis="x", color="#eeeeee")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_wilson_top.png"), dpi=160)
plt.close(fig)
print("wrote figures to", OUT)
