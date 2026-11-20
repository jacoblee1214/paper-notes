"""WorldVLA 분석용 그래프 (matplotlib). 실행: python3 analysis/plot_figures.py
입력: analysis/recompute.py 와 analysis/toy_power.py 의 값을 그대로 옮긴 상수.
출력: report/figures/own_ablation.png, report/figures/own_power.png (우리가 그린 그래프)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def korean_font():
    for path in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    return "DejaVu Sans"

plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False

# 표 3 (LIBERO SR %, Goal, Object, Spatial, Long)
rows = {
    "1 행동만": [67.3, 82.9, 77.8, 23.0],
    "2 +세계모델": [73.1, 88.0, 80.2, 27.3],
    "3 청크": [79.6, 82.9, 36.7, 16.9],
    "4 청크+마스크": [84.4, 90.9, 81.8, 49.3],
    "5 청크+마스크+세계모델": [85.1, 90.9, 84.0, 52.4],
}
suites = ["Goal", "Object", "Spatial", "Long"]
fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=160)
width = 0.15
for k, (name, vals) in enumerate(rows.items()):
    xs = [i + (k - 2) * width for i in range(len(suites))]
    ax.bar(xs, vals, width=width, label=name)
ax.set_xticks(range(len(suites)))
ax.set_xticklabels(suites)
ax.set_ylabel("성공률 (%) · 과제당 50회")
ax.set_ylim(0, 100)
ax.set_title("WorldVLA 절제 실험 (논문 표 3 값을 옮김)", fontsize=11)
ax.legend(fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.36), frameon=False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "own_ablation.png")
plt.close(fig)

# 검출력 곡선 (toy_power.py 와 같은 모델: 200회 평균, 양측 z 0.05)
import json
res = json.loads((ROOT / "analysis" / "toy_power_results.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=160)
for base in [0.63, 0.77]:
    pts = [r for r in res["power"] if r["p_base"] == base]
    ax.plot([r["true_diff_pp"] for r in pts], [100 * r["power_95"] for r in pts], marker="o",
            label=f"기준 성공률 {int(base*100)}%")
ax.axvline(4.4, color="gray", ls="--", lw=1)
ax.text(4.6, 8, "표 3 행 1→2\n(+4.4pp)", fontsize=7.5, color="gray")
ax.axhline(80, color="gray", ls=":", lw=1)
ax.set_xlabel("참 차이 (pp)")
ax.set_ylabel("검출력 (%)")
ax.set_title("과제 4개 x 50회에서 차이를 찾을 확률 (모의실험)", fontsize=10.5)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "own_power.png")
plt.close(fig)
print("saved", sorted(p.name for p in OUT.glob("own_*.png")))
