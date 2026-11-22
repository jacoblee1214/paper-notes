"""우리가 그린 그래프: 사용자 연구 선호율과 Wilson 95% 구간 (논문 그림은 쓰지 않음).

실행: python3 analysis/plot_figures.py  -> report/figures/own_user_study.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
res = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))["user_study"]
for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

labels = {"overall": "전체 (1,540 판단)", "walking": "걷기 (770)", "running": "달리기 (770)"}
keys = ["overall", "walking", "running"]
pct = [res[k]["ours_pct_recomputed"] for k in keys]
lo = [res[k]["wilson95"][0] for k in keys]
hi = [res[k]["wilson95"][1] for k in keys]
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=160)
y = range(len(keys))
ax.barh(list(y), pct, color="#1f5fae", height=0.5)
ax.errorbar(pct, list(y), xerr=[[p - l for p, l in zip(pct, lo)], [h - p for p, h in zip(pct, hi)]],
            fmt="none", ecolor="#222", capsize=4, lw=1.2)
ax.axvline(50, color="#888", ls="--", lw=1)
ax.set_yticks(list(y))
ax.set_yticklabels([labels[k] for k in keys])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("BeyondMimic 선호 비율 (%), 막대 끝: Wilson 95% 구간")
for i, p in enumerate(pct):
    ax.text(p + 1.5, i - 0.28, f"{p:.1f}%", va="center", fontsize=9)
ax.set_title("사용자 연구: 판단을 독립으로 본 재계산", fontsize=10)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = ROOT / "report" / "figures" / "own_user_study.png"
fig.savefig(out)
print("saved", out)
