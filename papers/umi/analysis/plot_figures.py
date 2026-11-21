"""UMI 분석 그래프 두 장 (matplotlib). 실행: python3 analysis/plot_figures.py
입력: analysis/recompute_results.json (recompute.py 결과)
출력: report/figures/own_success_ci.png, report/figures/own_pvalues.png
한글 폰트는 Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, MUTE = "#1b1b1a", "#1f5fae", "#9a9890"

# 그림 1: 성공률과 Wilson 95% 구간 (논문 보고 값의 재계산)
rows = [r for r in RES["rates"] if not r["name"].startswith("Cup in-the-wild, 전체")]
fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=180)
for i, r in enumerate(rows):
    lo, hi = r["wilson95"]
    y = len(rows) - 1 - i
    color = ACC if "UMI" in r["name"] or "전체 방법" in r["name"] else INK
    ax.plot([lo, hi], [y, y], color=color, lw=2)
    ax.plot(r["rate"], y, "o", color=color, ms=5)
    ax.text(1.02, y, f"{r['k']}/{r['n']}", va="center", fontsize=8, color=MUTE, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["name"] for r in rows][::-1], fontsize=8)
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("성공률 (Wilson 95% 구간)", fontsize=9)
ax.axvline(0.5, color=MUTE, lw=0.6, ls=":")
ax.grid(axis="x", color="#eeeeea", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("UMI 보고 성공률의 재계산 (논문 v3 본문과 그림 8, 9)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_success_ci.png")
plt.close(fig)

# 그림 2: 대비 검정의 -log10(p) (양측 Fisher, 가로선 p = 0.05)
import math
pairs = [p for p in RES["fisher"] if "n=10" not in p["name"]]
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
vals = [-math.log10(max(p["p_two_sided"], 1e-12)) for p in pairs]
labels = [p["name"] for p in pairs]
order = sorted(range(len(pairs)), key=lambda i: vals[i])
ax.barh(range(len(order)), [vals[i] for i in order], color=[ACC if vals[i] > -math.log10(0.05) else MUTE for i in order])
ax.axvline(-math.log10(0.05), color=INK, lw=0.8, ls="--")
ax.text(-math.log10(0.05) + 0.05, len(order) - 0.6, "p = 0.05", fontsize=8, color=INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([labels[i] for i in order], fontsize=8)
ax.set_xlabel("-log10(p), Fisher 정확 검정 양측", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("보고된 횟수에서 계산한 대비 검정 (시행 독립 가정)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_pvalues.png")
plt.close(fig)
print("wrote", FIG / "own_success_ci.png", FIG / "own_pvalues.png")
