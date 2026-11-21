"""Octo 직접 그린 그래프 두 장 (논문 그림 아님). 실행: python3 analysis/plot_figures.py
입력: analysis/recompute_results.json → 출력: report/figures/own_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = json.loads((HERE / "recompute_results.json").read_text())
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다
from matplotlib import font_manager
for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 1) 표 I: 도메인별 Octo 성공률과 Wilson 95% 구간, 가장 강한 기준선 표시
d = R["table_I"]["per_domain"]
names = list(d.keys())
ok = [int(d[n]["octo"].split("/")[0]) / 20 for n in names]
lo = [d[n]["octo_wilson95"][0] for n in names]
hi = [d[n]["octo_wilson95"][1] for n in names]
base = [int(d[n]["baseline"].split("/")[0]) / 20 for n in names]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
y = range(len(names))
ax.errorbar(ok, y, xerr=[[o - l for o, l in zip(ok, lo)], [h - o for o, h in zip(ok, hi)]], fmt="o", color="#1f5fa8", capsize=3, label="Octo (20회, Wilson 95%)")
ax.scatter(base, y, marker="x", color="#b5442b", label="최강 기준선 (20회)")
ax.set_yticks(list(y)); ax.set_yticklabels(names)
ax.invert_yaxis(); ax.set_xlim(-0.02, 1.02); ax.set_xlabel("성공률")
ax.set_title("표 I 재계산: 도메인별 성공률과 95% 구간")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "own_table1_ci.png", dpi=200); plt.close(fig)

# 2) 표 II/VI: 구성 요소별 WidowX 성공률 (40회 합산) 과 95% 구간
r = R["table_VI_II"]["rows"]
keys = list(r.keys())
vals = [r[k]["successes_of_40"] / 40 for k in keys]
lo = [r[k]["wilson95"][0] for k in keys]
hi = [r[k]["wilson95"][1] for k in keys]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = range(len(keys))
ax.bar(x, vals, color=["#1f5fa8"] + ["#9aa7b8"] * (len(keys) - 1), width=0.6)
ax.errorbar(x, vals, yerr=[[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]], fmt="none", ecolor="#333", capsize=3)
ax.set_xticks(list(x)); ax.set_xticklabels(keys, rotation=12, fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("성공률 (40회 합산)")
ax.set_title("표 VI 재계산: WidowX 구성 요소 비교 (Wilson 95%)")
fig.tight_layout(); fig.savefig(OUT / "own_table6_ablation.png", dpi=200); plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
