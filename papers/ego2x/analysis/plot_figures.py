"""우리가 만든 그래프. 논문 그림은 쓰지 않는다(라이선스 nonexclusive-distrib).
입력: analysis/recompute_results.json. 출력: report/figures/own_*.png
실행: python3 analysis/plot_figures.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

w = RES["wilson_95"]
rows = [
    ("Bread: 제로샷", "xvla_zero_shot"),
    ("Bread: 필터 없음", "xvla_unfiltered_bread"),
    ("Bread: P70 검증", "xvla_verified_P70"),
    ("Bread: 55 layout-weighted 검증", "xvla_verified_55lw"),
    ("Bell: 제로샷", "xvla_bell_zero_shot"),
    ("Bell: 필터 없음", "xvla_bell_unfiltered"),
    ("Bell: 제로리트라이 + 프롬프트", "xvla_bell_zero_retry_prompt"),
]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
for i, (label, key) in enumerate(rows):
    r = w[key]
    y = len(rows) - i
    ax.errorbar(r["p"] * 100, y, xerr=[[(r["p"] - r["ci"][0]) * 100], [(r["ci"][1] - r["p"]) * 100]],
                fmt="o", color="#1f5fae" if ("verified" in key or "retry" in key) else "#55544f", capsize=3)
    ax.text(101, y, f'{r["k"]}/{r["n"]}', va="center", fontsize=7, color="#55544f")
ax.set_yticks([len(rows) - i for i in range(len(rows))])
ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlim(0, 100)
ax.set_xlabel("RoboTwin 2.0 X-VLA 성공률 (%), Wilson 95% 구간", fontsize=8)
ax.axvline(40.5, color="#bbb", lw=0.8, ls="--")
ax.set_title("표 7 성공률 재계산 (개수는 보고된 비율 x 장면 수로 역산)", fontsize=8.5)
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "own_xvla_wilson.png")

pr = RES["mcnemar_check"]
names = [("bread_55lw_vs_zero_shot", "Bread 55lw"), ("bread_P70_vs_zero_shot", "Bread P70"),
         ("bell_zero_retry_prompt_vs_zero_shot", "Bell 제로리트라이+프롬프트"),
         ("bell_fixed_prompt_vs_zero_shot_195", "Bell 고정 프롬프트 (195장)"),
         ("bell_unfiltered_vs_zero_shot", "Bell 필터 없음")]
fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=200)
for i, (k, label) in enumerate(names):
    y = len(names) - i
    obs = pr[k]["b_minus_c"]
    imp = pr[k]["implied_b_minus_c_from_SR"]
    ax.plot([imp], [y], "o", color="#55544f", label="SR 차이에서 역산 (b-c)" if i == 0 else None)
    ax.plot([obs], [y], "s", color="#1f5fae", label="논문 표 7의 쌍 수 (b-c)" if i == 0 else None)
ax.set_yticks([len(names) - i for i in range(len(names))])
ax.set_yticklabels([n[1] for n in names], fontsize=7.5)
ax.axvline(0, color="#999", lw=0.6)
ax.set_xlabel("b - c (Ego2X만 성공 - 기준선만 성공)", fontsize=8)
ax.set_title("표 7 McNemar 쌍 수 점검: 3-4쌍 차이, 필터 없음 Bell은 부호까지 다름", fontsize=8.5)
ax.legend(fontsize=7, loc="lower right", frameon=False)
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "own_mcnemar_check.png")
print("ok", sorted(p.name for p in OUT.glob("own_*.png")))
