"""리포트용 그래프 세 장 (직접 그림). 실행: cd analysis && python3 plot_figures.py

1) throughput.png: Table 1 손·물체별 SPS(로그축)와 Fig. 1의 기준선 범위
2) fswo_scale.png: 같은 접점 배치를 키울 때 FSWO 통과 비율 (fswo_toy.py F2b) + PGD 15회의 오판정
3) cf_occupancy.png: contact field 1 cm box당 벡터 수 분포 (contact_field_toy.py)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import contact_field_toy as cft
import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e4e7", "#9a9a9f"
COL = {"Allegro": "#2a78d6", "LEAP": "#eb6834", "Shadow": "#1baf7a", "DClaw": "#eda100"}
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


# 1) throughput
fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
x = np.arange(len(tc.OBJECTS))
for k, (hand, (vals, _)) in enumerate(tc.TABLE1.items()):
    ax.scatter(x + (k - 1.5) * 0.12, vals, s=26, color=COL[hand], edgecolor="white", linewidth=1, zorder=3, label=hand)
ax.axhspan(30, 50, color=MUTED, alpha=0.25, lw=0)
ax.text(7.45, 38, "BODex 30–50", fontsize=7.5, color=INK2, va="center", ha="right")
ax.axhline(3, color=MUTED, lw=1, ls="--")
ax.text(7.45, 3.4, "DexGraspNet·SpringGrasp < 3", fontsize=7.5, color=INK2, ha="right")
ax.axhline(1000 / 6, color=INK2, lw=0.8, ls=":")
ax.text(-0.35, 1000 / 6 * 0.7, "6초에 1,000개가 되는 선 (≈167 SPS)", fontsize=7.5, color=INK2, va="top")
ax.set_yscale("log")
ax.set_ylim(2, 5000)
ax.set_xticks(x, tc.OBJECTS, fontsize=8)
ax.set_ylabel("유효 grasp / 초 (A100, 로그축)", color=INK2)
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
style(ax)
fig.text(0.01, 0.01, "출처: Table 1(손·물체별 amortized effective SPS), Fig. 1(기준선 범위). 기준선은 논문이 적은 범위 표기만 사용.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "throughput.png")
print("wrote", OUT / "throughput.png")

# 2) fswo scale
res = json.loads((ROOT / "analysis" / "fswo_toy_results.json").read_text())
f2b = {k: v for k, v in res["F2b_scaled_same_configuration"].items() if k.startswith("x")}
sizes = [float(k.split("_")[-1].replace("cm", "")) for k in f2b]
kept = [v["still_accepted"] for v in f2b.values()]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), dpi=200)
ax = axes[0]
ax.plot(sizes, kept, color="#2a78d6", lw=2, marker="o", ms=6, mec="white")
for s, k in zip(sizes, kept):
    ax.annotate(f"{k:.0%}", (s, k), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK2)
ax.set_xlabel("가장 긴 변 (cm), 접점 배치는 4 cm 때와 같은 모양", color=INK2)
ax.set_ylabel("FSWO < 0.15 통과 비율", color=INK2)
ax.set_ylim(0, 1.08)
ax.set_title("(a) 같은 grasp을 키우면 불안정으로 판정", fontsize=9.5, color=INK)
style(ax)
ax = axes[1]
f12 = res["F1_F2_pgd_vs_exact_by_size"]
for sname, c in (("cube", "#2a78d6"), ("box", "#eb6834"), ("tool", "#1baf7a")):
    ks = [k for k in f12 if k.startswith(sname + "_")]
    xs = [int(k.split("_")[1].replace("cm", "")) for k in ks]
    ys = [f12[k]["false_reject_among_exact_accept"] for k in ks]
    ax.plot(xs, ys, color=c, lw=2, marker="o", ms=5, mec="white", label=sname)
ax.set_xlabel("가장 긴 변 (cm)", color=INK2)
ax.set_ylabel("정확히는 통과인데 코드가 떨어뜨린 비율", color=INK2)
ax.set_ylim(0, 0.4)
ax.set_title("(b) PGD 15회의 과대추정 (오통과는 0)", fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=8)
style(ax)
fig.text(0.01, 0.01, "공식 코드의 점수식(토크 가중 10 m⁻¹, PGD lr 0.15)을 numpy로 옮겨 scipy NNLS와 비교. 무작위 3접점, 직육면체. analysis/fswo_toy.py", fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "fswo_scale.png")
print("wrote", OUT / "fswo_scale.png")

# 3) contact field occupancy (palmar patch)
q = cft.rng.uniform(cft.LIMITS[:, 0], cft.LIMITS[:, 1], size=(cft.N_CFG, 4))
R, c = cft.fk_tip(q)
pos_l, _ = cft.patch_keyvectors(np.array([1.0, 0, 0.3]))
pos = (np.einsum("nij,kj->nki", R, pos_l) + c[:, None, :]).reshape(-1, 3)
_, _, counts = cft.occupancy(pos)
fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=200)
ax.hist(counts, bins=np.arange(0, 900, 25), color="#2a78d6", edgecolor="white", linewidth=0.6)
for v, lab in ((50, "논문: 대부분 ≤ 50"), (256, "leaf 상한 256")):
    ax.axvline(v, color=INK2, lw=1, ls="--")
    ax.text(v + 8, ax.get_ylim()[1] * 0.9, lab, fontsize=7.5, color=INK2)
ax.set_xlabel("1 cm box 하나에 들어간 접촉 벡터 수", color=INK2)
ax.set_ylabel("box 수", color=INK2)
style(ax)
fig.text(0.01, 0.01, "Allegro 비슷한 손가락(가정), 관절 배치 80,000개 × 패치당 6개(공식 코드 설정). analysis/contact_field_toy.py", fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "cf_occupancy.png")
print("wrote", OUT / "cf_occupancy.png")
