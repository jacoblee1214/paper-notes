"""공개 리포트와 게시본에 넣을 그래프 세 장을 만든다 (우리가 그림, CPU, 수 초).

입력: analysis/recompute_results.json (recompute.py 결과)
출력: report/figures/own_pearson_ci.png, own_fvd_change.png, own_planning_ci.png
한글 폰트는 Linux(Noto CJK) 다음 macOS(Apple SD Gothic Neo) 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))

available = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleSDGothicNeo"]:
    if cand in available:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, BASE, AFF = "#1b1b1a", "#8a8880", "#9aa7b8", "#1f5fae"

# 그림 1: Pearson 상관과 n=8 근사 95% CI (Table 1, 2)
keys = [k for k in R["pearson_n8"] if k.endswith("|base")]
labels = [k.split("|")[0] + " " + k.split("|")[1] for k in keys]
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
for i, k in enumerate(keys):
    b = R["pearson_n8"][k]
    a = R["pearson_n8"][k.replace("|base", "|AFFECT")]
    y = len(keys) - i
    ax.plot([b["ci95_lo"], b["ci95_hi"]], [y + 0.12, y + 0.12], color=BASE, lw=2)
    ax.plot(b["r"], y + 0.12, "o", color=BASE, ms=6)
    ax.plot([a["ci95_lo"], a["ci95_hi"]], [y - 0.12, y - 0.12], color=AFF, lw=2)
    ax.plot(a["r"], y - 0.12, "o", color=AFF, ms=6)
ax.set_yticks(range(len(keys), 0, -1))
ax.set_yticklabels(labels, color=INK, fontsize=9)
ax.axvline(0, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("Pearson r (실제 대 세계모델 성공률)", color=INK)
ax.set_xlim(-0.8, 1.0)
ax.set_title("논문 Table 1, 2의 상관: 점 = 논문 값, 선 = n=8 가정의 Fisher z 95% CI", color=INK, fontsize=10)
ax.legend(handles=[plt.Line2D([], [], color=BASE, marker="o", label="기반 모델 (flow loss만)"),
                   plt.Line2D([], [], color=AFF, marker="o", label="+ AFFECT")], loc="lower right", fontsize=8, frameon=False)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_pearson_ci.png")
plt.close(fig)

# 그림 2: FVD 상대 변화 (논문 값에서 계산)
fvd = R["fvd_rel_change_pct"]
names, vals = list(fvd.keys()), list(fvd.values())
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=180)
colors = [AFF if v < 0 else "#b5533c" for v in vals]
ax.barh(range(len(names)), vals, color=colors, height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9, color=INK)
ax.invert_yaxis()
ax.axvline(0, color=INK, lw=0.8)
for i, v in enumerate(vals):
    ax.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
ax.set_xlabel("FVD 상대 변화 (AFFECT 대 기반, %, 낮을수록 좋음)", color=INK)
ax.set_xlim(-30, 8)
ax.set_title("FVD 변화: DROID와 ALOHA-2 Cosmos3가 크고, ALOHA-2 DreamDojo는 나빠짐", color=INK, fontsize=10)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_fvd_change.png")
plt.close(fig)

# 그림 3: 계획 성공률과 Wilson 95% CI (UR5e n=60, ALOHA-2 n=36 역산)
rows = [
    ("UR5e 균등 선택", R["ur5e_planning"]["Uniform"]),
    ("UR5e DreamDojo", R["ur5e_planning"]["DreamDojo"]),
    ("UR5e DreamDojo+AFFECT", R["ur5e_planning"]["DreamDojo+AFFECT"]),
    ("ALOHA-2 DreamDojo (n=36 추정)", R["aloha2_planning_inferred_n36"]["DreamDojo"]),
    ("ALOHA-2 DreamDojo+AFFECT (n=36 추정)", R["aloha2_planning_inferred_n36"]["DreamDojo+AFFECT"]),
]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=180)
for i, (name, d) in enumerate(rows):
    y = len(rows) - i
    lo, hi = d["wilson95"]
    c = AFF if "AFFECT" in name else BASE
    ax.plot([lo, hi], [y, y], color=c, lw=2.2)
    ax.plot(d["pct"], y, "o", color=c, ms=7)
    ax.text(hi + 1.2, y, f"{d['pct']:.1f}% ({d['k']}/{d['n']})", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(rows), 0, -1))
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
ax.set_xlim(20, 105)
ax.set_xlabel("성공률 (%) 와 Wilson 95% CI", color=INK)
ax.set_title("계획 성공률: UR5e 차이 8.3pp는 60시도에서 5번 차이, 구간이 크게 겹침", color=INK, fontsize=10)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_planning_ci.png")
plt.close(fig)
print("ok", sorted(p.name for p in FIG.glob("own_*.png")))
