"""우리가 그린 그래프 세 장 (report/figures/own_*.png).

1. own_ladder.png  : 적응 사다리 18개 시뮬레이터의 정책 걷기 점수 (저장 파일 analysis/inputs/d_ladder.json)
2. own_compress.png: 압축 복사본의 스킬 보유율 K(h) (논문 표 8.2 값을 옮김)
3. own_checks.png  : 검사별 걷기 순위 상관 (논문 §9.5 표와 본문 값, 저장 파일 d_stats.json과 대조)

실행: python3 analysis/plot_figures.py
한글 폰트: macOS AppleSDGothicNeo, Linux Noto CJK 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INP = HERE / "inputs"

for name in ["AppleSDGothicNeo", "Noto Sans CJK KR", "NanumGothic"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.family"] = name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

INK, GREY, BLUE, RED, GREEN = "#1b1b1a", "#8a8880", "#1f5fae", "#b8432f", "#2e7d4f"

# ---------- 1. 적응 사다리 ----------
ladder = json.loads((INP / "d_ladder.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.axhspan(0.573, 0.610, color=GREY, alpha=0.18, lw=0, label="released model, own two policies (0.573–0.610)")
ax.axhline(0.15, color=RED, ls="--", lw=1, label="policy fails below 0.15")
styles = {"W": ("walking data, 1e-4", BLUE, "o"), "Wtf": ("walking data, one-step windows, 1e-4", GREEN, "s"),
          "M": ("mixed data, 1e-4", "#7a4fa3", "^"), "Wg": ("walking data, 1e-5", BLUE, "D"),
          "Mg": ("mixed data, 1e-5", "#7a4fa3", "v")}
seen = set()
for t in ladder["tested"]:
    key = t["run"].split("_")[0]
    label, color, marker = styles.get(key, (key, GREY, "x"))
    for p in t["policies"]:
        lab = label if key not in seen else None
        seen.add(key)
        ax.scatter(t["steps"], p["score"], color=color, marker=marker, s=28, alpha=0.85, label=lab,
                   edgecolor="white", lw=0.4, zorder=3)
ax.set_xscale("log")
ax.set_xticks([5, 20, 80, 160, 320])
ax.set_xticklabels(["5", "20", "80", "160", "320"])
ax.set_xlabel("fine-tuning steps on MuJoCo data (log scale)")
ax.set_ylabel("walking score of each policy")
ax.set_ylim(-0.02, 0.7)
ax.set_title("Each dot: one policy in a fine-tuned copy of the released model (n = 40)", fontsize=9.5, loc="left")
ax.legend(fontsize=7, loc="upper right", frameon=False)
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(OUT / "own_ladder.png", dpi=170)
plt.close(fig)

# ---------- 2. 압축 복사본의 스킬 보유율 (논문 표 8.2) ----------
horizons = ["1 step", "24 steps\n(0.48 s)", "100 steps\n(2 s)", "368 steps\n(7.4 s)"]
rows = {
    "8-bit weights": [99.7, 99.3, 97.0, 95],
    "20% pruned": [97.4, 95, 80, 57],
    "30% pruned": [90, 83, 39, -41],
    "30% pruned, fine-tuned 500": [102.3, 100.9, 102.2, 103.9],
}
fig, ax = plt.subplots(figsize=(7.2, 3.6))
colors = [BLUE, GREEN, RED, GREY]
x = list(range(len(horizons)))
for (name, vals), c in zip(rows.items(), colors):
    ax.plot(x, vals, marker="o", color=c, lw=1.6, label=name)
    ax.annotate(f"{vals[0]:g}%", (0, vals[0]), textcoords="offset points", xytext=(-22, 3), fontsize=7, color=c)
    ax.annotate(f"{vals[2]:g}%", (2, vals[2]), textcoords="offset points", xytext=(4, -10), fontsize=7, color=c)
ax.axhline(100, color=INK, lw=0.6, ls=":")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(horizons, fontsize=8)
ax.set_ylabel("skill retained K(h), %")
ax.set_title("Same 30% pruning: 90% at one step, 39% at 2 s (paper Table 8.2 values)", fontsize=9.5, loc="left")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(OUT / "own_compress.png", dpi=170)
plt.close(fig)

# ---------- 3. 검사별 걷기 순위 상관 ----------
stats = json.loads((INP / "d_stats.json").read_text(encoding="utf-8"))
wt = stats["secondary"]["walk_test"]
items = [
    ("1-step error", -0.42, None, GREY),
    ("8-step error", -0.43, None, GREY),
    ("100-step error (premise)", list(stats["premise"].values())[0]["rho"], list(stats["premise"].values())[0]["ci95"], GREY),
    ("368-step error", 0.32, None, GREY),
    ("growth rate", 0.07, None, BLUE),
    ("kinematic residual", 0.32, None, BLUE),
    ("walk test: tracking reward", wt["tracking"]["rho"]["rho"], wt["tracking"]["rho"]["ci95"], GREEN),
    ("walk test: head disagreement", 0.84, [0.53, 0.96], GREEN),
]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (name, rho, ci, c) in enumerate(items):
    y = len(items) - 1 - i
    if ci:
        ax.plot([ci[0], ci[1]], [y, y], color=c, lw=2, alpha=0.45)
    ax.scatter(rho, y, color=c, s=36, zorder=3)
    ax.annotate(f"{rho:+.2f}", (rho, y), textcoords="offset points", xytext=(6, 2), fontsize=7.5)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([n for n, *_ in items][::-1], fontsize=8)
ax.axvline(0, color=INK, lw=0.6)
ax.set_xlim(-1, 1.05)
ax.set_xlabel("Spearman rank correlation with the mean walking score (19 simulators)")
ax.set_title("Checks vs walking (lines: 95% intervals where the paper gives them)", fontsize=9, loc="left")
ax.grid(alpha=0.25, lw=0.5, axis="x")
fig.tight_layout()
fig.savefig(OUT / "own_checks.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
