"""verify_logs_results.json으로 리포트 그래프를 그린다(직접 그린 그림, 공개 로그 CC BY 4.0 기반).

실행: python3 analysis/verify_logs.py && python3 analysis/plot_figures.py
→ report/figures/exp4_edit_size.png, exp5_seeds.png, exp6_budget.png, exp8_paired.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
INK2, GRID = "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "verify_logs_results.json").read_text())


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 실험 4: 바뀐 개수는 3.5%지만 크기는 수십~수천 배
c4 = R["C4_exp4"]
names = ["embed.W", "embed.b", "actor0.W", "actor0.b", "actor1.W", "actor1.b", "actor2.W", "actor2.b",
         "actor3.W", "actor3.b", "actor4.W", "actor4.b", "actor_head.W", "actor_head.b"]
fig, ax = plt.subplots(figsize=(8.2, 3.2))
x = range(len(names))
ax.bar([i - 0.2 for i in x], [c4["max_abs"][n]["max_abs_base"] for n in names], 0.4, color=MUTED, label="PPO 원래 가중치 |w| 최대")
ax.bar([i + 0.2 for i in x], [c4["max_abs"][n]["max_abs_edited"] for n in names], 0.4, color=ORANGE, label="Astra 편집 후 |w| 최대")
ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("|w| 최대 (로그)"); grid(ax)
ax.set_title(f"실험 4: 바뀐 칸은 {c4['frac_changed'] * 100:.1f}%지만, 전체 가중치 노름 대비 변화량은 {c4['recomputed_rel_change_fro']:.1f}배", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "exp4_edit_size.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 실험 5: 씨앗별 PPO 점수와 Astra
mn = json.loads((ROOT / "logs_official" / "exp5_write_from_scratch" / "eval" / "minatar_eval.json").read_text()) \
    if (ROOT / "logs_official").exists() else None
if mn:
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.0))
    for ax, g in zip(axs, ["breakout", "seaquest", "asterix"]):
        d = mn[g]
        p1 = [v["mean"] for k, v in d.items() if k.startswith("ppo100")]
        p5 = [v["mean"] for k, v in d.items() if k.startswith("ppo500")]
        ax.scatter([0] * len(p1), p1, color=MUTED, s=18, label="PPO 100M (씨앗별)")
        ax.scatter([1] * len(p5), p5, color=BLUE, s=18, label="PPO 500M (씨앗별)")
        ax.scatter([2], [d["hand"]["mean"]], color=ORANGE, s=40, marker="D", label="Astra 한 번에 작성")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["PPO\n100M", "PPO\n500M", "Astra"])
        ax.set_title(g, fontsize=10); grid(ax)
        if g == "breakout":
            ax.set_yscale("log"); ax.set_ylabel("평균 점수 (1,000 에피소드)")
    axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("실험 5: PPO 씨앗 사이 편차가 크다 (블로그 표는 각 칸의 최고 씨앗)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "exp5_seeds.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 실험 6: forward pass 예산별 승률(95% 신뢰구간)
c6 = R["C6_exp6"]
b = [0, 10, 25, 50]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
for key, col, lab in (("ppo_s0", BLUE, "작은 PPO seed 0"), ("ppo_s2", AQUA, "작은 PPO seed 2")):
    y = [c6[f"budget_{k}"][key]["score"] * 100 for k in b]
    lo = [y[i] - c6[f"budget_{k}"][key]["wilson95"][0] * 100 for i, k in enumerate(b)]
    hi = [c6[f"budget_{k}"][key]["wilson95"][1] * 100 - y[i] for i, k in enumerate(b)]
    ax.errorbar(b, y, yerr=[lo, hi], color=col, marker="o", ms=4, lw=1.6, capsize=3, label=lab)
h = c6["h2h_same_size"]
for k, bb in (("0", 0), ("50", 50)):
    vals = [h[f"v1_b{k}_vs_lr3e-5"], h[f"v1_b{k}_vs_lr3e-5_s1"], h[f"v1_b{k}_vs_lr3e-5_s2"]]
    ax.scatter([bb] * 3, [v * 100 for v in vals], color=ORANGE, marker="x", s=30,
               label="같은 크기 PPO 3개와 맞대결" if k == "0" else None)
ax.axhline(50, color=INK2, lw=0.8, ls="--")
ax.set_xlabel("Astra가 쓴 forward pass 수"); ax.set_ylabel("승률 (%)"); ax.set_xticks(b); grid(ax)
ax.set_title("실험 6: 9x9 바둑, 2,000판 승률", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "exp6_budget.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 실험 8: 문제별 짝 비교 차이와 95% 부트스트랩 구간
c8 = R["C8_exp8"]
bench = [("math500", "MATH-500"), ("gsm8k", "GSM8K"), ("amc23", "AMC23"), ("minerva", "Minerva"), ("olympiad", "Olympiad"),
         ("ifeval", "IFEval"), ("gpqa", "GPQA-D"), ("mmlupro1k", "MMLU-Pro 1k")]
runs = [("p10c", "10 pass", MUTED), ("p25c", "25 pass", BLUE), ("p50c", "50 pass", ORANGE)]
fig, ax = plt.subplots(figsize=(8.4, 3.3))
w = 0.26
for j, (r, lab, col) in enumerate(runs):
    xs = [i + (j - 1) * w for i in range(len(bench))]
    ys = [c8[k][r]["diff_pp"] for k, _ in bench]
    lo = [ys[i] - c8[k][r]["ci95_pp"][0] for i, (k, _) in enumerate(bench)]
    hi = [c8[k][r]["ci95_pp"][1] - ys[i] for i, (k, _) in enumerate(bench)]
    ax.bar(xs, ys, w * 0.92, color=col, label=lab)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#333", lw=0.8, capsize=2)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(len(bench))); ax.set_xticklabels([n for _, n in bench], fontsize=8)
ax.set_ylabel("편집 - 원본 (%p)"); grid(ax)
cut = c8["reported_merged"]
ax.set_title(f"실험 8: Qwen3.5-0.8B 편집 효과. MATH-500에서 토큰 상한에 걸린 답 {cut['unedited']['math500_cut_share']:.0f}% → "
             f"{cut['p10c']['math500_cut_share']:.0f} / {cut['p25c']['math500_cut_share']:.0f} / {cut['p50c']['math500_cut_share']:.0f}%", fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "exp8_paired.png", dpi=170); plt.close(fig)
print("wrote figures to", OUT)
