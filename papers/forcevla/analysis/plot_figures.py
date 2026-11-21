"""우리가 그린 그래프 두 장을 만든다 (report/figures/own_*.png).

  own_setting_ci.png : 표 2의 방법별 풀링 성공률과 Wilson 95% 구간, 설정별 ForceVLA 대 π0-base w/ F
  own_fig5_average.png : 그림 5 막대 라벨을 옮겨 적은 값으로 평균을 다시 계산한 결과 (6막대 평균 대 5과제 평균)

입력: analysis/recompute_results.json (recompute.py가 만든다). 그림 5 값은 원문 7쪽 막대 라벨을 읽은 것이다.
실행: python3 analysis/plot_figures.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
font_path = next((p for p in FONT_CANDIDATES if pathlib.Path(p).exists()), None)
if font_path:
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

INK = "#1b1b1a"
GREY = "#8a8880"
ACCENT = "#1f5fae"
LIGHT = "#c9c6bd"

# ---------- 그림 1: 표 2 ----------
methods = ["pi0-fast w/o F", "pi0-fast w/ F", "pi0-base w/o F", "pi0-base w/ F", "ForceVLA"]
t2 = RES["table2"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
ys = list(range(len(methods)))[::-1]
for y, m in zip(ys, methods):
    lo, hi = t2[m]["pooled_wilson95_pct"]
    rate = t2[m]["pooled_rate_pct"]
    color = ACCENT if m == "ForceVLA" else GREY
    ax1.plot([lo, hi], [y, y], color=color, lw=2)
    ax1.plot(rate, y, "o", color=color, ms=7)
    ax1.text(hi + 1.5, y, f"{rate:.1f}% ({t2[m]['pooled_k']}/{t2[m]['pooled_n']})", va="center", fontsize=9, color=INK)
ax1.set_yticks(ys)
ax1.set_yticklabels(methods, fontsize=9)
ax1.set_xlim(0, 100)
ax1.set_xlabel("풀링 성공률 (%) 와 Wilson 95% 구간", fontsize=9)
ax1.set_title("표 2 설정 5개 합산 (독립 가정)", fontsize=10, color=INK, loc="left")
ax1.grid(axis="x", color=LIGHT, lw=0.5, ls="--")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

settings = ["Object Gen.1", "Object Gen.2", "Height Gen.", "Visual Occlusion", "Unstable Socket"]
n_set = [RES["table2_trials_per_setting"][s] for s in settings]
fv = t2["ForceVLA"]["k"]
bf = t2["pi0-base w/ F"]["k"]
x = range(len(settings))
w = 0.38
ax2.bar([i - w / 2 for i in x], [100 * k / n for k, n in zip(bf, n_set)], w, color=GREY, label="pi0-base w/ F")
ax2.bar([i + w / 2 for i in x], [100 * k / n for k, n in zip(fv, n_set)], w, color=ACCENT, label="ForceVLA")
for i, (k1, k2, n) in enumerate(zip(bf, fv, n_set)):
    ax2.text(i - w / 2, 100 * k1 / n + 2, f"{k1}/{n}", ha="center", fontsize=7.5, color=INK)
    ax2.text(i + w / 2, 100 * k2 / n + 2, f"{k2}/{n}", ha="center", fontsize=7.5, color=INK)
ax2.set_xticks(list(x))
ax2.set_xticklabels(settings, fontsize=7.5, rotation=15)
ax2.set_ylim(0, 118)
ax2.set_ylabel("성공률 (%)", fontsize=9)
ax2.set_title("설정별 (시행 수는 막대 위)", fontsize=10, color=INK, loc="left")
ax2.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
fig.text(0.01, 0.01, "시행 수 n: 설정 1=25, 설정 2=20, 높이=9, 가림=20, 불안정=20 (백분율에서 역산한 가정, 본문에 없음)",
         fontsize=7.5, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_setting_ci.png", dpi=170)
plt.close(fig)

# ---------- 그림 2: 그림 5 평균 점검 ----------
cols = ["Insert USB", "Insert Plug", "Pump Bottle", "Wipe Board-1", "Wipe Board-2", "Peel Cucumber"]
bars = {
    "pi0-base w/o F": [5.0, 45.0, 33.0, 45.0, 9.0, 87.0],
    "pi0-base w/ F": [5.0, 60.0, 83.0, 0.0, 0.0, 93.0],
    "ForceVLA": [25.0, 80.0, 67.0, 64.0, 27.0, 100.0],
}
colors = {"pi0-base w/o F": LIGHT, "pi0-base w/ F": "#b9a77f", "ForceVLA": ACCENT}
fig, ax = plt.subplots(figsize=(10.5, 4.2))
w = 0.26
for j, (m, vals) in enumerate(bars.items()):
    xs = [i + (j - 1) * w for i in range(len(cols))]
    ax.bar(xs, vals, w, color=colors[m], label=m)
    for xx, v in zip(xs, vals):
        ax.text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, fontsize=8.5)
ax.set_ylim(0, 112)
ax.set_ylabel("성공률 (%)", fontsize=9)
ax.set_title("그림 5 막대 값 (원문 라벨을 옮김). 평균은 6막대의 산술 평균과 일치", fontsize=10, color=INK, loc="left")
f5 = RES["figure5_average_check"]
txt = []
for m in bars:
    txt.append(f"{m}: 보고 {f5[m]['reported_average']:.1f} / 6막대 {f5[m]['mean_of_6_bars']:.1f} / 5과제(와이프 평균) {f5[m]['mean_of_5_tasks_wipe_avg']:.1f}")
ax.text(0.0, -0.2, "\n".join(txt), transform=ax.transAxes, fontsize=7.5, color=INK, va="top")
ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
ax.grid(axis="y", color=LIGHT, lw=0.5, ls="--")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(FIG / "own_fig5_average.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
