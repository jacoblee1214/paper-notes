"""논문 보고 수치를 우리가 다시 그린 그래프 세 장 (report/figures/own_*.png).

논문 그림은 쓰지 않는다. 입력은 논문 표의 숫자와 recompute_results.json의 Wilson 구간이다.
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 한글 폰트: macOS 후보, Linux 후보 순서로 찾는다
names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"

R = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


# 1. 표 V 절제: GSR (논문 값 그대로)
labels = ["전역 정렬\n최적화 없음", "전역 정렬\n필터 없음", "지역 정제\n최적화 없음",
          "지역 정제\n필터 없음", "템플릿\n업데이트 없음", "전체 (Ours)"]
vals = [45.7, 18.6, 17.8, 62.8, 41.5, 60.5]
colors = [GREY] * 5 + [ACC]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
ax.axhline(60.5, color=ACC, lw=0.8, ls="--")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=7.6)
ax.set_ylabel("GSR (%)", fontsize=9, color=INK)
ax.set_ylim(0, 75)
ax.set_title("표 V 재정리: 모듈 제거 시 GSR (Allegro 손, 손끝 파지, 시드 한 번)", fontsize=9.5, color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_ablation.png")
plt.close(fig)

# 2. 표 III: 시도 20 -> 100에서 OSR (GSR은 거의 평평)
att = ["20", "100"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=200, sharey=True)
data = {
    "BODex": {"DGN": [71.30, 89.84], "Objaverse": [43.48, 73.53]},
    "Ours": {"DGN": [91.28, 95.13], "Objaverse": [84.17, 94.63]},
}
for ax, dset in zip(axes, ["DGN", "Objaverse"]):
    for name, c in [("BODex", GREY), ("Ours", ACC)]:
        ys = data[name][dset]
        ax.plot(att, ys, marker="o", color=c, lw=1.6, label=name)
        for x, y in zip(att, ys):
            ax.text(x, y + 2, f"{y:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_title(f"OSR, {dset} 객체", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("시도 수 (객체당)", fontsize=8.5, color=INK)
    ax.set_ylim(35, 102)
    style(ax)
axes[0].set_ylabel("OSR (%)", fontsize=9, color=INK)
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.text(0.01, 0.01, "GSR은 BODex 14.8 → 14.8 (DGN), 6.9 → 6.9 (Objaverse)로 평평하다. 표 III 원본 값.",
         fontsize=7.5, color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIG / "own_attempts_osr.png")
plt.close(fig)

# 3. 실제 로봇 성공률과 Wilson 95% 구간 (163/198 역산)
rw = R["V9_realworld"]["wilson95_at_163_of_198"]
p, lo, hi = rw
fig, ax = plt.subplots(figsize=(7.2, 2.2), dpi=200)
ax.errorbar([p], [0], xerr=[[p - lo], [hi - p]], fmt="o", color=ACC, ecolor=ACC, capsize=5, ms=7)
ax.text(p, 0.22, f"{p:.1f}%  (163/198)", ha="center", fontsize=9, color=INK)
ax.text(lo, -0.27, f"{lo:.1f}", ha="center", fontsize=8, color=INK)
ax.text(hi, -0.27, f"{hi:.1f}", ha="center", fontsize=8, color=INK)
ax.set_xlim(70, 95)
ax.set_ylim(-0.6, 0.5)
ax.set_yticks([])
ax.set_xlabel("성공률 (%), Wilson 95% 구간", fontsize=8.5, color=INK)
ax.set_title("실제 로봇 (Shadow 손, 12 유형, 13 물체): 분모 198은 본문에서 역산", fontsize=9, color=INK, loc="left")
style(ax)
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_realworld_ci.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
