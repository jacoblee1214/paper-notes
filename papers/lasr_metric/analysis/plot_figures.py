"""논문 표 값으로 우리가 그린 그래프 세 장 (논문 그림은 쓰지 않음).

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
입력은 논문 표 10, 표 3(b)·표 8 값을 이 파일에 옮겨 적은 것이다 (논문 그림 파일 없음).
한글 폰트는 Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

INK, MUTED, ACCENT, WARN, LINE = "#1b1b1a", "#55544f", "#1f5fae", "#b5532a", "#dcdad3"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED)


# 1. 매칭 패널 GeoLASR 점추정과 부트스트랩 구간 (표 10, 부록 H.4)
routes = [
    ("DiLA 16프레임", 0.688, 0.668, 0.708, 10),
    ("DiLA 쌍 전용", 0.685, 0.665, 0.704, 10),
    ("LAPA 토큰", 0.831, 0.810, 0.851, 10),
    ("LaWAM", 0.860, 0.838, 0.883, 10),
    ("MVP-LAM", 0.988, 0.981, 0.995, 10),
    ("UniVLA 제어 가능", 1.010, 0.998, 1.025, 4),
]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (name, est, lo, hi, cells) in enumerate(routes):
    y = len(routes) - 1 - i
    color = ACCENT if est < 1 else WARN
    ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="butt")
    ax.plot(est, y, "o", color=color, ms=6)
    ax.text(1.035, y, f"지원 {cells}/10", va="center", color=MUTED, fontsize=9)
ax.axvline(1.0, color=INK, lw=1, ls="--")
ax.text(1.0, len(routes) - 0.35, "경계 1", ha="center", color=INK, fontsize=9)
ax.set_yticks(range(len(routes)))
ax.set_yticklabels([r[0] for r in routes][::-1])
ax.set_xlim(0.64, 1.09)
ax.set_xlabel("GeoLASR (낮을수록 좋음), 점추정과 95% 부트스트랩 구간")
ax.set_title("매칭 패널 GeoLASR (논문 표 10)", loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "own_matched_geo_ci.png", dpi=200)
plt.close(fig)

# 2. 쌍체 차이 (표 8) 와 점추정 차이 (표 3b) 의 비교, Worst 기준 (U1)
comps = ["원래 표현", "무작위 잔차", "섞은 잔차", "MSE 잔차"]
point_worst = [-0.011, -0.030, -0.049, -0.025]
paired_worst = [-0.00488, -0.01713, -0.01456, -0.01404]
point_geo = [-0.005, -0.017, -0.016, -0.014]
paired_geo = [-0.00563, -0.01804, -0.01617, -0.01466]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, pts, prs, title in [(axes[0], point_geo, paired_geo, "GeoLASR"),
                            (axes[1], point_worst, paired_worst, "WorstLASR")]:
    y = list(range(len(comps)))[::-1]
    ax.barh([v + 0.18 for v in y], pts, height=0.34, color=LINE, label="표 3(b) 점추정 차")
    ax.barh([v - 0.18 for v in y], prs, height=0.34, color=ACCENT, label="표 8 쌍체 차")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("어댑터 - 비교 대상 (음수가 어댑터 우세)")
    style(ax)
axes[0].set_yticks(list(range(len(comps)))[::-1])
axes[0].set_yticklabels(comps)
fig.suptitle("잔차 어댑터: 점추정 차와 쌍체 차 (논문 표 3b, 표 8)", x=0.02, ha="left", color=INK)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "own_paired_vs_point.png", dpi=200)
plt.close(fig)

# 3. 기록 행동과 누출 반례, 무작위 대조의 판정 (표 10)
ctrl = [
    ("기록 행동 (목표 Y_act, Y_Δ)", 0.002, 10),
    ("무작위 대조", 1.000, 3),
    ("쌍 안에서 섞은 대조", 0.999, 8),
    ("현재 프레임 시각", 1.000, 0),
    ("다음 프레임 결과 (누출 반례)", 0.985, 10),
]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for i, (name, geo, cells) in enumerate(ctrl):
    y = len(ctrl) - 1 - i
    color = WARN if "누출" in name else (ACCENT if geo < 1 else MUTED)
    ax.barh(y, max(geo, 0.004), color=color, height=0.55)
    ax.text(1.02, y, f"지원 {cells}/10", va="center", color=MUTED, fontsize=9)
ax.axvline(1.0, color=INK, lw=1, ls="--")
ax.set_yticks(range(len(ctrl)))
ax.set_yticklabels([c[0] for c in ctrl][::-1])
ax.set_xlim(0, 1.15)
ax.set_xlabel("GeoLASR (1 미만이면 통과)")
ax.set_title("통제 실험의 판정: 누출 반례도 1 미만 (논문 표 10)", loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "own_controls_pass.png", dpi=200)
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
