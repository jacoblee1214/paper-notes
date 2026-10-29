"""리포트용 그래프 세 장. 실행: cd analysis && python3 plot_figures.py

1) checkpoints.png: 그림 6 판독값(25K·50K)과 그림 5(b) 막대, 계산량 비율
2) kernels.png: 시작 상태 샘플링 커널 모양과 학습 후반 프런티어 표본 비율 (sampling_toy.py와 같은 식)
3) reward_delay.png: 성공 시 종료 + 양의 보조 보상에서 최적 정책의 완료 시점과 성공률 (reward_toy.py와 같은 식)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import reward_toy as rt
import sampling_toy as st

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, YELLOW, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DIG = json.loads((ROOT / "analysis" / "fig_digitize_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


# 1) checkpoint 대응
cv = DIG["fig6_curves"]
names = [("shared_sparse", "공유 정책\nsparse"), ("separate_sparse", "분리 20개\nsparse"),
         ("shared_dense", "공유 정책\ndense"), ("separate_dense", "분리 20개\ndense")]
reported = {"shared_sparse": 98.2, "separate_sparse": 81.4, "shared_dense": 17.8, "separate_dense": 78.3}
fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=200)
x = np.arange(len(names))
w = 0.34
b1 = ax.bar(x - w / 2, [cv[k]["25K"] for k, _ in names], w, color=BLUE, label="그림 6 판독, 25K iteration")
b2 = ax.bar(x + w / 2, [cv[k]["50K"] for k, _ in names], w, color=MUTED, label="그림 6 판독, 50K iteration")
ax.scatter(x, [reported[k] for k, _ in names], marker="D", s=34, color=ORANGE, zorder=3, edgecolor="white",
           linewidth=1.2, label="그림 5(b)에 적힌 값 (캡션: 25K)")
for xi, (k, _) in zip(x, names):
    ax.annotate(f"{reported[k]}", (xi, reported[k]), xytext=(9, 4), textcoords="offset points", fontsize=7.5, color=INK2)
ax.set_xticks(x, [n for _, n in names])
ax.set_ylim(0, 112)
ax.set_ylabel("학습 성공률 (%)", color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
style(ax)
ax.annotate("분리 dense의 78.3은 50K 값", (3 + w / 2, cv["separate_dense"]["50K"]), xytext=(2.05, 55),
            fontsize=7.5, color=ORANGE, arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.8})
fig.text(0.01, 0.01, "판독: analysis/fig_digitize.py. 계산량은 같은 25K 기준이면 약 20배, 분리 정책을 50K까지 돌린 비용이면 39.3배(본문).",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "checkpoints.png")
print("wrote", OUT / "checkpoints.png")

# 2) 샘플링 커널
p = np.linspace(0, 1, 401)
kern = [("균등", st.q_uniform, MUTED, "-"), ("식 4, ε_q = 1e-4", lambda v: st.q_alphanist(v, 1e-4), BLUE, "-"),
        ("식 4, ε_q = 0.1", lambda v: st.q_alphanist(v, 0.1), BLUE, "--"), ("SGS 원 설정 (T = 2)", st.q_sgs, ORANGE, "-")]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), dpi=200)
for label, f, c, ls in kern:
    q = f(p)
    axes[0].plot(p, q / f(np.array([0.5]))[0], color=c, ls=ls, lw=2, label=label)
axes[0].set_xlabel("항목의 추정 성공률 $\\hat{p}$", color=INK2)
axes[0].set_ylabel("상대 샘플링 가중치 ($\\hat{p}$ = 0.5에서 1)", color=INK2)
axes[0].set_title("커널 모양", fontsize=10, color=INK)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower center")
style(axes[0])
mastered = np.linspace(0, 0.99, 60)
for label, f, c, ls in kern:
    share = []
    for m in mastered:
        bank = np.concatenate([np.ones(int(2000 * m)), np.linspace(0.05, 0.95, 2000 - int(2000 * m))])
        q = f(bank)
        q = q / q.sum()
        share.append(q[bank < 1].sum())
    axes[1].plot(100 * mastered, 100 * np.array(share), color=c, ls=ls, lw=2, label=label)
axes[1].set_xlabel("다 익힌 항목($\\hat{p}$ = 1)의 비율 (%)", color=INK2)
axes[1].set_ylabel("프런티어 항목에 가는 표본 (%)", color=INK2)
axes[1].set_title("학습 후반의 배분", fontsize=10, color=INK)
style(axes[1])
fig.text(0.01, 0.01, "AlphaNIST 식 4와 SGS(arXiv:2610.12465 식 1–2, κ = 1, t = 0.5, ε = 1e-4, T = 2). AlphaNIST의 ε_q는 미기재.",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "kernels.png")
print("wrote", OUT / "kernels.png")

# 3) 보상 구조와 완료 시점
cases = [("이진 보상만 (식 6, 초록과 겹침)", dict(shaping=0.0, success=1.0, terminate_on_success=True), BLUE),
         ("dense + 성공 시 종료 (식 7)", dict(shaping=0.02, success=1.0, terminate_on_success=True), ORANGE),
         ("OmniReset식 (성공 매 step, 종료 없음)", dict(shaping=0.02, success=0.0, terminate_on_success=False,
                                                    success_per_step=0.1), AQUA)]
hazards = [0.0, 0.005, 0.01, 0.02]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), dpi=200)
for label, kw, c in cases:
    _, pi = rt.solve(0.99, hazard=0.0, **kw)
    r = rt.rollout(pi, 0.0, n=1500)
    axes[0].barh(label, r["median_completion_step"], color=c, height=0.55)
    axes[0].text(r["median_completion_step"] + 5, label, f"{r['median_completion_step']} step", va="center", fontsize=7.5, color=INK2)
    succ = []
    for h in hazards:
        _, pi_h = rt.solve(0.99, hazard=h, **kw)
        succ.append(100 * rt.rollout(pi_h, h, n=1500)["success_within_T"])
    axes[1].plot(hazards, succ, marker="o", color=c, lw=2, label=label)
axes[0].axvline(rt.T, color=INK2, ls=":", lw=1)
axes[0].text(rt.T - 4, 2.35, "horizon 375", fontsize=7, color=INK2, ha="right")
axes[0].set_xlim(0, 430)
axes[0].set_xlabel("최적 정책의 완료 시점 중앙값 (step)", color=INK2)
axes[0].invert_yaxis()
style(axes[0])
axes[1].set_xlabel("완료 직전 대기 중 미끄러짐 확률 (step당)", color=INK2)
axes[1].set_ylabel("horizon 안 성공률 (%)", color=INK2)
axes[1].set_ylim(88, 101)
axes[1].legend(frameon=False, fontsize=7.2, loc="lower left")
style(axes[1])
fig.text(0.01, 0.01, "진행도 60칸 사슬 MDP, γ = 0.99, 전진 확률 0.9, backward induction(analysis/reward_toy.py). 미끄러짐이 크면(0.02) 최적 정책이 대기를 포기한다. 보조 항 꼴·γ·성공 보상 크기는 가정.",
         fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "reward_delay.png")
print("wrote", OUT / "reward_delay.png")
