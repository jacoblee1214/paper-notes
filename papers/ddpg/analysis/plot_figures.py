"""리포트 그래프와 toy 통계. 실행: cd analysis && python3 plot_figures.py

입력: table_check.py의 표 값, ddpg_toy_results.json (ddpg_toy.py를 먼저 실행)
출력: report/figures/{table1_paired, toy_final, toy_curves, ou_noise}.png, analysis/toy_stats_results.json
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import stats

import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, VIOLET, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TOY = json.loads((ROOT / "analysis" / "ddpg_toy_results.json").read_text())
LABELS = {"paper": "논문 설정", "no_target": "표적망 없음 (τ = 1)", "tau_0.005": "τ = 0.005",
          "no_l2": "critic L2 없음", "ou_dt_0.01": "OU dt = 0.01", "gauss_0.1": "가우시안 σ = 0.1",
          "no_noise": "잡음 없음", "retuned": "재조정 (TD3식)"}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=6.8, color=INK2)


# 1) Table 1 과제 단위 비교 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=200)
pairs = [("av_cntrl", "av_lowd", "표적망 효과: DPG + BN (cntrl) → DDPG", "R_av, cntrl (표적망 없음)", "R_av, lowd (DDPG)"),
         ("av_lowd", "av_pix", "저차원 관측 → 픽셀 관측", "R_av, lowd", "R_av, pix")]
for ax, (cx, cy, title, xl, yl) in zip(axes, pairs):
    x, y = tc.col(cx), tc.col(cy)
    ax.plot([-0.6, 1.5], [-0.6, 1.5], color=INK2, lw=0.8, ls="--")
    ax.scatter(x, y, s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    above = sum(b > a for a, b in zip(x, y))
    ax.text(0.97, 0.04, f"대각선 위: {above} / {len(x)}개 과제", transform=ax.transAxes, fontsize=8.5, color=INK, ha="right")
    for name in ("reacherSingle", "hardCheetah", "blockworld3da", "canada2d"):
        i = list(tc.TABLE1).index(name)
        ax.annotate(name, (x[i], y[i]), xytext=(5, -3), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlim(-0.6, 1.5); ax.set_ylim(-0.6, 1.5)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel(xl, color=INK2); ax.set_ylabel(yl, color=INK2)
    style(ax)
note(fig, "출처: 논문 Table 1(5개 시드 평균, 무작위 정책 0 · iLQG 계획기 1로 정규화). 점 하나가 과제 하나. 점선은 같은 점수.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "table1_paired.png")

# 2) toy 최종 return ───────────────────────────────────────────────────────────
names = list(TOY["summary"])
fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=200)
for i, n in enumerate(names):
    vals = [v for v in TOY["summary"][n]["final_per_seed"] if v is not None]
    c = BLUE if n == "paper" else MUTED
    ax.scatter(vals, [i] * len(vals), s=30, color=c, edgecolor="white", linewidth=1, zorder=3)
    m = float(np.mean(vals))
    ax.plot([m, m], [i - 0.28, i + 0.28], color=INK if n == "paper" else INK2, lw=2, solid_capstyle="round")
    ax.text(-90, i, f"{m:,.0f}", va="center", fontsize=7.5, color=INK2)
zt = TOY["zero_torque_policy_return"]
ax.axvline(zt, color=ORANGE, lw=1, ls=":")
ax.text(zt + 10, len(names) - 0.4, "토크 0 정책", fontsize=7.5, color=ORANGE)
ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
ax.invert_yaxis()
ax.set_xlim(-1500, 0)
ax.set_xlabel("마지막 평가 return (40k step, 고정 초기 상태 10개 평균, 높을수록 좋음)", color=INK2)
style(ax)
note(fig, "numpy DDPG(64-64), 진자 swing-up, 설정마다 5개 시드. 점 = 시드, 막대 = 평균. 논문 설정 대비 한 항목씩만 바꿈. analysis/ddpg_toy.py")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "toy_final.png")

# 3) toy 학습 곡선 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
for n, c in (("paper", BLUE), ("no_target", ORANGE), ("tau_0.005", VIOLET), ("ou_dt_0.01", AQUA)):
    runs = [r for r in TOY["runs"] if r["config"] == n]
    steps = [p[0] for p in runs[0]["curve"]]
    arr = np.array([[p[1] for p in r["curve"]] for r in runs])
    ax.fill_between(steps, arr.min(0), arr.max(0), color=c, alpha=0.12, lw=0)
    ax.plot(steps, arr.mean(0), color=c, lw=2)
    ax.text(steps[-1] + 600, arr.mean(0)[-1], LABELS[n], fontsize=7.5, color=INK2, va="center")
ax.axhline(zt, color=INK2, lw=0.8, ls=":")
ax.set_xlim(0, 52_000)
ax.set_xlabel("환경 step", color=INK2)
ax.set_ylabel("평가 return", color=INK2)
style(ax)
note(fig, "선 = 5개 시드 평균, 띠 = 최소–최대. 점선 = 토크 0 정책. analysis/ddpg_toy.py")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "toy_curves.png")

# 4) OU 잡음 표본 경로 ──────────────────────────────────────────────────────────
rng = np.random.default_rng(3)
T = 200
fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
for label, dt, c in (("OU θ 0.15, σ 0.2, dt = 1 (rllab식)", 1.0, BLUE), ("OU θ 0.15, σ 0.2, dt = 0.01 (baselines·SB3 기본)", 0.01, ORANGE)):
    x, xs = 0.0, []
    for _ in range(T):
        x = x - 0.15 * x * dt + 0.2 * math.sqrt(dt) * rng.standard_normal()
        xs.append(x)
    ax.plot(range(T), xs, color=c, lw=1.6, label=label)
ax.plot(range(T), 0.1 * rng.standard_normal(T), color=MUTED, lw=1, label="가우시안 σ = 0.1 (Spinning Up·TD3 저장소)")
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlabel("step (진자 한 에피소드)", color=INK2)
ax.set_ylabel("행동 잡음 (정규화 단위)", color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
style(ax)
note(fig, "같은 θ, σ라도 dt에 따라 상관 시간이 6 step과 666 step으로 100배 다르다(table_check.py C9). 논문에는 dt가 없다.")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "ou_noise.png")

# toy 통계: 논문 설정 대비 Mann–Whitney U (양측), 5 vs 5 시드 ─────────────────────
base = TOY["summary"]["paper"]["final_per_seed"]
st = {}
for n in names:
    if n == "paper":
        continue
    other = TOY["summary"][n]["final_per_seed"]
    u = stats.mannwhitneyu(base, other, alternative="two-sided")
    st[n] = {"mean_diff_vs_paper": round(float(np.mean(other) - np.mean(base)), 1), "mannwhitney_p": round(float(u.pvalue), 4),
             "seeds_solved(return>-250)": sum(v > -250 for v in other)}
st["paper_seeds_solved(return>-250)"] = sum(v > -250 for v in base)
st["q_bias_paper"] = {"rel_bias_per_seed": TOY["summary"]["paper"]["q_rel_bias"],
                      "all_positive": all(b > 0 for b in TOY["summary"]["paper"]["q_rel_bias"]),
                      "wilcoxon_p_vs_0": round(float(stats.wilcoxon(TOY["summary"]["paper"]["q_rel_bias"]).pvalue), 4)}
pooled = [b for n in ("paper", "no_l2", "gauss_0.1") for b in TOY["summary"][n]["q_rel_bias"]]
st["q_bias_solved_configs_pooled"] = {"n": len(pooled), "mean": round(float(np.mean(pooled)), 3),
                                      "positive": sum(b > 0 for b in pooled)}
(ROOT / "analysis" / "toy_stats_results.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
print(json.dumps(st, indent=1, ensure_ascii=False))
print("wrote figures to", OUT)
