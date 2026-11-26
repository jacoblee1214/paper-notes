"""우리가 만든 그래프를 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_fig4_ci.png        그림 4a의 성공률 라벨 + Wilson 95% 구간(칸당 20회 가정), 방법 셋 x 과제 넷
  own_progress_vs_sr.png 그림 4a(성공률)와 4b(진행도)의 칸별 관계. 진행도 < 성공률인 두 칸 표시
  own_ttc_counts.png     표 II의 칸별 성공 횟수(20회)와 Wilson 구간, 같은 확률의 4후보 이상적 선택 상한
  own_min_detectable.png 20회 대 20회에서 p < 0.05가 되는 최소 차이(기준 성공 횟수별, Fisher 정확 검정)
  own_toy_rcs.png        토이 T1: RCS 선택의 성공률(무작위 / RCS / 이상적), 두 시나리오
  own_params.png         공개 코드 설정에서 센 파라미터(영상 백본, 행동 가지, 보상 가지)와 논문 표기

입력: analysis/recompute_results.json, toy_rcs_selection_results.json, param_count_results.json (먼저 각 스크립트를 실행할 것)
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ANA = ROOT / "analysis"
RC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
TY = json.loads((ANA / "toy_rcs_selection_results.json").read_text(encoding="utf-8"))
PC = json.loads((ANA / "param_count_results.json").read_text(encoding="utf-8"))

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACC, ACC2, WARN, GREY = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a", "#b9b8b0"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


# 1) Fig.4 성공률 + Wilson
tasks = ["Toolbox", "School Bag", "Badminton", "Faucet"]
v5 = RC["V5_fig4_tests"]
fig, ax = plt.subplots(figsize=(8.2, 4.0))
cols = {"Fast-WAM": GREY, "pi0.5": ACC2, "tau0-WM": ACC}
w = 0.26
for j, m in enumerate(["Fast-WAM", "pi0.5", "tau0-WM"]):
    xs = np.arange(4) + (j - 1) * w
    k = np.array(v5["counts"][m]); p = k / 20
    ci = np.array(v5["wilson"][m])
    ax.bar(xs, p, w * 0.92, color=cols[m], label=m.replace("pi0.5", "π0.5").replace("tau0-WM", "τ0-WM"))
    ax.errorbar(xs, p, yerr=[p - ci[:, 0], ci[:, 1] - p], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
ax.set_xticks(range(4)); ax.set_xticklabels(tasks)
ax.set_ylabel("성공률 (그림 4a 라벨, 20회 가정)"); ax.set_ylim(0, 1.05); style(ax)
ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_title("과제별 성공률과 Wilson 95% 구간", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "own_fig4_ci.png", dpi=170); plt.close(fig)

# 2) 진행도 대 성공률
fig, ax = plt.subplots(figsize=(5.2, 4.6))
mk = {"Fast-WAM": "s", "pi0.5": "^", "tau0-WM": "o"}
for c in RC["V4_progress_vs_sr"]["cells"]:
    bad = not c["progress_ge_sr"]
    ax.scatter(c["sr"], c["progress"], marker=mk[c["method"]], s=60, color=WARN if bad else cols[c["method"]], zorder=3)
    if bad:
        ax.annotate(f'{c["method"].replace("pi0.5","π0.5").replace("tau0-WM","τ0-WM")}\nSchool Bag', (c["sr"], c["progress"]), textcoords="offset points", xytext=(6, -22), fontsize=8, color=WARN)
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
ax.set_xlabel("성공률 (그림 4a)"); ax.set_ylabel("진행도 (그림 4b)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); style(ax)
ax.text(0.04, 0.93, "점선 위: 진행도 ≥ 성공률", fontsize=8.5, color=MUTED)
ax.set_title("칸별 진행도 대 성공률 (붉은 점 2개는 점선 아래)", color=INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "own_progress_vs_sr.png", dpi=170); plt.close(fig)

# 3) Table II
v7 = RC["V7_table2_tests"]
names = ["w/o TTC", "w. CFG", "w. ACG", "w. RCS", "w. RCS+LAR"]
fig, ax = plt.subplots(figsize=(8.2, 4.0))
for j, (t, lab) in enumerate([(0, "Tissue→Box"), (1, "Pen→Box")]):
    xs = np.arange(5) + (j - 0.5) * 0.36
    k = np.array([v7["counts"][nm][t] for nm in names]); p = k / 20
    ci = np.array([v7["wilson"][nm][t] for nm in names])
    ax.bar(xs, p, 0.34, color=[ACC, ACC2][j], label=lab)
    ax.errorbar(xs, p, yerr=[p - ci[:, 0], ci[:, 1] - p], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
orc = v7["oracle_best_of_4_if_iid"]["per_task"]
for j, o in enumerate(orc):
    ax.hlines(o, -0.5, 4.5, colors=[ACC, ACC2][j], linestyles=":", lw=1.2)
    ax.text(4.55, o, f"4후보 이상적 선택 상한 {o:.2f}", fontsize=7.5, color=MUTED, va="center")
ax.set_xticks(range(5)); ax.set_xticklabels(names); ax.set_ylim(0, 1.05); ax.set_xlim(-0.6, 6.3)
ax.set_ylabel("성공률 (표 II, 20회)"); style(ax); ax.legend(frameon=False, loc="center right")
ax.set_title("표 II 칸별 성공률, Wilson 95% 구간, 후보 4개 이상적 선택의 상한(점선)", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_ttc_counts.png", dpi=170); plt.close(fig)

# 4) 최소 구분 차이
mdd = {int(k): v for k, v in v7["min_detectable_pp_n20"].items()}
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(list(mdd), list(mdd.values()), "o-", color=ACC)
ax.set_xlabel("기준 쪽 성공 횟수 (20회 중)"); ax.set_ylabel("p < 0.05에 필요한 최소 차이 (%p)")
ax.set_ylim(0, 45); style(ax)
ax.set_title("20회 대 20회의 최소 구분 차이 (Fisher 정확 검정)", color=INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "own_min_detectable.png", dpi=170); plt.close(fig)

# 5) 토이 RCS
fig, ax = plt.subplots(figsize=(6.6, 3.9))
labs = ["A 품질-밀도 일치", "B 품질-밀도 불일치"]
for j, (key, c) in enumerate([("random", GREY), ("rcs", ACC), ("oracle", ACC2)]):
    vals = [TY[s]["mean"][key] for s in ("A", "B")]
    ax.bar(np.arange(2) + (j - 1) * 0.26, vals, 0.24, color=c, label={"random": "무작위 선택", "rcs": "RCS 선택", "oracle": "이상적 선택"}[key])
    for x, v in zip(np.arange(2) + (j - 1) * 0.26, vals):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(2)); ax.set_xticklabels(labs); ax.set_ylim(0, 1.1); ax.set_ylabel("후보 4개 중 선택의 성공률"); style(ax)
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.set_title("토이: RCS는 품질이 밀도와 같은 방향일 때만 돕는다", color=INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "own_toy_rcs.png", dpi=170); plt.close(fig)

# 6) 파라미터
fig, ax = plt.subplots(figsize=(6.6, 3.8))
vals = [PC["video_backbone_params"] / 1e9, PC["action_branch_params"] / 1e9, PC["reward_branch_params"] / 1e9]
paper = [5.0, 0.5, 0.5]
x = np.arange(3)
ax.bar(x - 0.18, vals, 0.34, color=ACC, label="공개 코드 설정에서 센 값")
ax.bar(x + 0.18, paper, 0.34, color=GREY, label="논문 표기 (그림 2, IV-B)")
for xi, v in zip(x - 0.18, vals):
    ax.text(xi, v + 0.08, f"{v:.2f}B", ha="center", fontsize=8.5)
for xi, v in zip(x + 0.18, paper):
    ax.text(xi, v + 0.08, f"{v:.1f}B", ha="center", fontsize=8.5, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels(["영상 백본", "행동 가지 (VAM)", "보상 가지 (ACVS)"]); ax.set_ylabel("파라미터 (10억)"); style(ax)
ax.legend(frameon=False, fontsize=8)
ax.set_title("파라미터 수: 코드 설정으로 센 값과 논문 표기", color=INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "own_params.png", dpi=170); plt.close(fig)
print("wrote 6 figures to", FIG)
