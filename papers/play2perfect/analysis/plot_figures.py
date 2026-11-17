"""직접 그린 그림 (공개 가능). 논문 그림은 포함하지 않는다(라이선스: arXiv 비독점 배포).

입력: analysis/recompute_results.json, analysis/toy_goal_tolerance_results.json
출력: report/figures/own_real_ci.png, report/figures/own_tolerance.png
실행: python3 -I analysis/plot_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

KO_CANDIDATES = ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleSDGothicNeo", "NanumGothic"]
avail = {f.name for f in font_manager.fontManager.ttflist}
ko = next((k for k in KO_CANDIDATES if k in avail), None)
plt.rcParams["font.family"] = ko or "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

rc = json.load(open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8"))
tol = json.load(open(os.path.join(HERE, "toy_goal_tolerance_results.json"), encoding="utf-8"))

# 그림 1: 실로봇 셀별 성공률과 Wilson 95% 구간
cells = [
    ("삽입 10 mm", "Tight 10mm"), ("삽입 2 mm", "Tight 2mm"), ("삽입 0.5 mm", "Tight 0.5mm"),
    ("빔 Step 1", "Beam Step1 (end-to-end)"), ("빔 Step 2 | Step 1 성공", "Beam Step2 | Step1 ok"),
    ("빔 전체", "Beam overall (12/20)"), ("스크류 삽입", "Screw Insert"),
    ("스크류 나사 | 삽입 성공", "Screw | Insert ok"), ("스크류 전체 (유도)", "Screw overall (5/10, 유도)"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
for i, (label, key) in enumerate(cells):
    d = rc["real_wilson_v4"][key]
    lo, hi = d["ci95"]
    y = len(cells) - 1 - i
    ax.plot([lo * 100, hi * 100], [y, y], color="#1f5fae", lw=2.2)
    ax.plot([d["p"] * 100], [y], "o", color="#1f5fae", ms=6)
    ax.text(101, y, f'{d["k"]}/{d["n"]}', va="center", fontsize=8, color="#55544f")
ax.set_yticks(range(len(cells)))
ax.set_yticklabels([c[0] for c in cells][::-1], fontsize=8.5)
ax.axvline(50, color="#bbb", lw=0.8, ls="--")
ax.set_xlim(0, 112)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("성공률 (%), 점: 점추정, 선: Wilson 95% 구간", fontsize=9)
ax.set_title("표 1 실로봇 성공률 (v4 시행 수 기준, 재계산)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_real_ci.png"))
plt.close(fig)

# 그림 2: ε에 따른 허용 회전각 (toy)
xs = [1.0, 1.5, 3.0, 5.0, 10.0]
keys = ["paper_eps_1cm", "code_effective_1cm_x1.5", "code_curriculum_start_2cm_x1.5", "paper_eps_5cm", "paper_eps_10cm"]
ys_worst = [tol["cases"][k]["theta_worst_deg"] for k in keys]
ys_med = [tol["cases"][k]["theta_median_deg"] for k in keys]
ys_best = [tol["cases"][k]["theta_best_deg"] for k in keys]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
ax.fill_between(xs, ys_worst, ys_best, color="#1f5fae", alpha=0.12, label="축 방향 범위 (최악 ~ 최선)")
ax.plot(xs, ys_worst, "o-", color="#1f5fae", label="최악 축 (설계 기준)")
ax.plot(xs, ys_med, "s--", color="#55544f", ms=4, lw=1, label="중앙값 축")
for x, y, k in zip(xs, ys_worst, keys):
    ax.annotate(f"{y:.1f}°", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=7.5)
ax.set_xlabel("판정 허용오차 ε (cm), 키포인트 거리 기준 (x는 실제 적용값)", fontsize=8.5)
ax.set_ylabel("허용 회전각 θ* (°)", fontsize=9)
ax.set_title("목표 허용오차가 허용하는 회전각 (기하 toy, 식 (3))", fontsize=10)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_tolerance.png"))
plt.close(fig)
print("wrote", FIG, "font:", ko)
