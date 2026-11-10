"""리포트 그래프 (블로그 원본 그림이 아니라 재계산 값으로 직접 그린 그림).

입력: analysis/recompute_results.json, analysis/refinement_toy_results.json,
      repo/site/assets/analysis/budget-tracking.json, repo/site/experience-story.json
출력: report/figures/*.png
팔레트: dataviz 참조 팔레트의 categorical 1–3번 슬롯 (all-pairs 검증 통과 세트)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                         "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#a9a8a2"})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (light)
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#a9a8a2"
AN = ROOT / "analysis"
REC = json.loads((AN / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((AN / "refinement_toy_results.json").read_text(encoding="utf-8"))
SITE = ROOT / "repo" / "site"


def fig_outcomes():
    order = ["hoi_spider", "zero_shot", "codex", "orchestrator", "communicating", "codex120"]
    names = ["HOI-Retarget\n+SPIDER", "Astra\none-shot", "Codex\n30분", "독립 스웜\n30분", "협업 스웜\n30분", "Codex\n120분"]
    agg = REC["aggregate_recomputed"]
    passp = [agg[m]["all"]["pass_pct"] for m in order]
    add = [agg[m]["all"]["mean_add_cm"] for m in order]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(order))
    for a, vals, title, fmt in [(ax[0], passp, "의미 성공률 (40과제, %)", "{:.1f}"),
                                (ax[1], add, "평균 객체 추적 오차 (cm, 낮을수록 좋음)", "{:.1f}")]:
        bars = a.bar(x, vals, color=C1, width=0.62)
        bars[-1].set_color(C2)
        for xi, v in zip(x, vals):
            a.text(xi, v + max(vals) * 0.02, fmt.format(v), ha="center", fontsize=7.5, color=INK2)
        a.set_xticks(x, names, fontsize=6.8)
        a.set_title(title, fontsize=8.5, color=INK2, loc="left")
        a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    ax[0].set_ylim(0, 90); ax[1].set_ylim(0, 80)
    fig.text(0.01, 0.01, "값은 analysis/recompute_results.json에서 원자료로 재계산. 의미 성공률은 원래 LLM 판정 기준. "
             "Codex 120분 추적 값은 미완료 1과제를 뺀 39과제 평균.", fontsize=6.2, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig_outcomes.png", dpi=200)
    plt.close(fig)


def fig_budget():
    bud = json.loads((SITE / "assets" / "analysis" / "budget-tracking.json").read_text(encoding="utf-8"))["curves"]
    fig, a = plt.subplots(figsize=(6.6, 3.1))
    spec = [("codex120", "Codex 120분 (최선값)", C1, "-"),
            ("codex", "Codex 30분 (최선값)", C2, "-"),
            ("communicating", "협업 스웜 30분 (최선값)", C3, "-"),
            ("orchestrator", "독립 스웜 30분 (최선값)", MUTED, "--")]
    for key, lab, col, ls in spec:
        c = bud[key]
        m = np.asarray(c["minutes"]); v = np.asarray(c["mean_add_cm"])
        a.step(m, v, where="post", color=col, lw=1.6, ls=ls)
        dy = {"orchestrator": 1.6, "communicating": -1.4}.get(key, 0.0)
        a.text(m.max() + 0.6, v[-1] + dy, lab, fontsize=7, color=INK2, va="center")
    a.set_xlim(0, 150); a.set_ylim(0, 56)
    a.set_xlabel("활성 실행 시간 (분)", fontsize=8, color=INK2)
    a.set_ylabel("40과제 평균 최선 ADD (cm)", fontsize=8, color=INK2)
    a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    a.set_title("최선값 곡선 (실행 간 비교이며 같은 실행의 접두부가 아님)", fontsize=8.5, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_budget.png", dpi=200)
    plt.close(fig)


def fig_experience():
    exp = json.loads((SITE / "experience-story.json").read_text(encoding="utf-8"))["tasks"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for i, x in enumerate(exp):
        col = C1 if x["group"] == "related" else C2
        wo = x["metrics"]["without_experience"]; wi = x["metrics"]["with_experience"]
        for a, k1, k2 in [(ax[0], "mean_add_m", None), (ax[1], "orientation_mean_deg", None)]:
            v0 = wo[k1] * (100 if k1 == "mean_add_m" else 1); v1 = wi[k1] * (100 if k1 == "mean_add_m" else 1)
            a.plot([v0, v1], [i, i], color=MUTED, lw=0.8, zorder=1)
            a.scatter([v0], [i], color=INK2, s=14, zorder=2)
            a.scatter([v1], [i], color=col, s=18, zorder=3)
    short = {"carry_side_table": "사이드 테이블 운반", "lifting_dining_chair": "식당 의자 들기",
             "walk_with_low_chair": "낮은 의자 들고 걷기", "lift_shelf": "선반 세우기",
             "pulling_and_pushing": "의자 끌기·밀기", "tilting_trash_can": "쓰레기통 기울이기",
             "drag_wash_tub": "세탁 통 끌기", "carrying_mixing_bowl": "믹싱볼 운반",
             "subject_holding_plastic": "그릇 들고 걷기", "turn_on_floor_lamp": "플로어램프 켜기"}
    names = [next(v for k, v in short.items() if x["clip"].startswith(k)) for x in exp]
    ax[0].set_yticks(range(len(exp)), names, fontsize=7)
    ax[0].set_xscale("log")
    ax[0].set_title("평균 ADD (cm, 로그축)", fontsize=8.5, color=INK2, loc="left")
    ax[1].set_title("평균 방향 오차 (°)", fontsize=8.5, color=INK2, loc="left")
    ax[1].set_xscale("log")
    for a in ax:
        a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
    ax[0].scatter([], [], color=INK2, s=14, label="경험 없음")
    ax[0].scatter([], [], color=C1, s=18, label="경험 있음 (유사 과제)")
    ax[0].scatter([], [], color=C2, s=18, label="경험 있음 (무작위 과제)")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.075))
    fig.text(0.01, 0.005, "과제당 실행 1회씩 (반복 없음). 회색 점=경험 없음, 색 점=경험 있음.", fontsize=6.2, color=INK2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "fig_experience.png", dpi=200)
    plt.close(fig)


def fig_toys():
    A = TOY["toy_a_feedback_proposer_vs_fixed"]
    budgets = ["256", "512", "1024", "2048"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for a, (scen, title) in zip(ax, [("multimodal_wrong_init", "잘못된 초기 기저 (다봉)"), ("unimodal", "단봉")]):
        for sg, ls in [("sigma_0.0", "-"), ("sigma_2.0", "--")]:
            for arm, col, lab in [("fixed", C2, "고정 가중치"), ("proposer", C1, "피드백 제안기")]:
                med = [A[scen][sg][arm][b]["median_true_cm"] for b in budgets]
                a.plot([int(b) for b in budgets], med, color=col, ls=ls, lw=1.5,
                       label=f"{lab} σ={sg.split('_')[1]}")
        a.set_title(title, fontsize=8.5, color=INK2, loc="left")
        a.set_xlabel("시뮬레이터 호출 수", fontsize=8, color=INK2)
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    ax[0].set_ylabel("진짜 목적값 중앙값 (낮을수록 좋음)", fontsize=7.5, color=INK2)
    ax[0].legend(frameon=False, fontsize=6.2, loc="upper right")
    fig.text(0.01, 0.01, "Toy A: 규칙 기반 제안기(LLM 아님), 100 시드, 시드마다 같은 초기값.", fontsize=6.2, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig_toy_a.png", dpi=200)
    plt.close(fig)

    B = TOY["toy_b_selection_optimism"]
    fig, a = plt.subplots(figsize=(5.2, 2.8))
    for (sg, rows), col in zip(B.items(), [C3, C1, C2]):
        Ks = [int(k) for k in rows]
        bias = [-rows[k]["optimism_bias_cm"] for k in rows]
        a.plot(Ks, bias, marker="o", ms=3.5, color=col, lw=1.5, label=f"σ={sg.split('_')[1]} cm")
    a.set_xscale("log"); a.set_xlabel("후보 수 K (보고 최솟값 기준)", fontsize=8, color=INK2)
    a.set_ylabel("낙관 편향 (cm)", fontsize=8, color=INK2)
    a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=7)
    a.set_title("Toy B: 같은 지표로 고르고 보고할 때의 편향", fontsize=8.5, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_toy_b.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_outcomes(); fig_budget(); fig_experience(); fig_toys()
    print("figures written to", FIG)
