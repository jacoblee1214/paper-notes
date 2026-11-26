"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림 조각은 report/crop_figures.py 가 만든다.

  own_corr_noise.png    BT 대 수동 평가 TP 산점도(표 1과 표 7 값)와 수동 평균의 표본 잡음을 넣은 r 분포 (V2, V3, V5)
  own_bt_gaps.png       인접 정책 BT 격차: 보고 값, 평균 TP 로그 비 환산, 토이 1 평균과 ±1 SE (V6, 토이 1)
  own_grader_audit.png  AutoGrader 감사: 정책별 자동 점수와 사람 교정 점수, 과대 평가 배율, 복원한 혼동행렬 (V10)
  own_prompt_power.png  프롬프트당 8회 롤아웃으로 정확한 순서를 맞힐 확률과 10개 중 9개 이상 확률 (토이 2)

입력: analysis/recompute_results.json, toy_bt_matching_results.json, toy_prompt_order_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
T1 = json.loads((ROOT / "analysis" / "toy_bt_matching_results.json").read_text(encoding="utf-8"))
T2 = json.loads((ROOT / "analysis" / "toy_prompt_order_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
POL = ["DreamZero", "π0.5", "GR00T N1.7", "MolmoAct2", "GR00T N1.6"]
BT = [1580, 1539, 1500, 1456, 1453]


def fig_corr():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    m1 = [54.4, 45.0, 43.9, 27.0, 34.3]
    m7 = RC["V3_table7_means"]["manual_TP_from_table7_in_POL_order"]
    m7 = [m7[k] for k in ["DreamZero", "pi0.5", "GR00T N1.7", "MolmoAct2", "GR00T N1.6"]]
    a = ax[0]
    a.scatter(m1, BT, color=ACC, s=40, label="표 1의 수동 평균 TP", zorder=3)
    a.scatter(m7, BT, facecolor="white", edgecolor=ACC2, s=40, linewidth=1.5, label="표 7 열 평균", zorder=3)
    for i in range(5):
        a.annotate(POL[i], (m1[i], BT[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
    a.annotate("", xy=(m7[1], BT[1]), xytext=(m1[1], BT[1]), arrowprops=dict(arrowstyle="->", color=ACC2, lw=1))
    a.set_xlabel("수동 평가 평균 TP (%)")
    a.set_ylabel("ScaleEvals BT (Elo)")
    r1 = RC["V2_correlation_n5"]["table1_values"]["pearson_BT_vs_manualTP"]
    r7 = RC["V3_table7_means"]["pearson_BT_vs_table7_means"]
    a.set_title(f"점 5개: r = {r1['r']:.2f} (표 1), {r7['r']:.2f} (표 7)\nFisher 95% 구간 {r1['fisher95'][0]:.2f}–{r1['fisher95'][1]:.2f}", fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=7.5, loc="lower right")
    h = RC["V5_manual_noise"]["bootstrap"]["pearson_r_hist"]
    edges, counts = np.array(h["edges"]), np.array(h["counts"])
    b = ax[1]
    b.bar(edges[:-1] + 0.025, counts / counts.sum(), width=0.045, color=ACC, alpha=0.8)
    b.axvline(0.94, color=WARN, lw=1.2)
    b.text(0.93, b.get_ylim()[1] * 0.97, "보고 r = 0.94", color=WARN, ha="right", fontsize=8, va="top")
    st = RC["V5_manual_noise"]["bootstrap"]["pearson_r_BT_vs_bootstrapped_manual"]
    b.set_xlabel("과제 20개를 재표집한 수동 평균과의 Pearson r")
    b.set_ylabel("비율")
    b.set_title(f"중앙값 {st['median']:.2f}, 5–95% = {st['p5']:.2f}–{st['p95']:.2f}\nr ≥ 0.94인 비율 {100 * st['frac_ge_0.94']:.0f}% (과제 독립 가정)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_corr_noise.png", dpi=200)
    plt.close(fig)


def fig_gaps():
    v6 = RC["V6_BT_vs_log_TP"]
    labels = ["DZ vs π0.5", "π0.5 vs N1.7", "N1.7 vs MA2", "MA2 vs N1.6"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    rep = v6["gap_reported_elo"]
    pred = v6["gap_if_BT_equals_ELO_ln_TP"]
    toy = T1["runs"][1]["mean_adjacent_gap"]
    se = [math.sqrt(16 ** 2 + 16 ** 2) if i in (0, 3) else math.sqrt(16 ** 2 + 17 ** 2) for i in range(4)]
    w = 0.26
    ax.bar(x - w, rep, w, color=ACC, label="보고된 BT 격차 (표 1)")
    ax.errorbar(x - w, rep, yerr=se, fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.bar(x, pred, w, color=ACC2, label="400/ln10 × ln(평균 TP 비) 환산")
    ax.bar(x + w, toy, w, color="#9aa4b2", label="토이 1 평균 (무작위 매칭, κ = 1.5)")
    for i in range(4):
        ax.text(x[i] - w, rep[i] + se[i] + 1, f"{rep[i]:.0f}", ha="center", fontsize=8)
        ax.text(x[i], pred[i] + 1, f"{pred[i]:.0f}", ha="center", fontsize=8)
        ax.text(x[i] + w, toy[i] + 1, f"{toy[i]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("인접 정책 BT 격차 (Elo)")
    ax.set_title("BT 격차는 평균 TP 로그 비와 가깝고, MolmoAct2–N1.6 격차 3은 표준오차 안", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.text(0.01, 0.02, "오차막대: 표의 SE 두 개를 독립으로 합침(앵커 공분산 무시)", transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
    fig.tight_layout()
    fig.savefig(FIG / "own_bt_gaps.png", dpi=200)
    plt.close(fig)


def fig_audit():
    v = RC["V10_autograder_audit"]["fig16"]
    names = ["DreamZero", "π0.5", "GR00T N1.7", "MolmoAct2", "GR00T N1.6"]
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    x = np.arange(5)
    a = ax[0]
    a.bar(x - 0.2, v["auto"], 0.4, color=ACC2, label="자동 채점")
    a.bar(x + 0.2, v["human"], 0.4, color=ACC, label="사람 교정")
    a.set_xticks(x)
    a.set_xticklabels(["DZ", "π0.5", "N1.7", "MA2", "N1.6"])
    a.set_ylabel("정책 TP (%)")
    a.set_title("감사 525 롤아웃의 정책별 TP (그림 16 값)", fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=8)
    b = ax[1]
    b.bar(x, v["ratio_auto_over_human"], color=WARN, alpha=0.85)
    for i, r in enumerate(v["ratio_auto_over_human"]):
        b.text(i, r + 0.05, f"{r:.2f}", ha="center", fontsize=8)
    b.axhline(1, color=MUTED, lw=0.8)
    b.set_xticks(x)
    b.set_xticklabels(["DZ", "π0.5", "N1.7", "MA2", "N1.6"])
    b.set_ylabel("자동 TP / 사람 교정 TP")
    b.set_title("낮은 정책일수록 배율이 큼", fontsize=9, color=INK)
    c = ax[2]
    cm = RC["V10_autograder_audit"]["implied_confusion_matrix(assume 'pass' is positive, N=1454)"]
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    c.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.3)
    for i in range(2):
        for j in range(2):
            c.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=10, color=INK)
    c.set_xticks([0, 1])
    c.set_xticklabels(["자동 No", "자동 Yes"])
    c.set_yticks([0, 1])
    c.set_yticklabels(["사람 No", "사람 Yes"])
    c.set_title(f"복원한 혼동행렬(가정)\n정확도 90.7%, 항상 No 기준 {100 * cm['always_fail_baseline_accuracy']:.1f}%, κ = {cm['cohen_kappa']:.2f}", fontsize=9, color=INK)
    c.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_grader_audit.png", dpi=200)
    plt.close(fig)


def fig_power():
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for kappa, col in ((0.7, WARN), (1.5, ACC), (3.0, ACC2)):
        rows = [c for c in T2["combined_10_prompts"] if c["kappa"] == kappa]
        ax.plot([r["g"] for r in rows], [100 * r["P(>=9 of 10 exact)"] for r in rows], marker="o", color=col,
                label=f"롤아웃 TP 표준편차 약 {100 * math.sqrt(0.1875 / (kappa + 1)):.0f}%p (κ = {kappa})")
    ax.axhline(50, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("g: 프롬프트 계열 격차 / 태그 수준 격차 (표 11b)")
    ax.set_ylabel("10개 중 9개 이상 정확한 순서일 확률 (%)")
    ax.set_title("g = 1(태그 격차 그대로)에서는 거의 안 나오고, 2–3배 격차를 가정해야 흔해짐", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "own_prompt_power.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_corr()
    fig_gaps()
    fig_audit()
    fig_power()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
