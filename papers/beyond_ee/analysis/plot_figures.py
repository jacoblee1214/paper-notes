"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_ci.png   : 표 1의 과제별·합산 성공률과 Wilson 95% 구간 (KALM, WACL xArm, WACL FR3)
own_mcnemar.png     : 구속 설정의 과제별 쌍 분해(보고된 p에서 역산)와 Holm 보정 전후 p
own_ablation_ci.png : 표 2의 절제·대조 7행, 변형 대 전체 모델 성공 수와 Wilson 구간
own_triangulation.png: 토이, 두 카메라 사이 각도에 따른 삼각측량 오차(가정값)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; recompute.py와 toy_triangulation.py를 먼저 실행)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c4651b"
TASKS = ["OpenCoffee", "CloseCabinet", "CloseFaucet", "PressButton", "합산(40)"]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def plot_table1():
    T1 = {("unc", "KALM"): [4, 7, 6, 6], ("unc", "WACL"): [6, 7, 8, 9], ("unc", "FR3"): [5, 8, 8, 9],
          ("con", "KALM"): [2, 5, 3, 3], ("con", "WACL"): [7, 8, 7, 9], ("con", "FR3"): [4, 7, 8, 8]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    cols = {"KALM": (GREY, "KALM (xArm)"), "WACL": (ACC, "WACL (xArm)"), "FR3": (ORG, "WACL (FR3)")}
    for ax, s, title in zip(axes, ("unc", "con"), ("구속 없음", "구속 있음")):
        for j, m in enumerate(("KALM", "WACL", "FR3")):
            ks = T1[(s, m)] + [sum(T1[(s, m)])]
            ns = [10] * 4 + [40]
            xs = np.arange(5) + (j - 1) * 0.22
            p = [k / n for k, n in zip(ks, ns)]
            lo = [wilson(k, n)[0] for k, n in zip(ks, ns)]
            hi = [wilson(k, n)[1] for k, n in zip(ks, ns)]
            ax.errorbar(xs, p, yerr=[np.array(p) - lo, np.array(hi) - p], fmt="o", color=cols[m][0], ms=5,
                        capsize=2.5, lw=1.2, label=cols[m][1])
        ax.axvline(3.5, color="#dcdad3", lw=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(TASKS, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#eee")
    axes[0].set_ylabel("성공률 (Wilson 95%)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].text(4, 0.02, "KALM 대 WACL: 구간 분리\n(쌍체 McNemar p=4.0e-5)", fontsize=7.5, ha="center", color=INK)
    fig.suptitle("표 1 재계산: 과제당 10회, 합산 40회", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=170)
    plt.close(fig)


def plot_mcnemar():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V3_mcnemar"]
    rows = rec["constrained_per_task_inverse"]
    names = [r["task"] for r in rows]
    b = [r["solutions"][0]["b"] for r in rows]
    c = [r["solutions"][0]["c"] for r in rows]
    both = [r["wacl"] - bb for r, bb in zip(rows, b)]
    nei = [10 - x - y - z for x, y, z in zip(b, c, both)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    x = np.arange(4)
    bottom = np.zeros(4)
    for vals, col, lab in ((both, "#8fb0d9", "둘 다 성공"), (b, ACC, "WACL만 성공 (b)"), (c, ORG, "KALM만 성공 (c)"),
                           (nei, "#d9d7d0", "둘 다 실패")):
        ax.bar(x, vals, bottom=bottom, color=col, label=lab, width=0.6)
        for xi, v, bt in zip(x, vals, bottom):
            if v:
                ax.text(xi, bt + v / 2, str(v), ha="center", va="center", fontsize=8, color="white" if col in (ACC, ORG) else INK)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("배치 수 (과제당 10)")
    ax.set_title("구속 설정의 쌍 분해 (보고된 p에서 역산)", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax = axes[1]
    raw, hol = rec["constrained_per_task_p_raw"], rec["constrained_per_task_p_holm"]
    ax.bar(x - 0.18, raw, 0.34, color=GREY, label="비보정")
    ax.bar(x + 0.18, hol, 0.34, color=ACC, label="Holm (4개)")
    ax.axhline(0.05, color=ORG, ls="--", lw=1)
    ax.text(3.45, 0.06, "0.05", color=ORG, fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["OC", "CC", "CF", "PB"], fontsize=8)
    ax.set_ylabel("정확 McNemar p")
    ax.set_title("과제별 p: Holm 뒤 모두 0.05 초과", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_mcnemar.png", dpi=170)
    plt.close(fig)


def plot_ablation():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V9_table2_ablation"]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    labels = {"second_constraint": "두 번째 구속(CloseFaucet)", "no_sliding_anchor": "슬라이딩 앵커 제거(OpenCoffee)",
              "link5_only": "링크 5 고정(PressButton)", "link6_only": "링크 6 고정(PressButton)",
              "geometric_arc": "기하 원호(OpenCoffee)", "geometric_line": "기하 직선(PressButton)",
              "registered_replay": "ICP 재생(OpenCoffee)"}
    y = np.arange(len(rec))[::-1]
    for yi, r in zip(y, rec):
        for off, k, col, lab in ((0.12, r["full_succ"], ACC, "전체 WACL"), (-0.12, r["variant_succ"], ORG, "변형")):
            lo, hi = wilson(k, 10)
            ax.errorbar(k / 10, yi + off, xerr=[[k / 10 - lo], [hi - k / 10]], fmt="o", color=col, capsize=2.5, ms=5,
                        label=lab if yi == y[0] else None)
        ax.text(1.03, yi, f"Fisher p={r['fisher_p']:.2f}\n최선 McNemar p={r['best_case_mcnemar_p']:.3f}", fontsize=7,
                va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[r["variant"]] for r in rec], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("성공률 (10회, Wilson 95%)")
    ax.grid(axis="x", color="#eee")
    ax.legend(fontsize=8, loc="lower left")
    fig.subplots_adjust(right=0.78, left=0.28, bottom=0.14)
    fig.savefig(FIG / "own_ablation_ci.png", dpi=170)
    plt.close(fig)


def plot_triangulation():
    rec = json.loads((ANA / "toy_triangulation_results.json").read_text(encoding="utf-8"))
    sw = rec["two_camera_angle_sweep"]
    ang = [s["angle_deg"] for s in sw]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(ang, [s["rms_mm"] for s in sw], "o-", color=ACC, label="융합 3D RMS")
    ax.plot(ang, [s["worst_axis_mm"] for s in sw], "s--", color=ORG, label="가장 나쁜 축 1-sigma")
    ax.axhline(30, color=GREY, ls=":", label="단안 깊이 오차(가정 30 mm)")
    ax.set_xlabel("두 장면 카메라 사이 각도 (도)")
    ax.set_ylabel("손끝 위치 오차 (mm, 가정값)")
    ax.set_title("토이: 광선 삼각측량 오차(2대, 3 px 방위 오차, 가중치 1)", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(color="#eee")
    fig.tight_layout()
    fig.savefig(FIG / "own_triangulation.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_mcnemar()
    plot_ablation()
    plot_triangulation()
    print("wrote", FIG)
