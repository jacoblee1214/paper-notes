"""우리가 그린 그래프 네 장. 논문 그림은 쓰지 않는다(라이선스 CC BY-NC-ND, 04 §5).
입력: analysis/table_check_results.json, analysis/toy_symmetry_results.json, analysis/toy_tolerance_results.json
출력: report/figures/own_*.png
실행: python3 analysis/plot_figures.py  (matplotlib, numpy; 한글 폰트는 Apple SD Gothic Neo → Noto CJK KR 순서)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREY = "#1f5fae", "#d9822b", "#7a7a75"


def fig_real_world():
    r = json.loads((AN / "table_check_results.json").read_text())["real_world"]
    names = list(r["per_object"])
    succ = [r["per_object"][n]["successes"] for n in names]
    lo = [r["per_object"][n]["wilson95"][0] for n in names]
    hi = [r["per_object"][n]["wilson95"][1] for n in names]
    labels = names + ["전체(ours)", "포즈 기반 최강(27/100)"]
    rates = [s / 20 for s in succ] + [r["ours"]["rate"], 0.27]
    lows = lo + [r["ours"]["wilson95"][0], r["pose_baseline_27_of_100_wilson95"][0]]
    highs = hi + [r["ours"]["wilson95"][1], r["pose_baseline_27_of_100_wilson95"][1]]
    err = np.array([np.array(rates) - np.array(lows), np.array(highs) - np.array(rates)])
    colors = [BLUE] * len(names) + [BLUE, GREY]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = np.arange(len(labels))
    ax.bar(x, rates, color=colors, width=0.6)
    ax.errorbar(x, rates, yerr=err, fmt="none", ecolor="#333", capsize=4, lw=1)
    ax.axhline(0.8, ls="--", color=ORANGE, lw=1)
    ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("성공률 (Wilson 95%)")
    ax.set_title("실로봇 성공률: 객체별 20회, 전체 100회 (논문 Table 5, 본문)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_world_ci.png", dpi=200)
    plt.close(fig)


def fig_symmetry():
    r = json.loads((AN / "toy_symmetry_results.json").read_text())
    c, b = r["cube"], r["bottle_of_revolution"]
    labels = ["정육면체\n기하 성공", "정육면체\n이미지상 성공", "정육면체\n이미지 성공·기하 실패", "회전체 병\n축 정렬 후 기하 실패"]
    vals = [c["P_geometric_success_random_goal"], c["P_image_equivalent_success_random_goal"],
            c["P_image_success_but_geometric_fail"], b["P_geometric_fail_given_axis_aligned"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(range(4), vals, color=[GREY, BLUE, ORANGE, ORANGE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v * 100:.2f}%" if v < 0.05 else f"{v * 100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(range(4), labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("비율 (Haar 무작위 자세, τ = 0.4 rad)")
    ax.set_title("대칭 물체: 같은 이미지를 내는 자세 중 목표에 가까운 비율 (토이, 설계 가정)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_symmetry.png", dpi=200)
    plt.close(fig)


def fig_tolerance():
    r = json.loads((AN / "toy_tolerance_results.json").read_text())
    taus = np.linspace(0.05, 0.6, 111)
    chance = (taus - np.sin(taus)) / math.pi
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    ax = axes[0]
    ax.plot(taus, chance * 100, color=GREY, lw=2)
    ax.axvline(0.4, ls="--", color=ORANGE, lw=1)
    ax.set_xlabel("허용오차 τ (rad)")
    ax.set_ylabel("무작위 목표의 성공 확률 (%)")
    ax.set_title("기준선: 허용오차에 따른 무작위 성공률", fontsize=10)
    ax = axes[1]
    for s_key, col in zip(r["tau_sweep_sigma_h_ref"]["rows"], (BLUE, ORANGE, GREY)):
        rows = r["tau_sweep_sigma_h_ref"]["rows"][s_key]
        xs = [float(t) for t in rows]
        ys = [rows[t]["PFR"] * 100 if rows[t]["PFR"] is not None else np.nan for t in rows]
        ax.plot(xs, ys, marker="o", color=col, label=s_key)
    ax.axhline(8.8, ls=":", color="#333", lw=1)
    ax.axhline(12.0, ls=":", color="#b00", lw=1)
    ax.set_xlabel("허용오차 τ (rad)")
    ax.set_ylabel("예측 실패율 (%)")
    ax.set_title("모형: 같은 머리 오차에서 τ별 예측 실패율", fontsize=10)
    ax.legend(title="실제 오차 척도 s", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_tolerance.png", dpi=200)
    plt.close(fig)


def fig_sim_cs():
    # 논문 Table 1 (CS 평균 ± 표준편차, 3 시드) 과 Table 3 대안 정렬
    rows = [
        ("고유감각만", 1.05, 0.00),
        ("포즈 목표\n(JP+P+C)", 11.23, 9.65),
        ("이미지+고유감각\n포즈 목표", 69.20, 5.47),
        ("이미지+고유감각\n이미지 목표(ours)", 77.30, 3.25),
        ("InfoNCE 정렬", 63.01, 15.51),
        ("EMA 목표 인코더", 67.95, 6.80),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    x = np.arange(len(rows))
    means = [m for _, m, _ in rows]
    sds = [s for _, _, s in rows]
    colors = [GREY, GREY, BLUE, ORANGE, GREY, GREY]
    ax.bar(x, means, yerr=sds, capsize=4, color=colors, width=0.6, ecolor="#333")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
    ax.set_ylabel("연속 성공 수 CS (↑)")
    ax.set_title("시뮬레이션 CS (논문 Table 1, 3; 3 시드, 막대 = 평균, 세로선 = 표준편차 표기값)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_cs.png", dpi=200)
    plt.close(fig)


def main():
    fig_real_world()
    fig_symmetry()
    fig_tolerance()
    fig_sim_cs()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))


if __name__ == "__main__":
    main()
