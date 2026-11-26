"""우리가 그리는 그래프 세 장 (report/figures/own_*.png).

  own_return_ci.png : 표 1, 2, 9, 10의 z축 평가 Return과 95% CI (논문 값)
  own_real_robot.png: 표 6의 실로봇 성공률(Wilson 95%)과 표 3의 RR 대 SR 정수 하한
  own_toy_clean_target.png: 토이 T1 결과 (analysis/toy_clean_target_results.json)
실행: python3 analysis/plot_figures.py   (matplotlib 필요. recompute_results.json이 먼저 있어야 함)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREY, RED = "#1f5fae", "#c8641e", "#8a8984", "#b3261e"


def fig_return():
    rows = [
        ("Blind RL", 304.1, 3.8, GREY), ("In-Hand Rotation", 236.1, 2.7, GREY), ("Touch Dexterity", 386.9, 3.1, GREY),
        ("RL (깊이, 촉각 없음)", 362.2, 4.7, GREY), ("WM 처음부터", 414.3, 15.7, ORANGE),
        ("LSTM (예측 목표 없음)", 621.0, 4.3, ORANGE), ("GRU (예측 목표 없음)", 497.8, 4.5, ORANGE),
        ("보조 헤드 없음", 688.1, 3.7, ORANGE), ("깨끗한 깊이 대신 잡음 깊이 목표", 708.0, 3.5, ORANGE),
        ("h(t-1) 입력 제거", 705.4, 4.3, ORANGE), ("zt 정책 입력(인코더-디코더)", 667.6, 4.7, ORANGE),
        ("촉각 전부 0 (테스트)", 724.9, 4.4, ORANGE), ("촉각 25% 드롭아웃 (테스트)", 743.4, 4.2, ORANGE),
        ("Full WSM (사전학습)", 753.3, 3.6, BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y = list(range(len(rows)))[::-1]
    for yi, (name, m, h, c) in zip(y, rows):
        ax.barh(yi, m, xerr=h, color=c, alpha=0.9, ecolor="#222", capsize=2)
        ax.text(m + h + 8, yi, f"{m:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("Return (z축, 아홉 물체, 평균 ± 95% CI; 논문 표 1, 2, 9, 10)")
    ax.set_xlim(0, 840)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("표의 CI는 평가 에피소드 변동을 담습니다. 학습 시드 수는 논문에 없습니다", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_return_ci.png", dpi=170)
    plt.close(fig)


def fig_real():
    r = json.load(open(ROOT / "analysis" / "recompute_results.json"))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axs[0]
    names = ["Touch Dexterity", "Blind RL", "RL (no tactile)", "In-Hand Rotation", "WM-Craftnet"]
    labels = ["Touch\nDexterity", "Blind RL", "RL\n(촉각 없음)", "In-Hand\nRotation", "WM-Craftnet"]
    for i, n in enumerate(names):
        d = r["V4_real_table6"][n]
        lo, hi = d["wilson95"]
        ax.bar(i, d["rate"] * 100, color=BLUE if n == "WM-Craftnet" else GREY)
        ax.errorbar(i, d["rate"] * 100, yerr=[[d["rate"] * 100 - lo * 100], [hi * 100 - d["rate"] * 100]], color="#222", capsize=3)
        ax.text(i, hi * 100 + 2, f"{d['rate'] * 200:.0f}/200", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("실로봇 성공률 % (Wilson 95%)")
    ax.set_ylim(0, 105)
    ax.set_title("표 6: 20개 물체 x 10회", fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axs[1]
    feas = r["V5_table3_feasibility_halfspin_pi"]
    pts = [f for f in feas if f["k"] > 0 and f["method"] != "Ours"]
    for f in pts:
        ok = f["feasible"]
        ax.scatter(f["min_mean_RR_if_halfspin_is_pi"], f["RR"], color=BLUE if ok else RED, s=34, zorder=3)
    mx = 3.6
    ax.plot([0, mx], [0, mx], color=GREY, lw=1, ls="--")
    ax.fill_between([0, mx], [0, mx], [0, 0], color=RED, alpha=0.07)
    for f in pts:
        if not f["feasible"]:
            ax.annotate(f"{f['method']}\n{f['object']} {f['k']}/10", (f["min_mean_RR_if_halfspin_is_pi"], f["RR"]),
                        textcoords="offset points", xytext=(8, -16), fontsize=7.5, color=RED)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, 3.6)
    ax.set_xlabel("SR에서 오는 평균 RR 하한 = k x pi / 10 (rad, 반 바퀴 = pi 가정)")
    ax.set_ylabel("보고된 평균 RR (rad / 20 s)")
    ax.set_title("표 3: 점이 대각선 아래(붉은 영역)이면 정수 시행으로 설명되지 않음", fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_robot.png", dpi=170)
    plt.close(fig)


def fig_toy():
    d = json.load(open(ROOT / "analysis" / "toy_clean_target_results.json"))
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    groups = [("고장 픽셀 + 작은 잡음\n(분포 안)", "clean", "noisy", "r2"), ("같은 모델, 고장 확률 2.67배\n(분포 이동)", "clean", "noisy", "r2_shift"),
              ("가우시안 잡음만\n(분포 안)", "clean_gauss_only", "noisy_gauss_only", "r2")]
    w = 0.34
    for i, (lab, a, b, key) in enumerate(groups):
        for j, (arm, col, nm) in enumerate(((a, BLUE, "깨끗한 깊이 목표"), (b, ORANGE, "입력과 같은 오염 깊이 목표"))):
            m, s = d[arm][key]["mean"], d[arm][key]["sd"]
            ax.bar(i + (j - 0.5) * w, m, w, yerr=s, color=col, capsize=3, label=nm if i == 0 else None)
            ax.text(i + (j - 0.5) * w, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("h(t)로 숨은 상태를 선형 복원한 R² (키 r2)")
    ax.set_ylim(0, 1.22)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    ax.set_title("토이 T1: 목표가 오염 깊이일 때 잠재 상태에서 숨은 상태를 덜 복원합니다 (5 시드, ± SD)", fontsize=9.2, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_clean_target.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_return()
    fig_real()
    fig_toy()
    print("wrote 3 figures to", FIG)
