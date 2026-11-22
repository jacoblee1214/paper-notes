"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_forest.png      Robot-only 대비 Clean·Rand 변화량과 불확실성 구간 (표 5 기반)
  own_ablation_real.png  그림 3 절제와 그림 4 실로봇 값에 표본오차 막대
  own_exposure.png    200K 스텝 x 96이 데이터를 몇 번 훑는가(프레임률 가정별)
  own_parity.png      표 5의 홀수 값 개수 (50 에피소드면 0이어야 함)

입력: analysis/recompute_results.json (먼저 recompute.py 실행)
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
RS = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

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
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 1. forest
def forest():
    names = ["Ego2R(1:3)", "Ego2R(3:1)", "Ego2R(1:1)"]
    labels = {"Ego2R(1:3)": "1:3", "Ego2R(3:1)": "3:1", "Ego2R(1:1)": "1:1"}
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ypos = 0
    ticks, tl = [], []
    for key, setting in (("clean", "Clean"), ("rand", "Rand")):
        for n in names:
            v7 = RS["V7_paired_over_tasks"][n][key]
            z50 = RS["V8_diff_z_binomial"][f"{n}|{key}"]
            z100 = RS["V8_diff_z_binomial_n100"][f"{n}|{key}"]
            d = z50["diff"]
            ax.plot([d - 1.96 * z50["se"], d + 1.96 * z50["se"]], [ypos + 0.14] * 2, color=ACC, lw=3, solid_capstyle="butt")
            ax.plot([d - 1.96 * z100["se"], d + 1.96 * z100["se"]], [ypos] * 2, color=ACC2, lw=3, solid_capstyle="butt")
            ax.plot(v7["boot95"], [ypos - 0.14] * 2, color=WARN, lw=3, solid_capstyle="butt")
            ax.plot(d, ypos, "o", color=INK, ms=5, zorder=5)
            ticks.append(ypos)
            tl.append(f"{setting} {labels[n]}")
            ypos -= 1
        ypos -= 0.5
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tl)
    ax.set_xlabel("Robot-only 대비 성공률 변화 (%p), 95% 구간")
    ax.plot([], [], color=ACC, lw=3, label="에피소드 이항(과제당 50)")
    ax.plot([], [], color=ACC2, lw=3, label="에피소드 이항(과제당 100 가정)")
    ax.plot([], [], color=WARN, lw=3, label="과제 부트스트랩(과제 50개)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("표 5로 다시 계산한 변화량과 불확실성(단일 학습 실행)", fontsize=10.5, loc="left")
    save(fig, "own_forest.png")


# ---------------------------------------------------------------- 2. ablation + real robot
def ablation_real():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    vals = RS["V12_ablation"]["values"]
    names = list(vals.keys())
    y = [vals[k] for k in names]
    se = [math.sqrt(v / 100 * (1 - v / 100) / 2500) * 100 for v in y]
    cols = [MUTED, ACC2, ACC2, ACC2, ACC2, WARN]
    a.bar(range(len(y)), y, color=cols, yerr=[1.96 * s for s in se], capsize=3, ecolor=INK)
    for i, v in enumerate(y):
        a.text(i, v + 1.96 * se[i] + 0.3, f"{v}", ha="center", fontsize=8)
    a.set_xticks(range(len(y)))
    a.set_xticklabels(["raw", "1", "5", "10", "15", "15+raw"], fontsize=8.5)
    a.set_xlabel("모폴로지 수 (Ego2R)")
    a.set_ylim(20, 42)
    a.set_ylabel("RoboTwin Randomized 성공률 (%)")
    a.set_title("그림 3 값 + 이항 95% (2,500 에피소드 가정)", fontsize=10, loc="left")

    fig4 = {"Put Fruits": [3, 23, 25, 31], "Put Blocks": [12, 35, 38, 49], "Fold Towel": [0, 25, 25, 30],
            "Sweep Trash": [0, 1, 5, 14], "Insert Screw": [0, 2, 6, 15]}
    conf = ["Scratch", "Robot-only", "Mix", "Mix+Play"]
    colors = ["#a9b4c4", ACC, ACC2, WARN]
    x = np.arange(len(fig4))
    w = 0.2
    sd = 35.0
    err = 1.96 * sd / math.sqrt(20)
    for j, c in enumerate(conf):
        vv = [fig4[t][j] for t in fig4]
        b.bar(x + (j - 1.5) * w, vv, w, color=colors[j], label=c)
        if c in ("Robot-only", "Mix+Play"):
            b.errorbar(x + (j - 1.5) * w, vv, yerr=err, fmt="none", ecolor=INK, capsize=2, lw=0.8)
    b.set_xticks(x)
    b.set_xticklabels(list(fig4.keys()), fontsize=8.5)
    b.set_ylabel("점수 (%)")
    b.set_title("그림 4 값. 막대는 시행 20, 점수 표준편차 35 가정의 95% 구간", fontsize=10, loc="left")
    b.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    b.set_ylim(0, 75)
    save(fig, "own_ablation_real.png")


# ---------------------------------------------------------------- 3. exposure
def exposure():
    ex = RS["V11_data_and_exposure"]["exposure"]
    fps_list = ["10", "15", "30"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = ["로봇만 (로봇 데이터)", "1:1 (로봇 몫)", "1:1 (Ego2R 몫)", "3:1 (Ego2R 몫)"]
    keys = ["robot_only_epochs", "1:1_robot_epochs", "1:1_ego_epochs", "3:1_ego_epochs"]
    x = np.arange(len(labels))
    w = 0.25
    for j, f in enumerate(fps_list):
        vv = [ex[f][k] * 100 for k in keys]
        ax.bar(x + (j - 1) * w, vv, w, label=f"{f} fps 가정", color=[ACC, ACC2, WARN][j])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("한 번의 사전학습이 훑는 비율 (%, 에폭 x 100)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("200K 스텝 x 배치 96 = 19.2M 샘플이 각 데이터를 훑는 비율", fontsize=10.5, loc="left")
    save(fig, "own_exposure.png")


# ---------------------------------------------------------------- 4. parity
def parity():
    pm = RS["V2_parity"]["per_model"]
    names = list(pm.keys())
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, [pm[n]["clean_odd"] for n in names], w, label="Clean", color=ACC)
    ax.bar(x + w / 2, [pm[n]["rand_odd"] for n in names], w, label="Rand", color=ACC2)
    ax.axhline(25, color=MUTED, ls="--", lw=1)
    ax.text(len(names) - 0.5, 25.8, "무작위 정수면 약 25개", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("표 5의 홀수 값 개수 (50개 중)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("과제당 50 에피소드면 모든 값이 짝수여야 한다", fontsize=10.5, loc="left")
    save(fig, "own_parity.png")


if __name__ == "__main__":
    forest()
    ablation_real()
    exposure()
    parity()
    print("wrote 4 figures to", FIG)
