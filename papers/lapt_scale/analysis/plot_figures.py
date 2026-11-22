"""우리가 만든 그래프 네 장 (report/figures/own_*.png).

  own_scaling_ci.png       : 표 14의 스케일링 곡선과 Wilson 95% 구간 (3과제 x 20회 = 60회)
  own_generalization_ci.png: 표 6의 영역별 성공률과 Wilson 95% 구간 (영역당 80회)
  own_ablation_diag.png    : 표 5의 PSNR, LARY 대 RoboTwin Hard 성공률
  own_toy_motion.png       : 토이 — 위치 오차별 PSNR, 코드 수별 변위 분해능

실행: python3 analysis/plot_figures.py   (recompute.py, toy_psnr_motion.py를 먼저 실행; matplotlib)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/Library/Fonts/AppleGothic.ttf"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

C = {"lapt": "#1f5fae", "dream": "#d9822b", "hand": "#7a7a7a", "pi": "#2a9d8f", "dp": "#b5483f"}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def scaling():
    ks = {"hand": [33, 35, 45, 33], "dream": [33, 37, 45, 46], "lapt": [33, 38, 48, 51]}
    lab = {"hand": "손 동작 의사 행동", "dream": "DreamDojo LAM", "lapt": "LAPT-Scale-LAM"}
    xs = [0, 1, 3, 5]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for i, key in enumerate(("hand", "dream", "lapt")):
        off = (i - 1) * 0.09
        ys, lo, hi = [], [], []
        for k in ks[key]:
            a, b = wilson(k, 60)
            ys.append(100 * k / 60)
            lo.append(100 * k / 60 - a)
            hi.append(b - 100 * k / 60)
        ax.errorbar([x + off for x in xs], ys, yerr=[lo, hi], marker="o", ms=4, lw=1.6, capsize=2.5, color=C[key], label=lab[key])
    ax.set_xticks(xs)
    ax.set_xticklabels(["0 h", "1K", "3K", "5K"])
    ax.set_ylabel("실로봇 평균 성공률 (%)")
    ax.set_xlabel("1단계 사람 영상 시간")
    ax.set_ylim(25, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("스케일링: 3과제 x 20회 = 60회, Wilson 95% 구간", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "own_scaling_ci.png", dpi=160)
    plt.close(fig)


def generalization():
    data = {"DP": [48, 16, 12, 12], "π0.5": [68, 40, 36, 32], "LAPT-Scale": [68, 60, 56, 56]}
    cols = {"DP": C["dp"], "π0.5": C["pi"], "LAPT-Scale": C["lapt"]}
    dom = ["Seen", "위치", "조명", "배경"]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for i, (m, ks) in enumerate(data.items()):
        off = (i - 1) * 0.12
        ys, lo, hi = [], [], []
        for k in ks:
            a, b = wilson(k, 80)
            ys.append(100 * k / 80)
            lo.append(100 * k / 80 - a)
            hi.append(b - 100 * k / 80)
        ax.errorbar([j + off for j in range(4)], ys, yerr=[lo, hi], marker="o", ms=4, lw=1.4, capsize=2.5, color=cols[m], label=m)
    ax.set_xticks(range(4))
    ax.set_xticklabels(dom)
    ax.set_ylabel("4과제 평균 성공률 (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("표 6: 영역당 4과제 x 20회 = 80회, Wilson 95% 구간", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "own_generalization_ci.png", dpi=160)
    plt.close(fig)


def ablation():
    t5 = {"MSE만": (33.9844, 0.675, 74.2), "LPIPS 제거": (34.2105, 0.691, 82.5), "DINO 제거": (34.15, 0.698, 84.1),
          "flow 제거": (34.3012, 0.685, 77.8), "전체": (34.4217, 0.713, 89.3)}
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for ax, idx, xl in ((axs[0], 0, "LAM PSNR (dB)"), (axs[1], 1, "LARY 정확도")):
        for name, v in t5.items():
            ax.scatter(v[idx], v[2], color=C["lapt"], s=26, zorder=3)
            ax.annotate(name, (v[idx], v[2]), textcoords="offset points", xytext=(4, -9 if name in ("LPIPS 제거", "전체") else 3), fontsize=8)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("RoboTwin Hard 성공률 (%)")
    axs[0].set_title("PSNR와는 순위가 어긋남 (ρ = 0.60)", fontsize=9)
    axs[1].set_title("LARY와는 순위가 같음 (ρ = 1.0, 점 5개)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "own_ablation_diag.png", dpi=160)
    plt.close(fig)


def toy():
    r = json.loads((HERE / "toy_psnr_motion_results.json").read_text(encoding="utf-8"))
    rows = [x for x in r["A"] if x["label"] == "점 작음" and x["sigma"] == 0.02] + \
           [x for x in r["A"] if x["label"] == "점 작음" and x["sigma"] == 0.05]
    names = [("psnr_perfect", "정답"), ("psnr_pos_err_1px", "1화소 오차"), ("psnr_pos_err_2px", "2화소 오차"),
             ("psnr_copy_no_motion", "이동 없음"), ("psnr_wrong_direction", "반대 방향")]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.7))
    ax = axs[0]
    w = 0.38
    for i, row in enumerate(rows):
        ax.bar([j + (i - 0.5) * w for j in range(len(names))], [row[k] for k, _ in names], w,
               color=[C["lapt"], C["dream"]][i], label=f"잡음 σ = {row['sigma']}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n for _, n in names], fontsize=8)
    ax.set_ylim(24, 40)
    ax.set_ylabel("다음 프레임 PSNR (dB)")
    ax.set_title("점 면적 0.2%: 이동 없음과 반대 방향이 PSNR로 구분되지 않음", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    ks = list(r["B"].keys())
    ax.plot([int(k) for k in ks], [r["B"][k]["r2_of_quantized_displacement"] for k in ks], marker="o", color=C["lapt"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("코드 수 K")
    ax.set_ylabel("2차원 변위의 R²")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title("코드 K개로 양자화한 변위 ([-3, 3]² 균일)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "own_toy_motion.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    scaling()
    generalization()
    ablation()
    toy()
    print("wrote", OUT)
