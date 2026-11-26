"""우리가 그리는 그래프 세 장 (report/figures/own_*.png). 입력: analysis/recompute_results.json, toy_leakage_results.json.
실행: python3 analysis/plot_figures.py   (matplotlib)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/System/Library/Fonts/Supplemental/AppleGothic.ttf"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = False
BLUE, ORANGE, GRAY = "#1f5fae", "#d9822b", "#8a8a85"
R = json.loads((HERE / "recompute_results.json").read_text())


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)


# 1) 표 2: 정확도와 Wilson 95% 구간
v2 = R["V2_table2_task_outcome"]
rows = [("RoboCasa-GR1\n선택 160", v2["robocasa_n160"]["adaln"], v2["robocasa_n160"]["ours"]),
        ("RoboCasa-GR1\n전체 256", v2["robocasa_full_n256"]["adaln"], v2["robocasa_full_n256"]["ours"]),
        ("RoboLab\n100", v2["robolab_n100"]["adaln"], v2["robolab_n100"]["ours"])]
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
for i, (lab, a, o) in enumerate(rows):
    for j, (d, c, name) in enumerate(((a, GRAY, "AdaLN"), (o, BLUE, "Ours"))):
        acc = d.get("reported_pct", d.get("accuracy_pct"))
        lo, hi = d["wilson95"]
        x = i + (j - 0.5) * 0.28
        ax.errorbar(x, acc, yerr=[[acc - lo], [hi - acc]], fmt="o", color=c, ms=6, capsize=3, lw=1.6,
                    label=name if i == 0 else None)
ax.axhline(50, color=ORANGE, lw=1, ls="--")
ax.text(2.45, 50.8, "우연 50%", color=ORANGE, fontsize=8, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylabel("정확도 (%)"); ax.set_ylim(35, 90); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("과제 결과 예측 정확도와 Wilson 95% 구간 (표 값, 시행 수는 표 기준)", fontsize=9)
style(ax); fig.tight_layout(); fig.savefig(FIG / "own_task_outcome.png"); plt.close(fig)

# 2) v1 표 2 대 v2 표 3, 표 4의 같은 설정
rows = ["raw", "nominal", "depth"]
lab = ["원시 행동 메시", "명목 메시", "명목 메시 + 깊이"]
v1 = [21.57, 22.44, 23.08]; v2p = [21.34, 21.76, 21.96]
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
w = 0.34
b1 = ax.bar([i - w / 2 for i in range(3)], v1, w, color=GRAY, label="v1 표 2")
b2 = ax.bar([i + w / 2 for i in range(3)], v2p, w, color=BLUE, label="v2 표 3")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}", ha="center", fontsize=8)
ax.plot([0.5 - 0.5, 1.5 + 0.5], [21.74, 21.74], color=ORANGE, lw=1.2, ls="--")
ax.text(-0.45, 21.8, "점선: 표 4\nnominal/nominal 21.74", color=ORANGE, fontsize=7.5, ha="left", va="bottom")
ax.set_ylim(20.5, 23.6); ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("DROID PSNR (dB)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("절제 표의 PSNR: v1과 v2 (같은 DROID, 같은 변형 이름)", fontsize=9)
style(ax); fig.tight_layout(); fig.savefig(FIG / "own_ablation_v1_v2.png"); plt.close(fig)

# 3) 토이 학습 곡선
T = json.loads((HERE / "toy_leakage_results.json").read_text())
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
for kind, c, name in (("raw", GRAY, "원시 행동"), ("nominal", BLUE, "명목 궤적"), ("logged", ORANGE, "기록된 실현 궤적")):
    ys = [T["runs"][f"{kind}_N{n}"]["rmse_mean"] for n in T["sizes"]]
    sd = [T["runs"][f"{kind}_N{n}"]["rmse_sd"] for n in T["sizes"]]
    ax.errorbar(T["sizes"], ys, yerr=sd, color=c, marker="o", ms=5, lw=1.6, capsize=2, label=name)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(T["sizes"]); ax.set_xticklabels([str(n) for n in T["sizes"]])
ax.set_yticks([0.005, 0.01, 0.02, 0.05]); ax.set_yticklabels(["0.005", "0.01", "0.02", "0.05"]); ax.minorticks_off()
ax.set_xlabel("학습 표본 수 N"); ax.set_ylabel("블록 이동량 RMSE (m)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("토이: 조건 종류별 학습 곡선 (시드 3개 평균 ± SD, 우리가 만든 모형)", fontsize=9)
style(ax); fig.tight_layout(); fig.savefig(FIG / "own_toy_leakage.png"); plt.close(fig)
print("wrote figures")
