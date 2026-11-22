"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sr_ci.png     : 표 1 성공률(과제별, 20회)과 Wilson 95% 구간
own_ir.png        : 표 1 개입률(IR)과 방법별 평균, 과제별 감소율
own_toy_tvr.png   : 토이, 이상적 선형 가치에서 채굴 라벨 양성 비율과 TVR 부호 양성 비율의 에피소드 길이 의존성
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 python3 analysis/toy_tvr_labels.py를 한 번 돌려 둔다.
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c2571a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


tasks = ["Pick\nEggplant", "Tube\nInsertion", "RAM\nInsertion", "Wipe\nWhiteboard", "Fold\nTowel"]
SR = {"π0.5 SFT": [95, 30, 10, 65, 70], "HiL-SERL": [90, 60, 85, 85, 85],
      "HiL-SERL+FA-RL": [85, 60, 75, 80, 85], "HiL-SERL+UniIntervene": [95, 70, 95, 90, 90]}
IR = {"HiL-SERL": [28.7, 30.2, 32.3, 30.5, 49.8], "HiL-SERL+FA-RL": [20.4, 22.1, 27.9, 21.9, 30.9],
      "HiL-SERL+UniIntervene": [10.0, 15.8, 12.1, 10.9, 24.1]}
COL = {"π0.5 SFT": GREY, "HiL-SERL": "#6f8fb8", "HiL-SERL+FA-RL": "#a9bfd8", "HiL-SERL+UniIntervene": ORG}

# 1. 성공률 + Wilson
fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.2
for j, (m, v) in enumerate(SR.items()):
    x = np.arange(5) + (j - 1.5) * w
    ks = [round(a * 20 / 100) for a in v]
    lo = [a - wilson(k, 20)[0] for a, k in zip(v, ks)]
    hi = [wilson(k, 20)[1] - a for a, k in zip(v, ks)]
    ax.bar(x, v, w, color=COL[m], label=m, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8, "ecolor": INK})
ax.set_xticks(np.arange(5)); ax.set_xticklabels(tasks)
ax.set_ylabel("성공률 (%)"); ax.set_ylim(0, 122)
ax.set_title("표 1 성공률과 Wilson 95% 구간 (과제당 20회)", fontsize=11)
ax.legend(fontsize=8, ncol=2, loc="upper center", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "own_sr_ci.png", dpi=170); plt.close(fig)

# 2. IR
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9), gridspec_kw={"width_ratios": [3, 1.3]})
ax = axs[0]
w = 0.26
for j, (m, v) in enumerate(IR.items()):
    ax.bar(np.arange(5) + (j - 1) * w, v, w, color=COL[m], label=m)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(tasks)
ax.set_ylabel("개입률 IR (%)"); ax.set_title("과제별 개입률 (3회 학습, 표준편차 없음)", fontsize=10)
ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
ax = axs[1]
avg = {m: np.mean(v) for m, v in IR.items()}
ax.bar(range(3), list(avg.values()), color=[COL[m] for m in avg])
for i, a in enumerate(avg.values()):
    ax.text(i, a + 0.8, f"{a:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(["HiL-\nSERL", "+FA-RL", "+Uni"], fontsize=8)
ax.set_title("평균 (감소 57%: 대 HiL-SERL)", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "own_ir.png", dpi=170); plt.close(fig)

# 3. 토이
res = json.loads((ANA / "toy_tvr_labels_results.json").read_text(encoding="utf-8"))["A_healthy_linear_ramp"]
Ts = sorted(int(t) for t in res)
fig, ax = plt.subplots(figsize=(7.6, 3.9))
ax.plot(Ts, [res[str(t)]["mined_positive_frac"] for t in Ts], "o-", color=ORG, label="채굴 라벨(부록 E 규칙) 양성 비율")
ax.plot(Ts, [res[str(t)]["R_positive_frac(sign)"] for t in Ts], "s-", color=ACC, label="R_t > 0 (σ(R_t) ≥ 0.5) 비율")
ax.axvline(60, color=GREY, ls=":", lw=1)
ax.text(62, 0.3, "T ≥ 60", fontsize=9, color=GREY)
ax.axvline(200, color=GREY, ls=":", lw=1)
ax.text(202, 0.3, "ε = 0.005\n↔ T = 200", fontsize=9, color=GREY)
ax.set_xlabel("에피소드 길이 T (선형 가치 0→1)"); ax.set_ylabel("양성 비율")
ax.set_title("토이: 건강한(정체 없는) 선형 가치 곡선에서의 트리거 양성 비율", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(0.2, 0.62)); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "own_toy_tvr.png", dpi=170); plt.close(fig)
print("saved:", sorted(p.name for p in FIG.glob("own_*.png")))
