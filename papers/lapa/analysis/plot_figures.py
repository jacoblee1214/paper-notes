"""recompute_results.json을 그래프 세 장으로 그린다.

출력: report/figures/own_language_table.png, own_realworld_paired.png, own_efficiency.png
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
한글 폰트는 Linux(Noto CJK)와 macOS(Apple SD Gothic Neo) 후보를 차례로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))

for cand in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"Scratch": "#9a9a95", "UniPi": "#c9b27c", "VPT": "#6f9fcf", "LAPA": "#1f5fae", "ActionVLA": "#3c7a52"}

# 1. Language Table: 표 1 값과 ± (부분 과제 5개의 표준오차로 확인한 값)
conds = ["in_seen", "in_unseen", "xtask_seen", "xtask_unseen", "xenv_seen", "xenv_unseen"]
labels = ["In-domain\nSeen", "In-domain\nUnseen", "Cross-task\nSeen", "Cross-task\nUnseen", "Cross-env\nSeen", "Cross-env\nUnseen"]
rep = {  # 표 1 (논문 값 그대로)
    "Scratch": [(15.6, 9.2), (15.2, 8.3), (27.2, 13.6), (22.4, 11.0), (15.6, 9.2), (15.2, 8.3)],
    "UniPi": [(22.0, 12.5), (13.2, 7.7), (20.8, 12.0), (16.0, 9.1), (13.6, 8.6), (12.0, 7.5)],
    "VPT": [(44.0, 7.5), (32.8, 4.6), (72.0, 6.8), (60.8, 6.6), (18.0, 7.7), (18.4, 9.7)],
    "LAPA": [(62.0, 8.7), (49.6, 9.5), (73.2, 6.8), (54.8, 9.1), (33.6, 12.7), (29.6, 12.0)],
    "ActionVLA": [(77.0, 3.5), (58.8, 6.6), (77.0, 3.5), (58.8, 6.6), (64.8, 5.2), (54.0, 7.0)],
}
fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.16
for k, m in enumerate(C):
    xs = [i + (k - 2) * w for i in range(len(conds))]
    ys = [rep[m][i][0] for i in range(len(conds))]
    es = [rep[m][i][1] for i in range(len(conds))]
    ax.bar(xs, ys, w, yerr=es, color=C[m], label=m, capsize=2, error_kw={"lw": 0.8})
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("평균 성공률 (%)")
ax.set_ylim(0, 100)
ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
ax.set_title("Language Table 평균 성공률과 ± (논문 표 1 값)", fontsize=10, pad=24)
ax.grid(axis="y", lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(FIG / "own_language_table.png", dpi=170)
plt.close(fig)

# 2. 실로봇 짝지은 차이 (부트스트랩 95% 구간)
pairs = RES["rw_paired_tests"]
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, p in enumerate(pairs):
    lo, hi = p["boot95_pp"]
    m = p["mean_diff_pp"]
    ax.plot([lo, hi], [i, i], color="#1f5fae", lw=2)
    ax.plot([m], [i], "o", color="#1f5fae")
    ax.text(hi + 0.6, i, f"p(부호)={p['sign_test_p_ties_dropped']:.2f}", va="center", fontsize=7.5)
ax.axvline(0, color="#333", lw=0.8, ls="--")
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels([p["comparison"] for p in pairs], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("평균 차이 (퍼센트 포인트), 과제 3개 × 18회 = 54쌍, 95% 부트스트랩 구간")
ax.set_xlim(-12, 34)
ax.set_title("실로봇 짝지은 차이: 0을 포함하는 구간이 대부분", fontsize=10)
ax.grid(axis="x", lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(FIG / "own_realworld_paired.png", dpi=170)
plt.close(fig)

# 3. 사전학습 효율 환산 (본문 4.6): H100이 A100보다 s배 빠르다고 가정
import numpy as np
s = np.linspace(1.0, 3.0, 100)
ratio = 21500 / (272 * s)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(s, ratio, color="#1f5fae")
ax.axhline(30, color="#b04a3a", ls="--", lw=1)
ax.text(1.02, 31.5, "본문 주장 30배", color="#b04a3a", fontsize=8)
ax.axvspan(2.0, 3.0, color="#dfe9f6", alpha=0.7)
ax.text(2.05, 70, "본문 가정 2–3배 구간", fontsize=8)
ax.set_xlabel("H100 대 A100 속도 배수 s")
ax.set_ylabel("OpenVLA 대 LAPA 사전학습 시간 비 (배)")
ax.set_ylim(0, 85)
ax.set_title("효율 주장의 민감도: 21,500 A100시간 대 34시간 × 8 H100", fontsize=9.5)
ax.grid(lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(FIG / "own_efficiency.png", dpi=170)
plt.close(fig)
print("wrote figures to", FIG)
