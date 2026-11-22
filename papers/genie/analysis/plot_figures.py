"""우리가 그린 그래프 두 장 (논문 그림은 쓰지 않는다: arXiv 라이선스가 비독점, PDF에 저작권 표시 있음).

  report/figures/own_ablation.png     표 2, 표 3의 FVD와 ΔtPSNR (단일 실행, 오차 막대 없음)
  report/figures/own_flops_check.png  표 10의 계산량을 6·N·D 근사와 비교한 비율 (V6)
실행: python3 analysis/plot_figures.py
한글 폰트: macOS는 Apple SD Gothic Neo, Linux는 Noto Sans CJK를 차례로 찾는다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#8a8880"

# 표 2 (LAM 입력)와 표 3 (토크나이저)
rows2 = [("Platformers\n토큰 입력", 38.8, 1.33), ("Platformers\n픽셀 입력(Genie)", 40.1, 1.91),
         ("Robotics\n토큰 입력", 257.8, 1.65), ("Robotics\n픽셀 입력(Genie)", 136.4, 2.07)]
rows3 = [("ViT", 114.5, 1.39), ("C-ViViT", 272.7, 1.37), ("ST-ViViT\n(논문 제안)", 81.4, 1.66)]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
for col, (title, rows) in enumerate([("표 2: LAM 입력", rows2), ("표 3: 토크나이저 구조", rows3)]):
    names = [r[0] for r in rows]
    fvd = [r[1] for r in rows]
    dp = [r[2] for r in rows]
    colors = [GREY if n.startswith(("Platformers\n토큰", "Robotics\n토큰", "ViT", "C-ViViT")) else ACC for n in names]
    ax = axes[0][col]
    ax.bar(range(len(rows)), fvd, color=colors)
    ax.set_xticks(range(len(rows)), names, fontsize=8)
    ax.set_ylabel("FVD (낮을수록 좋음)")
    ax.set_title(title + " · FVD", fontsize=11, loc="left", color=INK)
    for i, v in enumerate(fvd):
        ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
    ax = axes[1][col]
    ax.bar(range(len(rows)), dp, color=colors)
    ax.set_xticks(range(len(rows)), names, fontsize=8)
    ax.set_ylabel("ΔtPSNR (높을수록 제어성 큼)")
    ax.set_title(title + " · ΔtPSNR", fontsize=11, loc="left", color=INK)
    for i, v in enumerate(dp):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.005, "논문 표 2, 3의 점 값을 옮겼습니다. 시드 수와 분산이 보고되지 않아 오차 막대가 없습니다. "
         "표 2의 Platformers에서 두 입력의 파라미터는 2.3B 대 2.5B입니다.", fontsize=7.5, color=GREY)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "own_ablation.png", dpi=170)
plt.close(fig)

# 표 10: 계산량 대 6·N·D (D = 750B 토큰)
N = [41e6, 96e6, 192e6, 404e6, 811e6, 1.6e9, 2.7e9]
F = [2.05e20, 3.58e20, 6.4e20, 1.2e21, 2.2e21, 4.04e21, 6.91e21]
D = 750e9
ratio = [f / (6 * n * D) for n, f in zip(N, F)]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.semilogx(N, ratio, "o-", color=ACC, lw=1.6)
for n, r in zip(N, ratio):
    ax.annotate(f"{r:.2f}", (n, r), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.axhline(1.0, color=GREY, ls="--", lw=1)
ax.set_xlabel("동역학 모델 파라미터 수 (표 10, 로그 축)")
ax.set_ylabel("표 10 계산량 ÷ 6·N·D")
ax.set_ylim(0.4, 1.3)
ax.set_title("표 10 계산량이 6·N·D 근사와 일정한 비율이 아님", fontsize=11, loc="left", color=INK)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "D = 750B 토큰(표 10 캡션). 비율이 1.11에서 0.56으로 줄어듭니다. 계산량 집계 방식이 명시되지 않았습니다.",
         fontsize=7.5, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "own_flops_check.png", dpi=170)
plt.close(fig)
print("wrote", OUT / "own_ablation.png", OUT / "own_flops_check.png")
