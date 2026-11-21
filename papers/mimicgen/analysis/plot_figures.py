"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림은 만들거나 자르지 않는다.

  own_dgr_sr.png        표 P.1의 D0 생성 성공률과 그림 4의 D0 정책 성공률 (과제 16개, 순위 상관)
  own_binomial_sd.png   보고된 ± (3 시드 SD)와 롤아웃 50회 한 번의 이항 표준오차
  own_real_ci.png       실로봇 비율과 Wilson 95% 구간 (본문 6.4)
  own_robot_transfer.png 로봇 팔별 성공률 (부록 F, 이미지 기반, 평균 ± SD)

입력: analysis/recompute_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
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
INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

# ------------------------------------------------ 그래프 1: DGR 과 D0 성공률
d = R["plot_dgr_sr_D0"]
rho = R["dgr_vs_sr_spearman"]
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
OFFS = {"Square": (6, 6), "Gear Assembly": (-58, 6), "Hammer Cleanup": (4, 4), "Threading": (-40, -14), "Coffee Preparation": (6, -4),
        "Nut-and-Bolt Assembly": (-74, 4), "Stack Three": (8, -14), "Three Pc. Assembly": (-92, 4), "Frame Assembly": (4, -10)}
for t, (x, y) in d.items():
    ax.scatter(x, y, s=28, color=ACC, zorder=3)
    ax.annotate(t, (x, y), xytext=OFFS.get(t, (4, 3)), textcoords="offset points", fontsize=6.6, color=INK)
ax.set_xlabel("데이터 생성 성공률 DGR (%), 표 P.1 D0")
ax.set_ylabel("정책 성공률 (%), 그림 4 D0 (이미지)")
ax.set_xlim(0, 108)
ax.set_ylim(40, 105)
ax.set_title(f"과제 {rho['D0_n_tasks']}개 D0: Spearman ρ = {rho['D0_rho']:.2f} (p = {rho['D0_p']:.3f})\n"
             "본문 6.3의 '꼭 같지는 않다'는 예외 사례에 대한 말이고, 전체 순위는 양의 상관입니다",
             fontsize=8.2, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_dgr_sr.png")
plt.close(fig)

# ------------------------------------------------ 그래프 2: 보고된 ± 대 이항 표준오차
rows = R["binomial_vs_reported_sd"]
sd = np.array([v["reported_sd_3seeds"] for v in rows.values()])
se = np.array([v["binomial_se_one_eval_n50"] for v in rows.values()])
fig, ax = plt.subplots(figsize=(5.8, 4.0), dpi=200)
ax.scatter(se, sd, s=22, color=ACC, zorder=3)
lim = [0, 14]
ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
ax.text(9.6, 9.0, "같은 값", color=MUTED, fontsize=7.5, rotation=0)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("롤아웃 50회 한 번의 이항 표준오차 (%p)")
ax.set_ylabel("보고된 ± 값 (3 시드 SD, %p)")
s = R["binomial_vs_reported_sd_summary"]
ax.set_title(f"{s['n_cells']}개 셀 중 {s['share_cells_reported_sd_below_binomial_se']*100:.0f}%에서 보고된 ±가 평가 한 번의 표본 잡음보다 작습니다",
             fontsize=8.2, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_binomial_sd.png")
plt.close(fig)

# ------------------------------------------------ 그래프 3: 실로봇 구간
real = R["real_robot"]
labels = ["Stack BC-RNN", "Stack 확산 정책", "Stack 원천 데모 10회", "Coffee BC-RNN 전체", "Coffee 삽입 단계"]
keys = ["Stack BC-RNN 18/50", "Stack DiffPolicy 38/50", "Stack source 0/50", "Coffee policy 7/50", "Coffee pod insertion 10/50"]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
for i, (lab, k) in enumerate(zip(labels, keys)):
    r = real[k]
    lo, hi = r["wilson95"]
    rate = r["rate"]
    ax.errorbar(rate, i, xerr=[[rate - lo], [hi - rate]], fmt="o", color=ACC if i < 3 else ACC2, capsize=3, lw=1.2)
    ax.text(hi + 2, i, f"{rate:.0f}%", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7.4)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("성공률 (%), 평가 50회, Wilson 95% 구간")
ax.set_title("실로봇 (본문 6.4): 평가 50회 한 번씩, 체크포인트 선택 없음", fontsize=8.4, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png")
plt.close(fig)

# ------------------------------------------------ 그래프 4: 로봇 팔별 성공률
rt = R["plot_robot_transfer"]
cells = ["Square D0", "Square D1", "Threading D0", "Threading D1"]
arms = ["Panda", "Sawyer", "IIWA", "UR5e"]
cols = [ACC, "#8fa9c9", "#c9b48f", "#b7c5d8"]
fig, axes = plt.subplots(1, 4, figsize=(8.2, 3.0), dpi=200, sharey=True)
for ax, cell in zip(axes, cells):
    for j, arm in enumerate(arms):
        m, s = rt[cell][arm]
        ax.bar(j, m, 0.7, color=cols[j], edgecolor="white")
        ax.errorbar(j, m, yerr=s, fmt="none", ecolor=INK, capsize=2.5, lw=0.9)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=6.8, rotation=35)
    ax.set_title(cell, fontsize=8.2, color=INK)
axes[0].set_ylabel("성공률 (%)")
axes[0].set_ylim(0, 110)
fig.suptitle("부록 F 표 F.2 (이미지 기반): D1에서 IIWA는 Panda보다 25 %p 낮습니다. 막대 = 평균, 선 = 보고된 SD", fontsize=8.2, x=0.02, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "own_robot_transfer.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
