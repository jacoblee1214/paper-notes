"""리포트·게시본용 직접 그린 그래프 세 장 (논문 그림이 아님).
실행: python3 analysis/plot_figures.py -> report/figures/{toy_curves,sig_vs_rho,table3_rows}.png
입력: analysis/align_toy_results.json, analysis/table_check_results.json (모두 CPU 재계산 결과)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FONT = next((p for p in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
             if Path(p).exists()), None)
if FONT:
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, TEAL, ORANGE, GRAY = "#1b1b1a", "#55544f", "#1f5fae", "#c95a24", "#9a9891"

toy = json.loads((ROOT / "analysis" / "align_toy_results.json").read_text())
runs = [r for r in toy["runs"] if r["gamma"] == 0.5 and r["delta"] == 0.3]
N = [r["N_pairs"] for r in runs]
series = [("M2 실측 라벨 직접 회귀", "M2 실측 라벨 N개 직접 회귀", TEAL, "-"),
          ("M3 CCA 공유 잠재 (쌍)", "M3 CCA 공유 잠재", ORANGE, "-"),
          ("M4 InfoNCE 정렬 (쌍)", "M4 InfoNCE 정렬", ORANGE, "--"),
          ("M0 시뮬 회귀 그대로", "M0 시뮬 회귀 그대로", GRAY, ":"),
          ("M1 도메인 무작위화", "M1 도메인 무작위화(시뮬 매핑 근처 섭동)", GRAY, "-.")]
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
for label, key, color, ls in series:
    m = [r[key][0] for r in runs]
    s = [r[key][1] for r in runs]
    ax.plot(N, m, ls, color=color, marker="o", ms=3.5, lw=1.6, label=label)
    ax.fill_between(N, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], color=color, alpha=0.12, lw=0)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xscale("log")
ax.set_xlabel("쌍 또는 라벨 수 N (로그)", color=INK)
ax.set_ylabel("실측 테스트 R²", color=INK)
ax.set_title("비선형 간극 (γ=0.5): 쌍이 N개일 때", color=INK, fontsize=10)
ax.set_ylim(-1.2, 1.0)
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(FIG / "toy_curves.png")
plt.close(fig)

# 2) 궤적 단위 유효 표본 수에 따른 InfoNCE 대 MSE 정확도 z
sig = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())["significance"]
rhos = [0.0, 0.05, 0.2]
zs = [sig[f"Acc InfoNCE vs MSE, ID, rho={r}"]["z"] for r in rhos]
zo = [sig[f"Acc InfoNCE vs MSE, OOD-Size, rho={r}"]["z"] for r in rhos]
fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=200)
ax.plot(rhos, zs, "o-", color=TEAL, lw=1.6, label="분포 내")
ax.plot(rhos, zo, "s-", color=ORANGE, lw=1.6, label="OOD-Size")
ax.axhline(1.96, color=INK2, lw=0.7, ls="--")
ax.text(0.205, 2.05, "z = 1.96 (p = 0.05)", fontsize=7, color=INK2, ha="right")
ax.set_xlabel("궤적 내 상관 ρ (가정)", color=INK)
ax.set_ylabel("z (InfoNCE 대 MSE, 정확도)", color=INK)
ax.set_title("궤적 상관을 넣으면 정렬 우위가 사라진다", color=INK, fontsize=10)
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(FIG / "sig_vs_rho.png")
plt.close(fig)

# 3) 논문 Table III의 행별 분포 내 깊이와 힘 (수치는 표에서 옮긴 값)
rows = [("R1 복원 없음", 45, 60), ("R2 정렬 Mc,Mp", 58, 78), ("R3 정렬 Mc,Ms", 48, 63),
        ("R4 TactSpace", 26, 65), ("R5 OOD 포함 정렬", 25, 62), ("R6 실측 상한", 18, 20)]
fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=200)
for name, pen, force in rows:
    color = ORANGE if name.startswith("R4") else (GRAY if name.startswith("R6") else TEAL)
    ax.scatter(pen, force, s=46, color=color, zorder=3)
    ax.annotate(name, (pen, force), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK)
ax.set_xlabel("분포 내 깊이 오차 (um, 낮을수록 좋음)", color=INK)
ax.set_ylabel("분포 내 힘 오차 (mN, 낮을수록 좋음)", color=INK)
ax.set_title("표 III 행 비교: 깊이 최저(R4)가 힘 최저는 아님", color=INK, fontsize=10)
ax.set_xlim(10, 52)
ax.set_ylim(15, 85)
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(FIG / "table3_rows.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("*.png")))
