"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 값은 recompute.py의 표 값을 그대로 가져온다.

  own_realworld_ci.png    : 실로봇 성공률 (과제 3개 합계, 60회)과 Wilson 95% 구간, 방법 4개
  own_libero_plus_total.png: 표 3의 Total(보고값)과 일곱 교란 범주의 단순 평균
  own_wan_dct_vs_vae.png  : DCT 인코딩 − VAE 인코딩 상대 재구성 오차 (σ별, Cosmos·Wan2.1·Wan2.2)

실행: python3 analysis/plot_figures.py   (matplotlib)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recompute as rc  # noqa: E402

FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

KOREAN_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",            # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux (Noto CJK)
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
for cand in KOREAN_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
INK, ACCENT, GREY = "#1b1b1a", "#1f5fae", "#8a8880"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


# 1. 실로봇 성공률과 Wilson 구간 (ID 합계, OOD 합계; 과제당 20회 × 3과제 = 60회)
methods = ["π0", "π0.5", "Cosmos Policy", "MetaWAM"]
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
for ax, key, title in ((axes[0], "id", "ID (분포 내)"), (axes[1], "ood", "OOD (분포 외)")):
    for i, m in enumerate(methods):
        per_task = rc.T4[m][0] if key == "id" else rc.T4[m][1]
        k = sum(round(x * 20 / 100) for x in per_task)  # 과제당 20회
        lo, hi = rc.wilson(k, 60)
        p = k / 60
        color = ACCENT if m == "MetaWAM" else GREY
        ax.errorbar(100 * p, i, xerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="o", color=color,
                    ecolor=color, capsize=3, ms=6)
        ax.text(100 * p + 2, i + 0.18, f"{k}/60", fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlim(20, 100)
    ax.set_xlabel("성공률 (%) · Wilson 95% 구간", fontsize=9, color=INK)
    style(ax)
axes[0].set_yticks(range(len(methods)))
axes[0].set_yticklabels(methods)
axes[0].invert_yaxis()
fig.suptitle("실로봇 과제 합계 (논문 표 4 값에서 역산한 성공 수)", fontsize=11, color=INK)
fig.text(0.5, 0.01, "우리가 그린 그래프. 방법 간 차이는 60회에서 Fisher p ≥ 0.38이라 구분되지 않음",
         ha="center", fontsize=8, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig(FIG / "own_realworld_ci.png", dpi=200)
plt.close(fig)

# 2. 표 3 Total(보고값) 대 일곱 범주 단순 평균
rows = {k: v for k, v in rc.PAPER_T3.items()}
rows["π0.5"] = rc.SRC_T3["π0.5"]
rows["π0 (rerun)"] = rc.SRC_T3["π0 (rerun)"]
names = sorted(rows, key=lambda k: rows[k][8])
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for i, n in enumerate(names):
    rep = rows[n][8]
    avg = sum(rows[n][1:8]) / 7
    hl = n == "MetaWAM"
    ax.plot([rep, avg], [i, i], color="#d9d7cf", lw=2, zorder=1)
    ax.scatter(rep, i, color=ACCENT if hl else INK, s=34, zorder=4, marker="o")
    ax.scatter(avg, i, color="white", edgecolor=ACCENT if hl else GREY, s=34, zorder=3, marker="o")
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n + ("  (표 3에 없음)" if n in ("π0.5", "π0 (rerun)") else "") for n in names], fontsize=8.5)
ax.set_xlabel("LIBERO-Plus 성공률 (%)", fontsize=9, color=INK)
ax.set_xlim(40, 92)
style(ax)
ax.scatter([], [], color=INK, s=34, label="보고된 Total")
ax.scatter([], [], color="white", edgecolor=GREY, s=34, label="일곱 범주 단순 평균")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title("표 3의 Total은 단순 평균이 아니다 (MetaWAM과 Cosmos Policy의 차이 1.2 → 0.45 pp)",
             fontsize=10, color=INK)
fig.text(0.5, 0.01, "우리가 그린 그래프. 원자료 arXiv:2603.22078 표 4 값. 범주별 시도 수가 없어 가중치는 알 수 없음.",
         ha="center", fontsize=8, color=GREY)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "own_libero_plus_total.png", dpi=200)
plt.close(fig)

# 3. DCT - VAE (상대 재구성 오차, 음수면 DCT가 낮음)
series = [
    ("Cosmos-Predict2 (표 1a)", rc.SIGMAS,
     [d - v for d, v in zip(rc.T1A["DCT Encoding"], rc.T1A["VAE Encoding"])]),
    ("Wan2.1-I2V-14B (표 6a)", rc.WAN_SIG,
     [d - v for d, v in zip(rc.W21A["DCT Encoding"], rc.W21A["VAE Encoding"])]),
    ("Wan2.2-TI2V-5B (표 7a)", rc.WAN_SIG,
     [d - v for d, v in zip(rc.W22A["DCT Encoding"], rc.W22A["VAE Encoding"])]),
]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
colors = [ACCENT, "#7a6a3a", "#4f8a6b"]
for (label, xs, ys), c in zip(series, colors):
    ax.plot(xs, ys, marker="o", color=c, lw=1.6, label=label)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xscale("log")
ax.set_xticks([0.01, 0.05, 0.2, 0.4, 1.0, 5.0])
ax.set_xticklabels(["0.01", "0.05", "0.2", "0.4", "1.0", "5.0"])
ax.set_xlabel("노이즈 수준 σ (로그 축)", fontsize=9, color=INK)
ax.set_ylabel("DCT - VAE (상대 재구성 오차)", fontsize=9, color=INK)
ax.annotate("Wan2.1, σ=0.2: VAE가 낮음\n(본문 A.2 '일관되게 낮다'와 불일치)",
            xy=(0.2, 0.0167), xytext=(0.012, -0.125), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
style(ax)
ax.set_ylim(-0.18, 0.04)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title("DCT 인코딩이 VAE 인코딩보다 낮은가 (음수면 DCT가 낮음)", fontsize=10, color=INK)
fig.text(0.5, 0.01, "우리가 그린 그래프. 값은 논문 표 1a, 6a, 7a 그대로 (단일 신호 주입).",
         ha="center", fontsize=8, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_wan_dct_vs_vae.png", dpi=200)
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
