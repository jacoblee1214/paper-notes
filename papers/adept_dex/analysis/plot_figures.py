"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다(논문 그림 조각은 만들지 않는다).

  own_real_stages.png     표 4 실로봇 단계별 누적 성공 (10회, 마지막 단계 Wilson 95% 구간)
  own_zeroshot_adr.png    표 2 영샷 성공률 곡선과 1024 에피소드 정수 가능성(Flexiv ADR 50 의 0.70%)
  own_ablation.png        표 3 절제: 단계 시간·학습 SR (± 를 SD 로 읽은 구간)
  own_breakeven.png       사전학습 비용의 손익분기 (사전학습 8B 대 누적 계산)

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
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


# 1. 표 4 단계별 누적 성공
STAGES = ["Reach", "Grasp", "Lifting", "Reorient", "Align", "Insert"]
T4 = {
    "Kuka 비전 · star": ([10, 9, 8, 8, 7, 5], "#6aa2d8", "o"),
    "Kuka 비전 · sq/round": ([10, 8, 6, 4, 3, 3], ACC, "s"),
    "Kuka 비전 · dish": ([10, 10, 8, 7, 6, 6], "#7a9e55", "^"),
    "Flexiv 비전 · sq/round": ([10, 7, 5, 3, 3, 3], ACC2, "v"),
    "Flexiv 시각촉각 · sq/round": ([10, 10, 10, 9, 8, 8], WARN, "D"),
}
fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=170)
x = np.arange(6)
for i, (name, (vals, col, mk)) in enumerate(T4.items()):
    ax.plot(x + (i - 2) * 0.04, [v * 10 for v in vals], marker=mk, color=col, label=name, lw=1.6, ms=5)
ax.set_xticks(x)
ax.set_xticklabels(STAGES, fontsize=9)
ax.set_ylabel("누적 성공 (%), 조건당 10회", fontsize=9)
ax.set_ylim(0, 108)
clean(ax)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
w = RES["V1_table4"]["Flexiv visuo-tactile / FMB square-round"]["insert_wilson95_pct"]
w2 = RES["V1_table4"]["Flexiv vision / FMB square-round"]["insert_wilson95_pct"]
ax.annotate(f"Insert 8/10 (Wilson {w[0]:.0f}–{w[1]:.0f}%)", (5, 80), (3.1, 94), fontsize=7.5, color=WARN,
            arrowprops=dict(arrowstyle="-", color=WARN, lw=0.6))
ax.annotate(f"Insert 3/10 (Wilson {w2[0]:.0f}–{w2[1]:.0f}%)", (5, 30), (3.0, 12), fontsize=7.5, color=ACC2,
            arrowprops=dict(arrowstyle="-", color=ACC2, lw=0.6))
ax.set_title("표 4 재구성: 단계별 누적 성공 (시행이 10회라 구간이 넓음)", fontsize=9.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_real_stages.png")
plt.close(fig)

# 2. 표 2 영샷 곡선
ADR = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
K = [98.6, 94.1, 84.7, 78.6, 71.9, 67.9, 58.7, 52.5, 39.6, 18.0, 0.0]
F = [94.5, 87.1, 80.3, 76.8, 74.9, 70.4, 66.8, 59.2, 52.3, 22.9, 0.70]
fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=170)
ax.plot(ADR, K, "-o", color=ACC, ms=4, label="Kuka-Allegro")
ax.plot(ADR, F, "-s", color=ACC2, ms=4, label="Flexiv-Sharpa")
ax.axhline(50, color=MUTED, lw=0.6, ls=":")
ax.axvline(20, color=MUTED, lw=0.6, ls="--")
ax.text(20.5, 90, "사후학습 시작 ADR 20\n(영샷 72–75%)", fontsize=7.5, color=MUTED)
ax.annotate("0.70% 는 1024 에피소드로\n불가(7/1024 = 0.68%)", (50, 0.7), (37, 12), fontsize=7.5, color=WARN,
            arrowprops=dict(arrowstyle="-", color=WARN, lw=0.6))
ax.set_xlabel("ADR 수준", fontsize=9)
ax.set_ylabel("영샷 성공률 (%)", fontsize=9)
clean(ax)
ax.legend(fontsize=8, frameon=False)
ax.set_title("표 2 재구성: 사전학습 정책의 영샷 성공률은 ADR 50 에서 0%", fontsize=9.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_zeroshot_adr.png")
plt.close(fig)

# 3. 표 3 절제
labs = ["(a) 전체", "(b) BC 제거", "(c) 워밍업 제거", "(f) 느슨한 클립"]
sr = [(46.0, 1.1), (38.5, 1.2), (28.4, 3.7), (46.8, 2.0)]
tm = [(19.9, 1.0), (35.2, 4.9), (20.2, 1.5), (17.6, 1.8)]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=170)
cols = [ACC, "#8a8a85", "#8a8a85", "#7a9e55"]
for ax, data, ttl, yl in ((axs[0], sr, "학습 성공률 (%)", ""), (axs[1], tm, "ADR 50 도달 시간 (h)", "")):
    ax.bar(range(4), [d[0] for d in data], yerr=[d[1] for d in data], color=cols, capsize=3, error_kw=dict(lw=0.8))
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=7, rotation=18)
    ax.set_title(ttl, fontsize=9)
    clean(ax)
fig.suptitle("표 3 절제 (± 를 SD 로 읽음, 시드 5개; LR 1e-3 행은 모두 0/5라 제외)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "own_ablation.png")
plt.close(fig)

# 4. 손익분기
fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=170)
Ss = [9, 20, 50]
labels = ["논문 표기(8B)", "누적 Kuka(108B)", "누적 Flexiv(432B)"]
k1 = RES["K1_breakeven"]
keys = list(k1.keys())[:3]
wd = 0.26
for i, key in enumerate(keys):
    vals = [k1[key][f"scratch {s}B"] for s in Ss]
    ax.bar(np.arange(3) + (i - 1) * wd, vals, wd, label=labels[i], color=[ACC, ACC2, WARN][i])
    for j, v in enumerate(vals):
        ax.text(j + (i - 1) * wd, v + 1, f"{v:.1f}", ha="center", fontsize=7)
ax.set_xticks(range(3))
ax.set_xticklabels([f"scratch 가 {s}B 스텝에 성공한다고 가정" for s in Ss], fontsize=8)
ax.set_ylabel("손익분기 과제 수 N", fontsize=9)
clean(ax)
ax.legend(fontsize=8, frameon=False)
ax.set_title("사전학습을 몇 개 과제에 나눠야 본전인가 (N > 사전학습/(S-3B))", fontsize=9.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_breakeven.png")
plt.close(fig)
print("wrote 4 figures to", FIG)
