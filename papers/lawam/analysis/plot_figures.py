"""우리가 그린 그래프 네 장 (report/figures/own_*.png). 값은 recompute.py의 표 값을 그대로 가져온다.

  own_robotwin_means.png : RoboTwin 청정·무작위 평균과 Wilson 95% 구간(5,000회), 원 논문 보고값 표시
  own_realworld_ci.png   : 실로봇 여섯 조건의 성공률과 Wilson 95% 구간 (LaWAM, π0.5, Fast-WAM)
  own_ablation_ci.png    : LIBERO-10 절제 (500회 가정) 성공률과 Wilson 95% 구간
  own_latency_vs_sr.png  : 표 1의 지연 대 LIBERO 평균 (보고값)

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

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
INK, ACCENT, GREY, ORANGE = "#1b1b1a", "#1f5fae", "#8a8880", "#c4651a"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


# 1. RoboTwin
t = rc.OUT["V4_robotwin"]["per_method"]
names = ["LaWAM", "Fast-WAM", "LingBot-VA", "Motus", "GigaWorld", "pi0.5"]
labels = ["LaWAM", "Fast-WAM*", "LingBot-VA*", "Motus", "GigaWorld-Policy", "π0.5"]
orig = {"Fast-WAM": (91.88, 91.78), "LingBot-VA": (92.93, 91.55)}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
for ax, key, title, oi in ((axes[0], "clean_successes_of_5000", "청정 장면", 0),
                          (axes[1], "rand_successes_of_5000", "무작위 장면", 1)):
    for i, (n, lab) in enumerate(zip(names, labels)):
        k = t[n][key]
        lo, hi = rc.wilson(k, 5000)
        p = 100 * k / 5000
        c = ACCENT if n == "LaWAM" else GREY
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=c, ecolor=c, capsize=3, ms=6)
        ax.text(p + 0.6, i - 0.28, f"{p:.2f}", fontsize=8, color=INK)
        if n in orig:
            ax.plot(orig[n][oi], i, marker="D", mfc="white", mec=ORANGE, ms=7, ls="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(len(names) - 0.3, -0.7)
    ax.set_xlim(74, 96)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("성공률 (%) · Wilson 95% 구간 (5,000회)", fontsize=9, color=INK)
    style(ax)
axes[1].plot([], [], marker="D", mfc="white", mec=ORANGE, ls="none", label="원 논문 보고값 (* 표 2는 재평가)")
axes[1].legend(loc="lower left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "own_robotwin_means.png", dpi=170)
plt.close(fig)

# 2. 실로봇
conds = ["Pick-and-Place", "Open Drawer", "Fold Towel", "Fold Towel-Light", "Clean Table", "Clean Table-Light"]
clab = ["집어 놓기 (30)", "서랍 열기 (30)", "수건 접기 (30)", "수건 접기-Light (50)", "테이블 치우기 (50)", "테이블 치우기-Light (50)"]
meths = [("pi0.5", "π0.5", GREY, -0.2), ("Fast-WAM", "Fast-WAM", "#b9b7ae", 0.0), ("LaWAM", "LaWAM", ACCENT, 0.2)]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
for j, c in enumerate(conds):
    n, d = rc.T3[c]
    for key, lab, col, off in meths:
        k = d[key]
        lo, hi = rc.wilson(k, n)
        p = 100 * k / n
        ax.errorbar(p, j + off, xerr=[[p - lo], [hi - p]], fmt="o", color=col, ecolor=col, capsize=2.5, ms=5,
                    label=lab if j == 0 else None)
ax.set_yticks(range(len(conds)))
ax.set_yticklabels(clab, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-2, 102)
ax.set_xlabel("성공률 (%) · Wilson 95% 구간 (괄호는 시행 수)", fontsize=9, color=INK)
ax.legend(fontsize=8, frameon=False, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_realworld_ci.png", dpi=170)
plt.close(fig)

# 3. 절제
ab = rc.T4
order = ["Full LaWAM", "w/o Stage-2 Pretraining", "LA Distillation Only", "w/o Subgoal Conditioning", "Direct Future Prediction",
         "w/o KI", "w/o LA Distillation", "Future Prediction", "w/o KI & LA Distillation", "Capacity-Matched Adapter", "Base VLA"]
kolabel = {"Full LaWAM": "전체 (Stage-2 사전학습 포함)", "w/o Stage-2 Pretraining": "Stage-2 사전학습 없음 (기준)",
           "LA Distillation Only": "증류만", "w/o Subgoal Conditioning": "부분 목표 조건 제거",
           "Direct Future Prediction": "직접 미래 예측 (표 6)", "w/o KI": "KI 제거", "w/o LA Distillation": "증류 제거",
           "Future Prediction": "미래 예측 보조만", "w/o KI & LA Distillation": "KI·증류 제거",
           "Capacity-Matched Adapter": "용량 일치 어댑터 (표 6)", "Base VLA": "기본 VLA"}
fig, ax = plt.subplots(figsize=(8.0, 4.6))
ref = ab["w/o Stage-2 Pretraining"]
for i, k in enumerate(order):
    v = ab[k]
    lo, hi = rc.wilson(round(v * 5), 500)
    col = ACCENT if k == "w/o Stage-2 Pretraining" else GREY
    ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=col, ecolor=col, capsize=2.5, ms=5)
    if k != "w/o Stage-2 Pretraining":
        p = rc.fisher(round(ref * 5), 500, round(v * 5), 500)
        ax.text(99.3, i, f"{v:.1f}  (기준과 p " + ("<0.01" if p < 0.01 else f"={p:.2f}") + ")", fontsize=7.5, color=INK, va="center")
    else:
        ax.text(99.3, i, f"{v:.1f}  (기준)", fontsize=7.5, color=INK, va="center")
ax.axvline(ref, color=ACCENT, lw=0.8, ls=":")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([kolabel[k] for k in order], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(86, 108)
ax.set_xticks([88, 90, 92, 94, 96, 98, 100])
ax.set_xlabel("LIBERO-10 성공률 (%) · Wilson 95% 구간 (500회 가정, Fisher p는 독립 가정)", fontsize=9, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_ablation_ci.png", dpi=170)
plt.close(fig)

# 4. 지연 대 LIBERO 평균
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for m, r in rc.T1.items():
    if r[5] is None:
        continue
    col = ACCENT if m == "LaWAM" else GREY
    ax.scatter(r[5], r[4], s=60, color=col, zorder=3)
    off = {"pi0": (5, -11), "pi0.5": (-30, 5)}.get(m, (5, 4))
    ax.annotate(m.replace("pi0", "π0"), (r[5], r[4]), textcoords="offset points", xytext=off, fontsize=8, color=INK)
ax.set_xscale("log")
ax.set_xticks([200, 500, 1000, 2000, 5000])
ax.set_xticklabels(["200", "500", "1,000", "2,000", "5,000"])
ax.minorticks_off()
ax.set_xlabel("모델 지연 (ms, 로그 축, 표 1: 방법별 기본 설정)", fontsize=9, color=INK)
ax.set_ylabel("LIBERO 평균 성공률 (%)", fontsize=9, color=INK)
ax.set_ylim(93.5, 99.3)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_latency_vs_sr.png", dpi=170)
plt.close(fig)
print("saved figures to", FIG)
