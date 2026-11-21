"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_success_ci.png  : 표 1 평균 성공률과 풀링 Wilson 95% 구간 (80회)
own_mae_diff_ci.png : 표 S7 행동 오차 차이 (WiTac-NW 빼기 WiTac) 와 논문 보고 95% 구간
own_noise_red.png   : 표 3 노이즈 상대 감소 (특징 드리프트, GR00T·π0.5 행동 MAE)
입력: analysis/recompute_results.json (recompute.py가 만든 값)
실행: python3 analysis/plot_figures.py   (matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#b23b3b"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


# 1. 표 1 성공률 (풀링 구간)
t1 = RES["table1_average_recomputed"]
order = ["FTP-1", "T-Rex", "GR00T base", "GR00T + WiTac-NW", "GR00T + VT-MUSE", "GR00T + WiTac",
         "pi05 base", "pi05 + WiTac-NW", "pi05 + WiTac"]
labels = ["FTP-1", "T-Rex", "GR00T 기본", "GR00T + WiTac-NW", "GR00T + VT-MUSE", "GR00T + WiTac",
          "π0.5 기본", "π0.5 + WiTac-NW", "π0.5 + WiTac"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, key in enumerate(order):
    v = t1[key]
    lo, hi = v["pooled_ci95_wilson_pct"]
    mean = v["avg_pct_recomputed"]
    color = RED if "WiTac" in key and "NW" not in key else (ACC if key.startswith("GR00T") or key.startswith("pi") else GREY)
    ax.plot([lo, hi], [i, i], color=color, lw=2.2, solid_capstyle="round")
    ax.plot(mean, i, "o", color=color, ms=6)
    ax.text(hi + 1.2, i, f"{mean + 1e-9:.1f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(20, 85)
ax.set_xlabel("과제 평균 성공률 (%), 과제 4개 × 20회 풀링 Wilson 95% 구간", fontsize=9, color=INK)
ax.set_title("표 1 재계산: 평균과 구간 (검정은 본문 수치 밖)", fontsize=10.5, color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_success_ci.png", dpi=200)
plt.close(fig)

# 2. 표 S7 행동 오차 차이 (NW − WiTac, 도) 와 논문 보고 구간
s7 = RES["tableS7_noise"]
keys = list(s7.keys())
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, k in enumerate(keys):
    lo, hi = s7[k]["reported_CI_diff"]
    d = s7[k]["dMAE_diff_NW_minus_W"]
    sig = not (lo <= 0 <= hi)
    color = RED if sig and d < 0 else (ACC if sig else GREY)
    ax.plot([lo, hi], [i, i], color=color, lw=2.2, solid_capstyle="round")
    ax.plot(d, i, "o", color=color, ms=6)
ax.axvline(0, color=INK, lw=0.8, ls="--")
ax.set_yticks(range(len(keys)))
ax.set_yticklabels([k.replace("pi05", "π0.5").replace("insert", "배터리").replace("wipe", "그릇")
                    .replace("cabinet", "캐비닛") for k in keys])
ax.invert_yaxis()
ax.set_xlabel("노이즈 ΔMAE 감소: WiTac-NW 빼기 WiTac (도). 양수는 WiTac 우위", fontsize=9, color=INK)
ax.set_title("표 S7: 가우시안 촉각 노이즈 (σ = 3)에서의 행동 오차 차이와 95% 구간", fontsize=10, color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_mae_diff_ci.png", dpi=200)
plt.close(fig)

# 3. 표 3 노이즈 상대 감소 (특징 드리프트 = 정책 공통, GR00T·π0.5 행동 MAE)
tasks = ["배터리", "그릇", "USB", "캐비닛"]
gd = [37.4, 35.6, 37.0, 44.1]
g_gr = [22.76, 7.35, 9.18, 21.59]
g_pi = [-0.56, -1.90, 1.91, 2.19]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
y = range(len(tasks))
h = 0.26
ax.barh([i - h for i in y], gd, height=h, color=ACC, label="특징 드리프트 (정책 공통)")
ax.barh([i for i in y], g_gr, height=h, color="#7a9cc6", label="GR00T 행동 MAE")
ax.barh([i + h for i in y], g_pi, height=h, color=GREY, label="π0.5 행동 MAE")
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(list(y))
ax.set_yticklabels(tasks)
ax.invert_yaxis()
ax.set_xlabel("WiTac-NW 대비 상대 감소 (%). 음수는 악화 (σ = 3)", fontsize=9, color=INK)
ax.set_title("표 3: 촉각 노이즈에서 WiTac의 상대 감소", fontsize=10.5, color=INK, loc="left")
ax.legend(fontsize=8, frameon=False, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_noise_red.png", dpi=200)
plt.close(fig)

print("wrote figures to", FIG)
