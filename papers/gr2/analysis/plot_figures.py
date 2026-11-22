"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_calvin_ci.png    CALVIN 연속 k과제 성공률(그림 10 숫자)과 1000 사슬 가정의 95% Wilson 구간
  own_trials_ci.png    논문에 없는 시도 수 n에 따른 95% 구간 폭 (71.4%, 55.8%)
  own_scaling_read.png 규모 실험(그림 11d) 눈 읽기 값과 ±3 pp 띠

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
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

INK, MUTED, ACC, ACC2 = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def wilson_pp(p_pct, n):
    p = p_pct / 100
    z = 1.959964
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


# 1) CALVIN 연속 k과제 (그림 10 숫자, n = 1000 가정)
calvin = R["C3_calvin"]
ks = [r["tasks_in_row"] for r in calvin]
gr2 = [r["GR-2_pct"] for r in calvin]
lo = [wilson_pp(v, 1000)[0] for v in gr2]
hi = [wilson_pp(v, 1000)[1] for v in gr2]
fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=200)
ax.errorbar(ks, gr2, yerr=[np.array(gr2) - lo, np.array(hi) - np.array(gr2)], fmt="o-", color=ACC,
            ecolor=ACC, capsize=3, label="GR-2 (논문 그림 10 숫자)")
ax.plot([1, 5], [94.9, 73.1], "s--", color=ACC2, label="GR-1 (본문 1과제·5과제 값)")
ax.set_xticks(ks)
ax.set_xlabel("연속으로 완료한 과제 수 k")
ax.set_ylabel("성공률 (%)")
ax.set_ylim(60, 102)
ax.set_title("CALVIN ABCD→D: 연속 k과제 성공률 (n = 1000 가정, 95% 구간)", fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "own_calvin_ci.png")
plt.close(fig)

# 2) 시도 수 n에 따른 구간 폭 (논문에 n이 없다)
ns = np.array([10, 20, 30, 50, 75, 100, 200, 400])
fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=200)
for p, color, label in [(71.4, ACC, "GR-2 Unseen BG 71.4%"), (55.8, ACC2, "GR-2 Unseen Manip. 55.8%")]:
    w = np.array([wilson_pp(p, n)[1] - wilson_pp(p, n)[0] for n in ns])
    ax.plot(ns, w, "o-", color=color, label=label)
ax.axhline(10, color=MUTED, lw=0.8, ls=":")
ax.text(ns[-1], 10.8, "10 pp", color=MUTED, fontsize=7, ha="right")
ax.set_xscale("log")
ax.set_xlabel("설정당 시도 수 n (논문에 없음, 가정)")
ax.set_ylabel("95% 구간 폭 (pp)")
ax.set_title("시도 수에 따른 95% 구간 폭: n이 작으면 구간이 수십 pp", fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "own_trials_ci.png")
plt.close(fig)

# 3) 규모 실험 (그림 11d 눈 읽기)
sc = R["D1_scaling"]
params = np.array(sc["params_M"])
succ = np.array(sc["succ_read_pct"])
fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=200)
ax.fill_between(params, succ - 3, succ + 3, color=ACC, alpha=0.15, lw=0, label="읽기 오차 ±3 pp")
ax.plot(params, succ, "o-", color=ACC, label="그림 11(d) 눈 읽기")
for x, y, name in zip(params, succ, ["S", "B", "L", "XL"]):
    ax.annotate(f"GR-2-{name}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=7, color=MUTED)
ax.set_xscale("log")
ax.set_xlabel("학습 가능 파라미터 수 (M, 로그 축)")
ax.set_ylabel("실로봇 성공률 (%)")
ax.set_ylim(20, 75)
ax.set_title("규모 실험: 네 점, 오차막대와 설정 수 없음", fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "own_scaling_read.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
