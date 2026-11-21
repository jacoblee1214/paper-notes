"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_heldout_error.png   독립 검증 오차 분포와 임계 0.08 (V3)
  own_repeat_consistency.png  같은 후보로 봐야 할 두 방법의 교정 오차 쌍 (V2)
  own_paired_heldout.png  파지 held-out 성공률 (Wilson 95% CI)과 쌍별 불일치 (V5)

입력: repo/paper/analysis_data/filter_long.csv, application_long.csv (공식 저장소의 파생 CSV)
실행: python3 -I analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repo" / "paper" / "analysis_data"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

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


def rd(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1. 독립 검증 오차 분포
flt = rd("filter_long.csv")
ok = np.array([float(r["heldout_error"]) for r in flt if r["status_success"] == "True"])
bad = np.array([float(r["heldout_error"]) for r in flt if r["status_success"] == "False"])
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
bins = np.linspace(0, 0.6, 61)
ax.hist(ok, bins=bins, color=ACC, alpha=0.85, label=f"통과 {len(ok):,}건")
ax.hist(bad, bins=bins, color=WARN, alpha=0.85, label=f"실패 {len(bad):,}건")
ax.axvline(0.08, color=INK, lw=1, ls="--")
ax.text(0.085, ax.get_ylim()[1] * 0.92, "임계 0.08", color=INK, fontsize=8)
ax.set_xlabel("독립 검증 오차 (식 4, 300 bag × 28 실행)", fontsize=8.5)
ax.set_ylabel("실행 수", fontsize=8.5)
ax.legend(frameon=False, fontsize=8)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_heldout_error.png")
plt.close(fig)

# 2. 같은 후보로 봐야 할 두 방법의 교정 오차 (피드백 없는 fixed profile, one-shot prior)
idx = {(r["sample_id"], r["budget_allowed"], r["method"]): r for r in flt}
xs, ys = [], []
for (s, b, m), r in idx.items():
    if m != "fixed_profile":
        continue
    o = idx.get((s, b, "one_shot_prior"))
    if o is None:
        continue
    xs.append(float(r["calibration_error"]))
    ys.append(float(o["calibration_error"]))
xs, ys = np.array(xs), np.array(ys)
fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=200)
ax.scatter(xs, ys, s=10, color=ACC, alpha=0.6, edgecolor="none")
lo, hi = min(xs.min(), ys.min()) * 0.8, max(xs.max(), ys.max()) * 1.2
ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("fixed profile의 교정 오차", fontsize=8.5)
ax.set_ylabel("one-shot prior의 교정 오차", fontsize=8.5)
ax.text(0.02, 0.92, f"같은 값: {int(np.sum(xs == ys))} / {len(xs):,} 그룹", transform=ax.transAxes, fontsize=8, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_repeat_consistency.png")
plt.close(fig)

# 3. 파지 held-out 성공률 (Wilson 95%)과 쌍별 불일치
app = rd("application_long.csv")
names = {"fixed": "고정 명령", "reactive_slip": "슬립 피드백", "deligrasp_prior": "힘 사전", "privileged_oracle": "특권 기준"}
held = defaultdict(dict)
for r in app:
    if r["split"] == "hidden_test":
        held[r["method"]][r["benchmark_instance_id"]] = r["task_success"] == "True"
labels, rates, los, his = [], [], [], []
for m in ["fixed", "reactive_slip", "deligrasp_prior", "privileged_oracle"]:
    k = sum(held[m].values())
    n = len(held[m])
    lo_, hi_ = wilson(k, n)
    labels.append(names[m])
    rates.append(100 * k / n)
    los.append(100 * (k / n - lo_))
    his.append(100 * (hi_ - k / n))
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=200)
y = np.arange(len(labels))[::-1]
ax.errorbar(rates, y, xerr=[los, his], fmt="o", color=ACC, ecolor=ACC, capsize=3, ms=5)
for yi, r_ in zip(y, rates):
    ax.text(r_, yi + 0.22, f"{r_:.1f}%", ha="center", fontsize=8, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlim(40, 85)
ax.set_xlabel("held-out 작업 성공률 (%, Wilson 95%)", fontsize=8.5)
ax.text(0.99, 0.02, "고정 명령 대 슬립 피드백: 불일치 0 / 340\n고정 명령 대 힘 사전: 불일치 45 / 0 (모두 고정 명령 쪽)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=MUTED)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_paired_heldout.png")
plt.close(fig)
print("wrote 3 figures to", FIG)
