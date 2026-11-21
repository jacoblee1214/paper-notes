"""논문 수치와 recompute.py 결과로 그래프 두 장을 그린다 (우리가 그린 그림, 논문 그림 아님).

출력: report/figures/own_baseline_ci.png  (기준선 비교, Wilson 95% 구간)
      report/figures/own_vla_ci.png       (VLA 과제 성공률, Wilson 95% 구간, FSQ 대 SMPL)
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
FIG.mkdir(parents=True, exist_ok=True)


def pick_font():
    for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:  # noqa: BLE001
            continue
    return "DejaVu Sans"


plt.rcParams["font.family"] = pick_font()
plt.rcParams["axes.unicode_minus"] = False
INK, GREY = "#1b1b1a", "#8a8a85"
ACC = {"SONIC": "#1f5fae", "BeyondMimic": "#9aa7b8", "Any2Track": "#c9ced8"}


def wilson(k, n, z=1.959964):
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return c - h, c + h


# ---------- 그림 1: 기준선 비교 (그림 2 D의 수치, 개수는 백분율 × N) ----------
splits = [("test_content", "Test-content (OOD)", 7016),
          ("test_repetition", "Test-repetition", 9395),
          ("phuma", "PHUMA", 68326)]
rates = {
    "SONIC": [98.5, 99.2, 97.2],
    "BeyondMimic": [82.0, 85.4, 73.8],
    "Any2Track": [61.6, 69.4, 78.5],
}
fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=170)
w = 0.26
for j, method in enumerate(["SONIC", "BeyondMimic", "Any2Track"]):
    xs, ys, lo_err, hi_err = [], [], [], []
    for i, (_, _, n) in enumerate(splits):
        p = rates[method][i]
        k = round(p / 100 * n)
        lo, hi = wilson(k, n)
        xs.append(i + (j - 1) * w)
        ys.append(100 * k / n)
        lo_err.append(100 * k / n - 100 * lo)
        hi_err.append(100 * hi - 100 * k / n)
    ax.bar(xs, ys, width=w * 0.92, color=ACC[method], label=method, edgecolor="none")
    ax.errorbar(xs, ys, yerr=[lo_err, hi_err], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    for x, y in zip(xs, ys):
        ax.text(x, y + 2.2, f"{y:.1f}", ha="center", va="bottom", fontsize=6.8, color=INK)
ax.set_xticks(range(len(splits)))
ax.set_xticklabels([s[1] for s in splits], fontsize=8)
ax.set_ylim(0, 120)
ax.set_ylabel("성공률 (%)", fontsize=8)
ax.tick_params(axis="y", labelsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.01, "막대: 논문 본문 백분율 × N. 오차 막대: Wilson 95% 구간. 논문의 P < 0.001과 맞음.",
         fontsize=6.5, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_baseline_ci.png")
plt.close(fig)

# ---------- 그림 2: VLA 과제 성공률 (표 1 A, Wilson 95%) 과 FSQ 대 SMPL (표 1 B, 가정 n) ----------
tasks = RES["V2_vla_table1A"]["tasks"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.5), dpi=170, gridspec_kw={"width_ratios": [1.55, 1]})
labels = [t["task"].replace("Object pickup ", "pickup ").replace(" relocation", "") for t in tasks]
ys = [100 * t["rate"] for t in tasks]
lo_e = [100 * t["rate"] - 100 * t["wilson95"][0] for t in tasks]
hi_e = [100 * t["wilson95"][1] - 100 * t["rate"] for t in tasks]
xs = range(len(tasks))
a1.bar(xs, ys, color="#1f5fae", width=0.6, edgecolor="none")
a1.errorbar(xs, ys, yerr=[lo_e, hi_e], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
for x, t, y in zip(xs, tasks, ys):
    a1.text(x, 100 * t["wilson95"][1] + 2, f"{y:.0f}%\nn={t['n']}", ha="center", va="bottom", fontsize=6.2, color=INK)
a1.set_xticks(list(xs))
a1.set_xticklabels(labels, rotation=25, ha="right", fontsize=6.8)
a1.set_ylim(0, 128)
a1.set_ylabel("성공률 (%)", fontsize=7.5)
a1.tick_params(axis="y", labelsize=7)
a1.spines[["top", "right"]].set_visible(False)
a1.set_title("VLA 과제 성공률 (Wilson 95%)", fontsize=8.5, color=INK)

v3 = RES["V3_fsq_vs_smpl_table1B"]["tasks"]
names = ["carrot", "trash", "soda"]
disp = ["pickup (carrot)", "trash can", "soda → trash"]
for i, (nm, lab) in enumerate(zip(names, disp)):
    f_k, f_n = v3[nm]["fsq"]
    s_k, s_n = v3[nm]["smpl"]
    for off, (k, n), col in ((-0.17, (f_k, f_n), "#1f5fae"), (0.17, (s_k, s_n), "#b5654a")):
        p = k / n
        lo, hi = wilson(k, n)
        a2.errorbar([i + off], [100 * p], yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="o",
                    color=col, ms=4.5, elinewidth=0.9, capsize=2)
a2.set_xticks(range(3))
a2.set_xticklabels(disp, rotation=25, ha="right", fontsize=6.8)
a2.set_ylim(-5, 105)
a2.set_xlim(-0.6, 2.6)
a2.spines[["top", "right"]].set_visible(False)
a2.tick_params(axis="y", labelsize=7)
a2.set_title("FSQ(파랑) 대 SMPL(주황)", fontsize=8.5, color=INK)
a2.text(0.02, 0.02, "SMPL 시도 수는 논문에 없어\n표 1 A 시도 수로 가정", transform=a2.transAxes,
        fontsize=5.8, color=GREY)
fig.text(0.01, 0.01, "Wilson 95% 구간. 과제별 시도 수가 10–20회라 구간이 넓습니다.", fontsize=6.3, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_vla_ci.png")
plt.close(fig)
print("wrote", FIG / "own_baseline_ci.png", "and", FIG / "own_vla_ci.png")
