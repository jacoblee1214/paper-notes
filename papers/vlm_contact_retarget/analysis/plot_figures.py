"""논문 수치로 우리가 그린 그래프 네 장 (report/figures/own_*.png).

입력: analysis/recompute_results.json (표 3, 7, 8의 읽은 값과 계산 결과), 표 4 상수(5, 8, 5, 5, 8 / 10).
논문 그림은 쓰지 않는다.
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

KO_FONT = next((f.name for f in font_manager.fontManager.ttflist
                if any(k in f.name for k in ("Apple SD Gothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun"))), "DejaVu Sans")
plt.rcParams["font.family"] = KO_FONT
plt.rcParams["axes.unicode_minus"] = False
INK, GREY = "#1b1b1a", "#8a8880"
ACTS = ["hug", "high-five", "hand-shake", "link-arms", "pat-on-back"]
KO = {"hug": "hug", "high-five": "high-five", "hand-shake": "hand-shake", "link-arms": "link-arms", "pat-on-back": "pat-on-back"}


def style(ax):
    ax.grid(axis="y", color="#e5e3dc", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# 그림 1: 표 3 학생 절제 (보류 분할, 30 cm 접촉 재현율)
T3 = RES["parsed_tables"]["table3"]
variants = [("Ours", "#1f5fae"), ("(-) future generation", "#6b9bd1"), ("generation → regression", "#c9a227"), ("(-) pose prediction", GREY)]
fig, ax = plt.subplots(figsize=(9.4, 4.2))
w = 0.2
x = np.arange(len(ACTS))
for i, (v, c) in enumerate(variants):
    vals = [T3[v][a] for a in ACTS]
    ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=v)
    if i == 0:
        for xi, val in zip(x, vals):
            ax.text(xi + (i - 1.5) * w, val + 1.2, f"{val:.1f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(ACTS, fontsize=9)
ax.set_ylabel("보류 분할 접촉 재현율 @30 cm (%)")
ax.set_ylim(0, 95)
ax.set_title("학생 정책 절제 (표 3): 전체 모델이 가장 높은 행동은 5개 중 3개", fontsize=10.5, color=INK)
ax.legend(fontsize=8, frameon=False, ncol=2)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_student_ablation.png", dpi=170)
plt.close(fig)

# 그림 2: 학생의 훈련 분할(표 7) 대 보류 분할(표 3), 30 cm, 교사 대비
T7 = RES["parsed_tables"]["table7"]
km = {"hug": "Hug", "high-five": "High-five", "hand-shake": "Hand-shake", "link-arms": "Link-arms", "pat-on-back": "Pat-on-back"}
fig, ax = plt.subplots(figsize=(9.4, 4.2))
w = 0.26
for i, (lab, vals, c) in enumerate([
        ("교사 (표 7, 훈련 모션)", [T7[km[a]]["T30"] for a in ACTS], GREY),
        ("학생, 훈련 분할 (표 7)", [T7[km[a]]["S30"] for a in ACTS], "#6b9bd1"),
        ("학생, 보류 분할 (표 3)", [T3["Ours"][a] for a in ACTS], "#1f5fae")]):
    ax.bar(x + (i - 1) * w, vals, w, color=c, label=lab)
    for xi, val in zip(x, vals):
        ax.text(xi + (i - 1) * w, val + 1.0, f"{val:.0f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(ACTS, fontsize=9)
ax.set_ylabel("접촉 재현율 @30 cm (%)")
ax.set_ylim(0, 108)
ax.set_title("'교사의 75%'는 훈련 분할 값이다 (보류 분할에서는 평균 약 54%, high-five 15%)", fontsize=10.5, color=INK)
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_teacher_student_split.png", dpi=170)
plt.close(fig)

# 그림 3: 실로봇 성공 횟수 (표 4)의 Wilson 95% 구간과 시뮬 보류 재현율
real = {"high-five": 5, "hug": 8, "hand-shake": 5, "link-arms": 5, "pat-on-back": 8}
W = RES["V10_real_robot_table4"]["wilson95"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
order = ["hug", "high-five", "hand-shake", "link-arms", "pat-on-back", "total"]
for i, k in enumerate(order):
    kk, n, p, lo, hi = W[k]
    col = "#1f5fae" if k != "total" else INK
    ax.errorbar(i, p * 100, yerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="o", color=col, ms=7, capsize=4, lw=1.6)
    ax.text(i + 0.12, hi * 100 + 1.5, f"{kk}/{n}", fontsize=8.5, color=INK)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, fontsize=8.5)
ax.set_ylim(0, 105)
ax.set_ylabel("실로봇 성공률 (%)")
ax.set_title("실로봇 성공 횟수 (표 4)와 Wilson 95% 구간", fontsize=10, color=INK)
style(ax)
ax = axes[1]
sx = [T3["Ours"][a] for a in real]
sy = [real[a] * 10 for a in real]
ax.scatter(sx, sy, s=45, color="#1f5fae")
for a, xx, yy in zip(real, sx, sy):
    ax.annotate(a, (xx, yy), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlim(0, 100)
ax.set_ylim(30, 100)
ax.set_xlabel("시뮬 보류 접촉 재현율 @30 cm (%)")
ax.set_ylabel("실로봇 성공률 (%)")
ax.set_title("시뮬 재현율과 실로봇 성공률 (n=5)", fontsize=10, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_real_robot_wilson.png", dpi=170)
plt.close(fig)

# 그림 4: 표 8 F1 보고값과 옌센 상한
B = RES["V11_table8_label_quality"]["bound_checks"]
names = ["Hug", "Hand-shake", "Pat-on-back", "High-five", "Link-arms", "Total"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
for ax, tag, title in zip(axes, ["distance", "vlm"], ["거리 후보만 (Ours w/o VLM filtering)", "VLM 검증 (Ours)"]):
    xs = np.arange(len(names))
    rep = [B[f"{n}_{tag}"]["F1_reported"] for n in names]
    up = [B[f"{n}_{tag}"]["max_possible_mean_F1"] for n in names]
    hm = [B[f"{n}_{tag}"]["harmonic_of_means"] for n in names]
    ax.bar(xs - 0.2, rep, 0.4, color="#1f5fae", label="보고된 평균 F1")
    ax.bar(xs + 0.2, up, 0.4, color="#c9a227", label="가능한 최대 평균 F1 (평균 P·R 기준)")
    ax.scatter(xs + 0.2, hm, color=INK, s=14, zorder=3, label="평균 P·R의 조화평균")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7.5, rotation=20)
    ax.set_title(title, fontsize=9.5, color=INK)
    style(ax)
axes[0].set_ylabel("F1")
axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("표 8: 12개 행 중 10개에서 보고된 평균 F1이 상한 2P/(1+P) (P=평균 정밀도)을 넘는다", fontsize=10.5, color=INK, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "own_f1_bound.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
