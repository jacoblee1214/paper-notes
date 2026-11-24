"""우리가 만든 그래프를 report/figures/에 저장한다 (논문 그림과 별개).

own_table1_ci.png : 표 1 (v2) 셀별 성공률과 Wilson 95% 구간(n=20 가정), 물체 9개 x 4개 조건
own_v1_v2.png     : 표 1의 v1 대 v2 변경 셀 6개
own_vlm_ci.png    : 표 4 Qwen3-VL 라벨 정확도와 Wilson 95% 구간(n=10)
own_fig_readings.png : 그림 4·5 판독 근사값에서 본 공격 성공 수준 (우연 수준과 비교)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
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
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#c2570c", "#9a9890"
OBJ = ["Bomb", "Dynamite", "Gun", "Oil", "Cardboard", "CerealBox", "Fork", "Knife", "Grater"]

# 1) 표 1 셀별 CI
ci = R["V2_cell_wilson_ci"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, cam, title in zip(axes, ("ood", "id"), ("카메라 분포 밖 (OOD)", "카메라 분포 안 (ID)")):
    x = np.arange(9)
    for off, pr, col in ((-0.2, "vague", ACC), (0.2, "specific", ORG)):
        rows = ci[f"{cam}_{pr}"]
        y = np.array([r["pct"] for r in rows], float)
        lo = y - np.array([r["wilson95"][0] for r in rows])
        hi = np.array([r["wilson95"][1] for r in rows]) - y
        ax.errorbar(x + off, y, yerr=[lo, hi], fmt="o", ms=4, lw=1.2, color=col, label="모호한 프롬프트" if pr == "vague" else "구체적 프롬프트", capsize=2)
    base = np.array([r["pct"] for r in ci[f"{cam}_none"]], float)
    ax.plot(x, base, "_", color=GREY, ms=14, mew=2, label="공격 없음")
    ax.set_xticks(x)
    ax.set_xticklabels(OBJ, rotation=40, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("공격 성공률 (%)")
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("표 1: 셀별 성공률과 Wilson 95% 구간 (시행 수 20 가정, 무공격 행 n도 20 가정)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "own_table1_ci.png", dpi=170)
plt.close(fig)

# 2) v1 대 v2
chg = R["V5_v1_vs_v2_table1"]["changed_cells"]
labels = {"ood_vague": "OOD 모호", "ood_specific": "OOD 구체", "id_vague": "ID 모호"}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for i, c in enumerate(chg):
    ax.plot([i, i], [c["v1"], c["v2"]], color=GREY, lw=1.5)
    ax.plot(i, c["v1"], "o", color=GREY, label="v1" if i == 0 else None)
    ax.plot(i, c["v2"], "o", color=ACC, label="v2" if i == 0 else None)
ax.set_xticks(range(len(chg)))
ax.set_xticklabels([f'{c["obj"]}\n{labels[c["cond"]]}' for c in chg], fontsize=8)
ax.set_ylabel("공격 성공률 (%)")
ax.set_title("표 1에서 v1과 v2 값이 다른 셀 6개 (54셀 중, 나머지 48셀은 같음)", fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_v1_v2.png", dpi=170)
plt.close(fig)

# 3) 표 4
cells = R["V6_table4_vlm_accuracy"]["cells"]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
y = np.array([c["pct"] for c in cells], float)
lo = y - np.array([c["wilson95"][0] for c in cells])
hi = np.array([c["wilson95"][1] for c in cells]) - y
ax.errorbar(range(9), y, yerr=[lo, hi], fmt="o", color=ACC, capsize=3)
ax.set_xticks(range(9))
ax.set_xticklabels(OBJ, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("VLM 라벨 정확도 (%)")
ax.set_title("표 4: Qwen3-VL 라벨 정확도와 Wilson 95% 구간 (물체당 10개)", fontsize=10)
ax.set_ylim(0, 105)
ax.grid(axis="y", color="#e5e3dc", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_vlm_ci.png", dpi=170)
plt.close(fig)

# 4) 그림 4/5 판독
f4 = R["V8_fig4_readings_approx"]
f5 = R["V8_fig5_readings_approx"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
names = ["무작위 행동\n(균일 가정)", "무오염 정책\n(그림 4 끝)", "VTH 정책\n(그림 4 끝)", "VTH 정책\n(최고점)"]
vals = [R["V8_attack_success_chance_level"]["uniform_random_hit_prob"], f4["attack_success_none_end"], f4["attack_success_vth_end"], f4["attack_success_vth_peak"]]
ax.bar(names, vals, color=[GREY, GREY, ACC, ACC])
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("공격 성공률 (목표 행동 +-0.2 이내)")
ax.set_title("Dino WM 백도어 (그림 판독 값)", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax = axes[1]
ax.plot(f5["x"], f5["embedding_l2_mean"], "o-", color=ACC)
ax.fill_between(f5["x"], [35.5, 37.0, 35.7, 27.0, 25.3], [54.0, 56.5, 48.5, 45.0, 43.7], color=ACC, alpha=0.15)  # 띠 가장자리 그림 판독 값
ax.set_ylim(0, 60)
ax.axhline(0, color=INK, lw=0.6)
ax.set_xlabel("입력 행동 값 (모든 차원)")
ax.set_ylabel("깨끗한 생성과의 임베딩 L2 거리")
ax.set_title("Cosmos AC (그림 5, 축을 0에서 시작)", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_fig_readings.png", dpi=170)
plt.close(fig)
print("saved", sorted(p.name for p in FIG.glob("own_*.png")))
