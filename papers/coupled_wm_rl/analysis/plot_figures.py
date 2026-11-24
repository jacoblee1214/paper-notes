"""우리가 만든 그래프를 report/figures/ 에 저장한다(논문 그림 조각은 여기서 만들지 않는다).

  own_toy_alignment.png   토이: 호라이즌별 기울기 코사인(국소 단독 대 분리)
  own_cost_ratio.png      그림 8 판독 값: 과제별 표본 비·벽시계 비·GPU 가중 비
  own_agg_ends.png        그림 5 집계 곡선의 끝점이 과제별 끝점의 평균이 아니라 최솟값과 맞는 정도(그림 판독 값)
  own_success_ci.png      표 1–3 성공 횟수와 Wilson 95% 구간(시행 10회)

입력: analysis/toy_decoupled_gradient_results.json, recompute_results.json, digitize_fig8_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TOY = json.loads((ROOT / "analysis" / "toy_decoupled_gradient_results.json").read_text(encoding="utf-8"))
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

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
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1) 토이: 기울기 코사인
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
for ax, row in zip(axes, TOY["결과"]):
    Hs = list(row["호라이즌별"].keys())
    x = [int(h) for h in Hs]
    ax.plot(x, [row["호라이즌별"][h]["A 국소 단독 코사인"] for h in Hs], "o-", color=WARN, label="국소 모델 단독 롤아웃")
    ax.plot(x, [row["호라이즌별"][h]["B 분리 코사인"] for h in Hs], "s-", color=ACC, label="분리(전방 참, 야코비안 국소)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_title(f"{row['국소 모델']}\n1스텝 RMSE {row['1스텝 RMSE(분포 내)']}", fontsize=8.5)
    ax.set_xlabel("호라이즌 H")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("참 기울기와의 코사인(평균)")
axes[0].legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "own_toy_alignment.png", dpi=170)
plt.close(fig)

# 2) 그림 8 판독: 과제별 비용 비
v5 = RC["V5_그림판독_과제별"]
tasks = list(v5.keys())
fig, ax = plt.subplots(figsize=(7.4, 3.5))
w = 0.2
keys = [("표본 비(PPO/DMO)", "표본 수 비 (PPO ÷ DMO)", ACC),
        ("벽시계 비(PPO/DMO)", "벽시계 시간 비 (PPO ÷ DMO)", ACC2),
        ("GPU-초 비(PPO/DMO, GPU 종류 무시)", "GPU 수 곱한 시간 비 (PPO ÷ DMO)", "#4b8b4b"),
        ("표본당 비용 비(DMO/PPO)", "표본당 시간 비 (DMO ÷ PPO)", WARN)]
for i, (k, lab, col) in enumerate(keys):
    vals = [v5[t][k] for t in tasks]
    b = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=lab, color=col)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.15, f"{v:g}", ha="center", fontsize=7.5)
ax.axhline(1, color=MUTED, lw=0.8, ls="--")
ax.set_xticks(range(3))
ax.set_xticklabels(tasks)
ax.set_ylabel("비율 (1이면 같음)")
ax.set_title("그림 8 판독 값으로 본 DMO와 PPO의 비용 (그림 판독 값, 표 4의 GPU 수는 셀 위치 가정)", fontsize=8.5)
ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "own_cost_ratio.png", dpi=170)
plt.close(fig)

# 3) 그림 5 끝점
v6 = RC["V6_그림5_끝점"]
labs = ["DMO 표본", "DMO 시간", "SAC 표본", "SAC 시간", "NoRSSM 시간"]
keys = ["DMO_s", "DMO_t", "SAC_s", "SAC_t", "NoRSSM_t"]
fig, ax = plt.subplots(figsize=(7.4, 3.4))
for i, k in enumerate(keys):
    per = v6["과제별 정규화 끝점 목록"][k]
    ax.scatter([i] * 3, per, color=MUTED, s=26, zorder=3, label="과제별 끝점(그림 8 판독)" if i == 0 else None)
    ax.plot([i - 0.22, i + 0.22], [v6["과제 평균"][k]] * 2, color=ACC2, lw=2, label="세 과제 평균" if i == 0 else None)
    ax.scatter([i], [v6["그림 5 집계 곡선 끝점(판독)"][k]], marker="D", color=ACC, s=40, zorder=4, label="그림 5 집계 곡선의 끝(판독)" if i == 0 else None)
ax.set_xticks(range(5))
ax.set_xticklabels(labs)
ax.set_ylabel("PPO 끝으로 나눈 정규화 위치")
ax.set_title("그림 5의 곡선 끝은 과제별 끝점의 최솟값과 맞음 (그림 판독 값)", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_agg_ends.png", dpi=170)
plt.close(fig)

# 4) 성공 횟수와 Wilson
rows = [("Push-T DMO", 9), ("Push-T PPO", 1), ("Push-T PPO(완화 기준)", 4), ("Push-T No Diffusion", 0),
        ("Push Cube DMO", 9), ("Push Cube BC†", 6), ("Push Cube PPO", 4), ("Push Cube No Diffusion", 0),
        ("G1 DMO", 8), ("G1 BC", 7), ("G1 PPO", 1), ("G1 No Diffusion", 0)]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for i, (name, k) in enumerate(rows):
    lo, hi = wilson(k, 10)
    ax.plot([lo, hi], [i, i], color=ACC, lw=2)
    ax.scatter([k / 10], [i], color=ACC2, zorder=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("성공률 (시행 10회, Wilson 95% 구간)")
ax.set_title("표 1–3의 성공 횟수는 구간이 넓게 겹침 (†BC는 목표 근처 도달을 성공으로 정의)", fontsize=8.8)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_success_ci.png", dpi=170)
plt.close(fig)
print("wrote 4 figures")
