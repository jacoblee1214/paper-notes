"""재계산 결과와 SAC 토이의 그래프를 만든다. 입력은 JSON뿐이다.

실행: python3 analysis/plot_figures.py → report/figures/fig_*.png
입력: analysis/recompute_results.json, analysis/sac_stabilizer_toy_results.json, analysis/derivation_checks_results.json
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())
T = json.loads((ROOT / "analysis" / "sac_stabilizer_toy_results.json").read_text())

# 한글 글꼴: Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#8a8a85"
C_FS, C_PPO, C_FTD = "#2a9d4b", "#e0743a", "#3f7fd0"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


# 1) GPU 시뮬레이터 과제별 최종 정규화 점수 (평균 ± sd, 시드)
rows = sorted(R["gpu_state"]["rows"], key=lambda r: r["flashsac_final"])
fig, ax = plt.subplots(figsize=(7.2, 6.0))
y = np.arange(len(rows))
ax.errorbar([r["flashsac_final"] for r in rows], y - 0.22, xerr=[r["flashsac_sd"] for r in rows],
            fmt="o", color=C_FS, ms=4, lw=1, label="FlashSAC (50M)")
ax.scatter([r["ppo200M_final"] for r in rows], y, marker="s", s=14, color=C_PPO, label="PPO (200M)")
ax.scatter([r["fasttd3_final"] for r in rows], y + 0.22, marker="^", s=14, color=C_FTD, label="FastTD3 (50M)")
ax.set_yticks(y)
ax.set_yticklabels([r["env"] for r in rows], fontsize=6.5)
ax.set_xlabel("최종 정규화 점수 (공개 CSV 재계산; 1 = 논문 Normalize Score)")
ax.axvline(1, color=GREY, lw=0.6, ls="--")
ax.legend(loc="lower right", fontsize=7, frameon=False)
ax.set_title("GPU 시뮬레이터 25과제: 최종 성능 (시드 5개, 평균 ± sd)", fontsize=9, color=INK)
save(fig, "fig_gpu_final.png")

# 2) 소거 실험: 정규화 점수 평균(4과제), 기본값 강조
groups = [("buffer", "replay buffer 크기"), ("batch", "배치 크기"), ("width", "네트워크 폭"),
          ("depth", "블록 수"), ("utd", "UTD (업데이트/1024 전이)"), ("exploration", "탐색 소거")]
default_mean = R["ablation_default_mean_norm"]
default_label = {"buffer": "10M", "batch": "2k", "width": "critic256/actor128", "depth": "2",
                 "utd": "2/1024", "exploration": "0.15"}
fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.4))
for ax, (g, title) in zip(axes.flat, groups):
    names = list(R["ablation_scaling_exploration"][g].keys())
    vals = [np.mean([v["mean"] for v in R["ablation_scaling_exploration"][g][n].values()]) for n in names]
    ax.bar(range(len(names)), vals, color=GREY)
    ax.axhline(default_mean, color=ACC, lw=0.8, ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("critic", "c").replace("actor", "a").replace("width_", "")
                        .replace("target_std_", "σ=").replace("_", " ") for n in names], rotation=35, ha="right", fontsize=6)
    ax.set_title(title + f" (기본: {default_label[g]})", fontsize=7.5)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("정규화 점수 평균", fontsize=7)
save(fig, "fig_ablation_scaling.png")

# 3) 안정성 지표: 비판자 조건수 중앙값(로그축), srank
arch = R["ablation_architecture"]
names = list(arch.keys())
cond = [R["condition_number_final_median"][n] for n in names]
srank = [arch[n]["stability_final"]["srank"] for n in names]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
axes[0].bar(range(len(names)), cond, color=[ACC if n == "f_flashSAC" else GREY for n in names])
axes[0].set_yscale("log")
axes[0].set_title("비판자 조건수 중앙값 (마지막 step, 로그)", fontsize=8)
axes[1].bar(range(len(names)), srank, color=[ACC if n == "f_flashSAC" else GREY for n in names])
axes[1].set_title("특징 srank (마지막 step)", fontsize=8)
for ax in axes:
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split("_", 1)[1] for n in names], rotation=35, ha="right", fontsize=7)
save(fig, "fig_stability.png")

# 4) SAC 토이: 평가 반환 곡선(평균 ± sd), 과대추정 지표
cfgs = list(T["summary"].keys())
labels = {"base": "SAC 기준선", "norm": "+ 단위 노름·RMSNorm", "norm_bn": "+ BN(교차 배치)",
          "fs_sched": "+ 큰 배치·적은 업데이트"}
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
for c in cfgs:
    runs = [r for r in T["runs"] if r["config"] == c]
    steps = np.array([p["env_steps"] for p in runs[0]["curve"]])
    ret = np.array([[p["return"] for p in r["curve"]] for r in runs])
    qb = np.array([[p.get("q_bias", np.nan) for p in r["curve"]] for r in runs])
    axes[0].plot(steps, ret.mean(0), label=labels[c])
    axes[0].fill_between(steps, ret.mean(0) - ret.std(0), ret.mean(0) + ret.std(0), alpha=0.15)
    axes[1].plot(steps, qb.mean(0), label=labels[c])
    t = [r["time_total_s"] for r in runs]
    axes[2].errorbar([np.mean(t)], [np.mean([r["final_return"] for r in runs])],
                     yerr=[np.std([r["final_return"] for r in runs], ddof=1)], fmt="o", label=labels[c])
axes[0].set_title("평가 반환 (평균 ± sd, 3 시드)", fontsize=8)
axes[0].set_xlabel("환경 step")
axes[1].axhline(0, color=GREY, lw=0.6)
axes[1].set_title("Q(s0,π) - MC 반환 (과대추정 > 0)", fontsize=8)
axes[1].set_xlabel("환경 step")
axes[2].set_title("최종 반환 대 CPU 시간 (1 스레드)", fontsize=8)
axes[2].set_xlabel("CPU 벽시계 시간 (s)")
axes[2].set_ylabel("최종 반환")
axes[0].legend(fontsize=6.5, frameon=False)
save(fig, "fig_toy_sac.png")

# 5) 실물 전 단계: 거친 지형 curriculum 단계, 환경 step (시드 1개)
import pandas as pd  # noqa: E402

RES = ROOT / "repo" / "results" / "sim2real"
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for rel, lab, col in [("flashsac_g1rough.csv", "FlashSAC (거친 지형)", C_FS), ("ppo_g1rough.csv", "PPO (거친 지형)", C_PPO)]:
    d = pd.read_csv(RES / rel, index_col=0)
    s = d[d.metric == "avg_terrain_level"].sort_values("env_step")
    ax.plot(s.env_step / 1e6, s.value.rolling(50, min_periods=1).mean(), color=col, label=lab)
ax.set_xscale("log")
ax.set_xlabel("환경 step (M, 로그)")
ax.set_ylabel("지형 단계 (이동 평균)")
ax.set_title("실물 전 단계 학습 곡선: 시드 1개", fontsize=8.5)
ax.legend(fontsize=7, frameon=False)
save(fig, "fig_sim2real_rough.png")
