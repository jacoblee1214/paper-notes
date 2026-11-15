"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_real_ci.png    표 1 실로봇 과제별 성공률과 Wilson 95% 구간 (과제당 20회) + 전체 42% 대 76%
  own_sim_ci.png     표 1 시뮬레이션 평균과 3 시드 기준 t 95% 구간
  own_toy_pose.png   토이 C: 포즈 오차 축별 성공률 (세 학습 설정, 기반 정책, 해석해 상한)

입력: analysis/recompute_results.json, analysis/toy_pose_residual_results.json
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
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_pose_residual_results.json").read_text(encoding="utf-8"))

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

BLUE, GREY, ORANGE = "#1f5fae", "#8a8a85", "#c77a12"


def real_ci():
    tasks = [r["task"] for r in RC["real_per_task"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=180)
    x = np.arange(len(tasks))
    w = 0.36
    for off, key, color, label in ((-w / 2, "base", GREY, "기반 VLA"), (w / 2, "res", BLUE, "기반 + 잔차")):
        vals = [r[key][0] / 20 * 100 for r in RC["real_per_task"]]
        lo = [r["wilson_" + key][0] * 100 for r in RC["real_per_task"]]
        hi = [r["wilson_" + key][1] * 100 for r in RC["real_per_task"]]
        yerr = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
        ax.bar(x + off, vals, w, color=color, label=label, yerr=yerr, capsize=3, error_kw={"lw": 0.8})
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 9, f"{v:.0f}", ha="center", fontsize=7.5, color="#333")
    ax.set_xticks(x, [t.replace("Stand Cup Up", "Stand").replace("Close Drawer", "Close")
                      .replace("Pick-and-Place", "PnP") for t in tasks], fontsize=8)
    ax.set_ylabel("실로봇 성공률 (%)", fontsize=8.5)
    ax.set_ylim(0, 118)
    ax.set_title("표 1 실로봇: 과제당 20회, Wilson 95% 구간", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    h = RC["headline_real"]
    ax.text(0.99, 0.02, f"전체 {h['base'][0]}% 대 {h['res'][0]}% (100회), Fisher p = {h['fisher_p']:.1e}",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#333")
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


def sim_ci():
    from scipy import stats
    tc = stats.t.ppf(0.975, 2)
    tasks = [r["task"] for r in RC["sim_table1"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=180)
    x = np.arange(len(tasks))
    w = 0.36
    for off, key, color, label in ((-w / 2, "base", GREY, "기반 VLA"), (w / 2, "res", BLUE, "기반 + 잔차")):
        means = [r[key][0] / 20 * 100 for r in RC["sim_table1"]]
        hw = [tc * r[key][1] / np.sqrt(3) / 20 * 100 for r in RC["sim_table1"]]
        ax.bar(x + off, means, w, color=color, label=label, yerr=hw, capsize=3, error_kw={"lw": 0.8})
    ax.set_xticks(x, [t.replace("Stand Cup Up", "Stand").replace("Close Drawer", "Close")
                      .replace("Pick-and-Place", "PnP") for t in tasks], fontsize=8)
    ax.set_ylabel("시뮬레이션 성공률 (%)", fontsize=8.5)
    ax.set_ylim(0, 118)
    ax.set_title("표 1 시뮬레이션: 3 시드 평균, t 95% 구간 (±를 표준편차로 가정)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png")
    plt.close(fig)


def toy():
    sweep = TOY["sweep"]
    axes = [("iid_sd_m", "포즈 iid 오차 σ (m)", "tab:blue"), ("episode_bias_sd_m", "에피소드 편향 σ_β (m)", "tab:red"),
            ("dropout_p", "드롭아웃 p", "tab:green")]
    names = {"none": "증강 없음", "aug": "논문식 증강 (iid 5 mm, p 0.1)", "aug+bias": "증강 + 편향 1 cm"}
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.3), dpi=180, sharey=True)
    for ax, (key, label, _) in zip(axs, axes):
        for mode, color in (("none", "#555"), ("aug", BLUE), ("aug+bias", ORANGE)):
            rows = sweep[mode][key]
            ax.plot([r["value"] for r in rows], [r["success"] for r in rows], "o-", color=color, ms=4, lw=1.3,
                    label=names[mode])
        ax.axhline(TOY["base_only"]["success"], color=GREY, ls="--", lw=1, label="기반 정책만")
        ax.axhline(TOY["oracle_analytic_clean_pose"]["success"], color="#2a8a4a", ls=":", lw=1,
                   label="해석해 상한 (깨끗한 포즈)")
        ax.set_xlabel(label, fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("성공률 (1000 에피소드)", fontsize=8.5)
    axs[0].set_ylim(0, 1.02)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.suptitle("토이 C: 포즈 오차 축별 성공률 (numpy 2차원 점 그리퍼)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.savefig(FIG / "own_toy_pose.png")
    plt.close(fig)


if __name__ == "__main__":
    real_ci()
    sim_ci()
    toy()
    print("wrote figures to", FIG)
