"""리포트·게시본용 그래프 네 장 (직접 계산한 결과만 사용; 논문 그림은 쓰지 않음).

1) study_forest.png: 사용자 연구 쌍별 차이와 Welch 95% 구간 (독립 근사, 쌍체 정보 없음)
2) success_counts.png: Fig. 3 성공률 막대에서 읽은 참가자별 성공 횟수 분포 (합계·평균 검산 포함)
3) contact_pareto.png: 2D 접촉 toy, 정상상태 접촉력 대 접선 추적 오차 (등방 스윕 vs 방향별)
4) label_toy.png: 라벨 toy, (a) 방향 편향·이방성 비율, (b) 접촉 거리 관측 여부에 따른 식별 오차

실행: python3 analysis/plot_figures.py   → report/figures/*.png
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
US = json.loads((ROOT / "analysis" / "user_study_check_results.json").read_text())
CT = json.loads((ROOT / "analysis" / "contact_toy_results.json").read_text())
LT = json.loads((ROOT / "analysis" / "label_toy_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def study_forest():
    rows = []
    for metric, label in (("force", "최대 접촉력"), ("time", "과제 시간")):
        for task in ("Vase", "Peg", "Door"):
            for pr in US["continuous"][metric][task]["pairs"]:
                if pr["pair"] in ("SC-Low", "SC-High"):
                    rows.append((f"{label} · {task} · {pr['pair']}", pr))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ys = np.arange(len(rows))[::-1]
    for y, (name, pr) in zip(ys, rows):
        lo, hi = pr["ci95_diff_welch"]
        ax.plot([lo, hi], [y, y], color=BLUE, lw=2)
        ax.plot(pr["diff"], y, "o", color=ORANGE, ms=5)
        ax.text(hi + 1.5, y, f"p={pr['holm_p_welch']:.2g}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("차이 = SC - 비교 조건 (N 또는 초), Welch 95% 구간, 표시 p는 과제·지표 안 Holm 보정값")
    ax.set_title("사용자 연구: 쌍별 차이 (독립 근사, 쌍체 자료 없음)", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "study_forest.png", dpi=200)
    plt.close(fig)


def success_counts():
    tasks = ("Vase", "Peg", "Door")
    conds = ("Low", "SC", "High")
    colors = ["#cfe8f5", "#8fcbe6", "#3fa2d4", "#0b6fa8"]  # 0/3 → 3/3
    labels = ["0/3", "1/3", "2/3", "3/3"]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), sharey=True)
    for ax, task in zip(axes, tasks):
        bottom = np.zeros(3)
        for k in range(4):
            vals = np.array([US["success"][task][c]["counts_k0_to_3"][k] for c in conds])
            ax.bar(range(3), vals, bottom=bottom, color=colors[k], edgecolor="white", lw=0.6, label=labels[k])
            for i, v in enumerate(vals):
                if v >= 2:
                    ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center", fontsize=7.5,
                            color="white" if k >= 2 else INK)
            bottom += vals
        means = [US["success"][task][c]["successes_of_54"] / 54 for c in conds]
        ax.set_title(f"{task}: 성공률 " + " / ".join(f"{m:.2f}" for m in means), fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Low", "SC", "High"])
        style(ax)
    axes[0].set_ylabel("참가자 수 (합 18)")
    axes[2].legend(title="성공 시행", fontsize=7, title_fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("참가자별 성공 횟수 분포 (Fig. 3 막대에서 읽음; 평균·SD가 Table I과 일치)", fontsize=9, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "success_counts.png", dpi=200)
    plt.close(fig)


def contact_pareto():
    g = CT["C_pareto"]
    grid = g["k_grid"]
    iso, dr = g["isotropic"], g["directional_kt3000"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(iso["Fss_N_mean"], iso["track_rms_mm_mean"], "o-", color=INK2, ms=4, lw=1.2, label="등방 K 스윕 (k = 300…3000)")
    ax.plot(dr["Fss_N_mean"], dr["track_rms_mm_mean"], "s-", color=BLUE, ms=4, lw=1.2, label="방향별: k_n 스윕, k_t = 3000")
    for name, key, col in (("고정 저 300", "fixed low (300/300)", AQUA), ("고정 고 3000", "fixed high (3000/3000)", ORANGE),
                           ("방향별 (300 법선, 3000 접선)", "directional (300 normal / 3000 tangent)", BLUE)):
        v = CT["A_named_delta1cm"][key]
        ax.plot(v["Fss_N_mean"], v["track_rms_mm_mean"], "*", color=col, ms=14, mec=INK, mew=0.5)
        ax.annotate(name, (v["Fss_N_mean"], v["track_rms_mm_mean"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color=INK)
    ax.set_xlabel("정상상태 접촉력 (N), 2–3 s 평균 (조작 과오버슛 δ = 1 cm)")
    ax.set_ylabel("접선 추적 오차 RMS (mm)")
    ax.set_title("2D 접촉 toy: 같은 접촉력에서 추적 오차가 갈린다", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "contact_pareto.png", dpi=200)
    plt.close(fig)


def label_toy():
    a = LT["partA_label_estimators"]
    b = LT["partB_identifiability"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    names = ["논문\n(log 평균)", "공분산\n(Duan식)"]
    vals = [a["skew1"]["paper_log_label"]["aniso_ratio"], a["skew1"]["covariance_duan_label"]["aniso_ratio"]]
    ax1.bar(names, vals, color=[BLUE, MUTED], width=0.55)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    eps_keys = list(a["eps_sweep_skew1"].keys())
    ax1.set_ylabel("K 고유값 비 λmax/λmin")
    ax1.set_title("(a) 같은 힘 자료, 방향 편향과 이방성", fontsize=9, loc="left")
    ins = ax1.inset_axes([0.58, 0.55, 0.4, 0.4])
    ins.plot(range(len(eps_keys)), [a["eps_sweep_skew1"][k] for k in eps_keys], "o-", color=BLUE, ms=3)
    ins.set_xticks(range(len(eps_keys)))
    ins.set_xticklabels(eps_keys, fontsize=6)
    ins.set_title("ε(N²) 스윕", fontsize=6.5)
    ins.tick_params(labelsize=6)
    style(ax1)
    ids = [("영상: 각도만", b["angle_only_no_distance"]),
           ("각도 + 거리 σ=0.1", b["angle_plus_distance_noise0.1"]),
           ("각도 + 거리 σ=0.02", b["angle_plus_distance_noise0.02"]),
           ("θ와 접촉 c 직접", b["oracle_theta_and_contact_c"])]
    ax2.barh([x[0] for x in ids][::-1], [x[1] for x in ids][::-1], color=[AQUA, BLUE, BLUE, MUTED][::-1], height=0.55)
    ax2.set_xlabel("kNN 예측 오차 (정규화 ‖K‖_F)")
    ax2.set_title("(b) 접촉 상태가 영상에 있는가", fontsize=9, loc="left")
    style(ax2)
    fig.tight_layout()
    fig.savefig(OUT / "label_toy.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    study_forest()
    success_counts()
    contact_pareto()
    label_toy()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
