"""우리가 만든 그래프를 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_top3.png          표 1·2에서 방법별 상위 3위 횟수(AUCPR, AUCPDT 따로)
  own_seed_vs_rollout.png  논문 표 1의 시드 구간 폭 대 롤아웃 재표본 부트스트랩(토이 1)
  own_stac_floor.png    X-VLA에서 STAC의 AUCPR 대 실패율(표 1, 표 16)과 신호 없는 검출기 기준선
  own_norm_pdt.png      AUCPDT를 최댓값 1 - 실패율로 나눈 값(표 1·2, 표 15–17의 성공률 사용)
  own_cov_toy.png       LLMD 공분산 토이(토이 3): 유효 표본 수와 λ 에 따른 OOD 분리

입력: analysis/recompute_results.json, toy_metrics_results.json, toy_llmd_cov_results.json
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
TM = json.loads((ROOT / "analysis" / "toy_metrics_results.json").read_text(encoding="utf-8"))
TL = json.loads((ROOT / "analysis" / "toy_llmd_cov_results.json").read_text(encoding="utf-8"))

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
METHODS = ["ACE", "Diff", "STAC", "ACC", "LLMD", "FAIL"]
COLORS = ["#9a9890", "#b9b7ae", "#c9c7bf", "#7a4a1d", "#4d3a2a", ACC2]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)


def fig_top3():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, key, n in [(axes[0], "V1_top3_real", 12), (axes[1], "V1_top3_libero", 8)]:
        s = RC[key]["summary"]
        x = np.arange(len(METHODS))
        w = 0.38
        ax.bar(x - w / 2, [s[m]["top3_PR"] for m in METHODS], w, color=ACC, label="AUCPR 상위 3위")
        ax.bar(x + w / 2, [s[m]["top3_PDT"] for m in METHODS], w, color=ACC2, label="AUCPDT 상위 3위")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_ylim(0, n + 1)
        ax.axhline(n, color=MUTED, lw=0.8, ls=":")
        ax.set_title(("실제 로봇 6과제 x 2모델 = 12칸" if n == 12 else "Libero-Plus 4스위트 x 2모델 = 8칸"), fontsize=10, color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("칸 수 (공동 순위 포함)", color=MUTED)
    fig.suptitle("표 1·2: 방법별 상위 3위에 든 칸 수 (평균값 2자리 기준, 우리가 센 값)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_top3.png", dpi=170)
    plt.close(fig)


def fig_seed_vs_rollout():
    T1 = RC["V5_ci_T1"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    cases = TM["T1"]
    names = [c["case"].replace(" (", "\n(") for c in cases]
    x = np.arange(len(cases))
    for ax, key, title in [(axes[0], "approx_95_halfwidth_AUCPR", "AUCPR"), (axes[1], None, "AUCPDT")]:
        if key:
            vals = [c[key] for c in cases]
        else:
            vals = [round(1.96 * c["AUCPDT_boot_sd"], 3) for c in cases]
        ax.bar(x, vals, color=ACC2)
        ax.axhline(T1["median_width"] / 2, color=ACC, lw=1.6, label="표 1 시드 구간 반폭의 중앙값(약 0.005)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5)
        ax.set_title(f"{title}: 롤아웃을 다시 뽑을 때 95% 반폭(토이)", fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("반폭", color=MUTED)
    fig.suptitle("롤아웃 표본에서 오는 변동(토이)과 논문의 시드 구간 비교. 성공·실패 수는 표 15·16에서 복원한 값(시행 수 가정)", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_seed_vs_rollout.png", dpi=170)
    plt.close(fig)


def fig_stac_floor():
    inf = RC["V8"]["stac_xvla_infer_failure_rate"]
    tasks = list(inf)
    pi = [inf[t]["pi_from_T16_success_rate"] for t in tasks]
    pr = [inf[t]["STAC_AUCPR"] for t in tasks]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.text(0.62, 0.66, "신호 없음 = 실패율", color=MUTED, fontsize=8, rotation=38)
    for t, a, b in zip(tasks, pi, pr):
        ax.scatter(a, b, color=ACC2 if t != "Cups" else ACC, s=36, zorder=3)
        ax.annotate(t, (a, b), textcoords="offset points", xytext=(5, -9), fontsize=8, color=INK)
    ax.set_xlim(0.2, 0.9)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("실패율 (1 - 성공률, 표 16)", color=MUTED)
    ax.set_ylabel("STAC AUCPR (표 1, X-VLA)", color=MUTED)
    ax.set_title("X-VLA의 STAC AUCPR은 실패율 아래에 놓입니다(5과제)", fontsize=9.5, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_stac_floor.png", dpi=170)
    plt.close(fig)


def fig_norm_pdt():
    rows = []
    for k, x in {**RC["V8_normalized_pdt_real"], **RC["V8_normalized_pdt_libero"]}.items():
        rows.append((k, [x["AUCPDT_over_(1-pi)"][m] for m in METHODS], x["pi"]))
    mat = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    im = ax.imshow(mat, cmap="YlOrBr", vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]} (실패율 {r[2]:.2f})" for r in rows], fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_title("AUCPDT / (1 - 실패율): 1은 탐지가 하나도 없는 값(우리가 계산)", fontsize=9.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "own_norm_pdt.png", dpi=170)
    plt.close(fig)


def fig_cov():
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    lams = sorted({r["lambda_rel_to_mean_eig"] for r in TL})
    ns = sorted({r["n_eff"] for r in TL})
    for lam, col in zip(lams, [ACC2, ACC, "#6b9a52", MUTED]):
        ys = [next(r["AUROC_OOD_vs_ID"] for r in TL if r["shift_c"] == 0.5 and r["n_eff"] == n and r["lambda_rel_to_mean_eig"] == lam) for n in ns]
        ax.plot(ns, ys, marker="o", color=col, label=f"λ = {lam:g} x 평균 고유값")
    ax.axvline(1024, color=MUTED, ls=":", lw=1)
    ax.text(1100, 0.55, "n = d = 1024", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("유효 표본 수 n (학습 특징)", color=MUTED)
    ax.set_ylabel("OOD 대 ID AUROC", color=MUTED)
    ax.set_title("LLMD 공분산 토이(d = 1024, 이동 크기 0.5)", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_cov_toy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_top3()
    fig_seed_vs_rollout()
    fig_stac_floor()
    fig_norm_pdt()
    fig_cov()
    print("wrote figures to", FIG)
