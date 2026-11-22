"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sr_ci.png     : 과제별 성공률(ID, OOD)과 Wilson 95% 구간. 로봇만 대 +EgoVerse, 방법 5개
own_pvalues.png   : 로봇만 대 +EgoVerse 성공률 차이의 Fisher 정확 검정 p값 (방법별, 과제별)
own_egomotion.png : 토이, 안정화 방식별 배경 앵커의 흐름 크기와 임계값 통과 비율
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 recompute.py, toy_egomotion.py 실행)
한글 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#d9822b"
METHODS = ["bc", "pixel", "pixelpt", "dino", "flow"]
NAMES = {"bc": "BC", "pixel": "Pixel", "pixelpt": "Pixel-PT", "dino": "DINO", "flow": "3D Flow"}
TASKS = {"cup": "컵 올리기", "bag": "봉지 담기", "fold": "옷 접기"}


def plot_sr():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    rows = rec["V4_sr_base_vs_egoverse"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0), sharey=True)
    for j, task in enumerate(TASKS):
        for i, split in enumerate(("id", "ood")):
            ax = axes[i, j]
            x = np.arange(len(METHODS))
            for off, key, col, lab in ((-0.19, "base", GREY, "로봇만"), (0.19, "flag", ACC, "+EgoVerse")):
                ys, lo, hi = [], [], []
                for m in METHODS:
                    r = next(z for z in rows if z["task"] == task and z["split"] == split and z["method"] == m)
                    k, n = (r["k_base"], r["n_base"]) if key == "base" else (r["k_flag"], r["n_flag"])
                    w = r["wilson_base"] if key == "base" else r["wilson_flag"]
                    ys.append(100 * k / n)
                    lo.append(100 * k / n - 100 * w[0])
                    hi.append(100 * w[1] - 100 * k / n)
                ax.bar(x + off, ys, 0.36, color=col, label=lab, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.9})
            ax.set_xticks(x)
            ax.set_xticklabels([NAMES[m] for m in METHODS], fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_title(f"{TASKS[task]} / {'ID' if split == 'id' else 'OOD'}", fontsize=10)
            ax.grid(axis="y", lw=0.3)
            if j == 0:
                ax.set_ylabel("성공률 (%)")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("성공률과 Wilson 95% 구간 (ID 20회, OOD 19~20회)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_sr_ci.png", dpi=170)
    plt.close(fig)


def plot_p():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    rows = rec["V4_sr_base_vs_egoverse"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    labels, ps, cols = [], [], []
    for task in TASKS:
        for split in ("id", "ood"):
            for m in METHODS:
                r = next(z for z in rows if z["task"] == task and z["split"] == split and z["method"] == m)
                labels.append(f"{TASKS[task]} {'ID' if split == 'id' else 'OOD'} {NAMES[m]}")
                ps.append(max(r["fisher_p"], 1e-4))
                cols.append(ACC if r["delta_pp"] > 0 else ("#b3342b" if r["delta_pp"] < 0 else GREY))
    y = np.arange(len(ps))[::-1]
    ax.scatter(ps, y, c=cols, s=24)
    ax.axvline(0.05, color=ORG, lw=1.2)
    ax.text(0.055, y.max() + 0.7, "p = 0.05", color=ORG, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0.0001 이하", "0.001", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xlabel("Fisher 정확 검정 p (로봇만 대 +EgoVerse, 로그 축)")
    n_sig = sum(1 for q in ps if q < 0.05)
    ax.set_title(f"{len(ps)}개 비교 중 p < 0.05는 {n_sig}개 (파랑: 증가, 빨강: 감소, 회색: 변화 없음)", fontsize=10)
    ax.grid(axis="x", lw=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_pvalues.png", dpi=170)
    plt.close(fig)


def plot_ego():
    t = json.loads((ANA / "toy_egomotion_results.json").read_text(encoding="utf-8"))
    names = [("no_stabilization", "안정화 없음"), ("rgb_only_pose", "RGB만 추정한 포즈\n(회전 1.9도, 이동 1.8 cm 오차)"),
             ("vio_exact", "정확한 포즈 (VIO)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    vals = [t[k]["static_median_mm"] for k, _ in names]
    ax.bar(range(3), vals, color=[GREY, ORG, ACC])
    ax.axhline(10, color="#b3342b", lw=1.1)
    ax.text(2.45, 11.5, "임계값 10 mm", color="#b3342b", fontsize=8, ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for _, n in names], fontsize=7.5)
    ax.set_ylabel("정지 배경 앵커의 흐름 크기, 중앙값 (mm)")
    ax.set_title("배경에 남는 흐름", fontsize=10)
    ax = axes[1]
    vals = [100 * t[k]["static_pass_threshold_frac"] for k, _ in names]
    ax.bar(range(3), vals, color=[GREY, ORG, ACC])
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for _, n in names], fontsize=7.5)
    ax.set_ylabel("임계값을 넘는 정지 앵커 비율 (%)")
    ax.set_ylim(0, 105)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_title("마스크가 정지 배경을 못 거름", fontsize=10)
    fig.suptitle("토이: 머리 회전 4.7도, 이동 2.5 cm, 깊이 0.5 m, 앵커 1120개, 300회 평균", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_egomotion.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_sr()
    plot_p()
    plot_ego()
    print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
