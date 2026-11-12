"""리포트 그래프 (직접 계산한 값만 그린다). 실행: python3 -I analysis/plot_figures.py → report/figures/*.png

입력: analysis/gate_toy_results.json, 그리고 gate_toy.py의 합성 궤적 함수.
"""
import importlib.util
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

FONT_CANDIDATES = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/System/Library/Fonts/AppleSDGothicNeo.ttc"]
font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if font_path:
    font_manager.fontManager.addfont(font_path)
    fam = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = [fam, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

spec = importlib.util.spec_from_file_location("gate_toy", ROOT / "analysis" / "gate_toy.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)
res = json.loads((ROOT / "analysis" / "gate_toy_results.json").read_text())
G = res["mean_over_seeds"]
PENS = res["config"]["pen_grid_mm"]
ACCS = res["config"]["acc_grid_mps2"]
BEST_P99 = (3.0, 2.5)
KIND_KO = {"normal": "정상", "deep_pen": "깊은 관통", "jitter": "떨림", "jump": "튐", "float": "떠 있음"}
COL = {"normal": "#1f5fae", "deep_pen": "#c95a24", "jitter": "#6a4c93", "jump": "#2a9d8f", "float": "#8d8d8d"}


def fig_traces():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    t = np.arange(gt.T) * gt.DT
    for kind in gt.KINDS:
        obj, fingers = gt.scene(rng)
        o, f = gt.corrupt(obj, fingers, rng, kind)
        g = gt.box_sdf(f - o[:, None, :]) - gt.R_FINGER
        depth = np.maximum(-g, 0).max(axis=1) * 1e3
        acc = np.r_[0, 0, np.linalg.norm(np.diff(o, n=2, axis=0), axis=1) * gt.FPS**2]
        axes[0].plot(t, depth, color=COL[kind], lw=1.1, label=KIND_KO[kind])
        axes[1].plot(t, acc, color=COL[kind], lw=1.1)
    axes[0].axhline(BEST_P99[0], color="k", ls="--", lw=0.8)
    axes[0].text(0.1, BEST_P99[0] + 0.4, f"관통 임계 {BEST_P99[0]:g} mm", fontsize=8)
    axes[1].axhline(BEST_P99[1], color="k", ls="--", lw=0.8)
    axes[1].text(0.1, BEST_P99[1] + 0.8, f"가속도 임계 {BEST_P99[1]:g} m/s²", fontsize=8)
    axes[0].set_ylabel("프레임별 최대 관통 깊이 (mm)")
    axes[1].set_ylabel("물체 가속도 크기 (m/s²)")
    axes[1].set_xlabel("시간 (s)")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylim(0, 60)
    axes[0].legend(ncol=5, fontsize=7.5, loc="upper right", frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("합성 궤적 하나씩의 지표 (시드 3, 결함 유형별)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "gate_traces.png", dpi=170)
    plt.close(fig)


def fig_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax = axes[0]
    for stat, ls in [("max", "--"), ("p99", "-")]:
        frr = [G[f"{stat}|{p}|2.5"]["frr_mean"] * 100 for p in PENS]
        far = [G[f"{stat}|{p}|2.5"]["far_mean"] * 100 for p in PENS]
        ax.plot(PENS, frr, ls, color="#1f5fae", marker="o", ms=3, label=f"FRR ({stat})")
        ax.plot(PENS, far, ls, color="#c95a24", marker="s", ms=3, label=f"FAR ({stat})")
    ax.set_xlabel("관통 임계 (mm)")
    ax.set_ylabel("%")
    ax.set_title("관통 임계 변화 (가속도 2.5 m/s²)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    ax = axes[1]
    frr = [G[f"p99|{BEST_P99[0]}|{a}"]["frr_mean"] * 100 for a in ACCS]
    far = [G[f"p99|{BEST_P99[0]}|{a}"]["far_mean"] * 100 for a in ACCS]
    ax.plot(ACCS, frr, "-o", ms=3, color="#1f5fae", label="FRR")
    ax.plot(ACCS, far, "-s", ms=3, color="#c95a24", label="FAR")
    ax.set_xlabel("가속도 p99 임계 (m/s²)")
    ax.set_title("가속도 임계 변화 (관통 3 mm, p99)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "gate_tradeoff.png", dpi=170)
    plt.close(fig)


def fig_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, key, title in [(axes[0], "frr", "정상 기각 비율 FRR (%)"), (axes[1], "far", "결함 수용 비율 FAR (%)")]:
        M = np.array([[G[f"p99|{p}|{a}"][f"{key}_mean"] * 100 for a in ACCS] for p in PENS])
        im = ax.imshow(M, cmap="viridis_r" if key == "frr" else "magma_r", aspect="auto", vmin=0, vmax=max(M.max(), 1))
        ax.set_xticks(range(len(ACCS)), [f"{a:g}" for a in ACCS], fontsize=7)
        ax.set_yticks(range(len(PENS)), [f"{p:g}" for p in PENS], fontsize=7)
        ax.set_xlabel("가속도 임계 (m/s²)")
        ax.set_ylabel("관통 임계 (mm)")
        ax.set_title(title, fontsize=9)
        for i in range(len(PENS)):
            for j in range(len(ACCS)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="w" if M[i, j] < M.max() * 0.6 else "k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "gate_heatmap.png", dpi=170)
    plt.close(fig)


def fig_defects():
    key = f"p99|{BEST_P99[0]}|{BEST_P99[1]}"
    kinds = [k for k in gt.KINDS if k != "normal"]
    far_gate = [np.mean([s["grid"][key]["far_by_kind"][k] for s in res["per_seed"]]) * 100 for k in kinds]
    contact_only = [res["contact_only_accept_rate_mean"][k] * 100 for k in kinds]
    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - 0.2, far_gate, 0.38, color="#c95a24", label="제안 게이트 (4개 지표 모두)")
    ax.bar(x + 0.2, contact_only, 0.38, color="#8d8d8d", label="접촉 규칙만 (관통·떨림·튐 검사 없음)")
    ax.set_xticks(x, [KIND_KO[k] for k in kinds])
    ax.set_ylabel("결함이 통과한 비율 (%)")
    ax.set_ylim(0, 125)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("결함 유형별 통과율 (관통 3 mm, 떨림 2.5 m/s², p99, 시드 3 평균)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "gate_defects.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_traces()
    fig_tradeoff()
    fig_heatmap()
    fig_defects()
    print("ok", sorted(p.name for p in FIG.glob("gate_*.png")))
