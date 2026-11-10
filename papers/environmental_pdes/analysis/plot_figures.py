"""environmental_pdes 직접 그린 그래프 세 장 (CPU, 1스레드). 실행: python3 analysis/plot_figures.py

1) toy_localization.png: 위치추정 toy의 설정별 RMSE(평균 ± SD, 30회)  ← pde_toy_results.json
2) toy_symmetry.png: 회전 판별력 D(θ)(왼쪽)와 자세 피셔 정보 고유값(오른쪽)  ← obs_toy_results.json
3) toy_field_decay.png: 참값 장의 최대값과 기울기 RMS의 시간 변화(장이 흩어져 정보가 줄어드는 과정)  ← pde_toy.solve
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "axes.formatter.use_mathtext": True})
INK, GRID = "#0b0b0b", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = (6.5, 3.6)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_localization():
    R = json.loads((ROOT / "analysis" / "pde_toy_results.json").read_text())["results"]
    order = [("asym_match", "일치 κ\n(시간 의존)", BLUE),
             ("asym_kmis_0.5x", "κ×0.5", ORANGE), ("asym_kmis_2x", "κ×2", ORANGE),
             ("asym_kmis_0.25x", "κ×0.25", ORANGE), ("asym_kmis_4x", "κ×4", ORANGE),
             ("asym_static_t0", "정적 지도\n(t=0 고정)", MUTED),
             ("asym_pf_nosensor", "PF, 센서 없음", MUTED),
             ("odometry_only", "주행기록계만", MUTED),
             ("sym_match", "대칭 장\n일치", AQUA), ("sym_static_t0", "대칭 장\n정적", MUTED)]
    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(order))
    means = [R[k]["rmse_mean"] for k, _, _ in order]
    sds = [R[k]["rmse_sd"] for k, _, _ in order]
    ax.bar(x, means, yerr=sds, color=[c for _, _, c in order], capsize=3, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in order], fontsize=6.6)
    ax.set_ylabel("위치 RMSE (m), 30회 평균 ± SD")
    style(ax)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "toy_localization.png", dpi=200)
    plt.close(fig)


def fig_symmetry():
    O = json.loads((ROOT / "analysis" / "obs_toy_results.json").read_text())["results"]
    thetas = np.rad2deg(np.linspace(-np.pi / 2, np.pi / 2, len(O["rotation_asym"]["D_curve"])))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(SIZE[0], 3.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a1.plot(thetas, np.maximum(O["rotation_asym"]["D_curve"], 1e-4), color=BLUE, label="비대칭 장")
    a1.plot(thetas, np.maximum(O["rotation_sym"]["D_curve"], 1e-4), color=ORANGE, label="방사 대칭 장")
    a1.axhline(5.99, color=MUTED, ls="--", lw=0.8)
    a1.text(-88, 8, "χ² 95% (2자유도)", color=MUTED, fontsize=7)
    a1.set_yscale("log")
    a1.set_yticks([1e-4, 1e-2, 1, 1e2])
    a1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlabel("경로 회전 θ (도), 중심 (10, 10) m")
    a1.set_ylabel("판별력 D(θ) / σ²")
    a1.legend(frameon=False, fontsize=7)
    style(a1)
    keys = [("pose_fim_asym_bias_known", "비대칭\n편향 없음", BLUE),
            ("pose_fim_asym_bias_unknown", "비대칭\n편향 미지", BLUE),
            ("pose_fim_sym_bias_known", "대칭\n편향 없음", ORANGE),
            ("pose_fim_sym_bias_unknown", "대칭\n편향 미지", ORANGE)]
    for i, (k, lab, c) in enumerate(keys):
        ev = O[k]["eig"]
        a2.bar(i, max(ev[0], 1e-4), color=c, edgecolor="none")
    a2.set_yscale("log")
    a2.set_yticks([0.01, 0.1, 1.0])
    a2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xticks(range(len(keys)))
    a2.set_xticklabels([lab for _, lab, _ in keys], fontsize=6.8)
    a2.set_ylabel("최소 고유값 λ_min (1/σ² 단위)")
    style(a2)
    fig.tight_layout()
    fig.savefig(OUT / "toy_symmetry.png", dpi=200)
    plt.close(fig)


def fig_field_decay():
    import pde_toy as pt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(SIZE[0], 3.0))
    for scen, c in (("asym", BLUE), ("sym", ORANGE)):
        F, _ = pt.solve(pt.N_TRUE, pt.KAPPA, scen)
        tt = np.arange(F.shape[0])
        dx = pt.L / pt.N_TRUE
        gmax = [np.sqrt(np.mean(sum(g**2 for g in np.gradient(F[t], dx)))) for t in tt]
        a1.plot(tt, F.max(axis=(1, 2)), color=c, label="비대칭" if scen == "asym" else "방사 대칭")
        a2.plot(tt, gmax, color=c)
    a1.set_xlabel("시간 (s)")
    a1.set_ylabel("장의 최대값")
    a1.legend(frameon=False, fontsize=7)
    a2.set_xlabel("시간 (s)")
    a2.set_ylabel("기울기 RMS (1/m)")
    a2.set_yscale("log")
    a2.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1])
    a2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(a1)
    style(a2)
    fig.tight_layout()
    fig.savefig(OUT / "toy_field_decay.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_localization()
    fig_symmetry()
    fig_field_decay()
    print("wrote", sorted(p.name for p in OUT.glob("toy_*.png")))
