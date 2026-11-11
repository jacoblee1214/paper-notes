"""toy 결과 그래프 두 장을 report/figures/에 만든다. 실행: python3 -I analysis/plot_figures.py
입력: analysis/sysid_toy_results.json (sysid_toy.py 결과). 그래프 안 글자는 영어(폰트 문제 회피)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sysid_toy as S  # noqa: E402

FIG = ROOT / "report" / "figures"
R = json.loads((Path(__file__).resolve().parent / "sysid_toy_results.json").read_text())
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# 그림 1: 식별성 (상관 행렬 + 상대 표준편차)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.1]})
C = np.array(R["identifiability"]["corr_CRLB"])
im = ax[0].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
ax[0].set_xticks(range(4), S.NAMES)
ax[0].set_yticks(range(4), S.NAMES)
for i in range(4):
    for j in range(4):
        ax[0].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if abs(C[i, j]) > 0.6 else "black")
ax[0].set_title("CRLB correlation (one pull)")
fig.colorbar(im, ax=ax[0], fraction=0.046)
rel = np.array(R["identifiability"]["rel_std_CRLB_pct"])
ax[1].bar(S.NAMES, rel, color="#1f5fae")
ax[1].set_yscale("log")
ax[1].set_ylabel("CRLB relative std (%)")
ax[1].set_title("Expected error per parameter")
for x, y in enumerate(rel):
    ax[1].text(x, y * 1.15, f"{y:.1f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "toy_identifiability.png", dpi=170)
plt.close(fig)

# 그림 2: 착탄 거리 오차 분포 (목표 1.925 m, 반복별 절댓값)
reps = len(R["fits_raw"]["full"]["theta_cma"])
goal = 1.925
TRUE = S.TRUE[None].repeat(reps, axis=0)
groups = {
    "default\nθ0": np.repeat(np.array([[10.0, 5.0, 0.5, 0.25]]), reps, axis=0),
    "full\nDE": np.array(R["fits_raw"]["full"]["theta_de"]),
    "full\nDE+CEM": np.array(R["fits_raw"]["full"]["theta_cma"]),
    "force\nDE": np.array(R["fits_raw"]["force_only"]["theta_de"]),
    "force\nDE+CEM": np.array(R["fits_raw"]["force_only"]["theta_cma"]),
}
data = []
for name, TH in groups.items():
    d_hat = S.solve_pull(TH, np.full(reps, goal))
    err = 100 * np.abs(S.landing(TRUE, d_hat) - goal)
    data.append(err)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.boxplot(data, labels=list(groups.keys()), showfliers=True)
ax.set_yscale("log")
ax.set_ylabel("|landing error| (cm), log")
ax.set_title(f"Toy: pull chosen with calibrated model, goal {goal} m, {reps} noise draws")
fig.tight_layout()
fig.savefig(FIG / "toy_landing.png", dpi=170)
plt.close(fig)
print("wrote", FIG / "toy_identifiability.png", FIG / "toy_landing.png")
