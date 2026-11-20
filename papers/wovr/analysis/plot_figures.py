"""WoVR 분석의 자체 그래프 두 장 (report/figures/own_*.png). 논문 그림은 쓰지 않는다.

1) own_libero_ci.png: LIBERO 스위트별 SR과 Wilson 95% 구간 (one-traj, n=500 가정)
2) own_kir_toy.png: 토이 T1, 롤아웃 깊이에 따른 가짜 성공 비율
실행: python3 analysis/plot_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for font in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"]:
    try:
        from matplotlib import font_manager
        if any(font in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


def wilson(p, n, z=1.96):
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


# 1) LIBERO
suites = ["Spatial", "Object", "Goal", "Long"]
data = {"OFT": [63.6, 36.4, 48.2, 13.8], "WMPO": [67.8, 65.4, 56.6, 13.8], "WoVR": [84.2, 80.8, 77.4, 35.8]}
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
width = 0.26
for k, (name, vals) in enumerate(data.items()):
    xs = [i + (k - 1) * width for i in range(len(suites))]
    lo_err, hi_err = [], []
    for v in vals:
        lo, hi = wilson(v / 100, 500)
        lo_err.append(v - 100 * lo)
        hi_err.append(100 * hi - v)
    color = {"OFT": "#9a9a94", "WMPO": "#7d9cc4", "WoVR": "#1f5fae"}[name]
    ax.bar(xs, vals, width, label=name, color=color, yerr=[lo_err, hi_err], capsize=2, error_kw={"lw": 0.7})
ax.set_xticks(range(len(suites)))
ax.set_xticklabels(suites)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 100)
ax.set_title("LIBERO 한 궤적 SFT: 스위트별 성공률과 Wilson 95% 구간 (n=500 가정)", fontsize=9)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_libero_ci.png")
plt.close(fig)

# 2) 토이 T1
res = json.loads((ROOT / "analysis" / "toy_kir_depth_results.json").read_text(encoding="utf-8"))
labels, full, kir = [], [], []
for key, v in res["sweep"].items():
    if "bias=0.0002" in key:
        continue
    labels.append(key.replace("_bias=0.0", "").replace("sigma=", "σ="))
    full.append(100 * v["full_rollout"]["false_positive_share_of_imagined_success"])
    kir.append(100 * v["keyframe_rollout"]["false_positive_share_of_imagined_success"])
fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
xs = range(len(labels))
ax.bar([x - 0.18 for x in xs], full, 0.36, label="초기 상태 롤아웃 (깊이 60)", color="#9a9a94")
ax.bar([x + 0.18 for x in xs], kir, 0.36, label="키프레임 롤아웃 (깊이 20)", color="#1f5fae")
ax.set_xticks(list(xs))
ax.set_xticklabels(labels)
ax.set_ylabel("상상 성공 중 가짜 성공 (%)")
ax.set_title("토이 T1: 오차 독립 랜덤워크 가정에서 깊이와 가짜 성공 (편향 0)", fontsize=9)
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_kir_toy.png")
plt.close(fig)
print("saved", sorted(p.name for p in FIG.glob("own_*.png")))
