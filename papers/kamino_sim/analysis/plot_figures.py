"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다: arXiv non-exclusive 라이선스).

own_table1_ratio.png : 표 1 재계산. Kamino 값 / (다른 솔버 중 최선) 의 로그 비율 (중앙값, p90)
own_ant_hetero.png   : 표 A.5 이종 세계 처리량과 단순 혼합 모형
own_toy_padmm.png    : 토이. 쌓인 상자 LCP에서 질량비에 따른 PGS와 PADMM 반복 횟수
own_rl_time.png      : DR Legs 학습 시간이 허용하는 반복당 롤아웃 길이 (v1 대 v2)
실행: python3 analysis/plot_figures.py   (recompute.py와 toy_padmm_stack.py를 먼저 실행; numpy, matplotlib)
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
INK, ACC, GREY, ORG, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#d9822b", "#b83b3b"

rc = json.loads((ANA / "recompute_results.json").read_text())
toy = json.loads((ANA / "toy_padmm_stack_results.json").read_text())

# ---- 1. 표 1 비율 ----
metrics = ["rJ", "rkin", "rC", "rv", "rprimal", "rdual", "rcmpl", "rvi"]
models = ["BDX", "Olaf", "DR Legs", "Iron Man"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
for ax, key, title in zip(axes, ("median_cells", "p90_cells"), ("중앙값", "p90")):
    M = np.full((len(metrics), len(models)), np.nan)
    zero = np.zeros_like(M, dtype=bool)
    for c in rc["V1_table1"][key]:
        i, j = metrics.index(c["metric"]), models.index(c["model"])
        r = c["ratio_kamino_over_best_other"]
        if r is None:  # 둘 다 0
            M[i, j] = 0.0
        elif r == float("inf") or c["best_other"] == 0 or c["kamino"] == 0:
            zero[i, j] = True
            M[i, j] = 3.0 if c["kamino"] > 0 else -3.0
        else:
            M[i, j] = np.clip(np.log10(r), -3, 3)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(models)), models)
    ax.set_yticks(range(len(metrics)), metrics)
    ax.set_title(f"표 1 {title}: log10(Kamino / 다른 솔버 중 최선)", fontsize=10)
    for i in range(len(metrics)):
        for j in range(len(models)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "–", ha="center", va="center", color=GREY)
            else:
                ax.text(j, i, "0 대비" if zero[i, j] else f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 1.8 else INK)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("파랑 = Kamino가 더 작음(좋음), 빨강 = 더 큼(나쁨), ±3에서 자름")
fig.savefig(FIG / "own_table1_ratio.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ---- 2. Ant 이종 세계 ----
a5 = rc["V3_ant_hetero"]
labels = ["3다리", "4다리", "5다리", "이종 혼합\n(보고)"]
vals = [a5["reported"]["3-leg"], a5["reported"]["4-leg"], a5["reported"]["5-leg"], a5["reported"]["hetero"]]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
bars = ax.bar(labels, vals, color=[ACC, ACC, ACC, ORG])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,}", ha="center", fontsize=8)
ax.axhline(a5["ideal_harmonic_mix_samples_per_s"], color=RED, ls="--", lw=1)
ax.text(3.45, a5["ideal_harmonic_mix_samples_per_s"] * 1.03,
        f"단순 혼합 모형 {a5['ideal_harmonic_mix_samples_per_s']:,.0f}", color=RED, ha="right", fontsize=8)
ax.set_ylabel("samples / s (2,048 환경)")
ax.set_title("표 A.5 재계산: 혼합 세계는 모형보다 약 14% 느림", fontsize=10)
fig.savefig(FIG / "own_ant_hetero.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ---- 3. 토이 ----
CAP = 20000
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
series = [("PGS_up", "PGS", GREY, "o"), ("PADMM_nesterov_rho1_f64", "PADMM+가속 ρ=1", ACC, "s"),
          ("PADMM_nesterov_rho1e-2_f64", "PADMM+가속 ρ=0.01", ORG, "^"),
          ("PADMM_nesterov_rho1e-4_f64", "PADMM+가속 ρ=1e-4", RED, "v")]
for ax, dist, title in zip(axes, ("heavy_top", "heavy_bottom"), ("위가 무거움", "아래가 무거움")):
    cases = [c for c in toy["cases"] if c["dist"] == dist]
    Rs = [c["R"] for c in cases]
    for key, lab, col, mk in series:
        its = [c[key]["iterations"] if c[key]["iterations"] is not None else CAP for c in cases]
        ax.plot(Rs, its, marker=mk, color=col, label=lab)
        for r, it, c in zip(Rs, its, cases):
            if c[key]["iterations"] is None:
                ax.annotate(">2e4", (r, it), textcoords="offset points", xytext=(0, 5), fontsize=7, color=col,
                            ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("질량비 R (무거운 상자 / 나머지)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("수렴까지 반복 (상대 잔차 < 1e-6)")
axes[0].legend(fontsize=8)
fig.suptitle("토이: 5개 상자 더미의 충돌 LCP (float64, 마찰 없음)", fontsize=10)
fig.savefig(FIG / "own_toy_padmm.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ---- 4. 학습 시간 ----
K = np.arange(1, 49)
iters, envs, dec = 4000, 4096, 5
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for name, hours, col in (("DR Legs v2 (3.5 h)", 3.5, ACC), ("Ant 이종 v2 (2.8 h)", 2.8, ORG)):
    ax.plot(K, iters * envs * K * dec / (hours * 3600) / 1e3, color=col, label=f"{name}: 필요한 시뮬 스텝/s")
ax.plot(K, iters * envs * K * dec / (31 * 3600) / 1e3, color=GREY, label="DR Legs v1 (31 h)")
ax.axhline(59, color=ACC, ls="--", lw=1)
ax.text(47, 61, "DR Legs 벤치 최대 약 59k (그림 A.6 판독)", ha="right", fontsize=8, color=ACC)
ax.axhline(86.2, color=ORG, ls="--", lw=1)
ax.text(47, 88, "Ant 혼합 벤치 86k (표 A.5)", ha="right", fontsize=8, color=ORG)
ax.axvline(24, color=INK, ls=":", lw=1)
ax.text(24.4, 5, "v1 추정 K ≈ 24", fontsize=8)
ax.set_xlabel("반복당 롤아웃 길이 K (정책 스텝, v2 미보고)")
ax.set_ylabel("필요한 시뮬레이터 처리량 (천 samples/s)")
ax.set_ylim(0, 220)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("학습 시간이 허용하는 롤아웃 길이: v2는 K ≲ 9", fontsize=10)
ax.grid(alpha=0.3)
fig.savefig(FIG / "own_rl_time.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote figures to", FIG)
