"""우리가 만든 그래프 네 장을 report/figures/ 에 저장한다. 논문 그림은 여기서 만들지 않는다.

  own_auc_gap.png      v1과 v2의 라이브 AUC: ContactGuard와 Direct-linear (표 1)
  own_operating.png    v2 표 7의 재현율과 거짓 중단률(FAR) Wilson 95% 구간
  own_toy_probe.png    토이: 라벨 수에 따른 프로브 AUC (현재 잠재, 직접 선형, 롤아웃, 오라클)
  own_latency.png      논문 표 5(RTX 5090)와 맥에서 잰 랜덤 가중치 구조의 지연

입력: analysis/recompute_results.json, toy_rollout_probe_results.json, bench_monitor_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
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
TOY = json.loads((ROOT / "analysis" / "toy_rollout_probe_results.json").read_text(encoding="utf-8"))["results"]
BENCH = json.loads((ROOT / "analysis" / "bench_monitor_results.json").read_text(encoding="utf-8"))

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
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
TASKS = ["Cup", "Box", "Pencil", "Towel"]

# ---------------------------------------------- 그래프 1: v1 대 v2 AUC
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=200, sharey=True)
rows = {r["task"]: r for r in RC["V13_v1_vs_v2"]["rows"]}
x = np.arange(4)
for ax, ver in zip(axes, ["v1", "v2"]):
    o = [rows[t]["ours_" + ver] for t in TASKS]
    d = [rows[t]["direct_" + ver] for t in TASKS]
    ax.bar(x - 0.18, d, 0.34, color="#b7c5d8", label="Direct-linear")
    ax.bar(x + 0.18, o, 0.34, color=ACC, label="ContactGuard")
    for i in range(4):
        ax.text(x[i], max(o[i], d[i]) + 0.065, f"{o[i] - d[i]:+.3f}", ha="center", fontsize=7.2, color=WARN)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylim(0.5, 1.1)
    ax.set_title({"v1": "v1 (2026-08-13)", "v2": "v2 (2026-11-14)"}[ver], fontsize=9, loc="left")
axes[0].set_ylabel("라이브 AUC (표 1)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=2)
fig.suptitle("두 판의 표 1: ContactGuard와 Direct-linear의 AUC 간격(붉은 숫자) 평균 0.225 → 0.048", fontsize=8.8, x=0.02, ha="left", color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(FIG / "own_auc_gap.png")
plt.close(fig)

# ---------------------------------------------- 그래프 2: Wilson 구간
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=200, sharey=True)
meths = ["Ours", "Single view", "Direct-linear"]
cols = {"Ours": ACC, "Single view": ACC2, "Direct-linear": MUTED}
lab = {"Ours": "ContactGuard", "Single view": "Single view", "Direct-linear": "Direct-linear"}
ci = {(r["task"], r["method"]): r for r in RC["V5_wilson"]}
for ax, key, title in zip(axes, ["recall", "far"], ["재현율 (실패를 중단)", "거짓 중단률 FAR (성공을 중단)"]):
    for j, m in enumerate(meths):
        for i, t in enumerate(TASKS):
            p, lo, hi = ci[(t, m)][key]
            xx = i + (j - 1) * 0.22
            ax.plot([xx, xx], [lo, hi], color=cols[m], lw=1.6, alpha=0.55)
            ax.scatter([xx], [p], color=cols[m], s=16, zorder=3, label=lab[m] if i == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels(TASKS)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("비율 (점: 값, 선: Wilson 95%)")
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.suptitle("표 7의 작동점: 시행 25–28개라서 구간이 서로 크게 겹칩니다 (각 방법의 임계값은 따로 정함)", fontsize=8.8, x=0.02, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "own_operating.png")
plt.close(fig)

# ---------------------------------------------- 그래프 3: 토이
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=200, sharey=True)
labs = [50, 100, 250]
style = {"current": ("현재 잠재 z_t", MUTED, "s"), "direct": ("직접 선형 [z_t, 행동]", ACC2, "^"),
         "rollout": ("롤아웃 z_K", ACC, "o"), "oracle": ("참 z_K (상한)", "#2e8b57", "x")}
for ax, nu in zip(axes, [500, 20000]):
    for m, (name, c, mk) in style.items():
        mean = [TOY[f"linear_zK|unlab={nu}|lab={n}"]["auc_mean"][m] for n in labs]
        sem = [TOY[f"linear_zK|unlab={nu}|lab={n}"]["auc_se"][m] for n in labs]
        ax.errorbar(labs, mean, yerr=sem, color=c, marker=mk, ms=5, lw=1.3, capsize=2, label=name)
    rel = TOY[f"linear_zK|unlab={nu}|lab=50"]["dynamics_rel_mse"]
    ax.set_title(f"라벨 없는 전이 {nu:,}개 (롤아웃 상대 오차 {rel:.3f})", fontsize=8.8, loc="left")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(labs)
    ax.set_xticklabels([str(n) for n in labs])
    ax.set_xlabel("라벨 있는 클립 수")
axes[0].set_ylabel("테스트 AUC")
axes[0].legend(frameon=False, fontsize=7.2, loc="lower right")
fig.suptitle("토이(합성, 실패가 마지막 잠재의 선형 규칙일 때): 롤아웃 이득은 동역학 정확도에 달려 있습니다", fontsize=8.8, x=0.02, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "own_toy_probe.png")
plt.close(fig)

# ---------------------------------------------- 그래프 4: 지연
fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=200)
t5 = {5: (3.08, 9.02), 15: (9.26, 13.24), 25: (14.31, 18.71), 30: (17.20, 21.48)}
ks = sorted(t5)
xx = np.arange(len(ks))
ax.bar(xx - 0.2, [t5[k][0] for k in ks], 0.38, color="#b7c5d8", label="논문 표 5 롤아웃 (RTX 5090)")
ax.bar(xx - 0.2, [t5[k][1] - t5[k][0] for k in ks], 0.38, bottom=[t5[k][0] for k in ks], color=ACC, label="논문 표 5 인코딩+분류기")
mps = [t for t in BENCH["timing"] if t["device"] == "mps"][0]
cpu = [t for t in BENCH["timing"] if t["device"] == "cpu"][0]
ax.bar([3 + 0.2], [mps["full_ms_median"] - mps["encoder_only_ms_median"]], 0.38, color="#e8b98a", label="맥 MPS 롤아웃 (K=30, 랜덤 가중치)")
ax.bar([3 + 0.2], [mps["encoder_only_ms_median"]], 0.38, bottom=[mps["full_ms_median"] - mps["encoder_only_ms_median"]], color=ACC2, label="맥 MPS 인코딩 9장")
ax.axhline(33.3, color=WARN, lw=1, ls="--")
ax.text(-0.45, 29.2, "제어 주기 33.3 ms (30 Hz)", fontsize=7.5, color=WARN)
ax.set_xticks(xx)
ax.set_xticklabels([f"K={k}" for k in ks])
ax.set_ylabel("ms")
ax.set_ylim(0, 56)
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.0))
ax.set_title("지연: 논문 RTX 5090 대 맥 M4 Pro (K=30 막대끼리 비교, 구현 세부가 달라 참고용)", fontsize=8.6, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_latency.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
