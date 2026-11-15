"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 입력: table_check_results.json, toy_streaming_results.json.
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FONT_CANDIDATES = ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "NanumGothic", "DejaVu Sans"]
from matplotlib import font_manager  # noqa: E402

_names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in FONT_CANDIDATES if f in _names), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

T = json.loads((ROOT / "analysis" / "table_check_results.json").read_text(encoding="utf-8"))
S = json.loads((ROOT / "analysis" / "toy_streaming_results.json").read_text(encoding="utf-8"))
ACC, GREY, RED = "#1f5fae", "#8a8a85", "#b03a2e"

# 1. 디코딩 패스 모델: N별, c0 비중별 체인 대비 속도 향상 (토이 A_grid)
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
for share in [0.0, 0.5, 0.9, 1.0]:
    xs, ys = [], []
    for row in S["A_grid"]:
        if row["c0_share"] == share:
            xs.append(row["N"])
            ys.append(row["decode_speedup_per_chunk"])
    ax.plot(xs, ys, marker="o", label=f"패스 고정비 비중 {int(share*100)}%")
ax.axhline(19.9, color=RED, ls="--", lw=1)
ax.text(2.6, 20.6, "Fig. 1b 막대 측정 19.9× (이 모델 최대 10×: 패스 수만 줄 때)", color=RED, fontsize=7)
ax.set_xlabel("버퍼 청크 수 N")
ax.set_ylabel("청크당 디코딩 속도 향상 (체인 10패스 대비)")
ax.set_title("스트리밍 1패스 모델의 디코딩 속도 향상 (토이)", fontsize=10)
ax.set_ylim(0, 22)
ax.legend(fontsize=7, loc="center right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "own_latency_grid.png")
plt.close(fig)

# 2. 신뢰구간 요약 (재계산 결과)
rows = [
    ("LIBERO 평균, 1회 실행 (n=2000)", T["libero_test_one_run"]["flash_d1_vs_pi05"]["diff_pp"],
     T["libero_test_one_run"]["flash_d1_vs_pi05"]["newcombe95_pp"]),
    ("LIBERO 평균, 5회 독립 상한 (n=10000)", T["libero_test_five_runs_upper_bound"]["diff_pp"],
     T["libero_test_five_runs_upper_bound"]["newcombe95_pp"]),
    ("RoboTwin 평균 (가정 n=5000)", T["robotwin_test_assumed_n5000"]["diff_pp"],
     T["robotwin_test_assumed_n5000"]["newcombe95_pp"]),
    ("실로봇: 동기 π0.5 대비 (n=45, 근사)", T["real_world_approx"]["flash_vs_pi05_sync"]["diff_pp"],
     T["real_world_approx"]["flash_vs_pi05_sync"]["newcombe95_pp"]),
    ("실로봇: 단순 비동기 대비 (n=45, 근사)", T["real_world_approx"]["flash_vs_naive"]["diff_pp"],
     T["real_world_approx"]["flash_vs_naive"]["newcombe95_pp"]),
]
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=160)
for i, (lab, d, (lo, hi)) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([lo, hi], [y, y], color=ACC, lw=2)
    ax.plot([d], [y], "o", color=ACC)
    ax.text(hi + 0.4, y, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=7)
ax.axvline(0, color=GREY, lw=1)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7)
ax.set_xlabel("FlashVLA 차이 (percentage point), 95% 근사 구간")
ax.set_xlim(-16, 36)
ax.set_title("차이의 불확실성 (근사, 실행 간 상관 무시)", fontsize=10)
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(FIG / "own_ci_summary.png")
plt.close(fig)

# 3. 경계 점프 토이: 지연 d와 rho
fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=160)
for E, ls in [(10, "-"), (20, "--")]:
    for rho, col in [(0.0, GREY), (0.5, "#4a7fc1"), (1.0, ACC)]:
        ds, vals = [], []
        for d in [0, 1, 2, 4]:
            ds.append(d)
            vals.append(S["B_boundary_jump_mean_abs"][f"E{E}_d{d}"][f"rho{rho}"])
        ax.plot(ds, vals, ls=ls, color=col, marker="o", label=f"E={E}, ρ={rho}")
ax.set_xlabel("추론 지연 d (스텝)")
ax.set_ylabel("청크 경계 평균 점프 (토이 단위)")
ax.set_title("비동기 경계 점프 토이 (가정 토이, 논문 검증 아님)", fontsize=10)
ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "own_boundary_toy.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
