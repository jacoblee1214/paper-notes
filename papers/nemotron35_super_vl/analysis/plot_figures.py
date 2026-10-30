"""리포트 그래프 네 장. 실행: cd analysis && python3 plot_figures.py (table_check.py, param_check.py를 먼저 실행)

1) t10_vs_ultra.png   : T10에서 Super 3.5 − Ultra 3 점수 차 (27개 공통 벤치마크)
2) t23_recovery.png   : T23의 NVFP4/BF16 정확도 비율과 중앙값(보고 99.62%, 표 전체 99.71%)
3) state_memory.png   : 요청당 상태 메모리, KV 캐시(FP8)와 Mamba 상태(FP16)의 문맥 길이별 비교
4) memory_fit.png     : 체크포인트·공개 양자화 크기와 24 GiB 맥·H100(80 GB)
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
P = json.load(open(ROOT / "analysis" / "param_check_results.json"))


def style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=6.8, color=INK2)


# 1) Super 3.5 − Ultra 3 (T10)
d = {k: v[0] - v[1] for k, v in tc.T10.items() if v[1] is not None}
d = {k: (v * 100 if k.startswith("GDPVal") else v) for k, v in d.items()}  # 정규화 ELO는 0–1 척도라 ×100
items = sorted(d.items(), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=200)
ys = np.arange(len(items))
ax.barh(ys, [v for _, v in items], color=[BLUE if v > 0 else ORANGE for _, v in items], height=0.66)
ax.set_yticks(ys, [k + (" (×100)" if k.startswith("GDPVal") else "") for k, _ in items], fontsize=7.4)
for y, (_, v) in zip(ys, items):
    ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=6.8, color=INK2)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("Nemotron 3.5 Super - Nemotron 3 Ultra (점수 차, %p)", color=INK2)
ax.set_title("T10: Super 3.5(활성 12B)가 Ultra 3(활성 55B)보다 나은 곳 17, 못한 곳 10", fontsize=9.5, color=INK)
ax.set_xlim(-16, 31)
style(ax)
note(fig, "출처: 보고서 Table 10. 파랑 = Super가 높음, 주황 = Ultra가 높음. 시각 벤치마크는 Ultra 값이 없어 제외.")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "t10_vs_ultra.png")
plt.close(fig)

# 2) T23 복원율
rows = sorted(((k, v[3] / v[2] * 100) for k, v in tc.T23.items()), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=200)
ys = np.arange(len(rows))
ax.scatter([v for _, v in rows], ys, s=22, color=[ORANGE if v < 99 else BLUE for _, v in rows], zorder=3)
ax.set_yticks(ys, [k for k, _ in rows], fontsize=7)
med_all = float(np.median([v for _, v in rows]))
ax.axvline(100, color=INK2, lw=0.8)
ax.axvline(99.62, color=AQUA, lw=1.2, ls="--")
ax.axvline(med_all, color=MUTED, lw=1.2, ls=":")
ax.text(99.62, len(rows) - 0.2, " 보고 중앙값 99.62%\n (36개 기준)", color=AQUA, fontsize=7, ha="right", va="top")
ax.text(med_all, len(rows) - 3.2, f" 표 37행 중앙값 {med_all:.2f}%", color=INK2, fontsize=7, ha="left", va="top")
ax.set_xlabel("NVFP4 정확도 / BF16 정확도 (%)", color=INK2)
ax.set_title("T23: 공개 NVFP4 체크포인트의 벤치마크별 정확도 복원율", fontsize=9.5, color=INK)
style(ax)
note(fig, "출처: 보고서 Table 23(FP8 KV 캐시). 주황 = 99% 미만. 중앙값은 높지만 최저 93.6%(SciCode problem), 94.2%(Tau3 Banking).")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "t23_recovery.png")
plt.close(fig)

# 3) 요청당 상태 메모리
sm = P["P4_state_memory"]
L = np.logspace(3, np.log10(2 ** 20), 200)
kv = sm["kv_bytes_per_token"]["FP8"] * L / 2 ** 20  # MiB
ssm = sm["ssm_state_MiB_per_request"]["FP16"] + sm["conv_state_MiB_per_request_bf16"]
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
ax.plot(L, kv, color=BLUE, lw=2, label="KV 캐시 (어텐션 8층, FP8)")
ax.plot(L, np.full_like(L, ssm), color=ORANGE, lw=2, label="Mamba 상태 (40층, FP16) + conv")
ax.plot(L, kv + ssm, color=INK2, lw=1, ls="--", label="합계")
x0 = sm["crossover_tokens_ssm_fp16_vs_kv_fp8"]
ax.axvline(x0, color=MUTED, lw=0.8, ls=":")
ax.text(x0 * 1.08, 3, f"교차점 ≈ {x0:,} 토큰", fontsize=7.2, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("문맥 길이 (토큰, 로그)", color=INK2)
ax.set_ylabel("요청당 메모리 (MiB, 로그)", color=INK2)
ax.set_title("요청당 상태 메모리: 1M 토큰에서도 KV 4 GiB(FP8) + Mamba 0.08 GiB", fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=7.4, loc="upper left")
style(ax, "both")
note(fig, "공개 config.json으로 계산(analysis/param_check.py). MTP의 어텐션 1층과 시각 토큰은 제외.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "state_memory.png")
plt.close(fig)

# 4) 24 GiB 맥에 들어가는가
fit = P["P5_local_fit"]["sizes_GiB"]
items = sorted(fit.items(), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=200)
ys = np.arange(len(items))
ax.barh(ys, [v for _, v in items], color=[AQUA if k.startswith("참고") else BLUE for k, _ in items], height=0.62)
ax.set_yticks(ys, [k for k, _ in items], fontsize=7.4)
for y, (_, v) in zip(ys, items):
    ax.text(v * 1.04, y, f"{v:.0f} GiB", va="center", fontsize=7, color=INK2)
for x, lab, col, ls in ((18, "24 GiB 맥의 GPU 한도 18 GiB (가정)", ORANGE, ":"), (24, "맥 통합 메모리 24 GiB", ORANGE, "--"),
                        (74.5, "H100 80 GB", MUTED, "--")):
    ax.axvline(x, color=col, lw=1, ls=ls, label=lab)
ax.legend(frameon=False, fontsize=6.8, loc="lower right")
ax.set_xlim(10, 400)
ax.set_xscale("log")
ax.set_xlabel("가중치 파일 크기 (GiB, 로그)", color=INK2)
ax.set_title("Super VL은 가장 작은 공개 양자화(3비트 MLX)도 24 GiB 맥에 들어가지 않는다", fontsize=9.5, color=INK)
style(ax)
note(fig, "크기: Hugging Face API(2026-10-30). 시각 모듈(mmproj)·MTP 별도 파일 제외. 초록 = 같은 계열 Lightning 30B-A3B(텍스트 전용).")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "memory_fit.png")
plt.close(fig)
print("wrote", [p.name for p in OUT.glob("*.png") if not p.name.startswith("paper_")])
