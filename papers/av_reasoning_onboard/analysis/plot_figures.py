"""리포트·게시본용 그래프 세 장 (자체 계산 결과만 그린다. 블로그 그림은 복사하지 않는다).

실행: python3 analysis/plot_figures.py  (latency_toy_results.json, config_check_results.json을 읽음)
1) latency_budget.png: 라운드 수에 따른 요청 지연(롤링·콜드, 5·10 Euler), 블로그·레시피 범위, 10 Hz 예산
2) decode_bandwidth.png: 토큰당 읽는 바이트와 이론 tok/s 상한, 레시피 측정 7 ms/토큰 위치
3) spec_decode.png: 수락률 α에 따른 라운드당 확정 토큰 E(α)와 대비 greedy 속도 배율, 손익분기
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#141b1a", "#555e5c", "#e2e5e4"
TEAL, ORANGE, GREEN, GRAY = "#1f5fae", "#c95a24", "#2e8b57", "#9a9fa0"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
LT = json.loads((ROOT / "analysis" / "latency_toy_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def latency_budget():
    rows = LT["latency"]["rows"]
    r = [x["rounds"] for x in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    ax.axhspan(160, 190, color=GRAY, alpha=0.18, lw=0, label="블로그 보고 범위 160–190 ms")
    ax.axhspan(180, 250, color=ORANGE, alpha=0.10, lw=0, label="레시피 검증 범위 180–250 ms (10 클립)")
    ax.axhline(100, color=INK2, lw=1, ls="--")
    ax.text(7.05, 103, "10 Hz (100 ms)", color=INK2, fontsize=8, ha="right")
    ax.plot(r, [x["rolling_5steps_ms"] for x in rows], "-o", color=TEAL, ms=4, label="롤링, Euler 5단계 (레시피 측정 조합)")
    ax.plot(r, [x["rolling_10steps_ms"] for x in rows], "--", color=TEAL, lw=1.2, label="롤링, Euler 10단계 (PyTorch 기본값 가정)")
    ax.plot(r, [x["cold_5steps_est_ms"] for x in rows], ":", color=ORANGE, lw=1.8, label="콜드 스타트 추정 (ViT 22프레임 선형 가정)")
    ax.set_xlabel("추측 디코딩 라운드 수 (초안+검증 1회 = 12 ms 가정)", color=INK)
    ax.set_ylabel("요청 지연 (ms)", color=INK)
    ax.set_xticks(range(0, 8))
    ax.set_ylim(60, 330)
    ax.set_title("Thor 단계 시간을 더한 요청 지연 모델 (실측 아님)", color=INK, fontsize=10, loc="left")
    style(ax)
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "latency_budget.png")
    plt.close(fig)


def decode_bandwidth():
    rows = LT["bandwidth"]["rows"]
    names = ["BF16\n(전체 head)", "FP8\n(전체 head)", "FP8·축소 head·FP8 KV\n(레시피)"]
    ceil = [x["ceiling_tok_s_가정 273 GB/s"] for x in rows]
    gb = [x["GB_per_token"] for x in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    y = np.arange(len(names))
    ax.barh(y, ceil, color=[GRAY, TEAL, TEAL], height=0.55)
    for i, (c, g) in enumerate(zip(ceil, gb)):
        ax.text(3, i, f"{c:.0f} tok/s  (토큰당 {g:.2f} GB)", va="center", color="white", fontsize=8)
    measured = 1000 / 7.0
    ax.axvline(measured, color=ORANGE, lw=1.4, ls="--")
    ax.text(measured + 2, -0.55, f"레시피 측정 7 ms/토큰 ≈ {measured:.0f} tok/s", color=ORANGE, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK)
    ax.set_xlabel("이론 상한 tok/s (대역폭 273 GB/s 가정, 사전 지식·확인 필요)", color=INK)
    ax.set_xlim(0, 260)
    ax.invert_yaxis()
    ax.set_title("디코드는 메모리 대역폭에 묶인다 (텍스트 디코더 기준)", color=INK, fontsize=10, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "decode_bandwidth.png")
    plt.close(fig)


def spec_decode():
    alpha = np.linspace(0.05, 0.95, 181)
    E = np.array([sum(a ** k for k in range(16)) for a in alpha])
    speed = E * 7.0 / 12.0
    be = LT["speculative"]["breakeven_alpha"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax.plot(alpha, E, color=TEAL, lw=2, label="라운드당 확정 토큰 E(α)")
    ax.set_xlabel("토큰 수락률 α (독립 가정)", color=INK)
    ax.set_ylabel("라운드당 확정 토큰 수", color=TEAL)
    ax.set_ylim(0, 9)
    ax2 = ax.twinx()
    ax2.plot(alpha, speed, color=ORANGE, lw=1.6, ls="--", label="greedy 대비 속도 배율 E·7/12")
    ax2.axhline(1.0, color=INK2, lw=0.9, ls=":")
    ax2.set_ylabel("greedy 대비 배율 (×)", color=ORANGE)
    ax2.set_ylim(0, 5.5)
    ax.axvline(be, color=INK2, lw=0.9, ls=":")
    ax.text(be + 0.015, 7.6, f"손익분기 α≈{be:.2f}", color=INK, fontsize=8)
    ax.set_title("초안 블록 16 (초안 15개) 추측 디코딩: 수락률이 이득을 정한다", color=INK, fontsize=10, loc="left")
    style(ax)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "spec_decode.png")
    plt.close(fig)


def main():
    latency_budget()
    decode_bandwidth()
    spec_decode()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
