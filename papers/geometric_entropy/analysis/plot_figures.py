"""직접 그린 그래프 (CPU). 실행: python3 analysis/plot_figures.py → report/figures/

공개 가능(직접 계산한 toy 결과):
  own_entropy_toy.png : H_G의 r·k 의존, 시작–목표 거리에 따른 r 효과, 격리 검사(k=2 대 k=0), N 편향과 노이즈
  own_bc_toy.png      : 2차원 BC toy에서 커버리지(E1)와 모드 평균화(E2) 시연, 실패 유형(충돌 대 시간 초과)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, GRID = "#0b0b0b", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
E = json.loads((ROOT / "analysis" / "entropy_toy_results.json").read_text())
B = json.loads((ROOT / "analysis" / "bc_toy_results.json").read_text())
SIZE = (6.5, 3.4)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def own_entropy():
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    # (a) 시작–목표 거리 0.3에서 r 효과, k=0 대 k=1
    ax = axs[0]
    for k, c, lab in [("k=0", BLUE, "k=0 (연속 θ)"), ("k=1", ORANGE, "k=1 (단일 전략)")]:
        d = E["T1b_H_vs_r_startdist0.3"][k]
        rs = [float(x) for x in d]
        ax.plot(rs, [d[x] for x in d], "o-", color=c, label=lab)
    ax.set_xlabel("경유점 반지름 r"); ax.set_ylabel("H_G (toy, nats)")
    ax.set_title("(a) 시작–목표 거리 0.3", fontsize=9); ax.legend(frameon=False, fontsize=8); style(ax)
    # (b) 격리 검사: 시작점 고정, 경유점 기하만
    ax = axs[1]
    d = E["T1c_isolated_waypoint_modes_r0.10"]
    ks = ["k=0", "k=1", "k=2", "k=3"]
    vals = [d[k]["H"] if k != "k=1" else np.nan for k in ks]
    deff = [d[k]["d_eff"] for k in ks]
    xs = np.arange(len(ks))
    ax.bar(xs - 0.2, [0 if np.isnan(v) else v for v in vals], 0.4, color=BLUE, label="H_G")
    ax.bar(xs + 0.2, deff, 0.4, color=AQUA, label="d_eff")
    ax.set_xticks(xs, ks); ax.set_title("(b) 격리: k=2는 k=0보다 ln2 낮음\n(k=1은 분산 0이라 H_G→-inf)", fontsize=8)
    ax.legend(frameon=False, fontsize=8); style(ax)
    # (c) N 편향: 노이즈가 큰 자료(T2c)와 등방 가우시안(T2d)
    ax = axs[2]
    Ns = [int(k) for k in E["T2c_noise0.03_bias_by_N"]]
    ax.plot(Ns, list(E["T2c_noise0.03_bias_by_N"].values()), "o-", color=BLUE, label="노이즈 0.03 (T=50)")
    Ns2 = [int(k) for k in E["T2d_isotropic_gaussian_D150_by_N"]["values"]]
    v2 = list(E["T2d_isotropic_gaussian_D150_by_N"]["values"].values())
    ax2 = ax.twinx()
    ax2.plot(Ns2, v2, "s--", color=ORANGE, label="등방 가우시안 D=150 (오른쪽 축)")
    ax2.set_ylabel("등방 가우시안 H_G", color=ORANGE)
    ax.set_xscale("log"); ax.set_xlabel("샘플 수 N (로그)"); ax.set_ylabel("H_G, 노이즈 자료")
    ax.set_title("(c) N에 따른 편향", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="lower right"); style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "own_entropy_toy.png", dpi=200)
    plt.close(fig)


def own_bc():
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    # E1: 커버리지
    ax = axs[0]
    for kind, c, lab in [("poly2", BLUE, "2차 최소제곱"), ("knn", ORANGE, "kNN")]:
        d = B[f"E1_coverage_single_mode_{kind}"]
        xs = [float(k) for k in d]
        ax.plot(xs, [d[k]["success_mean"] for k in d], "o-", color=c, label=lab)
    ax.set_xlabel("학습 시작점 퍼짐 s_tr"); ax.set_ylabel("성공률 (시험 시작 y0 ∈ ±0.6)")
    ax.set_title("(a) 커버리지: 넓을수록 좋아짐", fontsize=9); ax.legend(frameon=False, fontsize=8); style(ax)
    # E2: 모드 비율
    ax = axs[1]
    for kind, c, lab in [("poly2", BLUE, "2차 최소제곱: 성공"), ("knn", ORANGE, "kNN: 성공")]:
        d = B[f"E2_two_mode_fraction_q_{kind}"]
        xs = [float(k) for k in d]
        ax.plot(xs, [d[k]["success_mean"] for k in d], "o-", color=c, label=lab)
    d = B["E2_two_mode_fraction_q_poly2"]
    xs = [float(k) for k in d]
    ax.plot(xs, [d[k]["collision_mean"] for k in d], "x--", color=MUTED, label="2차 최소제곱: 충돌 비율")
    ax.set_xlabel("아래쪽 전략 비율 q"); ax.set_title("(b) 모드 평균화: q>0이면 충돌이 생김", fontsize=9)
    ax.legend(frameon=False, fontsize=7); style(ax)
    # E2 H_G
    ax = axs[2]
    d = B["E2_two_mode_fraction_q_poly2"]
    xs = [float(k) for k in d]
    ax.plot(xs, [d[k]["H_G_mean"] for k in d], "o-", color=AQUA)
    ax.set_xlabel("아래쪽 전략 비율 q"); ax.set_ylabel("H_G (2차원 toy, nats)")
    ax.set_title("(c) 같은 시연의 H_G는 q와 함께 단조 증가", fontsize=9); style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "own_bc_toy.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    own_entropy()
    own_bc()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
