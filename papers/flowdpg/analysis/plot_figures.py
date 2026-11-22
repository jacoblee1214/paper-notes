"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_success_ci.png   보고된 성공률의 Wilson 95% 구간 (표 1 n = 25, 그림 4 n = 50, 그림 5d n = 25로 추정)
  own_shift_phi.png    식 (8)–(9) 정상점의 속도 이동량 s(t)와 ODE 민감도 Phi(1,t)의 항등행렬에서의 거리
  own_toy_dpg.png      토이: FlowDPG, ODE 역전파 DPG, 원시 이동의 Q 이득과 분산, 크리틱 기울기 잡음의 영향

입력: analysis/recompute_results.json, analysis/toy_dpg_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
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
TD = json.loads((ROOT / "analysis" / "toy_dpg_results.json").read_text(encoding="utf-8"))

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
matplotlib.rcParams["axes.formatter.use_mathtext"] = False
INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---- 그림 1: Wilson 구간
def fig_success_ci():
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    names = ["BC", "DSRL", "RECAP", "RA-BC", "AWR", "PLD", "RLT", "QAM", "FlowDPG\n오프라인", "FlowDPG\n+온라인"]
    keys = ["BC", "DSRL", "RECAP", "RA-BC", "AWR", "PLD", "RLT", "QAM", "FlowDPG_off", "FlowDPG_on"]
    overall = {"BC": 64, "DSRL": 68, "RECAP": 72, "RA-BC": 76, "AWR": 76, "PLD": 76, "RLT": 80, "QAM": 80, "FlowDPG_off": 88, "FlowDPG_on": 92}
    for i, k in enumerate(keys):
        cnt = round(overall[k] * 25 / 100)
        lo, hi = wilson(cnt, 25)
        col = ACC if k.startswith("FlowDPG") else MUTED
        ax.plot([i, i], [lo * 100, hi * 100], color=col, lw=2.2)
        ax.plot(i, overall[k], "o", color=col, ms=6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("Overall 성공률 (%)")
    ax.set_ylim(35, 105)
    ax.set_title("표 1: 25 케이스의 Wilson 95% 구간", fontsize=9.5)
    ax.grid(axis="y", color="#e6e4dd")
    ax = axs[1]
    f4 = {"BC": 62, "QAM\n오프": 78, "QAM\n+온": 80, "FlowDPG\n오프": 86, "FlowDPG\n+온": 90}
    for i, (k, v) in enumerate(f4.items()):
        cnt = round(v * 50 / 100)
        lo, hi = wilson(cnt, 50)
        col = ACC if k.startswith("FlowDPG") else MUTED
        ax.plot([i, i], [lo * 100, hi * 100], color=col, lw=2.2)
        ax.plot(i, v, "o", color=col, ms=6)
    ax.set_xticks(range(len(f4)))
    ax.set_xticklabels(list(f4.keys()), fontsize=7.5)
    ax.set_ylim(35, 105)
    ax.set_title("그림 4: 50 롤아웃의 Wilson 95% 구간", fontsize=9.5)
    ax.grid(axis="y", color="#e6e4dd")
    p = RC["V3"]["fig4_fisher_p"]["FlowDPG_on_vs_QAM_on"]
    ax.text(0.02, 0.04, f"FlowDPG +온라인 대 QAM +온라인\nFisher p = {p:.2f}", transform=ax.transAxes, fontsize=8, color=WARN)
    fig.tight_layout()
    fig.savefig(FIG / "own_success_ci.png", dpi=170)
    plt.close(fig)


# ---- 그림 2: 이동량 s(t)와 Phi
def fig_shift_phi():
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7))
    t = np.linspace(0, 1, 200)
    lam, mu = 0.5, 1.0
    s_curve = lam / (lam * t + (1 - lam) + mu * (1 - t) ** 2)
    ax = axs[0]
    ax.plot(t, s_curve, color=ACC, lw=2, label="닫힌 형태 s(t)/‖Δ‖ (λ=0.5, μ=1)")
    k1 = TD["K1_stationary_shift"]
    ax.plot([x["t"] for x in k1], [x["shift_over_delta_numeric"] for x in k1], "o", color=ACC2, ms=6, label="경사하강 수치 해")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.text(0.02, 1.02, "Δ 전체 (이동이 Δ와 같다면)", color=MUTED, fontsize=8)
    ax.set_xlabel("흐름 시각 t")
    ax.set_ylabel("정상점의 속도 이동량 / ‖Δ‖")
    ax.set_ylim(0, 1.15)
    ax.set_title("증류 목표가 정하는 속도 이동 (식 8, 9)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(color="#e6e4dd")
    ax = axs[1]
    k2 = TD["K2_phi_vs_identity"]
    ax.plot([x["t"] for x in k2], [x["rel_dev_from_I"] for x in k2], "o-", color=WARN, label="‖Φ(1,t) - I‖ / √d")
    ax.plot([x["t"] for x in k2], [x["mean_singular_value"] for x in k2], "s--", color=ACC, label="Φ(1,t)의 평균 특이값 (I면 1)")
    ax.set_xlabel("흐름 시각 t")
    ax.set_title("근사 (i): BC로 학습한 토이 망의 ODE 민감도", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(color="#e6e4dd")
    fig.tight_layout()
    fig.savefig(FIG / "own_shift_phi.png", dpi=170)
    plt.close(fig)


# ---- 그림 3: 토이 방법 비교
def fig_toy():
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.8))
    K3 = TD["K3_methods_3seeds"]
    ax = axs[0]
    order = [("BC", "BC", MUTED), ("FlowDPG", "FlowDPG", ACC), ("FlowDPG_nocons", "FlowDPG\n(L_cons 없음)", "#6c9bd2"),
             ("BPTT_eta1", "ODE 역전파\nη=1", ACC2), ("BPTT_eta3", "ODE 역전파\nη=3", "#b25c10")]
    for i, (k, lab, col) in enumerate(order):
        m, sd = K3[k]["Q_mean"]["mean"], K3[k]["Q_mean"]["sd"]
        sp = K3[k]["spread_std"]["mean"]
        ax.errorbar(sp, m, yerr=sd, fmt="o", color=col, ms=7, capsize=3)
        ax.annotate(lab, (sp, m), textcoords="offset points", xytext=(6, -3 if i % 2 else 6), fontsize=7.5, color=col)
    ax.set_xlabel("생성 행동의 표준편차 (다양성)")
    ax.set_ylabel("생성 행동의 평균 Q (높을수록 좋음)")
    ax.set_title("토이: Q 이득과 다양성 (3 시드, 평균 ± SD)", fontsize=9.5)
    ax.grid(color="#e6e4dd")
    ax = axs[1]
    K4 = TD["K4_noisy_critic_gradient_seed0"]
    bc = K3["BC"]["Q_mean"]["mean"]
    xs = ["0.0", "3.0"]
    for key, lab, col in (("FlowDPG_normalized", "적응 이동 (정규화)", ACC), ("raw_alpha_g", "원시 αg", WARN)):
        ys = [K4[key][f"g_noise_{x}"]["Q_mean"] for x in xs]
        ax.plot(xs, ys, "o-", color=col, label=lab)
    ax.axhline(bc, color=MUTED, ls=":", lw=1)
    ax.text(0.02, bc + 0.01, "BC", color=MUTED, fontsize=8)
    ax.set_xlabel("크리틱 기울기 잡음 (‖g‖ 대비 표준편차 배수)")
    ax.set_ylabel("생성 행동의 평균 Q")
    ax.set_title("크리틱 기울기에 잡음이 있을 때 (시드 0)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#e6e4dd")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_dpg.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_success_ci()
    fig_shift_phi()
    fig_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
