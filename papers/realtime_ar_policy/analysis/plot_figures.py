"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림을 다시 그린 것이 아니라 표·그림에서 읽은 값을 다시 그림).

own_libero_eps.png   : 표 1 평균 열을 외부 지연 ε별로 다시 묶음 (REALFAST, π0+RTC, π0.5+RTC), v1 기준선 값은 속이 빈 점
own_droid_counts.png : DROID 80회 성공 횟수(그림 5, 그림 C에서 읽은 값)와 Wilson 95%, 과제 단위 부트스트랩 90% 구간
own_latency_model.png: 토큰 수 대 지연(그림 3a 점선에서 읽음)과 지연 증가율 대 N (그림 7a에서 읽음)
own_fig4b_noise.png  : 그림 4b(τ별 평균 성공률, 그림에서 읽은 값)와 단일 평가의 이항 오차 범위
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 extract_figures.py, recompute.py 실행)
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
FIGV = json.loads((ANA / "figure_values.json").read_text())
RES = json.loads((ANA / "recompute_results.json").read_text())


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#c0661a", "#2a8a5a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def libero_eps():
    # 표 1 (v2) 평균 열 (평균, SD). ε = 1Δt, 2Δt, 3Δt. RTC는 s = ε/Δt 이고 REALFAST는 s = ε/Δt + 1
    pi0 = [(95.0, 0.1), (94.8, 0.2), (95.3, 0.4)]
    pi05 = [(97.1, 0.5), (97.2, 0.6), (97.2, 0.1)]
    rf = [(96.0, 0.7), (95.6, 0.3), (95.7, 0.3)]
    pi0_v1 = [94.7, 94.3, 94.8]
    pi05_v1 = [96.9, 96.5, 97.0]
    x = np.array([100, 200, 300])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for lab, data, c, off in (("π0 + RTC (m=4)", pi0, ORG, -7), ("π0.5 + RTC (m=4)", pi05, GRN, 7), ("π0-REALFAST (m=4)", rf, ACC, 0)):
        ax.errorbar(x + off, [d[0] for d in data], yerr=[d[1] for d in data], marker="o", color=c, capsize=3, label=lab + " (v2, 3시드 평균 ± SD)")
    ax.plot(x - 7, pi0_v1, "o", mfc="none", color=ORG, label="π0 + RTC (v1 값)")
    ax.plot(x + 7, pi05_v1, "o", mfc="none", color=GRN, label="π0.5 + RTC (v1 값)")
    ax.set_xticks(x)
    ax.set_xticklabels(["ε = 100 ms\n(RTC s=1, REALFAST s=2)", "ε = 200 ms\n(s=2, s=3)", "ε = 300 ms\n(s=3, s=4)"])
    ax.set_ylabel("LIBERO 평균 성공률 (%)")
    ax.set_ylim(93.5, 98.3)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax.set_title("표 1을 같은 외부 지연 ε로 묶어 다시 그림", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_libero_eps.png", dpi=200)
    plt.close(fig)


def droid_counts():
    d = RES["V8_droid"]
    k = d["counts_of_80_from_figC"]
    boot = d["task_bootstrap90"]
    items = [("π0\n동기", "pi0_sync", "pi0_sync"), ("π0.5\n동기", "pi05_sync", "pi05_sync"), ("π0-FAST\n동기", "pi0fast_sync", "pi0fast_sync"),
             ("π0\n+RTC", "pi0_rtc", "pi0_rtc"), ("π0.5\n+RTC\n(반투명 막대)", "pi05_rtc_translucent", "pi05_rtc_translucent"), ("π0-\nREALFAST", "realfast", "realfast")]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    cols = [GREY, GREY, GREY, ORG, ORG, ACC]
    for i, (lab, key, bk) in enumerate(items):
        v = k[key]
        lo, hi = wilson(v, 80)
        ax.bar(i, 100 * v / 80, color=cols[i], alpha=0.85, width=0.6)
        ax.errorbar(i - 0.12, 100 * v / 80, yerr=[[100 * v / 80 - lo], [hi - 100 * v / 80]], color=INK, capsize=3, lw=1.2)
        b = {"pi0_sync": "pi0_sync", "pi05_sync": "pi05_sync", "pi0fast_sync": "pi0fast_sync", "pi0_rtc": "pi0_rtc", "pi05_rtc_translucent": "pi05_rtc_translucent", "realfast": "realfast"}[bk]
        blo, bhi = boot[b]
        ax.errorbar(i + 0.14, 100 * v / 80, yerr=[[100 * v / 80 - 100 * blo], [100 * bhi - 100 * v / 80]], color="#7a1f8a", capsize=3, lw=1.2, ls=":")
        ax.text(i, 2, f"{v}/80", ha="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([x[0] for x in items], fontsize=8.5)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.plot([], [], color=INK, label="Wilson 95% (80회를 독립으로 취급)")
    ax.plot([], [], color="#7a1f8a", ls=":", label="과제 단위 부트스트랩 90% (8과제 평균, 10회 가정)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("DROID 80회 성공 횟수 (그림 판독 값, 과제당 10회 가정)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_droid_counts.png", dpi=200)
    plt.close(fig)


def latency_model():
    tl = FIGV["fig3"]["token_len_at_latency_lines"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axs[0]
    L = np.linspace(4, 22, 50)
    ax.plot(L, 36 + 13.3 * L, color=GREY, lw=1, label="δ ≈ 36 + 13.3·L (ms), 두 점선에서 맞춤")
    ax.plot([tl["LIBERO_129ms"], tl["LIBERO_183ms"]], [129, 183], "o", color=ACC, label="LIBERO 점선 (평균 7, 99% 11 토큰)")
    ax.plot([tl["DROID_183ms"], tl["DROID_302ms"]], [183, 302], "s", color=ORG, label="DROID 점선 (평균 11, 99% 20 토큰)")
    ax.axhline(400, color=INK, ls="--", lw=0.8)
    ax.text(4.2, 408, "m·Δt = 400 ms (토큰화 구간)", fontsize=7.5)
    ax.set_xlabel("디코딩 토큰 수 L")
    ax.set_ylabel("지연 δ (ms, RTX 4090)")
    ax.set_ylim(80, 450)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_title("지연은 토큰 수에 거의 선형 (그림 3a 판독)", fontsize=9)
    ax = axs[1]
    for name, c in (("Jetson Orin", ACC), ("RTX 4090", ORG), ("A100", GRN)):
        pts = FIGV["fig7a"][name]
        ax.plot([p["N"] for p in pts], [p["rel_overhead"] for p in pts], "o-", color=c, ms=3, label=name)
    ax.set_xlabel("병렬 궤적 수 N")
    ax.set_ylabel("(δ(N) - δ(1)) / δ(1)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("N에 따른 지연 증가 (그림 7a 판독)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_latency_model.png", dpi=200)
    plt.close(fig)


def fig4b_noise():
    b = FIGV["fig4"]["b"]
    taus = sorted(b["N=1"].keys(), key=float)
    x = [float(t) for t in taus]
    se = 100 * np.sqrt(0.955 * 0.045 / 2000)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for nm, c in (("N=1", ACC), ("N=2", GRN), ("N=4", ORG)):
        ax.plot(x, [100 * b[nm][t] for t in taus], "o-", color=c, ms=4, label=nm)
    ax.axhspan(95.5 - 1.96 * se, 95.5 + 1.96 * se, color=GREY, alpha=0.2, label=f"95.5% ± 1.96 SE (한 번 평가 2000회, SE {se:.2f} %p)")
    ax.set_ylim(94.3, 97.3)
    ax.annotate("N=4, τ=0.3: 96.6 (부록 표 B의 선택값)", (0.3, 96.6), (0.42, 96.95), arrowprops={"arrowstyle": "->"}, fontsize=8)
    ax.set_xlabel("샘플링 온도 τ")
    ax.set_ylabel("LIBERO 평균 성공률 (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("그림 4b (그림 판독 값): 점마다 한 번 평가", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig4b_noise.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    libero_eps()
    droid_counts()
    latency_model()
    fig4b_noise()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
