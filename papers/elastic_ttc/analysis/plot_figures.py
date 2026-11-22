"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_dp_gain.png   : 확산 정책 6과제에서 ELASTIC 대 Fixed 차이(%p)와 Fisher 유의 여부 (읽은 값, recompute.py 결과)
own_hw_ci.png     : 실로봇 8개 설정의 성공률과 Wilson 95% 구간 (n=20), 읽은 지연
own_latency.png   : 그림 14 지연 곡선과 그림 5 점들의 비교
실행: python3 analysis/plot_figures.py   (matplotlib; 먼저 recompute.py 실행)
한글 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서로 찾는다.
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
R = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#d9822b"


def fig_dp_gain():
    rows = R["V7_dp_elastic_vs_fixed"]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(rows))
    cols = [ACC if r["fisher_p"] < 0.05 else GREY for r in rows]
    ax.bar(x, [r["gain_pp"] for r in rows], color=cols)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r["task"].replace("PushT ", "PushT\n").replace("Square ", "Square\n").replace("Can ", "Can\n")}\nL={r["L"]}' for r in rows],
                       fontsize=6.5)
    ax.set_ylabel("ELASTIC - Fixed (%p)")
    ax.set_title("확산 정책: 같은 평균 계산량에서 ELASTIC 대 Fixed (파랑 = Fisher p<0.05, n=200씩)", fontsize=10)
    ax.text(0.01, 0.95, "읽은 값(±0.3 %p), Fixed가 가려진 4개 비교는 가정값(표 V7 비고)", transform=ax.transAxes, fontsize=8, va="top", color="#55544f")
    fig.tight_layout()
    fig.savefig(FIG / "own_dp_gain.png", dpi=200)
    plt.close(fig)


def fig_hw_ci():
    v4 = R["V4_hw_counts"]
    lat = {"pi0.5 (10,1)": 69.1, "BoN (10,10)": 95.5, "ELASTIC (2.6,5.6)": 62.5, "ELASTIC (3.6,2.9)": 64.6,
           "ELASTIC (1.9,3.5)": 58.0, "Fixed (2,10)": 63.6, "Fixed (2,4)": 60.0, "Fixed (2,3)": 60.0}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for k, v in v4.items():
        c = ORG if k.startswith("ELASTIC") else (ACC if k.startswith("BoN") else (INK if k.startswith("pi") else GREY))
        # 같은 지연의 점이 겹치지 않도록 약간 옮긴다
        dx = {"Fixed (2,3)": 0.5, "Fixed (2,4)": -0.5}.get(k, 0)
        ax.errorbar(lat[k] + dx, v["pct"], yerr=[[v["pct"] - v["wilson95"][0]], [v["wilson95"][1] - v["pct"]]],
                    fmt="o", color=c, capsize=3, ms=6)
        off = {"ELASTIC (2.6,5.6)": (-92, 8), "ELASTIC (3.6,2.9)": (5, 8)}.get(k, (4, 5))
        ax.annotate(k.replace("pi0.5", "π0.5"), (lat[k] + dx, v["pct"]), fontsize=7, xytext=off, textcoords="offset points")
    ax.set_xlabel("추론 지연 (ms, 그림 5에서 읽음)")
    ax.set_ylabel("성공률 % (n=20, Wilson 95%)")
    ax.set_ylim(10, 100)
    ax.set_title("실로봇 pick-and-place: 구간이 크게 겹침", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_hw_ci.png", dpi=200)
    plt.close(fig)


def fig_latency():
    F14 = {1: [63.0, 64.2, 65.3, 65.5, 65.4, 66.3, 66.8, 67.1, 67.0, 67.7],
           5: [73.4, 74.6, 76.4, 77.9, 79.4, 80.1, 80.6, 81.0, 82.6, 83.7],
           10: [84.8, 86.9, 90.2, 92.0, 94.4, 96.6, 97.9, 99.0, 101.0, 102.8]}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (L, ys), c in zip(F14.items(), [INK, ORG, ACC]):
        ax.plot(range(1, 11), ys, "-o", color=c, ms=3.5, label=f"그림 14: L={L}")
    pts = R["V6_latency_model"]["fig5_vs_fig14"]
    for k, v in pts.items():
        P = {"pi0.5 (10,1)": 1, "BoN (10,10)": 10, "ELASTIC (2.6,5.6)": 5.6, "ELASTIC (3.6,2.9)": 2.9, "ELASTIC (1.9,3.5)": 3.5}[k]
        ax.plot(P, v["fig5_ms"], "s", color="#c0392b", ms=6)
        ax.annotate(k.replace("pi0.5", "π0.5"), (P, v["fig5_ms"]), fontsize=7, xytext=(4, -9), textcoords="offset points", color="#c0392b")
    ax.plot([], [], "s", color="#c0392b", label="그림 5의 같은 구성(읽은 값)")
    ax.set_xlabel("병렬 샘플 수 P")
    ax.set_ylabel("지연 (ms)")
    ax.set_title("같은 RTX 6000 지연 측정인데 그림 5가 그림 14보다 7–16 ms 낮음", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "own_latency.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_dp_gain()
    fig_hw_ci()
    fig_latency()
    print("wrote", [p.name for p in FIG.glob("own_*.png")])
