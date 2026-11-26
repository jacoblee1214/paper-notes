"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_libero_ci.png  : LIBERO 스위트별 성공률과 Wilson 95% 구간(스위트당 500회 가정)
own_real_ci.png    : 실로봇 과제별 성공률과 Wilson 95% 구간(셀당 20회 가정), 그림 4 평균 막대의 판독 차이
own_memory.png     : 메모리 1.4 GB의 부품별 구성(파라미터 수와 GGUF 파일 크기 메타데이터)
own_toy_quant.png  : 토이 Q1–Q2, 식 (4) 계수 비교와 W1.58A8 오차 분해

실행: python3 analysis/plot_figures.py   (numpy, matplotlib; analysis/recompute.py, analysis/toy_quant.py를 먼저 실행)
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
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
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"
COL = {"BitVLA": "#1f5fae", "BitVLA w/o PT": "#8fb3dd", "pi0": "#c0762b", "OpenVLA-OFT": "#4a8f5a"}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_libero():
    models = [("OpenVLA-OFT", [97.6, 98.4, 97.9, 94.5]), ("pi0", [96.8, 98.8, 95.8, 85.2]),
              ("BitVLA", [96.6, 99.0, 95.4, 92.8]), ("BitVLA w/o PT", [97.4, 99.6, 94.4, 87.6])]
    suites = ["Spatial", "Object", "Goal", "Long"]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    n = 500
    w = 0.19
    for i, (m, vals) in enumerate(models):
        xs = np.arange(4) + (i - 1.5) * w
        lo, hi = zip(*[wilson(round(v / 100 * n), n) for v in vals])
        ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="o", ms=5,
                    color=COL[m], capsize=3, label={"pi0": "π0", "BitVLA w/o PT": "BitVLA (사전학습 없음)"}.get(m, m))
    ax.set_xticks(range(4))
    ax.set_xticklabels(suites)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(80, 100.5)
    ax.grid(axis="y", color="#e5e3dc")
    ax.legend(ncol=4, fontsize=8, loc="lower left", frameon=False)
    ax.set_title("LIBERO 성공률과 Wilson 95% 구간 (스위트당 500회 가정, 표 I 값)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_libero_ci.png", dpi=200)
    plt.close(fig)


def plot_real():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V8_real_world_figures"]
    n = 20
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    names = ["BitVLA", "pi0", "OpenVLA-OFT", "BitVLA w/o PT"]
    for ax, key, tasks, title in ((axes[0], "fig4", ["수박 잡기", "빵 담기", "종 세우기"], "그림 4: 학습한 과제"),
                                  (axes[1], "fig5", ["스펀지 잡기", "수박 바구니", "종+식탁보"], "그림 5: 분포 밖(OOD) 과제")):
        w = 0.2
        for i, m in enumerate(names):
            vals = np.array(rec[key][m]["values"])
            lo, hi = zip(*[wilson(v / 100 * n, n) for v in vals])
            xs = np.arange(3) + (i - 1.5) * w
            ax.bar(xs, vals, width=w * 0.92, color=COL[m], label={"pi0": "π0", "BitVLA w/o PT": "BitVLA (사전학습 없음)"}.get(m, m))
            ax.errorbar(xs, vals, yerr=[vals - np.array(lo), np.array(hi) - vals], fmt="none", ecolor=INK, capsize=2, lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(tasks, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", color="#e5e3dc")
    axes[0].set_ylabel("성공률(점수) %  (n = 20 가정)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    ax = axes[2]
    labs = ["BitVLA", "π0", "OFT", "w/o PT"]
    keys = ["BitVLA", "pi0", "OpenVLA-OFT", "BitVLA w/o PT"]
    calc = [rec["fig4"][k]["mean_of_labels"] for k in keys]
    read = [rec["fig4"][k]["avg_bar_read(그림 판독)"] for k in keys]
    x = np.arange(4)
    ax.bar(x - 0.17, calc, width=0.32, color="#cfd6e0", label="세 과제 라벨 평균")
    ax.bar(x + 0.17, read, width=0.32, color=ACC, label="평균 막대(그림 판독)")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_title("그림 4 평균: 라벨 대 막대", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#e5e3dc")
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def plot_memory():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V12_memory_budget"]
    g = rec["gguf_bytes"]
    llm_lin = rec["gb_llm_linear_2bit"]
    emb = rec["gb_embedding_bf16"]
    vit = g["vision"] / 1e9
    act = g["action"] / 1e9
    pro = g["proprio"] / 1e9
    other = g["llm"] / 1e9 - llm_lin - emb
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    labels = ["OpenVLA-OFT\n(7.7B x 2 byte)", "BitVLA 미세조정 체크포인트\n(bf16 마스터 가중치)", "BitVLA I2S GGUF\n(2비트 묶음)"]
    vals = [15.4, rec["hf_bf16_gb_finetuned"], rec["gguf_total_gb"]]
    ax.barh(labels, vals, color=["#4a8f5a", "#c9b27a", ACC], height=0.55)
    for y, v in enumerate(vals):
        ax.text(v + 0.2, y, f"{v:.2f} GB", va="center", fontsize=9)
    ax.set_xlim(0, 18.5)
    ax.set_xlabel("가중치 크기 (GB, 활성과 KV 캐시 제외)")
    ax.grid(axis="x", color="#e5e3dc")
    ax.set_title("논문 메모리 1.4 GB는 GGUF 크기와 같음", fontsize=9.5)
    ax = axes[1]
    parts = [("LLM 선형 (2비트)", llm_lin, ACC), ("임베딩 (bf16)", emb, "#8fb3dd"), ("ViT", vit, "#4a8f5a"),
             ("액션 헤드", act, "#c0762b"), ("고유감각 투영 등", pro + other, GREY)]
    bottom = 0
    for nm, v, c in parts:
        ax.bar([0], [v], bottom=bottom, color=c, width=0.5, label=f"{nm}  {v:.2f} GB")
        bottom += v
    ax.set_xlim(-0.6, 0.9)
    ax.set_xticks([])
    ax.set_ylabel("GB")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_title(f"GGUF {bottom:.2f} GB의 구성", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_memory.png", dpi=200)
    plt.close(fig)


def plot_toy():
    t = json.loads((ANA / "toy_quant_results.json").read_text(encoding="utf-8"))["Q1_Q2_linear_layer"]["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    labels = list(t.keys())
    x = np.arange(2)
    ax.bar(x - 0.2, [t[k]["rel_err_W1.58_only"] for k in labels], 0.2, color=COL["pi0"], label="W 1.58만")
    ax.bar(x, [t[k]["rel_err_A8_only"] for k in labels], 0.2, color="#8fb3dd", label="A 8만")
    ax.bar(x + 0.2, [t[k]["rel_err_W1.58A8"] for k in labels], 0.2, color=ACC, label="W1.58A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("상대 출력 오차 (양자화 출력과 원 출력의 노름 비)")
    ax.set_title("난수 선형층의 양자화 오차 분해", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", color="#e5e3dc")
    ax = axes[1]
    vals = [t[k]["literal_over_correct_median"] for k in labels]
    ax.bar(x, vals, 0.45, color=GREY)
    ax.set_yscale("log")
    ax.axhline(1, color=INK, lw=0.8)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.15, f"{v:,.0f}배" if v > 50 else f"{v:.1f}배", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("식 (4) 계수 대 α β / 127 (비율)")
    ax.set_title("식 (4) 글자 그대로의 크기가 역양자화 값과 이루는 비", fontsize=9.5)
    ax.set_ylim(0.5, 5000)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_quant.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_libero()
    plot_real()
    plot_memory()
    plot_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
