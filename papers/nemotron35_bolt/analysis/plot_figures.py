"""리포트용 그래프 세 장. 실행: python3 analysis/plot_figures.py (gguf_header·config_check·table_check·spec_decode 결과 JSON을 읽음)

1) memory_budget.png: Table 2 메모리 예산(구성 요소별)과 24 GB·24 GiB·공식 레시피·이 맥 한도
2) per_token_bytes.png: 토큰 하나를 만들 때 읽는 바이트의 구성과, 초안 B개 검증 비용(라우팅 두 극단)
3) table7_deltas.png: Table 7 벤치마크별 BF16 → 2비트 변화와 부모 대비 변화
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, YELLOW, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
J = lambda n: json.loads((ROOT / "analysis" / n).read_text())
GG, CC, TC, SD = J("gguf_header_results.json"), J("config_check_results.json"), J("table_check_results.json"), J("spec_decode_results.json")


def style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def memory_budget():
    comp = GG["main"]["components"]
    iq2 = comp["Routed-expert projections"]["GB"] + comp["Shared-expert projections"]["GB"]
    nv = comp["Mamba input/output projections"]["GB"]
    q8 = sum(comp[k]["GB"] for k in ("Attention projections", "Latent MoE projections", "Token embedding", "LM head"))
    f32 = sum(v["GB"] for k, v in comp.items() if k not in ("Routed-expert projections", "Shared-expert projections",
                                                              "Mamba input/output projections", "Attention projections",
                                                              "Latent MoE projections", "Token embedding", "LM head"))
    bars = [("논문 Table 2\n(128K, FP8 KV)", [iq2, nv, q8, f32, 0.87, 0.96, 1.68]),
            ("Unsloth UD-IQ1_S\n(32K, F16 KV, 추정)", [18.51, 0, 0, 0, 0.27 + 0.17 + 0.5, 1.0, 1.73])]
    labels = ["IQ2_XXS experts", "NVFP4 Mamba 입출력", "Q8_0 (임베딩·LM head·attention·latent)", "F32 (router 등)",
              "런타임(KV·SSM·버퍼)", "LiLiCorr drafter", "비전 인코더·projector"]
    colors = [BLUE, ORANGE, AQUA, YELLOW, MUTED, "#6250d6", "#c9c8c4"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
    for i, (name, vals) in enumerate(bars):
        left = 0
        for j, v in enumerate(vals):
            if v <= 0:
                continue
            mixed = i == 1 and j == 0  # Unsloth 파일은 여러 형식이 섞인 가중치 한 덩어리
            ax.barh(i, v, left=left, color="#1f4e8c" if mixed else colors[j], edgecolor="white", linewidth=1.5,
                    height=0.55, label="Unsloth 혼합 양자화 가중치(IQ1_S 등)" if mixed else (labels[j] if i == 0 else None))
            left += v
        ax.text(left + 0.2, i + 0.38, f"합 {left:.2f} GB", va="center", fontsize=8, color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    mac = CC["mac"]
    for x, t, c, ls in ((24, "24 GB", INK2, "--"), (24 * 2**30 / 1e9, "24 GiB = 이 맥 RAM", INK, "-"),
                        (mac["gpu_default_assumed_GB"], "맥 GPU 기본 한도(2/3 가정)", ORANGE, ":"),
                        (28, "공식 28GB 레시피", MUTED, "-.")):
        ax.axvline(x, color=c, ls=ls, lw=1)
        ax.text(x, 1.55, t, rotation=90, va="bottom", ha="center", fontsize=7, color=c)
    ax.set_yticks([0, 1], [b[0] for b in bars], fontsize=8)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.5, 1.5)
    ax.set_clip_on(False)
    ax.set_xlabel("GB (1e9 바이트)", color=INK2)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "memory_budget.png", bbox_inches="tight")
    print("wrote memory_budget.png")


def per_token_bytes():
    comp = GG["main"]["components"]
    items = [("Mamba 입출력 (NVFP4)", comp["Mamba input/output projections"]["GB"]), ("LM head (Q8_0)", comp["LM head"]["GB"]),
             ("shared expert (IQ2)", comp["Shared-expert projections"]["GB"]),
             ("routed experts 9개/층 (IQ2)", SD["routed_GB_single_token"]),
             ("latent MoE (Q8_0)", comp["Latent MoE projections"]["GB"]), ("attention (Q8_0)", comp["Attention projections"]["GB"]),
             ("router (F32)", comp["MoE router weights"]["GB"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    names, vals = zip(*items)
    ys = range(len(items))[::-1]
    ax.barh(list(ys), vals, color=[ORANGE] + [BLUE] * (len(items) - 1), height=0.6)
    for y, v in zip(ys, vals):
        ax.text(v + 0.03, y, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(list(ys), names, fontsize=7.5)
    ax.set_xlabel("토큰 하나에 읽는 가중치 (GB)", color=INK2)
    ax.set_title(f"단일 토큰 디코드: 합 {SD['GB_per_token_no_spec']:.2f} GB", fontsize=9, color=INK)
    style(ax)
    ax = axes[1]
    rows = SD["verify_cost"]
    for mode, c, lab in (("uniform", ORANGE, "독립·균등 라우팅"), ("correlated", BLUE, "완전 상관 라우팅")):
        pts = [(r["B"], r["relative_to_single"]) for r in rows if r["routing"] == mode]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=c, lw=2, ms=5, label=lab)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlabel("한 번에 검증하는 토큰 수 B", color=INK2)
    ax.set_ylabel("단일 토큰 대비 읽는 바이트", color=INK2)
    ax.set_title("초안 검증 비용 (대역폭 모델)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5)
    style(ax, "y")
    fig.text(0.01, -0.02, "GGUF 헤더의 텐서별 바이트와 config의 층별 expert 수로 계산. 실제 라우팅은 두 극단 사이. analysis/spec_decode_toy.py",
             fontsize=6.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "per_token_bytes.png", bbox_inches="tight")
    print("wrote per_token_bytes.png")


def table7_deltas():
    per = TC["C4"]["per_benchmark"]
    order = sorted(per, key=lambda b: per[b]["bf16_to_2bit"])
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=200)
    ys = list(range(len(order)))
    for y, b in zip(ys, order):
        a, p = per[b]["bf16_to_2bit"], per[b]["parent_to_2bit"]
        ax.plot([p, a], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([a], [y], s=36, color=ORANGE, zorder=3, label="Bolt BF16 → 24GB(2비트)" if y == 0 else None)
        ax.scatter([p], [y], s=36, color=BLUE, zorder=3, label="부모 Super VL → Bolt 24GB" if y == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ys, order, fontsize=7.5)
    ax.set_xlabel("점수 변화 (%p, Table 7)", color=INK2)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    style(ax)
    fig.text(0.01, -0.01, "2비트가 BF16보다 높은 곳: AA-Omniscience non-hallucination(+5.13), WebArena(+0.61). 단일 측정, 분산 미보고.",
             fontsize=6.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "table7_deltas.png", bbox_inches="tight")
    print("wrote table7_deltas.png")


if __name__ == "__main__":
    memory_budget()
    per_token_bytes()
    table7_deltas()
