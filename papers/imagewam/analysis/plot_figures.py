"""우리가 만든 그래프를 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_robotwin_pairs.png : RoboTwin 과제별 쌍체 차이 (ImageWAM FLUX.2 4B 대 Fast-WAM, LingBot-VA; 랜덤화 설정)
own_real_ci.png        : 실로봇 과제별 성공률과 Wilson 95% 구간 (과제당 100회 가정)
own_latency.png        : 표 5와 표 11의 지연 (ms)
own_liberoplus.png     : LIBERO-Plus 차원별 성공률 (표 3, 표 7)
own_mask.png           : 공식 코드의 어텐션 마스크 규칙 (행: 질의, 열: 키). 행동 토큰은 목표 이미지 토큰을 보지 못한다
실행: python3 plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ANA = Path(__file__).resolve().parent
FIG = ANA.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c4651b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_pairs():
    d = json.loads((ANA / "table12_data.json").read_text(encoding="utf-8"))
    fw = json.loads((ANA / "fastwam_table3_data.json").read_text(encoding="utf-8"))
    V = np.array(d["values"], float)
    FW = np.array(fw["values"], float)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, (name, base, col) in zip(axes, [("Fast-WAM", FW[:, 1], GREY), ("LingBot-VA", V[:, 9], ORG)]):
        diff = V[:, 1] - base
        order = np.argsort(diff)
        ax.barh(np.arange(len(diff)), diff[order], color=[ACC if x > 0 else col for x in diff[order]], height=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.axvline(diff.mean(), color=INK, lw=1, ls="--")
        ax.set_title(f"ImageWAM - {name} (랜덤화)\n평균 {diff.mean():+.2f}점, 중앙값 {np.median(diff):+.0f}, 이김 {int((diff > 0).sum())}·짐 {int((diff < 0).sum())}", fontsize=9)
        ax.set_xlabel("과제별 성공률 차이 (점)")
        ax.set_yticks([])
    axes[0].set_ylabel("RoboTwin 과제 50개 (차이 순)")
    fig.tight_layout()
    fig.savefig(FIG / "own_robotwin_pairs.png", dpi=170)
    plt.close(fig)


def plot_real():
    T4 = {"π0": [57, 58, 54, 54], "π0.5": [83, 77, 74, 55], "Fast-WAM": [88, 75, 77, 76], "ImageWAM": [94, 84, 78, 82]}
    cols = [GREY, "#b5b2a8", ORG, ACC]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    w = 0.2
    for i, (k, v) in enumerate(T4.items()):
        x = np.arange(4) + (i - 1.5) * w
        lo = [wilson(c, 100)[0] for c in v]
        hi = [wilson(c, 100)[1] for c in v]
        ax.bar(x, v, width=w * 0.9, color=cols[i], label=k)
        ax.errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - np.array(v)], fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["T1 그릇 쌓기", "T2 수건 접기", "T3 서랍·마커", "T4 컵 걸기"])
    ax.set_ylim(30, 100)
    ax.set_ylabel("성공률 (%)")
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    ax.set_title("실로봇 성공률과 Wilson 95% 구간 (과제당 100회로 가정; 부록은 50회라고 적음)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_latency():
    items = [
        ("FastWAM-IDM\n(영상 전개)", 1081, GREY),
        ("FastWAM\n(1 스텝)", 302, GREY),
        ("ImageWAM\n(1 스텝)", 263, ACC),
        ("FastWAM\n접두부만", 194, GREY),
        ("ImageWAM\n접두부만", 198, ACC),
        ("FastWAM\n+컴파일", 80, GREY),
        ("ImageWAM\n+정적 그래프", 69, ACC),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.bar(range(len(items)), [v for _, v, _ in items], color=[c for *_, c in items], width=0.65)
    for i, (_, v, _) in enumerate(items):
        ax.text(i, v + 15, f"{v}", ha="center", fontsize=8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([n for n, *_ in items], fontsize=7.5)
    ax.set_ylabel("지연 (ms, A6000)")
    ax.set_title("표 5·표 11: 접두부만 쓰는 설정에서는 두 모델의 지연이 거의 같다", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_latency.png", dpi=170)
    plt.close(fig)


def plot_plus():
    dims = ["Camera", "Robot", "Language", "Light", "Backgr.", "Noise", "Layout"]
    rows = {
        "OpenVLA-OFT": ([56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2], GREY),
        "Fast-WAM": ([16.4, 44.5, 68.9, 78.2, 53.7, 37.7, 60.7], ORG),
        "ImageWAM (FLUX.2 4B)": ([80.8, 50.3, 91.4, 98.1, 85.5, 93.8, 80.5], ACC),
        "ImageWAM (FLUX.2 9B)": ([79.8, 58.7, 95.2, 96.1, 91.2, 93.3, 83.1], "#7fa6d6"),
    }
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    w = 0.2
    for i, (k, (v, c)) in enumerate(rows.items()):
        ax.bar(np.arange(7) + (i - 1.5) * w, v, width=w * 0.92, color=c, label=k)
    ax.set_xticks(range(7))
    ax.set_xticklabels(dims)
    ax.set_ylabel("성공률 (%)")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 112)
    ax.set_title("LIBERO-Plus 차원별 성공률 (표 3, 표 7)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_liberoplus.png", dpi=170)
    plt.close(fig)


def plot_mask():
    from mask_check import build_mask

    txt, cond, target, action = 6, 5, 4, 3
    m, (t0, r0, x0, a0, total) = build_mask(txt, cond, target, action)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1.4)
    names = ["텍스트", "참조(현재 관측)", "목표(노이즈) 이미지", "행동"]
    edges = [t0, r0, x0, a0, total]
    mids = [(edges[i] + edges[i + 1] - 1) / 2 for i in range(4)]
    ax.set_xticks(mids)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(mids)
    ax.set_yticklabels(names, fontsize=8)
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color=GREY, lw=0.8)
        ax.axvline(e - 0.5, color=GREY, lw=0.8)
    ax.set_xlabel("키 (읽히는 토큰)")
    ax.set_ylabel("질의 (읽는 토큰)")
    ax.set_title("공식 코드의 마스크: 행동 행의 '목표 이미지' 열이 비어 있다", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_mask.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_mask()
    plot_pairs()
    plot_real()
    plot_latency()
    plot_plus()
    print("그래프 5장 저장")
