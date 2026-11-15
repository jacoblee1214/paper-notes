"""우리가 그린 그래프 세 장 (논문 그림을 다시 그리지 않음). 입력: table_check_results.json, toy_readout_results.json.
출력: report/figures/own_*.png 와 publish/git/report/figures/ 에 복사하지 않음(build_publish.py가 복사).
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fig_readings():
    """그림 4·5 읽은 값과 막대 오차 범위 (본문 임계 구간 n=50 가정)."""
    tc = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
    rows = tc["success_critical_n50"]
    tasks = list(rows.keys())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    x = np.arange(len(tasks))
    for k, (key, color, off) in enumerate([("base", "#8fb573", -0.18), ("rlt", "#e8b923", 0.18)]):
        p = [rows[t][key]["p"] * 100 for t in tasks]
        lo = [rows[t][key]["bar"][0] * 100 for t in tasks]
        hi = [rows[t][key]["bar"][1] * 100 for t in tasks]
        ax.bar(x + off, p, 0.34, color=color, label={"base": "기준 정책", "rlt": "RLT"}[key])
        ax.errorbar(x + off, p, yerr=[np.array(p) - np.array(lo), np.array(hi) - np.array(p)],
                    fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.set_xticks(x, ["나사", "지퍼타이", "이더넷", "충전기"])
    ax.set_ylabel("임계 구간 성공률 (%)")
    ax.set_ylim(0, 128)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_title("그림 5 읽은 값 (임계 구간)", fontsize=10)
    ax = axes[1]
    ratios = [tc["throughput_ratio"]["critical"][t]["ratio"] for t in ["screw", "zip", "ethernet", "charger"]]
    lo = [tc["throughput_ratio"]["critical"][t]["ratio_95"][0] for t in ["screw", "zip", "ethernet", "charger"]]
    hi = [tc["throughput_ratio"]["critical"][t]["ratio_95"][1] for t in ["screw", "zip", "ethernet", "charger"]]
    ax.bar(x, ratios, 0.5, color="#e8b923")
    ax.errorbar(x, ratios, yerr=[np.array(ratios) - np.array(lo), np.array(hi) - np.array(ratios)],
                fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.axhline(3.45, color="#b03a2e", ls="--", lw=1)
    ax.text(-0.45, 3.9, "그림 9 중앙값 비 3.45", fontsize=7.5, color="#b03a2e", ha="left")
    ax.set_xticks(x, ["나사", "지퍼타이", "이더넷", "충전기"])
    ax.set_ylabel("처리량 비 (RLT / 기준)")
    ax.set_title("처리량 비, 근사 95% 구간 (그림 4 읽기)", fontsize=10)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "값은 그림을 눈으로 읽은 것(±1%p). 원자료 아님. 구간은 델타 방법 근사.", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_readings_check.png", dpi=200)
    plt.close(fig)


def fig_toy():
    res = json.loads((ROOT / "analysis" / "toy_readout_results.json").read_text())
    M = res["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    names = [("full", "전체 헤드(256차원), 묶음 없음", "#8e8e8e"),
             ("tok16", "readout 16차원, 묶음 없음", "#5b8bd0"),
             ("tok16+anchor", "readout 16차원 + 묶음", "#e8b923")]
    for key, label, color in names:
        x = np.array(M[key]["curve_x"])
        mu = np.array(M[key]["curve_mean"])
        sd = np.array(M[key]["curve_sd"]) / np.sqrt(res["config"]["EVAL_SEEDS"])
        ax.plot(x, mu, color=color, label=label, lw=1.6)
        ax.fill_between(x, mu - 1.96 * sd, mu + 1.96 * sd, color=color, alpha=0.2)
    ax.axhline(M["full"]["ref_return"], color="#333", ls=":", lw=1)
    ax.text(x[-1], M["full"]["ref_return"] + 0.01, "기준 행동 보상", fontsize=7.5, ha="right")
    ax.set_xlabel("에피소드 수 (샘플)")
    ax.set_ylabel("결정적 정책 보상 (평가 256문맥)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("toy: 학습 곡선 (평균 ±95%)", fontsize=10)
    ax = axes[1]
    keys = list(M.keys())
    tr = [M[k]["grad_trace_init"]["mean"] for k in keys]
    ax.bar(range(len(keys)), tr, color=["#8e8e8e" if k == "full" else "#5b8bd0" for k in keys])
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("초기 기울기 분산 (대각 합, 로그)")
    ax.set_title("toy: 초기 기울기 추정 분산", fontsize=10)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_readout.png", dpi=200)
    plt.close(fig)


def fig_budget():
    """실로봇 샘플 산술: 데이터 시간에 따른 전이 수와 업데이트 수 (본문 25 샘플/초, UTD 5 두 해석)."""
    tc = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
    arith = tc["sample_arithmetic"]
    labels = list(arith.keys())
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, [arith[k]["transitions"] for k in labels], w, color="#8e8e8e", label="저장된 전이 수 (25/초)")
    ax.bar(x, [arith[k]["updates_if_UTD_per_10step_chunk"] for k in labels], w, color="#5b8bd0",
           label="업데이트 (UTD 5, 청크 경계당)")
    ax.bar(x + w, [arith[k]["updates_if_UTD_per_transition"] for k in labels], w, color="#e8b923",
           label="업데이트 (UTD 5, 전이당)")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("개수 (로그)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("로봇 시간에 따른 샘플·업데이트 수 (계산, 본문 값 기반)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sample_budget.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_readings()
    fig_toy()
    fig_budget()
    print("ok", sorted(p.name for p in FIG.glob("own_*.png")))
