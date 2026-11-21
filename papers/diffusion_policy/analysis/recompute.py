"""Diffusion Policy (arXiv:2303.04137 v5) 표 수치 재계산과 검정.

논문 표 값을 그대로 옮기고(출처 표 번호를 주석으로 적음), 다음을 계산한다.
  1. 부록 B.2의 평균 향상률 46.9%를 여러 해석으로 다시 계산 (V1)
  2. 본문의 개별 향상률(32%, 213%) 재계산 (V2)
  3. 실제 로봇 성공률의 Wilson 95% 구간과 Fisher 정확 검정 (V3–V5)
  4. 시뮬 표의 근사 Wilson 구간 (V6)
  5. 본문과 표의 일관성 검사 (V7–V12)

실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)
결과: analysis/recompute_results.json
"""
import json
import math
import statistics as st
from pathlib import Path

from scipy.stats import fisher_exact

OUT = Path(__file__).resolve().parent / "recompute_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---- 표 1 (상태 기반). 열: Lift ph, Lift mh, Can ph, Can mh, Square ph, Square mh,
#      Transport ph, Transport mh, ToolHang ph, Push-T. 값은 (최대, 마지막 10개 평균). 표 1 원문.
COLS1 = ["Lift-ph", "Lift-mh", "Can-ph", "Can-mh", "Sq-ph", "Sq-mh", "Tr-ph", "Tr-mh", "ToolHang-ph", "PushT"]
T1 = {
    "LSTM-GMM": [(1.00, .96), (1.00, .93), (1.00, .91), (1.00, .81), (.95, .73), (.86, .59), (.76, .47), (.62, .20), (.67, .31), (.67, .61)],
    "IBC": [(.79, .41), (.15, .02), (0, 0), (.01, .01), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (.90, .84)],
    "BET": [(1, .96), (1, .99), (1, .89), (1, .90), (.76, .52), (.68, .43), (.38, .14), (.21, .06), (.58, .20), (.79, .70)],
    "DP-C": [(1, .98), (1, .97), (1, .96), (1, .96), (1, .93), (.97, .82), (.94, .82), (.68, .46), (.50, .30), (.95, .91)],
    "DP-T": [(1, 1), (1, 1), (1, 1), (1, .94), (1, .89), (.95, .81), (1, .84), (.62, .35), (1, .87), (.95, .79)],
}
# ---- 표 2 (시각 기반). 같은 열 구성. 표 2 원문.
T2 = {
    "LSTM-GMM": [(1.00, .96), (1.00, .95), (1.00, .88), (.98, .90), (.82, .59), (.64, .38), (.88, .62), (.44, .24), (.68, .49), (.69, .54)],
    "IBC": [(.94, .73), (.39, .05), (.08, .01), (0, 0), (.03, 0), (0, 0), (0, 0), (0, 0), (0, 0), (.75, .64)],
    "DP-C": [(1, 1), (1, 1), (1, .97), (1, .96), (.98, .92), (.98, .84), (1, .93), (.89, .69), (.95, .73), (.91, .84)],
    "DP-T": [(1, 1), (1, .99), (1, .98), (1, .98), (1, .90), (.94, .80), (.98, .81), (.73, .50), (.76, .47), (.78, .66)],
}
# ---- 표 4 (다단계, 상태 기반). 열: BlockPush p1, p2, Kitchen p1, p2, p3, p4. 값은 한 개.
COLS4 = ["BP-p1", "BP-p2", "K-p1", "K-p2", "K-p3", "K-p4"]
T4 = {
    "LSTM-GMM": [.03, .01, 1.00, .90, .74, .34],
    "IBC": [.01, .00, .99, .87, .61, .24],
    "BET": [.96, .71, .99, .93, .71, .44],
    "DP-C": [.36, .11, 1.00, 1.00, 1.00, .99],
    "DP-T": [.99, .94, 1.00, .99, .99, .96],
}
# ---- 표 6 (실로봇 Push-T, n = 20 시행). (성공률 표기, 시행 수 k/20 로 환산)
T6 = {"Human": 20, "IBC-pos": 0, "IBC-vel": 0, "LSTM-pos": 4, "LSTM-vel": 2, "DP-T-E2E": 13,
      "DP-ImgNet": 3, "DP-R3M": 16, "DP-E2E": 19}
N_REAL = 20
# 본문 실로봇 과제 (n = 20)
REAL_TASKS = {"Egg Beater (본문 55%)": 11, "Mat Unrolling (본문 75%)": 15,
              "Shirt Folding (본문 75%)": 15, "Mug Flip DP (본문 90%)": 18, "Mug Flip LSTM-GMM": 0}

BASE1 = ["LSTM-GMM", "IBC", "BET"]
OURS = ["DP-C", "DP-T"]
BASE2 = ["LSTM-GMM", "IBC"]


def b2_improvements(tbl, cols, base, ours, k):
    """k = 0: 최대 성능, k = 1: 마지막 10개 평균. 향상률 (ours - base) / base."""
    vals = []
    for i in cols:
        mb = max(tbl[b][i][k] for b in base)
        mo = max(tbl[o][i][k] for o in ours)
        vals.append((mo - mb) / mb)
    return vals


def main():
    res = {}

    # ---- V1: 부록 B.2 평균 향상률 재계산 (46.9%)
    ph_idx = [0, 2, 4, 6, 8, 9]  # 표 1 열 중 ph(또는 단일) 열
    all_idx = list(range(10))
    t4_idx = list(range(6))
    t4_imp = []
    for j in t4_idx:
        mb = max(T4[b][j] for b in BASE1)
        mo = max(T4[o][j] for o in OURS)
        t4_imp.append((mo - mb) / mb)

    readings = {}
    readings["B2 본문 절차: 최대 성능, mh 제외 (표1 ph 6열 + 표2 ph 6열 + 표4 6열, N=18)"] = (
        b2_improvements(T1, ph_idx, BASE1, OURS, 0) + b2_improvements(T2, ph_idx, BASE2, OURS, 0) + t4_imp)
    readings["최대 성능, mh 포함 (표1 10열 + 표2 10열 + 표4 6열, N=26)"] = (
        b2_improvements(T1, all_idx, BASE1, OURS, 0) + b2_improvements(T2, all_idx, BASE2, OURS, 0) + t4_imp)
    readings["마지막 10개 평균, mh 포함 (N=26)"] = (
        b2_improvements(T1, all_idx, BASE1, OURS, 1) + b2_improvements(T2, all_idx, BASE2, OURS, 1) + t4_imp)
    readings["마지막 10개 평균, mh 제외 (N=18)"] = (
        b2_improvements(T1, ph_idx, BASE1, OURS, 1) + b2_improvements(T2, ph_idx, BASE2, OURS, 1) + t4_imp)
    res["V1_avg_improvement"] = {k: {"N": len(v), "mean": round(st.mean(v), 5)} for k, v in readings.items()}
    res["V1_paper_claim"] = 0.46858

    # ---- V2: 본문 개별 향상률
    bp_p2 = (0.94 - 0.71) / 0.71
    kit_p4_bet = (0.99 - 0.44) / 0.44
    kit_p4_lstm = (0.99 - 0.34) / 0.34
    res["V2_text_improvements"] = {
        "BlockPush_p2_vs_BET_(DP-T_0.94)": round(bp_p2, 4),
        "Kitchen_p4_vs_BET_(DP-C_0.99)": round(kit_p4_bet, 4),
        "Kitchen_p4_vs_LSTM-GMM_(DP-C_0.99)": round(kit_p4_lstm, 4),
        "본문 주장": {"BlockPush p2": "32%", "Kitchen p4": "213%"},
    }

    # ---- V3: 실로봇 Push-T (n = 20). Wilson 구간과 Fisher 정확 검정
    res["V3_real_push_t"] = {}
    for name, k in T6.items():
        lo, hi = wilson(k, N_REAL)
        res["V3_real_push_t"][name] = {"k": k, "n": N_REAL, "rate": round(k / N_REAL, 3),
                                       "wilson95": [round(lo, 3), round(hi, 3)]}
    res["V4_fisher_E2E_vs"] = {}
    for other in ["DP-R3M", "DP-T-E2E", "DP-ImgNet", "LSTM-pos", "IBC-pos"]:
        table = [[T6["DP-E2E"], N_REAL - T6["DP-E2E"]], [T6[other], N_REAL - T6[other]]]
        odds, p = fisher_exact(table, alternative="two-sided")
        res["V4_fisher_E2E_vs"][other] = {"k_E2E": T6["DP-E2E"], "k_other": T6[other], "p_two_sided": float(f"{p:.4g}")}
    # 본문 과제 성공률의 Wilson 구간
    res["V5_real_tasks"] = {}
    for name, k in REAL_TASKS.items():
        lo, hi = wilson(k, N_REAL)
        res["V5_real_tasks"][name] = {"k": k, "n": N_REAL, "rate": round(k / N_REAL, 3),
                                      "wilson95": [round(lo, 3), round(hi, 3)]}
    odds, p = fisher_exact([[18, 2], [0, 20]])
    res["V5_real_tasks"]["Mug Flip DP vs LSTM-GMM Fisher p"] = float(f"{p:.4g}")

    # ---- V6: 시뮬 표 (마지막 10개 평균)의 근사 Wilson 구간.
    # 표 1 주석: 3 시드 x 50 초기조건 = 150. 단 본문 각주: robomimic은 22 초기조건만 사용 -> 3 x 22 = 66.
    # 체크포인트 평균을 이항 비율로 취급하는 근사다.
    approx = {}
    sel = [("Transport mh", 7, "DP-T", "LSTM-GMM", 66), ("Square mh", 5, "DP-T", "LSTM-GMM", 66),
           ("Lift mh", 1, "DP-T", "BET", 66)]
    for label, idx, o, b, n in sel:
        ko = round(T1[o][idx][1] * n)
        kb = round(T1[b][idx][1] * n)
        approx[label] = {"DP-T_avg10": T1[o][idx][1], "baseline_avg10": T1[b][idx][1], "baseline": b, "n_assumed": n,
                         "DP-T_wilson": [round(x, 3) for x in wilson(ko, n)],
                         "baseline_wilson": [round(x, 3) for x in wilson(kb, n)]}
    # 표 2 ToolHang (시각): DP-T 마지막 10개 평균 0.47 < LSTM-GMM 0.49
    approx["표2 ToolHang (시각) DP-T 대 LSTM-GMM, 마지막 10개 평균"] = {
        "DP-T": T2["DP-T"][8][1], "LSTM-GMM": T2["LSTM-GMM"][8][1],
        "note": "DP-T가 LSTM-GMM보다 낮다 (V9)"}
    res["V6_sim_approx"] = approx

    # ---- V7–V12: 일관성 검사 (본문 대 표)
    checks = {}
    # V7: 본문 "모든 과제와 변형에서 우위" (§5.3) — 최대 성능 기준으로 DP 최고가 기준선 최고보다 낮은 열
    def losing_cols(tbl, cols, base, ours, k):
        out = []
        for i, c in zip(cols, [COLS1[x] for x in cols]):
            mb = max(tbl[b][i][k] for b in base)
            mo = max(tbl[o][i][k] for o in ours)
            if mo <= mb:
                out.append({"col": c, "ours": mo, "base": mb, "tie" if mo == mb else "loss": True})
        return out
    checks["V7_T1_no_gain_max"] = losing_cols(T1, all_idx, BASE1, OURS, 0)
    checks["V7_T2_no_gain_max"] = losing_cols(T2, all_idx, BASE2, OURS, 0)
    checks["V7_T2_no_gain_avg10"] = losing_cols(T2, all_idx, BASE2, OURS, 1)
    checks["V7_T1_no_gain_avg10"] = losing_cols(T1, all_idx, BASE1, OURS, 1)
    # V8: 표 4의 DP-C 대 BET (BlockPush)
    checks["V8_T4_DPC_below_BET"] = {c: {"DP-C": T4["DP-C"][j], "BET": T4["BET"][j]}
                                    for j, c in enumerate(COLS4) if T4["DP-C"][j] < T4["BET"][j]}
    # V9: 표 2 ToolHang 시각, DP-T가 LSTM-GMM보다 낮음 (avg10)
    checks["V9_T2_ToolHang_DPT_below_LSTM_avg10"] = {"DP-T": T2["DP-T"][8][1], "LSTM-GMM": T2["LSTM-GMM"][8][1]}
    # V10: 표 5 (비전 인코더) 본문 대 표
    t5 = {"ResNet18": {"scratch": .94, "frozen": .58, "finetune": .92},
          "ResNet34": {"scratch": .92, "frozen": .40, "finetune": .94},
          "ViT-B/16 (CLIP)": {"scratch": .22, "frozen": .70, "finetune": .98}}
    checks["V10_T5_finetune_best_overall"] = {k: ("finetune 최고" if v["finetune"] == max(v.values()) else "finetune 아님")
                                             for k, v in t5.items()}
    checks["V10_T5_frozen_vs_scratch_ViT"] = {"frozen": .70, "scratch": .22}
    # V11: 실로봇 과제 수 (본문 §6 "4 tasks across 2 hardware setups" 대 실제 절 수)
    checks["V11_real_task_sections"] = ["Real Push-T", "Mug flip", "Sauce pouring", "Sauce spreading",
                                        "Egg beater", "Mat unrolling", "Shirt folding"]
    # V12: 시뮬 과제 수와 실로봇 과제 수의 합 (초록 15 대 본문 §6 4개)
    checks["V12_task_count"] = {"sim": 8, "real": 7, "sum": 15, "abstract_pdf": 15, "arxiv_abs_metadata": 12,
                                "v1_abstract": 11, "v4_abstract": 12}
    # V13: Table 7 대 Table 8: BlockPush 제어 방식
    checks["V13_blockpush_ctrl"] = {"Table 7 (CNN)": "Pos", "Table 8 (Transformer)": "Vel",
                                    "text §5.2": "DP는 Pos 사용"}
    # V14: Kitchen 시연 수 (본문 566 대 표 3 656)
    checks["V14_kitchen_demos"] = {"text §5.1": 566, "table 3 (#PH)": 656}
    # V15: DDIM 추론 단계 (본문 §3.4 10 대 표 7 시뮬 100, 실로봇 16)
    checks["V15_inference_steps"] = {"text §3.4": 10, "table 7 sim": 100, "table 7 real": 16}
    # V16: 실제 재현되는 최대-평균 차이의 이상값 (BET 대 Kitchen p4)
    checks["V16_kitchen_p4_improvement_vs_text"] = {"text": 2.13, "recomputed_vs_BET": round(kit_p4_bet, 3)}
    res["V7_V16_checks"] = checks

    # ---- Table 6 IoU, text checks
    res["V17_real_iou"] = {"human": 0.84, "E2E": 0.80, "text": "0.8 v.s. 0.84", "status": "일치"}
    res["V18_real_LSTM_IBC_text"] = {"text": "IBC 0%, LSTM-GMM 20%", "table": {"IBC-pos": 0, "LSTM-pos": 0.20},
                                     "status": "일치"}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 콘솔 요약
    for k, v in res["V1_avg_improvement"].items():
        print(f"{v['mean']:.4f}  N={v['N']:2d}  {k}")
    print("paper claim 0.46858")
    print("V2", res["V2_text_improvements"])
    print("V4 E2E vs", {k: v["p_two_sided"] for k, v in res["V4_fisher_E2E_vs"].items()})
    print("V5 mug p", res["V5_real_tasks"]["Mug Flip DP vs LSTM-GMM Fisher p"])
    print("V6", res["V6_sim_approx"])
    print("V7 T2 max tie/loss", checks["V7_T2_no_gain_max"])
    print("V7 T1 avg10 tie/loss", checks["V7_T1_no_gain_avg10"])
    print("V7 T2 avg10 tie/loss", checks["V7_T2_no_gain_avg10"])
    print("V8", checks["V8_T4_DPC_below_BET"])
    print("V9", checks["V9_T2_ToolHang_DPT_below_LSTM_avg10"])


if __name__ == "__main__":
    main()
