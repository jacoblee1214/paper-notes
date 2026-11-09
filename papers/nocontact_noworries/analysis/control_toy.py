"""NoContactNoWorries 제어 toy (CPU, numpy, 수 초). 실행: python3 analysis/control_toy.py

질문: 이진 접촉 신호에 오류(위양성 FPR, 위음성 FNR)가 있을 때, 신호로 멈추는 간단한 제어기는 얼마나 망가지나?
그리고 논문 표의 F1(0.71–0.93)은 그 오류율 중 어떤 조합과 맞는가?

T3. 접촉 기반 정지 제어기: 손가락이 20프레임(30 Hz) 동안 접근한 뒤 접촉이 시작된다.
    검출기가 K 프레임 연속 1을 내면 멈춘다(디바운스 K).
    성공 = 접촉 시작 후 0–Dmax 프레임(0–100 ms) 안에 멈춘 경우.
    실패 = 접촉 전에 멈춤(허공에서 멈춤, 위양성) 또는 Dmax 안에 못 멈춤(과압, 위음성).
    검출 오류는 프레임마다 독립이라고 가정했다. 논문 모델은 인과 transformer로 시간 상관이 있으므로
    이 가정은 위양성에 대해 비관적인 쪽이다.

T4. F1 ↔ (정밀도, 재현율, FPR) 변환: F1 = 2PR/(P+R)에서 P = F1·R/(2R − F1).
    접촉 비율 π(프레임 기준)가 있으면 FPR = R·π·(1/P − 1)/(1 − π).
    이 식으로 "F1이 같아도 위양성률이 얼마나 달라질 수 있는가"를 보고, 그 FPR/FNR을 T3에 넣는다.

결과: analysis/control_toy_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_PRE = 20   # 접촉 전 프레임 수
D_MAX = 3    # 접촉 후 허용 지연 프레임 (100 ms)
T = N_PRE + D_MAX + 1
N_EP = 20000


def success_rate(fpr, fnr, K, rng, n=N_EP):
    c = np.zeros((n, T), bool)
    c[:, N_PRE:] = True
    u = rng.random((n, T))
    d = np.where(c, u >= fnr, u < fpr)        # 접촉이면 1을 놓칠 확률 fnr, 아니면 1을 낼 확률 fpr
    run = np.zeros(n, int)
    stop = np.full(n, 10 ** 6)
    for t in range(T):
        run = np.where(d[:, t], run + 1, 0)
        newly = (stop == 10 ** 6) & (run >= K)
        stop[newly] = t
    ok = (stop >= N_PRE) & (stop <= N_PRE + D_MAX)
    early = stop < N_PRE
    return {"success": round(float(ok.mean()), 4), "early_stop_false_contact": round(float(early.mean()), 4),
            "missed_or_late": round(float(1 - ok.mean() - early.mean()), 4)}


def f1_to_rates(F1, R, pi):
    P = F1 * R / (2 * R - F1)
    if not (0 < P <= 1):
        return None
    fpr = R * pi * (1 / P - 1) / (1 - pi)
    return {"precision": round(P, 3), "recall": round(R, 3), "fpr": round(fpr, 3), "fnr": round(1 - R, 3)}


def main():
    rng = np.random.default_rng(1)
    res = {"settings": {"N_PRE": N_PRE, "D_MAX_frames": D_MAX, "episodes": N_EP, "fps": 30}}

    # T3 격자: FPR × FNR × 디바운스 K
    grid = {}
    for K in (1, 2, 3):  # K=5는 접촉 후 5프레임(167 ms)이 필요해 허용 지연 100 ms를 넘으므로 항상 실패 → 제외
        for fnr in (0.0, 0.1, 0.2, 0.3):
            for fpr in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05):
                grid[f"K={K},fnr={fnr},fpr={fpr}"] = success_rate(fpr, fnr, K, rng)
    res["T3_grid"] = grid

    # T3 요약: 위양성만 있을 때(fnr=0) 한 에피소드 기준 K별 성공률 감소
    res["T3_fpr_only_success"] = {f"K={K}": {str(f): grid[f"K={K},fnr=0.0,fpr={f}"]["success"]
                                              for f in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)} for K in (1, 2, 3)}

    # T4: 논문 Table I의 F1 값에 대응하는 (P, R, FPR) 범위 (π = 0.2, 0.4, 0.6; R = 0.7–0.95)
    t4 = []
    for F1 in (0.71, 0.74, 0.79, 0.84, 0.93):
        for pi in (0.2, 0.4, 0.6):
            row = {"F1": F1, "pi": pi, "R_grid": []}
            for R in (0.7, 0.8, 0.9, 0.95):
                m = f1_to_rates(F1, R, pi)
                if m is None:
                    continue
                m["controller_K1_success"] = success_rate(m["fpr"], m["fnr"], 1, rng)["success"]
                m["controller_K3_success"] = success_rate(m["fpr"], m["fnr"], 3, rng)["success"]
                row["R_grid"].append(m)
            t4.append(row)
    res["T4_f1_mapping"] = t4

    (ROOT / "analysis" / "control_toy_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["T3_fpr_only_success"], indent=1))
    for fnr in (0.0, 0.1, 0.2):
        print(f"fnr={fnr}:", {K: [grid[f"K={K},fnr={fnr},fpr={f}"]["success"] for f in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)] for K in (1, 2, 3)})
    print("K=3 fnr=0.1 by fpr:", {f: grid[f"K=3,fnr=0.1,fpr={f}"]["success"] for f in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)})
    print("K=1 fnr=0.1 by fpr:", {f: grid[f"K=1,fnr=0.1,fpr={f}"]["success"] for f in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)})
    for row in t4:
        if row["pi"] == 0.4:
            print(row["F1"], [(m["recall"], m["fpr"], m["controller_K1_success"], m["controller_K3_success"]) for m in row["R_grid"]])


if __name__ == "__main__":
    main()
