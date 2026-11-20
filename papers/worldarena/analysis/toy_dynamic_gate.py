"""WorldArena 동적 게이트 toy (numpy). 실행: python3 analysis/toy_dynamic_gate.py
검증 대상: 논문 식 (9)·(10)·(13)의 정적 영상 페널티(min(1, Sdyn/γ))와 코드 구현(dyn <= 0.1213 이면 곱함)이
같은 점수를 주는지, 그리고 임계 근처에서 점수가 얼마나 불연속인지 본다.
입력은 가상의 원점수(raw)와 Sdyn 값뿐이다. 실제 영상·모델은 쓰지 않는다.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_dynamic_gate_results.json"
GAMMA_CODE = 0.1213  # subject_consistency.py / background_consistency.py / flow_aepe_metrics.py 공통 상수


def paper_gate(raw, dyn, gamma):
    return raw * np.minimum(1.0, dyn / gamma)


def code_gate(raw, dyn, thr=GAMMA_CODE):
    return np.where(dyn <= thr, raw * dyn, raw)


dyn_grid = np.linspace(0.0, 0.5, 501)
raw = 0.90  # 정적에 가까운 영상도 raw 유사도는 높다고 가정
p = paper_gate(raw, dyn_grid, GAMMA_CODE)
c = code_gate(raw, dyn_grid, GAMMA_CODE)
jump_idx = np.argmax(np.abs(np.diff(c)))
res = {
    "max_abs_gap_paper_vs_code": round(float(np.max(np.abs(p - c))), 4),
    "code_jump_at_threshold": {
        "dyn_before": round(float(dyn_grid[jump_idx]), 4),
        "dyn_after": round(float(dyn_grid[jump_idx + 1]), 4),
        "score_before": round(float(c[jump_idx]), 4),
        "score_after": round(float(c[jump_idx + 1]), 4),
    },
    "paper_formula_continuous_at_gamma": bool(np.max(np.abs(np.diff(p))) < 0.01),
}

# 두 모델 비교: A는 정적 (dyn 0.10, raw 0.95), B는 움직임 있음 (dyn 0.13, raw 0.85)
# 코드 규칙에서는 A가 0.95*0.10=0.095로 떨어지고 B는 0.85로 남는다. 논문 규칙에서는 A가 0.95*0.10/0.1213=0.78.
A = dict(raw=0.95, dyn=0.10)
B = dict(raw=0.85, dyn=0.13)
res["two_model_ranking"] = {
    "paper_rule": {"A": round(float(paper_gate(A["raw"], A["dyn"], GAMMA_CODE)), 4),
                   "B": round(float(paper_gate(B["raw"], B["dyn"], GAMMA_CODE)), 4)},
    "code_rule": {"A": round(float(code_gate(A["raw"], A["dyn"])), 4),
                  "B": round(float(code_gate(B["raw"], B["dyn"])), 4)},
}

# 감도: 정적 영상 비율이 달라질 때 평균 점수(한 지표) 차이. 임계 0.1213을 0.10 ~ 0.20으로 옮기면
rng = np.random.default_rng(0)
N = 2000
dyn_sim = rng.uniform(0.0, 0.3, N)           # 가상 영상 Sdyn 분포
raw_sim = rng.uniform(0.80, 0.95, N)         # 가상 raw 유사도
sens = {}
for thr in [0.10, 0.1213, 0.15, 0.20]:
    sens[str(thr)] = {
        "mean_code_rule": round(float(code_gate(raw_sim, dyn_sim, thr).mean()), 4),
        "mean_paper_rule_with_same_gamma": round(float(paper_gate(raw_sim, dyn_sim, thr).mean()), 4),
    }
res["threshold_sensitivity_on_random_videos"] = sens
res["note"] = ("논문은 γ 값을 명시하지 않는다. 코드는 0.1213을 쓰며 식도 다르다(곱 대 min(1,·/γ)). "
               "toy는 영상 분포를 가정한 것이므로 절대 점수가 아니라 규칙 차이의 크기만 본다.")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
