"""논문 식 (2), (11), (4)와 공식 코드의 계산을 나란히 비교한다 (CPU, 1초 안팎).

코드 기준: Isaac-GR00T/gr00t/model/action_head/freebar.py, kinetix/src/model.py (freebar_action).
출력: analysis/formula_check_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("formula_check_results.json")
res = {}

# (1) 중간 구간 가중치: 식 (11)은 분모 hs-d, 코드는 h-d+1 (safe_span)
def w_paper(H, d, hs, beta):
    w = np.zeros(H)
    w[:d] = 1.0
    for i in range(d, hs):
        w[i] = math.exp(-beta * (i - d + 1) / (hs - d))
    return w

def w_code(H, d, h, beta):
    w = np.zeros(H)
    if d > 0:
        w[:d] = 1.0
    if h > d:
        for i in range(d, h):
            w[i] = math.exp(-beta * (i - d + 1.0) / max(h - d + 1, 1))
    return w

H, d, h, beta = 16, 3, 6, 3.0  # RoboCasa 설정 (README: --n_action_steps 6 --inference_delay 3 --override_action_horizon 16)
wp, wc = w_paper(H, d, h, beta), w_code(H, d, h, beta)
res["weights_robocasa_setting"] = {
    "paper_eq11": np.round(wp[:h + 1], 4).tolist(),
    "code": np.round(wc[:h + 1], 4).tolist(),
    "max_abs_diff": float(np.max(np.abs(wp - wc))),
    "note": "i=h에서 두 식 모두 0이 되지만 i=h-1의 값이 조금 다름",
}

# (2) 에너지: 논문 식 (2)는 가중치를 한 번 곱하고(w̃_i), 코드는 h_mid에서 가중치를 제곱한다(w̃_i^2).
e = np.ones((H, 1))  # 단위 오차 (모든 차원 1)
E_paper = 0.5 * np.sum(wp[:, None] * e ** 2)
E_code = 0.5 * np.sum((wc[:, None] * e) ** 2)
res["energy_w_vs_w2"] = {
    "E_paper_eq2": round(float(E_paper), 4),
    "E_code_h_total": round(float(E_code), 4),
    "ratio_code_over_paper": round(float(E_code / E_paper), 4),
    "note": "코드의 g는 w̃ (선형)에서 나오므로 g는 식 (3)과 같고 b만 다르다. 식 (2)의 ∇B와 코드의 g가 일치하지 않는다",
}

# (3) 시간 감쇠 τ(t): 논문 τ=1-t, 코드 τ=max(1-t, 1/max_guidance_weight)
sched = {}
for name, K, cap in [("GR00T (K=4, cap=10)", 4, 10.0), ("Kinetix (K=5, cap=5)", 5, 5.0), ("실로봇 (K=10, cap=10 가정)", 10, 10.0)]:
    ts = [k / K for k in range(K)]
    tau_paper = [1 - t for t in ts]
    tau_code = [max(1 - t, 1.0 / cap) for t in ts]
    sched[name] = {
        "t": [round(t, 3) for t in ts],
        "tau_paper": [round(x, 3) for x in tau_paper],
        "tau_code": [round(x, 3) for x in tau_code],
        "floor_active": any(tc > tp + 1e-12 for tp, tc in zip(tau_paper, tau_code)),
    }
res["tau_schedule"] = sched

# (4) 보정 상한의 대상: 코드는 λ 자체를 [0, cap]으로 자른다. 논문 본문은 '보정 크기'에 상한이라고 쓴다.
res["lambda_cap_note"] = "코드: clamp(λ, 0, cap). 논문 본문: 보정 크기 상한 10(GR00T). 상한의 대상이 λ인지 λ‖g‖인지 논문에 명시되지 않음"

# (5) 식 (4)-(5)의 닫힌형 해가 g^T ṽ ≥ b를 만족하는지 (임의 입력 1000개)
rng = np.random.default_rng(0)
ok = []
for _ in range(1000):
    g = rng.normal(size=(H, 3)); v = rng.normal(size=(H, 3)); b = rng.normal() * 2
    lam = max((b - np.sum(g * v)) / (np.sum(g * g) + 1e-8), 0.0)
    vt = v + lam * g
    if lam > 0:
        ok.append(bool(np.sum(g * vt) >= b - 1e-6))
res["closed_form_constraint_check"] = {"trials_with_lambda_positive": len(ok), "all_satisfied": all(ok)}

# (6) Kinetix 평가 경로: 참조 청크는 실행한 h개만큼 밀고 뒤를 0으로 채운다. RTC는 H-h 이후 가중치를 0으로 두지만
# FreeBAR는 그렇게 하지 않는다. 가중치가 0 목표(패딩)에 얼마나 실리는지 (H=8, 평가 격자 d=0..4, h=max(1,d)..8-d).
Hk = 8
rows = []
for dd in range(0, 5):
    for hh in range(max(1, dd), 8 - dd + 1):
        w = np.zeros(Hk)
        if dd > 0:
            w[:dd] = 1.0
        for i in range(dd, hh):
            w[i] = math.exp(-3.0 * (i - dd + 1) / max(hh - dd + 1, 1))
        pad = np.zeros(Hk, dtype=bool)
        pad[Hk - hh:] = True  # 밀고 남은 뒤쪽 h칸은 0
        tot = w.sum()
        on_pad = w[pad].sum()
        rows.append({"d": dd, "h": hh, "pad_indices_with_weight": [int(i) for i in np.where(pad & (w > 0))[0]],
                     "share_of_weight_on_zero_targets": round(float(on_pad / tot), 3) if tot > 0 else 0.0,
                     "max_weight_on_zero_target": round(float(w[pad].max()), 3) if pad.any() else 0.0})
res["kinetix_zero_padding_weight"] = {
    "H": Hk,
    "note": "RTC는 prefix_attention_horizon=H-h 이후를 0으로 막는다. FreeBAR는 막지 않는다. 0 목표에 실린 가중치 비율을 적었다",
    "rows_with_padding_weight": [r for r in rows if r["pad_indices_with_weight"]],
}

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=1))
