"""온보드 지연 예산 toy (CPU numpy, 수 초).

세 부분으로 나눈다.
  (1) 디코드 대역폭 상한: 토큰 하나를 만들 때 읽는 바이트 / 메모리 대역폭 = 이론 tok/s 상한
  (2) 지연 모델: 블로그 파이프라인의 단계별 시간(모델 카드의 Thor 레시피 README 표)을 더해 요청 지연을 세고,
      측정 조건(롤링 대 콜드 스타트, Euler 단계 수)이 얼마나 바뀌는지 본다
  (3) 추측 디코딩: 수락률 α에서 라운드당 확정 토큰 수 E(α) = sum_{k=0}^{15} α^k (블록 16, 초안 15개),
      그리고 몬테카를로로 같은 값을 확인한다. 또 CoT 길이 L에 따른 라운드 수와 지연 trade-off를 본다.

가정(모두 블로그·레시피에 없는 값이거나 사전 지식이라 확인이 필요함)
  - 메모리 대역폭 273 GB/s: Apple M4 Pro와 DRIVE/Jetson Thor 모두 사전 지식 값. 확인 필요. 민감도도 같이 본다.
  - 텍스트 디코더 파라미터 1.409B(층) + 축소 LM head 8192행×2048. config_check.py의 값을 쓴다.
  - 프롬프트 KV는 3300 토큰 가정(레시피 예시 클립 약 3300), FP8 KV 1 B/원소.
  - 콜드 스타트 ViT 시간은 프레임 수에 선형이라고 가정(22/6). 실제 값은 측정이 필요하다.
실행: python3 analysis/latency_toy.py  → analysis/latency_toy_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GB = 1e9
TEXT_LAYER_PARAMS = 1.409e9        # 28층 어텐션+MLP (config_check.py)
HEAD_FULL = 135168 * 2048          # 원래 LM head 행 수
HEAD_REDUCED = 8192 * 2048         # 레시피의 축소 head (coc_vocab 2211 토큰, 8192행 패딩)
KV_PER_TOKEN_BF16 = 114688         # 28층 × 8 KV heads × 128 × (K,V) × 2 B
CTX = 3300

# 실측 및 블로그/레시피 수치 (출처는 04 문서)
T_TOKEN_GREEDY_MS = 7.0            # 레시피 README: greedy 토큰당 약 7 ms (Thor, FP8)
T_ROUND_MS = 12.0                  # 레시피 README: 추측 라운드(초안+검증) 12 ms
T_VIT6_MS, T_PREFILL_MS, T_EXPERT_MS = 20.0, 75.0, 70.0   # expert = 5 Euler × 6 후보
EULER_STEPS_REF = 5
N_CAND = 6
N_FRAMES_ROLL, N_FRAMES_ALL = 6, 22
BW_GBPS = {"가정 273 GB/s": 273.0, "보수 200 GB/s": 200.0}
BLOG_RANGE = (160.0, 190.0)
RECIPE_RANGE = (180.0, 250.0)     # 레시피: 10개 내부 검증 클립, 1~7 라운드
TEN_HZ_MS = 100.0


def bytes_per_token(head_params, kv_bytes):
    """토큰 하나: 텍스트 층 + head 전체를 읽는다고 가정(FP8이면 1 B/원소)."""
    return (TEXT_LAYER_PARAMS + head_params) * kv_bytes[0] + CTX * KV_PER_TOKEN_BF16 * kv_bytes[1]


def bandwidth_part():
    rows = []
    fmts = {
        "BF16 가중치, 전체 head, BF16 KV": (HEAD_FULL, (2, 1.0)),
        "FP8 가중치, 전체 head, BF16 KV": (HEAD_FULL, (1, 1.0)),
        "FP8 가중치, 축소 head(레시피), FP8 KV": (HEAD_REDUCED, (1, 0.5)),
    }
    for name, (head, (wb, kvf)) in fmts.items():
        b = (TEXT_LAYER_PARAMS + head) * wb + CTX * KV_PER_TOKEN_BF16 * kvf
        row = {"format": name, "GB_per_token": round(b / GB, 3)}
        for dev, bw in BW_GBPS.items():
            row[f"ceiling_tok_s_{dev}"] = round(bw * GB / b, 1)
        rows.append(row)
    # 레시피의 측정 7 ms/토큰을 FP8 축소 구성의 바이트로 나누면 실효 대역폭
    b_recipe = (TEXT_LAYER_PARAMS + HEAD_REDUCED) * 1 + CTX * KV_PER_TOKEN_BF16 * 0.5
    implied = b_recipe / (T_TOKEN_GREEDY_MS / 1000) / GB
    return rows, {"recipe_implied_GBps": round(implied, 1),
                  "fraction_of_273": round(implied / 273.0, 3),
                  "note": "7 ms/토큰이 대역폭 바운드라고 보면 실효 대역폭은 약 230 GB/s (273 대비 약 85%). 커널 효율이 높다는 뜻이다."}


def request_latency(rounds, euler=EULER_STEPS_REF, cold=False):
    """요청 지연 (ms). 롤링은 최신 6프레임 ViT만, 콜드는 22프레임 전부(선형 가정)."""
    vit = T_VIT6_MS * (N_FRAMES_ALL / N_FRAMES_ROLL if cold else 1.0)
    expert = T_EXPERT_MS * euler / EULER_STEPS_REF
    return vit + T_PREFILL_MS + rounds * T_ROUND_MS + expert


def latency_part():
    rows = []
    for r in range(0, 8):
        rows.append({"rounds": r,
                     "rolling_5steps_ms": round(request_latency(r), 1),
                     "rolling_10steps_ms": round(request_latency(r, euler=10), 1),
                     "cold_5steps_est_ms": round(request_latency(r, cold=True), 1)})
    # 블로그 범위 160–190과 레시피 범위 180–250이 어떤 라운드 수에 해당하는지
    fit_blog = [r for r in range(0, 8) if BLOG_RANGE[0] <= request_latency(r) <= BLOG_RANGE[1]]
    fit_recipe = [r for r in range(0, 8) if RECIPE_RANGE[0] <= request_latency(r) <= RECIPE_RANGE[1]]
    summary = {
        "floor_one_round_ms": round(request_latency(1), 1),
        "blog_range_rounds": fit_blog,
        "recipe_range_rounds": fit_recipe,
        "example_clip_4_rounds_ms": round(request_latency(4), 1),
        "euler_5_to_10_extra_ms": round(request_latency(1, euler=10) - request_latency(1), 1),
        "cold_start_penalty_1_round_ms_est": round(request_latency(1, cold=True) - request_latency(1), 1),
        "ten_hz_budget_ms": TEN_HZ_MS,
        "ten_hz_gap_ms_at_1_round": round(request_latency(1) - TEN_HZ_MS, 1),
    }
    return rows, summary


def spec_expected(alpha, block=16):
    """라운드당 확정 토큰: 초안 block-1개 중 맞은 연속 길이 + 교정 토큰 1개 (독립 수락 가정)."""
    k = np.arange(block)  # 0..15
    return float(np.sum(alpha ** k))


def rounds_for_tokens(alpha, L, rng, trials=4000, block=16):
    """길이 L 토큰을 만드는 데 필요한 라운드 수의 평균 (몬테카를로)."""
    counts = []
    for _ in range(trials):
        made, r = 0, 0
        while made < L:
            # 맞은 초안 수 ~ min(Geom, block-1), 교정 1개 추가
            acc = 0
            while acc < block - 1 and rng.random() < alpha:
                acc += 1
            made += acc + 1
            r += 1
        counts.append(r)
    return float(np.mean(counts))


def spec_part():
    rng = np.random.default_rng(0)
    alphas = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for a in alphas:
        e_formula = spec_expected(a)
        mc = []
        for _ in range(20000):
            acc = 0
            while acc < 15 and rng.random() < a:
                acc += 1
            mc.append(acc + 1)
        rows.append({"alpha": a,
                     "E_formula": round(e_formula, 3),
                     "E_montecarlo": round(float(np.mean(mc)), 3),
                     "speedup_vs_greedy": round(e_formula * T_TOKEN_GREEDY_MS / T_ROUND_MS, 2)})
    # 손익분기: E*7/12 = 1  →  E = 12/7
    target = T_ROUND_MS / T_TOKEN_GREEDY_MS
    lo, hi = 0.0, 0.99
    for _ in range(60):
        mid = (lo + hi) / 2
        if spec_expected(mid) < target:
            lo = mid
        else:
            hi = mid
    breakeven_alpha = round(hi, 3)
    # 예시 클립: 4 라운드. CoT 길이 L을 모르므로 L별로 어느 α가 4 라운드인지 본다.
    rounds_table = []
    for L in (10, 25, 50, 100):
        for a in (0.5, 0.7, 0.8, 0.9):
            rounds_table.append({"L_tokens": L, "alpha": a,
                                 "rounds_MC": round(rounds_for_tokens(a, L, rng, trials=1500), 2)})
    # 지연: CoT L토큰, 롤링 5스텝, 라운드 = 평균 수
    lat_table = []
    for L in (10, 25, 50, 100, 256):
        greedy = request_latency(0) + L * T_TOKEN_GREEDY_MS
        a = 0.8
        rounds = rounds_for_tokens(a, L, rng, trials=1500)
        spec = request_latency(0) + rounds * T_ROUND_MS
        lat_table.append({"L_tokens": L, "greedy_ms": round(greedy, 1),
                          "spec_alpha0.8_rounds": round(rounds, 2), "spec_alpha0.8_ms": round(spec, 1)})
    return rows, {"breakeven_alpha": breakeven_alpha,
                  "rounds_for_L_alpha": rounds_table,
                  "latency_for_L_alpha0.8": lat_table}


def nominal_part():
    """NOMINAL(블로그 주장, 공개 코드에서는 미확인) 비율 f에 따른 평균 지연. 토큰 없는 프레임은 CoT 생략."""
    out = []
    for f in (0.0, 0.5, 0.9):
        avg = (1 - f) * request_latency(2) + f * (request_latency(0))
        out.append({"nominal_fraction": f, "avg_ms_CoT_2rounds_NOMINAL_0rounds": round(avg, 1)})
    return out


def main():
    bw_rows, bw_meta = bandwidth_part()
    lat_rows, lat_sum = latency_part()
    sp_rows, sp_sum = spec_part()
    out = {
        "assumptions": {"CTX": CTX, "bw_GBps": BW_GBPS, "T_token_greedy_ms": T_TOKEN_GREEDY_MS,
                        "T_round_ms": T_ROUND_MS, "stage_ms": [T_VIT6_MS, T_PREFILL_MS, T_EXPERT_MS]},
        "bandwidth": {"rows": bw_rows, **bw_meta},
        "latency": {"rows": lat_rows, **lat_sum},
        "speculative": {"rows": sp_rows, **sp_sum},
        "nominal_sensitivity": nominal_part(),
    }
    (ROOT / "analysis" / "latency_toy_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
