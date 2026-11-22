"""PLaW-VLA 월드 모델 가지의 파라미터 수와 토큰 수를 설정값에서 직접 센다(모델을 만들지 않는다, 표준 라이브러리만).

입력: 공식 코드의 설정값(Gemma 300M 전문가, V-JEPA 2 변형 표, 시간 오프셋)과 논문 4.4절의 수치.
출력: analysis/cost_accounting_results.json, 표준출력.
실행: python3 analysis/cost_accounting.py   (1초 미만)

이 계산은 정적 추정이다. 지연 시간은 GPU가 없어 측정하지 않았다.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "cost_accounting_results.json"


def gemma_params(width, depth, mlp_dim, heads, kv_heads, head_dim):
    attn = width * heads * head_dim + 2 * width * kv_heads * head_dim + heads * head_dim * width
    mlp = 3 * width * mlp_dim  # gate, up, down
    return depth * (attn + mlp)


def vit_params(hidden, layers, mlp_ratio, patch=16, tubelet=2, in_ch=3):
    attn = 4 * hidden * hidden  # q, k, v, o
    mlp = 2 * hidden * int(round(hidden * mlp_ratio))
    embed = in_ch * tubelet * patch * patch * hidden
    return layers * (attn + mlp) + embed


# 코드 값 (models/gemma.py, models/vjepa2.py)
wm_expert = gemma_params(1024, 18, 4096, 8, 1, 256)  # gemma_300m, 주석은 311M
variants = {
    "vitl-256": vit_params(1024, 24, 4.0),
    "vith-256": vit_params(1280, 32, 4.0),
    "vitg-256": vit_params(1408, 40, 48 / 11),
}
slots = 768 * 1024 * 2  # slot_embed + horizon_embed (wm_slot_max_len = 768)
pred_head = 1024 * 1024 + 1024  # LayerNorm 생략, Linear 1024 -> 1024 (vitl 기준)
res = {"wm_expert_params_M": round(wm_expert / 1e6, 1)}
res["added_params_B"] = {}
for k, v in variants.items():
    extra_proj = 0 if k == "vitl-256" else (1280 if k == "vith-256" else 1408) * 1024  # 입력 투영기(LayerNorm 제외)
    head = {"vitl-256": 1024, "vith-256": 1280, "vitg-256": 1408}[k]
    total = wm_expert + v + slots + (1024 * head + head) + extra_proj
    res["added_params_B"][k] = round(total / 1e9, 3)
res["paper_claim_added_params_B"] = 0.9
res["note_params"] = ("공식 기본값 vitl-256으로 세면 약 0.62B이고 논문 4.4절의 0.9B와 맞지 않는다. "
                      "vith-256으로 세면 약 0.94B로 0.9B에 가깝다. 논문은 V-JEPA 2의 크기를 적지 않는다. "
                      "HF 체크포인트의 예측기(predictor) 헤드를 포함하면 수십 M이 더해질 수 있다.")

# 토큰 수 (config.py의 시간 오프셋, tubelet 2, 256 공간 토큰)
tok_per_tubelet = (256 // 16) ** 2
hist_frames, fut_frames = 6, 6
hist_tokens = hist_frames // 2 * tok_per_tubelet
fut_tokens = fut_frames // 2 * tok_per_tubelet
res["tokens"] = {"tokens_per_tubelet": tok_per_tubelet, "history_tokens": hist_tokens, "future_slot_tokens": fut_tokens,
                 "future_units_in_code": fut_frames // 2, "future_units_in_paper_K": 2}
# 월드 모델 전문가는 한 번의 순방향에서 [history + slots] 1536 토큰, 그다음 [history + 예측] 1536 토큰을 다시 처리한다.
wm_pass_tokens = hist_tokens + fut_tokens
res["wm_expert_tokens_per_forward_pass"] = wm_pass_tokens
res["wm_expert_passes_per_control_step"] = 2
# 대략의 곱셈-덧셈 연산량 (2 * params * tokens, 어텐션 제외)
flops_wm = 2 * wm_expert * wm_pass_tokens * 2
flops_vjepa_l = 2 * variants["vitl-256"] * (hist_tokens)  # 인코더는 과거 6프레임만(추론)
res["approx_GFLOPs_wm_expert_2_passes"] = round(flops_wm / 1e9, 1)
res["approx_GFLOPs_vjepa_l_history_only"] = round(flops_vjepa_l / 1e9, 1)
# 비교: π0.5 접두부(SigLIP+Gemma 2B) 입력 토큰 수는 이미지 3장 x 256 + 언어 약 200 정도
res["note_flops"] = "어텐션 비용과 KV 캐시 재사용을 무시한 근사다. 실제 지연은 측정하지 않았다."
# 추론 시 과거 프레임 버퍼 요구
res["history_buffer_s"] = 1.0
res["history_frame_spacing_s"] = 0.2
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
