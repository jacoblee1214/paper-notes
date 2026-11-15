"""14B DiT의 순전파 FLOPs, 메모리, 7 Hz 목표 대비 필요한 실효 처리량을 계산한다(산술만, 실행 모델 없음).

입력 가정(모두 출력 JSON "assumptions"에 있음):
- Wan2.1-I2V-14B 구조: 블록 L=40, 은닉 d=5120 (공개 config 기준, 설치하지 않고 값만 가정)
- 480P 입력: VAE 시간 4배 · 공간 8배, 패치 (1,2,2) → 잠재 프레임 하나당 60×104/4 = 1560 토큰
- 청크 K = 2 잠재 프레임 (공식 AgiBot/DROID 학습 스크립트 num_frame_per_block=2, 논문 부록 C)
- 최대 문맥 8 잠재 프레임 → KV 캐시 12480 토큰 (논문 부록 C)
- 파라미터 14e9, 행동 토큰(24–48개)과 상태 토큰은 무시
- GPU 사양은 NVIDIA 공개 사양의 기억값이며 확인하지 않았다(추정 입력).

실행: python3 analysis/latency_arithmetic.py  →  analysis/latency_results.json  (표준 라이브러리만 사용)
"""
import json
from pathlib import Path

OUT = Path(__file__).with_name("latency_results.json")
P = 14e9          # 파라미터
L = 40            # 블록 수
D = 5120          # 은닉 차원
TOK_FRAME = 60 * 104 // 4  # 잠재 프레임당 토큰
N_Q = 2 * TOK_FRAME        # 새 청크(K=2)의 쿼리 토큰 = 3120
N_CTX_MAX = 8 * TOK_FRAME  # 최대 KV 문맥 = 12480

def forward_flops(n_q, n_k):
    lin = 2 * P * n_q                 # 선형층 (쿼리 토큰만)
    attn = L * 4 * D * n_q * n_k      # QK^T와 AV, 블록마다
    return lin + attn, lin, attn

f_max, lin, attn_max = forward_flops(N_Q, N_Q + N_CTX_MAX)
f_min, _, attn_min = forward_flops(N_Q, N_Q)
res = {
    "assumptions": [
        "L=40, d=5120, P=14e9 (Wan2.1-I2V-14B 구조, 값만 가정)",
        "480P, VAE 4×8×8, 패치 (1,2,2) → 1560 토큰/잠재 프레임, 청크 K=2 → 3120 쿼리 토큰",
        "최대 KV 문맥 8 잠재 프레임 = 12480 토큰 (논문 부록 C)",
        "GPU 사양(추정, 확인 안 함): B200/GB200 GPU 하나 BF16 dense 2.25 PF/s, FP8 4.5, FP4 9.0. H100 SXM BF16 dense 0.989, FP8 1.98 PF/s.",
        "사양의 MFU는 실측이 없다. 필요한 실효 처리량을 계산하고 사양과 비교한다.",
    ],
    "tokens": {"per_latent_frame": TOK_FRAME, "query_chunk": N_Q, "kv_context_max": N_CTX_MAX},
    "forward_flops_TF": {
        "linear_TF": round(lin / 1e12, 1),
        "attention_TF_max_context": round(attn_max / 1e12, 1),
        "total_TF_max_context": round(f_max / 1e12, 1),
        "total_TF_no_context": round(f_min / 1e12, 1),
    },
}

# 실효 처리량 = (DiT 순전파 수 × 순전파당 FLOPs) / 청크 지연. 청크당 지연 목표 150 ms(논문 Table 3).
# 논문 구성: CFG 두 분기를 GPU 두 개에 나눔 → GPU 하나당 순전파 = 단계 수 + KV 캐시 갱신 1회.
# 논문 구성은 두 GPU 모두 같은 순전파 수를 돈다고 가정.
def per_gpu_tflops_needed(steps, latency_s, flops=f_max):
    n_forward = steps + 1
    return n_forward * flops / latency_s / 1e12  # TF/s per GPU

cases = [
    ("논문 Table 3: 4단계(캐시), 350 ms", 4, 0.350),
    ("논문 Table 3: Flash 1단계, 150 ms", 1, 0.150),
    ("논문 목표: 7 Hz 청크 주기 143 ms, 4단계", 4, 1 / 7),
    ("공개 코드 기본 mask(8 DiT 단계), README GB200 0.6 s", 8, 0.6),
]
res["per_gpu_effective_TFs_needed"] = {}
for name, steps, lat in cases:
    need = per_gpu_tflops_needed(steps, lat)
    res["per_gpu_effective_TFs_needed"][name] = {
        "dit_steps": steps, "latency_s": round(lat, 3),
        "needed_TF_per_s": round(need, 0),
        "fraction_of_B200_bf16_dense_pct": round(100 * need / 2250, 0),
        "fraction_of_B200_fp8_dense_pct": round(100 * need / 4500, 0),
        "fraction_of_B200_fp4_dense_pct": round(100 * need / 9000, 0),
        "fraction_of_H100_bf16_dense_pct": round(100 * need / 989, 0),
    }
res["naive_single_gpu_5.7s_16_steps_cfg"] = {
    "forwards_per_chunk_incl_cache_update": 16 * 2 + 1,
    "needed_TF_per_s": round((16 * 2 + 1) * f_max / 5.7 / 1e12, 0),
    "fraction_of_B200_bf16_dense_pct": round(100 * (16 * 2 + 1) * f_max / 5.7 / 1e12 / 2250, 0),
    "note": "본문은 단일 GPU 종류를 밝히지 않았다. H100(989 TF)이면 이 값을 넘어서 불가능에 가깝다. 값이 높으면 토큰 수 또는 FLOPs 추정이 과대하다."}

# 선형 적합(표 3): t = a + b N. N = DiT 단계 수 (캐시 갱신 순전파는 a에 들어감)
b = (350 - 150) / 3
res["linear_fit_from_table3_ms"] = {"per_step_ms": round(b, 1), "intercept_ms": round(350 - 4 * b, 1)}

# 메모리 (GiB). 가중치 BF16 = 2 bytes, KV 캐시 BF16 = 2 bytes, 분기마다 하나
GiB = 2 ** 30
res["memory_GiB"] = {
    "weights_bf16_14B": round(P * 2 / GiB, 1),
    "weights_int4_14B": round(P * 0.5 / GiB, 1),
    "kv_cache_one_branch_bf16_max": round(L * 2 * N_CTX_MAX * D * 2 / GiB, 1),
    "kv_cache_two_branches_bf16_max": round(2 * L * 2 * N_CTX_MAX * D * 2 / GiB, 1),
    "checkpoint_files_DreamZero_DROID_GB": 64.79,
    "checkpoint_files_DreamZero_AgiBot_GB": 45.85,
    "mac_unified_memory_GiB": 24,
    "note": "체크포인트 파일 합계가 14B bf16(약 26 GiB)보다 크다. 저장 형식(fp32 또는 텍스트 인코더 포함)은 확인하지 않았다."}
mac = res["memory_GiB"]
res["memory_verdict_mac"] = {
    "bf16_weights_fit_24GiB": mac["weights_bf16_14B"] < 24,
    "int4_weights_plus_one_kv_branch_GiB": round(mac["weights_int4_14B"] + mac["kv_cache_one_branch_bf16_max"], 1),
    "int4_plus_two_kv_branches_GiB": round(mac["weights_int4_14B"] + mac["kv_cache_two_branches_bf16_max"], 1),
}

# 맥 추정: 실효 처리량 가정 5–10 TF/s(추정, 미측정). 순차 실행, CFG 두 분기 + 캐시 갱신 1회 × 2분기
mac_forwards_4 = 2 * 4 + 2
mac_forwards_1 = 2 * 1 + 2
res["mac_chunk_latency_estimate_s"] = {}
for tfs in (5, 10):
    res["mac_chunk_latency_estimate_s"][f"{tfs}TF_per_s"] = {
        "4_steps": round(mac_forwards_4 * f_max / (tfs * 1e12), 0),
        "1_step": round(mac_forwards_1 * f_max / (tfs * 1e12), 0),
        "ratio_to_150ms_4step": round(mac_forwards_4 * f_max / (tfs * 1e12) / 0.150, 0),
    }
res["mac_note"] = "가정 처리량은 측정하지 않았다. 결론은 처리량을 2배 바꿔도 변하지 않는다(청크당 수십 초에서 수 분). 메모리에도 들어가지 않는다."

# 반응 지연 구조(논문 3.2.2): 관측에서 행동까지 = 추론 시간 + 다음 청크로 교체되기까지 대기(최대 한 주기)
res["reactivity"] = {
    "inference_150ms_worst_wait_one_period_s": round(0.150 + 1 / 7, 3),
    "note": "비동기 실행이라도 관측이 행동에 닿는 데 추론 시간과 교체 대기가 더해진다. 논문의 '7 Hz'는 청크 주기이고 관측-행동 지연은 따로 보고하지 않았다."}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
print(json.dumps(res["forward_flops_TF"]))
for k, v in res["per_gpu_effective_TFs_needed"].items():
    print(k, v["needed_TF_per_s"], v["fraction_of_B200_bf16_dense_pct"], v["fraction_of_B200_fp4_dense_pct"])
print(json.dumps(res["naive_single_gpu_5.7s_16_steps_cfg"]))
print(json.dumps(res["memory_GiB"]), json.dumps(res["memory_verdict_mac"]))
print(json.dumps(res["mac_chunk_latency_estimate_s"]))
