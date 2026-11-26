"""공식 코드의 상수와 설정값으로 논문 부록의 파라미터 수 등을 산술로 다시 계산한다. 코드를 실행하거나 가져오지 않는다.

실행: python3 analysis/code_arithmetic.py   (표준 라이브러리만, 1초 미만)

읽은 저장소: https://github.com/Yoonkyo/mu0  (커밋 963d5f13a94df31921d22887ff8611763b48178d, 2026-10-27)
  git clone <url> repo && git -C repo checkout 963d5f13a94df31921d22887ff8611763b48178d  (분석할 때는 얕은 복제)
  - src/lerobot/policies/mu0/mu0_policy.py (GemmaActionExpert, GatedMotionMixer, Mu0Policy)
  - src/lerobot/policies/pi_gemma.py (PiGemmaRMSNorm: adaRMS 밀집층 dim*3)
  - src/lerobot/policies/mu0/smolvlm_with_expert.py (트레이스 전문가 구성)
읽은 저장소: https://github.com/Yoonkyo/TraceExtract (커밋 c02eb8efefd988dda7f0046e38efb00762a84e08)
  - run_traceextract.sh (데이터셋별 프레임률 프리셋)
SmolVLM2-2.2B 설정(hidden 2048, 24층, intermediate 8192, 어휘 49280, 비전 27층, 비전 intermediate 4304)은 Hugging Face
config.json에서 읽었고, config에 없는 값(비전 hidden 1152, 텍스트 헤드 32개, KV 헤드 32개)은 transformers 기본값을 가정했다.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "code_arithmetic_results.json"
R = {}


def lin(i, o, bias=True):
    return i * o + (o if bias else 0)


# ---- 1. 행동 전문가 (GemmaActionExpert, gemma_300m 변형): 폭 1024, 18층, mlp 4096, 헤드 8, KV 헤드 1, head_dim 256
W, DEPTH, MLP, HEADS, KV, HD = 1024, 18, 4096, 8, 1, 256
VLM_H = 2048                      # SmolVLM2-2.2B 텍스트 hidden (config.json)
attn = lin(W, HEADS * HD, False) + 2 * lin(W, KV * HD, False) + lin(HEADS * HD, W, False)
mlp = 3 * W * MLP
adarms = 2 * lin(W, 3 * W)        # 층마다 입력/사후 RMSNorm 각각 dense(cond_dim=W, 3W)
layer = attn + mlp + adarms
gemma = DEPTH * layer + lin(W, 3 * W)       # 마지막 norm 한 개 더
STATE_DIM, ACT_PAD = 16, 32
extras = (3 * (2 * VLM_H + lin(VLM_H, W))     # z, lang, visual: LayerNorm + 투영
          + lin(STATE_DIM, W) + lin(ACT_PAD, W) + lin(W, ACT_PAD) + 2 * lin(W, W))
action_expert = gemma + extras
R["action_expert_params"] = {"computed": action_expert, "paper_D4_Table5_M": 436.4}

# ---- 2. 기능 혼합기: GatedMotionMixer + 그리퍼 투영 + 그리퍼 위치 임베딩
EXPERT_H = int(VLM_H * 0.5)
mixer = (2 * 2 * VLM_H                         # norm_q, norm_kv (LayerNorm 가중치+편향)
         + lin(EXPERT_H, VLM_H)               # w_proj
         + 3 * VLM_H * VLM_H + 3 * VLM_H      # MultiheadAttention in_proj
         + lin(VLM_H, VLM_H)                  # out_proj
         + 1)                                 # gate 스칼라
gripper = lin(768, VLM_H) + 8 * 8 * VLM_H      # DINOv2-base 768 -> 2048, 8x8 위치 임베딩
R["feature_mixer_params"] = {"gated_mixer": mixer, "gripper_proj_and_pos": gripper, "sum": mixer + gripper,
                             "paper_D4_Table5_M": 20.6}
R["trainable_head_total_M"] = (action_expert + mixer + gripper) / 1e6

# ---- 3. 트레이스 백본 개산 (논문 2.59B). 가정: 비전 hidden 1152, 텍스트 32헤드(MHA), lm_head 포함
emb = 49280 * VLM_H
vlm_layers = 20 * (4 * VLM_H * VLM_H + 3 * VLM_H * 8192 + 2 * VLM_H)
vis_h, vis_inter, vis_layers = 1152, 4304, 27
vision = vis_layers * (4 * lin(vis_h, vis_h) + lin(vis_h, vis_inter) + lin(vis_inter, vis_h) + 4 * vis_h) + 14 * 14 * 3 * vis_h + 729 * vis_h
connector = lin(vis_h * 9, VLM_H, False)
E_H = 1024
E_INTER = 2816                    # get_intermediate_size(1024)
e_mlp = 3 * E_H * E_INTER
e_self = 4 * lin(E_H, 2048, False)             # q,k,v,o (o: 2048->1024)
e_cross = lin(E_H, 2048, False) * 2 + 2 * lin(2048, 2048, False)
expert = 10 * (e_self + e_mlp) + 10 * (e_cross + e_mlp)
adaln = 20 * lin(E_H, 6 * E_H)
misc = (lin(256, E_H) + lin(E_H, E_H)          # 시간 임베더
        + lin(3 * 8, E_H) + lin(3 * 10, E_H) + lin(E_H, 3 * 10)       # 입력/출력 투영
        + 40 * E_H + 2 * E_H                    # 그룹, 세그먼트 임베딩
        + lin(32, E_H) + lin(E_H, E_H)          # uv MLP
        + lin(2 * E_H, E_H) + lin(E_H, E_H)     # 시간 융합 MLP
        + lin(E_H, 32)                          # done 헤드
        + lin(E_H + 768, E_H) + lin(E_H, E_H))  # DINO 융합
dino = 86_580_000                               # DINOv2-base 약 86.6M (공개 수치, 근사)
backbone = emb + emb + vlm_layers + vision + connector + expert + adaln + misc + dino
R["trace_backbone_params_B"] = {"with_lm_head": backbone / 1e9, "without_lm_head": (backbone - emb) / 1e9,
                                "paper_B": 2.59,
                                "note": "근사. 비전 hidden, 헤드 수는 기본값 가정, 깊이 LoRA·스템 제외"}

# ---- 4. TraceExtract 프레임률 프리셋 (run_traceextract.sh)와 논문 A.7 문장
FPS_PRESET = {"EgoDex": 10, "EgoVerse": 10, "Agibot": 5, "DROID": 5}
FRAMES_M = {"EgoDex": 4.1, "EgoVerse": 2.7, "Agibot": 3.7, "DROID": 2.5}
days = {k: FRAMES_M[k] * 1e6 / FPS_PRESET[k] / 86400 for k in FPS_PRESET}
R["days_by_dataset_launcher_preset"] = {**days, "human_total": days["EgoDex"] + days["EgoVerse"],
                                        "robot_total": days["Agibot"] + days["DROID"],
                                        "paper_A7": {"human": 16, "robot": 7, "total": 23}}

# ---- 5. 학습 설정 대조에 쓰는 값 (코드 상수)
R["code_constants"] = {
    "history_dropout": {"full_mask_prob": 0.5, "kp_dropout_prob": 0.3, "effective_per_kp": 0.65},
    "flow_time_sampling": "Beta(1.5, 1.0) * 0.999 + 0.001 (modeling_smolvla.py sample_time)",
    "policy_query_grid": "20x20 = 400 uniform keypoints, 2x2 avg-pooled to 100 tokens (mu0_policy.py GRID_SIZE, _reduce_expert)",
    "mixer_gate_init": "g = 0.0 -> sigmoid(0) = 0.5",
    "policy_train_steps_script": 60000, "paper_Table6_steps": 50000,
    "policy_effective_batch_script": "8 per GPU x 4 GPUs = 32", "paper_Table6_batch": 32,
}
OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(R, indent=1, ensure_ascii=False))
