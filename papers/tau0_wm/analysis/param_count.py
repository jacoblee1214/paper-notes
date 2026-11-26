"""공개 코드의 모델 정의(models/wan_2_2_models/transformers/model.py, model_sim.py)와 설정 yaml 값으로
파라미터 수를 해석적으로 센다(코드를 import하거나 실행하지 않음, 가중치도 받지 않음).

repo/는 https://github.com/sii-research/tau-0-wm 의 커밋 e03902a4a8e0cf89b681e89920784ae8ca7b4b3c 이다:
  git clone --depth 1 <url> repo   (이 커밋이 최신일 때)
입력 크기는 Hugging Face API 메타데이터(파일 크기)에서 가져온 값이다.
실행: python3 analysis/param_count.py
"""
import json, os

def lin(i, o, bias=True): return i * o + (o if bias else 0)

def attn(dim, kv_dim=None, qk_norm=True):
    kv_dim = kv_dim or dim
    n = lin(dim, dim) + lin(kv_dim, dim) * 2 + lin(dim, dim)   # q, k, v, o
    if qk_norm: n += 2 * dim                                   # norm_q, norm_k (RMS)
    return n

def block(dim, ffn, cross_dim=None):
    n = attn(dim)                                  # self-attn
    n += attn(dim, cross_dim)                      # cross-attn (cross_attn_dim)
    n += 2 * dim                                   # norm3 (affine LayerNorm)
    n += lin(dim, ffn) + lin(ffn, dim)             # ffn
    n += 6 * dim                                   # modulation
    return n

def head(dim, out): return dim * 0 + lin(dim, out) + 2 * dim   # linear + modulation (norm has no affine)

L, D, F = 30, 3072, 14336          # Wan2.2-TI2V-5B 설정 (configs/tau_model/*.yaml)
video = L * block(D, F)
video += lin(4096, D) + lin(D, D)                     # text_embedding
video += lin(256, D) + lin(D, D) + lin(D, 6 * D)      # time embedding/projection
video += 48 * D * 1 * 2 * 2 + D                       # patch_embedding Conv3d (1,2,2)
video += head(D, 48 * 4)
# VAM 행동 가지: dim 1024, ffn 2048, 16 heads, 영상 토큰(3072)을 cross-attn
Da, Fa = 1024, 2048
act = L * block(Da, Fa, cross_dim=D)
act += lin(20, Da) + lin(256, Da) + lin(Da, Da) + lin(Da, 6 * Da) + head(Da, 20)
# ACVS 보상 가지: act_cond_dim 기본값 1536 (설정에 act_cond_dim 지정 없음), ffn 2048, 16 heads
Dr, Fr = 1536, 2048
rew = L * block(Dr, Fr, cross_dim=D)
rew += lin(1, Dr) + lin(256, Dr) + lin(Dr, Dr) + lin(Dr, 6 * Dr) + head(Dr, 1)
# 행동 조건(Eq.4): act_cond_proj_in, proj_in_2(->D), proj_in_2_6d(->6D), act_cond_in_dim=26(전신)
cin = 26
cond = lin(cin, Dr // 16) + lin(Dr // 16, Dr // 16) + lin(Dr, Dr) + lin(Dr, D) + lin(Dr, Dr) + lin(Dr, 6 * D)

meta = json.load(open(os.path.join(os.path.dirname(__file__), "hf_file_sizes.json")))
sz_vam = meta["diffusion_pytorch_model-00001-of-00002.bin"] + meta["diffusion_pytorch_model-00002-of-00002.bin"]
sz_sim = meta["simulator/diffusion_pytorch_model.bin"]
out = {
 "video_backbone_params": video, "action_branch_params": act, "vam_total_params": video + act,
 "reward_branch_params": rew, "action_condition_params": cond, "simulator_total_params": video + rew + cond,
 "hf_vam_bytes": sz_vam, "hf_sim_bytes": sz_sim,
 "hf_vam_params_if_bf16": sz_vam // 2, "hf_sim_params_if_bf16": sz_sim // 2,
 "ratio_vam": (video + act) / (sz_vam / 2), "ratio_sim": (video + rew + cond) / (sz_sim / 2),
}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "param_count_results.json"), "w"), indent=1)
for k, v in out.items(): print(f"{k:28s} {v:,.4f}" if isinstance(v, float) else f"{k:28s} {v:,}")
