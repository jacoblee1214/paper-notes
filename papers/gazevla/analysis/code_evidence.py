"""공식 코드의 근거 줄을 찾아 파일·줄 번호와 함께 저장한다(정적 읽기, 코드를 실행하거나 import하지 않는다).

repo/ 는 https://github.com/lichy2004/GazeVLA 의 얕은 clone이다:
    git clone --depth 1 https://github.com/lichy2004/GazeVLA repo && git -C repo checkout 0a834e8748b8d799166712a098fff91263109c43
data/GazeVLA-Data/ 는 https://github.com/lichy2004/GazeVLA-Data (커밋 e47f5dc2cadf8b0d5b7566a410fb0cd9c66e15a1)의 얕은 clone이다.
실행: python3 analysis/code_evidence.py   → analysis/code_evidence.json
함께 학습·추론 입력의 크기 차이를 산술로 확인한다(D 항목의 근거 숫자).
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo"
DATA = ROOT / "data" / "GazeVLA-Data"
M = "src/openpi/models_pytorch/VLIA_pytorch.py"
# (키, 기준 폴더, 파일, 정규식): 첫 번째로 맞는 줄 하나를 기록한다. 같은 줄이 여러 개면 n번째를 쓰도록 (키, ..., n)
PATTERNS = [
    ("flow_time_beta", REPO, M, r"sample_beta\(1\.5, 1\.0"),
    ("flow_xt_convention", REPO, M, r"x_t = time_expanded \* noise"),
    ("flow_target_noise_minus_action", REPO, M, r"u_t = noise - actions"),
    ("train_gaze_embed_scale_sqrt_dim", REPO, M, r"return gaze_emb \* math\.sqrt\(gaze_emb_dim\)"),
    ("infer_gaze_token_num_3", REPO, M, r"gaze_token_num = 3"),
    ("infer_gaze_init_scale", REPO, M, r"gaze_init_emb \* math\.sqrt\(gaze_token_num\)"),
    ("infer_prev_token_scale", REPO, M, r"prev_token_emb \* math\.sqrt\(gaze_token_num\)"),
    ("infer_prefix_len_includes_padding", REPO, M, r"current_prefix_len = prefix_pad_masks\.shape\[1\]"),
    ("train_position_ids_valid_only", REPO, M, r"position_ids = torch\.cumsum\(pad_masks, dim=1\) - 1"),
    ("infer_gaze_att_mask_all_zero", REPO, M, r"new_att_mask = torch\.zeros"),
    ("gaze_head_linear_224", REPO, M, r"self\.gaze_head = nn\.Linear"),
    ("gaze_init_token_param", REPO, M, r"self\.gaze_init_token = nn\.Parameter"),
    ("gaze_greedy_argmax", REPO, M, r"predicted_bin_id = torch\.argmax"),
    ("action_loss_mse_masked", REPO, M, r"action_loss_per_element = F\.mse_loss"),
    ("data_action_dim_mask", REPO, M, r"data_action_dim = self\.config\.data_action_dim"),
    ("tkinter_import", REPO, M, r"from tkinter\.constants import FALSE"),
    ("compile_sample_actions", REPO, M, r"torch\.compile\(self\.sample_actions"),
    ("denoise_steps_default_10", REPO, M, r"def sample_actions\(self, device, observation, noise=None, num_steps=10\)"),
    ("config_default_gaze_loss_weight", REPO, "src/openpi/models/VLIA_config.py", r"gaze_loss_weight: float = 1e-2"),
    ("config_default_gemma_2b", REPO, "src/openpi/models/VLIA_config.py", r'paligemma_variant: _gemma\.Variant = "gemma_2b"'),
    ("config_action_expert_300m", REPO, "src/openpi/models/VLIA_config.py", r'action_expert_variant: _gemma\.Variant = "gemma_300m"'),
    ("config_default_freeze_all_hot", REPO, "src/openpi/models/VLIA_config.py", r'"vlm": "hot"'),
    ("config_lfa_gaze_loss_weight_0.1", REPO, "src/openpi/training/config.py", r"gaze_loss_weight=0\.1"),
    ("config_lfa_action_dim_48", REPO, "src/openpi/training/config.py", r"action_dim=48,"),
    ("config_lfa_steps_100000", REPO, "src/openpi/training/config.py", r"num_train_steps=100000"),
    ("config_lfa_decay_steps_50000", REPO, "src/openpi/training/config.py", r"decay_steps=50000"),
    ("config_lfa_peak_lr_1e-4", REPO, "src/openpi/training/config.py", r"peak_lr=1e-4"),
    ("config_lfa_batch_32", REPO, "src/openpi/training/config.py", r"batch_size=32"),
    ("config_lfa_pi0_base_weights", REPO, "src/openpi/training/config.py", r"pi0_base/params"),
    ("config_lfa_single_camera", REPO, "src/openpi/training/config.py", r'image_keys=\("base_0_rgb",\)'),
    ("sim_gaze_mask_always_1", REPO, "examples/process_data_2.py", r'"gaze_mask": np\.array\(\[1\]'),
    ("sim_gaze_from_vr_eyes", REPO, "examples/process_data_1.py", r"avg_x = \(left_eye\[0\] \+ right_eye\[0\]\) / 2"),
    ("sim_action_is_next_state", REPO, "examples/process_data_1.py", r"action = states\[1:\]"),
    ("aug_crop_scale", REPO, "src/openpi/models_pytorch/preprocessing_human.py", r"AUG_CROP_SCALE = "),
    ("aug_rotation_deg", REPO, "src/openpi/models_pytorch/preprocessing_human.py", r"AUG_ROTATION_ANGLE = "),
    ("gaze_clamp_after_aug", REPO, "src/openpi/models_pytorch/preprocessing_human.py", r"processed_gaze = torch\.clamp"),
    ("optimizer_adamw_betas", REPO, "src/openpi/training/optimizer.py", r"b2: float = 0\.95"),
    ("lr_schedule_cosine_floor", REPO, "scripts/train/train.py", r"progress = min\(1\.0"),
    ("freeze_warmup_uses_lr_warmup", REPO, "scripts/train/train.py", r"warmup_steps_for_freeze = config\.lr_schedule\.warmup_steps"),
    ("human_config_no_named_instance", REPO, "src/openpi/training/config.py", r"class HumanRLDSDataConfig"),
    ("data_egodex_gaze_zero_mask", DATA, "src/dataset/EgoDex.py", r'data\["mask"\]\["gaze"\] = np\.zeros'),
    ("data_hot3d_gaze_from_aria", DATA, "src/dataset/HOT3D.py", r"get_eye_gaze_in_camera"),
]

out = {}
for key, base, rel, pat in PATTERNS:
    lines = (base / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    hit = next(((i + 1, ln.strip()) for i, ln in enumerate(lines) if re.search(pat, ln)), None)
    out[key] = {"file": rel if base == REPO else "data/GazeVLA-Data/" + rel, "line": hit[0] if hit else None, "text": hit[1] if hit else None}

# 학습과 추론의 입력 크기 차이 (D 항목 근거 산술)
dim = 2048
arith = {
    "train_gaze_token_embed_scale": math.sqrt(dim),
    "infer_gaze_token_embed_scale": math.sqrt(3),
    "ratio_train_over_infer": math.sqrt(dim) / math.sqrt(3),
    "text_len_padded_to": 48,
    "rope_position_shift_for_prompt_len": {str(n): 48 - n for n in (8, 12, 20, 30)},
    "note": "학습의 position_ids는 유효 토큰만 센다(cumsum(pad_mask)-1). 추론의 시선 토큰 위치는 패딩 포함 길이(256+48)에서 시작한다. 프롬프트 토큰 수 n은 가정(8–30)",
}
out["_arith"] = arith
(ROOT / "analysis" / "code_evidence.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
missing = [k for k, v in out.items() if k != "_arith" and v["line"] is None]
print("wrote analysis/code_evidence.json", "missing:", missing)
