# 04. 논문 ↔ 공식 코드 대조 (VLA-JEPA)

대상: 논문 arXiv:2602.10098 v2(2026-02-14), 공식 저장소 `ginwind/VLA-JEPA` 커밋 `ec907778` (2026-09-30 +08:00, 제목 "update license"). 저장소는 `repo/`에 얕은 clone으로 두었고, **정적으로만 읽었습니다.** 실행, import, 설치, 가중치 다운로드는 하지 않았습니다.

읽은 파일: `starVLA/model/framework/VLA_JEPA.py`(384줄, 핵심), `starVLA/model/modules/world_model/vj2_predictor.py`, `scripts/configs/vlajepa_pretrain.yaml`, `scripts/configs/vlajepa_libero_ft.yaml`, `scripts/vlajepa_pretrain.sh`, `README.md`, `LICENSE`.

판정 기준: 논문에 수치나 구조로 적힌 것은 **일치**, 값이 다르면 **불일치(D)**, 논문에 없는 설정이면 **미기재**, 코드가 확인되지 않는 부분(외부 라이브러리)은 **미검증**으로 씁니다.

## 1. 대조 요약

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 학생 경로 입력 (VLM) | 현재 관측과 언어 지시만 (식 2, 6) | `build_qwenvl_inputs(images=batch_images)`, `batch_images`는 `example["image"]` | 일치 |
| 타깃 인코더 | 고정, stop-gradient (식 4) | `with torch.no_grad()`로 V-JEPA2 호출 | 일치 (EMA 아님, 논문도 EMA라고 쓰지 않음) |
| 예측기 입력 상태 | "현재 관측의 세계 상태" (식 3) | 같은 고정 인코더를 **클립 전체**(T 프레임)에 적용한 뒤 앞 T−1 스텝을 입력으로 사용 | **D4 (누수 경로 가능성)** |
| 세계모델 손실 형식 | 식 (5): ŝ − s, 노름 미지정 | `F.l1_loss` (L1) | **D1** |
| 로봇 데이터 공동 손실 | 식 (9): L = L_FM + β L_WM, β 튜닝 | `wm_loss * 0.1` (하드코딩) | **D2** |
| 인간 영상 사전학습 손실 | 식 (5) | `return {"wm_loss": ...}` 가중치 1 | 일치 (경로별 가중치가 다름, D2에 포함) |
| 잠재 행동 토큰 수 | K = 24/T (A.1), T = 8이면 K = 3 (Table 5 "3") | 사전학습 설정 `num_action_tokens_per_timestep: 8` | **D3** |
| 임베디드 행동 토큰 반복 | 32 (A.1) | `num_embodied_action_tokens_per_instruction: 32` | 일치 |
| 미래 시계열 horizon | T = 8 (Table 5) | `num_frames: 8` | 일치 |
| 액션 청크 | 7 (Table 6, horizon 7) | `future_action_window_size: 6` + 현재 1 = 7 | 일치 |
| 상태, 행동 차원 | 8, 7 (Table 6) | `state_dim: 8`, `action_dim: 7` | 일치 |
| 흐름 매칭 샘플링 스텝 | 4 (Table 6) | `num_inference_timesteps: 4` | 일치 |
| 액션 헤드 | "DiT-B" 명칭 (A.1) | 16층, 1024차원, 헤드 12 (Table 6과 같음) | **D9 (명칭)** |
| 사전학습 데이터, 스텝 | SSv2 + DROID 공동 50K (A.2) | `max_train_steps: 50000`, `video_dir`(ssv2), `data_mix: droid` | 일치 |
| 사전학습 학습률 | VLM·WM 1e-5, 액션 1e-4 (A.2) | `base 3e-5`, `qwen_vl_interface 1e-5`, `action_model 1e-4`, 세계모델 전용 항목 없음 | **D5** |
| LIBERO 미세조정 | 30K 스텝 (A.2) | `max_train_steps: 120000`, `vj_predictor: 5e-4` | **D6** |
| 학습 시 디노이징 반복 | 논문 미기재 | `repeated_diffusion_steps: 8` (config), 코드 기본값 4 | **D7 (미기재)** |
| 재현 스크립트 | - | `vlajepa_pretrain.sh`가 `vlajepa_co_pretrain.yaml`을 참조하나 파일 없음 | **D8** |
| 라이선스 | 논문 CC BY 4.0 | 코드 Apache 2.0, 흐름 헤드는 GR00T N1.5 저작권 주석 | 해당 없음 (4절) |

## 2. 불일치 항목 (D1–D9)

**D1. 세계모델 손실의 거리 함수.** 논문 식 (5)는 `E[ŝ − s]`로만 쓰고 노름을 적지 않습니다. 식 (8)에서만 ℓ2를 씁니다. 코드는 `F.l1_loss(predicted_states, gt_states)`로 L1을 씁니다. 식 (5)가 L1을 뜻하는지 ℓ2를 뜻하는지는 논문만으로 알 수 없습니다. 논문 본문은 "JEPA alignment loss"라고만 합니다. 영향: 손실 크기와 가중치 β(D2)의 의미가 달라집니다. 확인하려면 저자에게 묻거나 실험 로그를 봐야 합니다.

**D2. 세계모델 손실의 가중치.** 논문 식 (9)는 `L = L_FM + β L_WM`이고 β는 "튜닝"한다고만 합니다. 값은 어디에도 없습니다. 코드는 로봇 데이터 경로에서 `wm_loss × 0.1`을 하드코딩합니다. 인간 영상 경로(`"action"`이 없는 배치)는 `wm_loss`를 그대로 반환합니다. 따라서 같은 세계모델 손실이 두 경로에서 다른 크기로 학습에 기여합니다. 논문은 이 점을 적지 않습니다. 또한 `trainer.loss_scale`에 `vla: 1.0`, `vlm: 0.1`이 있는데 이것이 세계모델 손실에 적용되는지는 이 파일에서 확인되지 않습니다.

**D3. 잠재 행동 토큰 수 K.** 논문 A.1은 "K = 24/T"이고 T = 8이면 K = 3이라고 합니다. Table 5의 "Number of action tokens per time step"도 3입니다. 저장소의 사전학습 설정은 `num_action_tokens_per_timestep: 8`입니다. 이 값이 K인지 다른 것인지 코드만으로는 확정할 수 없지만, 논문 표의 값과 다릅니다. LIBERO 미세조정 설정에는 같은 키가 보이지 않습니다.

**D4. 예측기 입력 상태의 누수 경로 (가장 중요).** 논문은 "미래 프레임은 입력이 아니라 타깃으로만 쓰인다", "학생 경로는 현재 관측만 본다"고 합니다(초록, 1절, 식 3). 코드를 보면 다음과 같습니다.

- VLM(학생)은 `example["image"]`, 즉 현재 관측만 받습니다. 이 부분은 논문과 일치합니다.
- 예측기 입력 `input_states`는 `video_embeddings[:, : (T−1)·dim]`입니다. `video_embeddings`는 `self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)`로 얻는데, 이 `input_videos`는 **T 프레임 전체 클립**입니다(`batch_videos[i]`, `T`는 `num_frames`=8).
- 즉 예측기가 받는 "과거" 상태는 미래 프레임을 포함해 인코딩된 특징입니다. 인코더가 프레임 사이에 양방향 어텐션을 쓴다면, 과거 스텝의 특징에 미래 정보가 섞입니다.
- 인코더 구현(`AutoModel`, V-JEPA2)은 이 저장소에 없고 HF `transformers` 쪽 코드입니다. 양방향 어텐션 여부는 **미검증**입니다. V-JEPA 2 ViT 계열은 일반적으로 전 시공간 토큰이 서로 보는 구조로 알려져 있지만, 이 저장소에서 확인한 것은 아닙니다.
- 예측기 내부의 인과 마스크(`is_frame_causal=True`, `build_action_block_causal_attention_mask`)는 **시간 축 어텐션**만 제한합니다. 입력 특징 자체가 미래 프레임을 본 뒤에 만들어졌다면 이 마스크로는 막히지 않습니다.

이것이 논문 주장을 뒤집는다는 뜻은 아닙니다. 논문의 "누수 없음"은 학생 경로(VLM)에 대한 주장으로 읽힙니다. 다만 세계모델의 예측 입력 쪽에 누수 경로가 있을 수 있고, 논문은 이를 다루지 않습니다. 토이(`analysis/toy_leakage.py`)는 양방향 인코더가 과거 특징에 미래 상태 정보를 더한다는 **메커니즘**만 보입니다. 실제 모델에서 그 크기는 측정하지 않았습니다(`03` §4).

**D5. 학습률.** 논문 A.2: VLM과 잠재 세계모델은 1e-5, 액션 헤드는 1e-4. 저장소 사전학습 설정: 기본 `base: 3e-5`, `qwen_vl_interface: 1e-5`, `action_model: 1e-4`입니다. 세계모델(`vj_predictor`)의 항목은 사전학습 설정에 없어 기본값 3e-5가 적용되는 것으로 보입니다. 미세조정 설정은 `vj_predictor: 5e-4`입니다. 논문 어디에도 5e-4는 없습니다.

**D6. 미세조정 길이.** 논문 A.2: LIBERO 미세조정은 "30K 스텝 이어서 학습". 저장소 `vlajepa_libero_ft.yaml`: `max_train_steps: 120000`, `num_warmup_steps: 20000`입니다. 사전학습 설정(50K)은 논문과 일치합니다. 실제로 쓴 체크포인트가 어떤 설정인지는 확인할 수 없습니다.

**D7. 학습 시 디노이징 반복 횟수.** `repeated_diffusion_steps`는 두 설정 모두 8입니다. 코드 기본값은 4입니다. 논문에 이 값은 없습니다. Table 6의 "Denoising timesteps 4"는 추론 스텝 수이며 `num_inference_timesteps: 4`와 일치합니다. 학습 반복 횟수와 추론 스텝을 논문이 구분하지 않아 생긴 모호함입니다.

**D8. 재현 스크립트의 깨진 경로.** `scripts/vlajepa_pretrain.sh`는 `CONFIG_PATH=${PROJECT_ROOT}/scripts/configs/vlajepa_co_pretrain.yaml`을 씁니다. 저장소에는 `vlajepa_pretrain.yaml`과 `vlajepa_libero_ft.yaml`만 있습니다. 스크립트가 공동 학습 설정을 가리키는지, 사전학습 설정을 가리켜야 하는지 확인이 필요합니다. 스크립트와 설정에는 원저자 서버의 절대 경로가 들어 있어 이 문서에는 옮기지 않았습니다.

**D9. 액션 헤드 명칭.** 논문은 "DiT-B"라고 하지만 Table 6의 값(16층, 1024차원, 12 헤드)은 DiT-B 표준(12층, 768차원)과 다릅니다. 저장소 설정은 `action_model_type: DiT-B`, `num_layers: 16`, `hidden_size: 1024`입니다. 값은 논문 표와 일치하고 이름만 다릅니다. 기능 문제는 아닙니다.

## 3. 일치 확인 (요약)

- 고정 타깃 인코더 `no_grad` (식 4의 stop-gradient와 일치).
- 학생(VLM) 입력은 현재 관측과 언어 지시. `<|embodied_action|>`는 32번 반복 (A.1의 32와 일치).
- 행동 토큰은 VLM의 마지막 은닉 상태에서 읽음 (식 6과 같은 구조).
- 흐름 매칭 헤드는 `FlowmatchingActionHead`(GR00T 계열). 인코더와 예측기는 `vj_predictor`로 따로 있음.
- 로봇 데이터 손실: `action_loss` + `wm_loss × 0.1`. 인간 영상 손실: `wm_loss`만.

## 4. 라이선스 메모 (코드)

- 저장소 `LICENSE`와 README는 **Apache 2.0**입니다. GitHub API가 `apache-2.0`을 반환했습니다.
- `VLA_JEPA.py` 헤더 주석: "Flow-matching header is copyright from GR00T N1.5". `THIRD_PARTY_NOTICES.md`와 소스 헤더를 따로 확인해야 합니다(이번에는 본문만 확인).
- 가중치는 HF `ginwind/VLA-JEPA`에 있고, 라이선스 표기는 확인하지 않았습니다. 다운로드하지 않았습니다.

## 5. 재현 관점의 한계

- 논문 수치 재현에는 8×A100, SSv2 220K 영상, DROID 76K 궤적, LIBERO와 SimplerEnv 시뮬레이터가 필요합니다. 이 맥에는 없습니다(`README.md` 구현 가능성 판정).
- 설정에 누락된 값(D2 β, D7 반복 횟수, D5 세계모델 학습률)이 있어서, 저장소만으로 논문 표를 그대로 재현하기는 어렵습니다.
- 외부 인코더 `AutoModel.from_pretrained(V-JEPA2)`의 어텐션 구조는 이 저장소에서 확인되지 않습니다(D4의 미검증 항목).
