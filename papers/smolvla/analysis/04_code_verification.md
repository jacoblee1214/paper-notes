# SmolVLA 논문 ↔ 공식 코드 대조 (04)

코드는 `repo/`(LeRobot HEAD `e5be0c1`, 2026-11-21)를 **정적으로만** 읽었습니다. 실행, import, 설치는 하지 않았습니다. 비동기 추론은 논문 시점의 구조가 현재 main에 없어서 태그 `v0.3.2`(2025-08-01 무렵)도 함께 읽었습니다. 파일 경로는 `repo/` 기준입니다.

## 0. 범위와 한계

- 대조한 파일: `src/lerobot/policies/smolvla/{configuration_smolvla.py, modeling_smolvla.py, smolvlm_with_expert.py}`, `src/lerobot/policies/common/flow_matching.py`, `src/lerobot/optim/schedulers.py`, `src/lerobot/remote_inference/`(현재), `src/lerobot/scripts/server/`(v0.3.2), `docs/source/smolvla.mdx`(현재).
- 공개된 가중치 설정 `lerobot/smolvla_base/config.json`(HF, 2026-10-06 수정)을 읽었습니다. 가중치는 받지 않았습니다.
- **재현 기준이 되는 사전학습 레시피(데이터 혼합, 어노테이션 스크립트, 카메라 매핑)는 repo에 없습니다.** 부록 A.1의 목록과 Qwen 어노테이션 스크립트는 찾지 못했습니다(`grep`으로 "481"과 목록 이름을 찾지 않음).
- 주의: `configuration_smolvla.py`의 기본값은 학습 레시피가 아닙니다. 실제 사전학습에 쓴 값은 확인할 수 없습니다.

## 1. 대조표 (논문 ↔ 코드)

| 번호 | 항목 | 논문 | 코드 (현재 main) | 판정 |
|---|---|---|---|---|
| D1 | 학습률 워밍업 | 100 스텝 (§4.3) | 기본 `scheduler_warmup_steps=1000`; 공개 config도 1000 | **불일치** (사전학습 설정이 다를 수 있음) |
| D2 | 학습률 감쇠 | 1e-4 → 2.5e-6 코사인, 감쇠 길이 미기재 | `scheduler_decay_steps=30000`, `decay_lr=2.5e-6`, 코사인 후 최저 유지 (`optim/schedulers.py`) | 값은 일치, 길이는 논문에 없음. 200k 스텝 학습이면 85%가 최저 학습률 |
| D3 | 학습 대상 | 서론: "trained end-to-end"; §4.3: VLM 동결, 액션 전문가만 학습 | `train_expert_only=True`(기본, 공개 config 포함)이면 VLM 전체가 `requires_grad=False`; `freeze_vision_encoder=True` | 코드는 §4.3과 일치. **서론과 불일치** |
| D4 | 흐름 매칭 규약 | §3.1: $A^\tau = \tau A + (1-\tau)\epsilon$, $u = \epsilon - A$, τ 방향 암묵적 | `NOISE_AT_ONE`: $x_t = t\cdot\text{noise} + (1-t)A$, $v = \text{noise} - A$; 추론은 t: 1→0, `dt = -1/num_steps` (`common/flow_matching.py`) | 코드 내부는 일관. **논문 식은 τ=1을 데이터로 두는데 u의 부호가 그 방향과 맞지 않음**(표기 문제로 보임) |
| D5 | τ 분포 | Beta 분포 (π0 설정) | `sample_time_beta(alpha=1.5, beta=1.0, scale=0.999, offset=0.001)` | 논문에 모수가 없음. 코드가 구체 값을 줌 |
| D6 | 추론 흐름 스텝 | 10 (§4.3) | `num_steps=10` | 일치 |
| D7 | 청크 크기 | n = 50 (§4.3) | `chunk_size=50`, `n_action_steps=50` | 일치 |
| D8 | 액션 전문가 폭 | VLM의 0.75배 (§3.1) | `expert_width_multiplier=0.75` | 일치 |
| D9 | 사용 VLM 층 수 | 첫 16층 (§4.3), N = L/2 | `num_vlm_layers=16` (기본), 층 잘라내기 `smolvlm_with_expert.py` | 일치. L = 32 여부는 transformers 쪽 설정이라 확인 못 함 |
| D10 | 교차·자기 어텐션 교차 | 매 층 교차, 자기 어텐션 교차 (§3.1) | `self_attn_every_n_layers=2`; `layer_idx % 2 == 0`이면 SA, 아니면 CA (`smolvlm_with_expert.py`) | 일치 (격층 교차) |
| D11 | 동작 토큰 마스크 | SA는 인과 마스크 (§3.1) | 접미부 `att_masks += [1]*chunk_size`로 인과 마스크 (`modeling_smolvla.py` embed_suffix) | 일치 |
| D12 | 상태 입력 위치 | VLM 접두 토큰 (§3.1, 표 11) | `state_proj`로 접두에 한 토큰 (`embed_prefix`) | 일치 |
| D13 | 입력 해상도 | 512×512 (§4.3) | `resize_imgs_with_padding=(512, 512)` | 일치 |
| D14 | 시각 토큰 수 | 프레임당 64 (픽셀 셔플) (§3.1) | 연결기(`connector`)는 transformers의 SmolVLM 코드에 있음, 이 repo에서 확인 못 함 | 미확인 |
| D15 | 사전학습 데이터 목록 | 481개 (§3.2, 표 1), 부록 A.1 목록 532개 | repo에 목록, 어노테이션, 카메라 매핑 스크립트 없음 | **재현 불가** (목록과 개수가 본문과 다름) |
| D16 | 옵티마이저 | AdamW, β = (0.9, 0.95) (§4.3) | `optimizer_betas=(0.9,0.95)`, `weight_decay=1e-10`, `grad_clip_norm=10` | 일치 (clip과 weight decay는 논문에 없음) |
| D17 | 카메라 순서 | OBS_IMAGE_1–3 (§3.2) | 공개 config의 입력 `camera1`, `camera2`, `camera3` | 일치 (이름 규약) |
| D18 | 입력 상태와 행동 차원 | 논문에 차원 없음 | `max_state_dim=32`, `max_action_dim=32`로 패딩 | 논문에 없음 |
| D19 | VLM 백본 이름 | SmolVLM-2 (§3.1, §4.3) | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 일치 |
| D20 | 설정 기본값과 공개 config 차이 | 해당 없음 | `prefix_length` -1 대 0, `pad_language_to` longest 대 max_length, `load_vlm_weights` False 대 True | 공개 설정이 학습 설정과 다를 수 있음 (사소) |
| D21 | 파라미터 수 | 450M, 액션 전문가 약 100M (§4.3) | 구조는 코드로 확인 가능하나 숫자는 재계산하지 않음 | 미확인 |
| D22 | 0.24B, 2.25B 변형 | 표 2에만 있음, 본문에 설명 없음 | `vlm_model_name`을 바꾸면 되는 구조. 해당 설정은 repo에 없음 | 논문에 없음 |
| D23 | 미세조정 스텝 | 시뮬 100k, 실로봇 200k (§4.3) | `docs/source/smolvla.mdx`: 20k 스텝, A100에서 약 4시간 | 문서가 권장하는 값이 논문과 다름 (문서는 "조정하라"고 함) |

### 1.1 비동기 추론 (Algorithm 1, §3.3)

논문의 Algorithm 1은 큐 임계 g(분율), 관절 공간 근접 필터 ε, 겹침 집계를 씁니다. 이 구조는 `v0.3.2`의 `src/lerobot/scripts/server/`에 있습니다. 현재 main에는 없습니다.

| 번호 | 항목 | 논문 | v0.3.2 | 현재 main | 판정 |
|---|---|---|---|---|---|
| D24 | 임계 g | Figure 3 예시 g = 0.7, 실험 값은 미기재 | `chunk_size_threshold` 기본 0.5 (`configs.py`) | `refill_seconds` 기본 0.5 s (남은 재생 시간 기준), 실효 값 = max(refill, 최근 왕복 시간 + 1/fps) (`docs/source/remote_inference.mdx`) | 실험 값은 논문에서 확인 불가. 단위가 분율에서 초로 바뀜 |
| D25 | 근접 필터 ε | 관절 공간 거리 임계 ε, 값 미기재 | `observations_similar`: 상태 차이의 L2 노름 < `atol`(기본 1.0). 단위 미기재 | 해당 로직 없음 (`grep`으로 near-duplicate·similarity·joint-space 검색, 없음) | 논문 값 미기재. **현재 main에서 제거된 것으로 보임** |
| D26 | 빈 큐 강제 처리 | 큐가 비면 최근 관측을 처리 (§3.3) | `must_go` 플래그 | 현재 구조는 다른 메커니즘 | v0.3.2 일치 |
| D27 | 집계 | 겹친 시점 집계 (f) | `aggregate_fn_name` 기본 `weighted_average` (문서 예) | `chunk_merge: append \| aligned`, 블렌딩, RTC (`inference/`) | 논문 구조와 다름 |
| D28 | 서버·클라이언트 | RobotClient, PolicyServer (gRPC) | `scripts/server/policy_server.py`, `robot_client.py` | `remote_inference/` (Zenoh 전송, 세션, 관리 상태) | 구조가 바뀜 |

## 2. 재현 관점에서 중요한 대조

1. **사전학습 레시피가 repo에 없습니다.** D15. 481개 데이터 목록, 어노테이션 프롬프트(부록 A.1은 있음), 카메라 매핑은 논문에만 있습니다. 목록 532개와 481개의 차이도 설명되지 않습니다.
2. **D1과 D2:** 워밍업과 감쇠 길이가 논문에 없거나 기본값과 다릅니다. 같은 학습률 곡선을 얻으려면 공개된 사전학습 config나 체크포인트 메타데이터가 필요합니다.
3. **D4 부호:** 코드는 openpi 계열의 `NOISE_AT_ONE` 규약을 씁니다. 학습과 추론이 같은 규약을 따르므로 구현은 맞습니다. 논문 식을 그대로 구현하면 부호가 뒤집힐 수 있습니다.
4. **D24–D27:** 논문의 비동기 실험(§4.6, 그림 5)은 v0.3.2 구조로 보입니다. 현재 main에서 같은 결과를 얻으려면 v0.3.2 코드를 써야 합니다. v0.3.2의 `observations_similar`는 논문 Algorithm 1의 ε에 대응하지만 기본 `atol = 1.0`의 단위는 확인하지 못했습니다.

## 3. 코드가 논문과 맞는 부분 (요약)

- 청크 50, 10 스텝 흐름 적분, 액션 전문가 0.75배, 16 VLM 층, 인터리브 CA/SA, 인과 마스크, 상태의 VLM 접두 입력, 512 입력, AdamW(0.9, 0.95), 2.5e-6 최저 학습률, 카메라 세 개 입력은 **모두 코드와 일치합니다.**
- VLM 동결, 액션 전문가만 학습합니다(§4.3).

## 4. 구현 가능성 (이 맥 기준)

| 작업 | 판정 | 근거 |
|---|---|---|
| 공개 가중치로 추론 (CPU 또는 MPS) | **가능, 측정 필요** | 450M 파라미터, 추론 메모리는 가중치 약 0.9 GB(bf16) + 활성값. 가중치 다운로드는 승인 필요 |
| 공개 가중치로 LIBERO 평가 | 조건부 | LIBERO 시뮬레이터와 렌더링 환경이 필요합니다(버전은 확인하지 않음). 설치 전 승인 필요 |
| 사전학습 재현 (4 GPU, 200k 스텝, 30k GPU 시간) | **불가** | 논문의 GPU 시간 규모. MPS는 이 규모를 감당하지 않음 |
| 미세조정 (시뮬 100k, 실로봇 200k) | **불가 (추정)** | CUDA 기준 설정, 데이터 수십 GB 가능성. 20k 스텝 문서 예도 A100 4시간 |
| 실로봇 재현 | **불가** | 로봇 하드웨어 필요 |
| 비동기 추론 구조의 CPU 검증 | 조건부 | 큐 로직과 `observations_similar`는 CPU로 시뮬레이션 가능. 필요한 코드는 v0.3.2 |

## 5. 저자에게 물을 질문

1. 사전학습 설정(워밍업, 감쇠 길이, 배치, 스텝)과 최종 체크포인트 config는 무엇인가요?
2. 481개와 532개 목록의 차이는 무엇인가요? 최종 목록을 공개할 수 있나요?
3. 실험에 쓴 g와 ε 값은 무엇인가요? 현재 main의 `refill_seconds` 대응값은 어떻게 되나요?
4. 흐름 매칭 식(τ 방향과 u의 부호)은 논문 표기가 맞나요, 아니면 구현 규약을 옮겨 쓴 것인가요?
5. 표 6의 CA+SA(85.5)와 표 10–13의 기준선(80.3)은 같은 설정인가요?
