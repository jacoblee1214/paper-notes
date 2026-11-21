# 04. 공식 코드 대조 (정적 읽기): 논문 ↔ 저장소, 차이 D1–D15

대상 코드: `AgibotTech/Genie-Envisioner-V1` 커밋 `d41358f` (2026-09-10), 얕은 복제 `repo/`.
방법: 파일을 읽기만 했습니다. 설치, 실행, import, 가중치 다운로드는 하지 않았습니다. 복제한 내용은 신뢰하지 않는 데이터로 다뤘습니다.
읽은 파일: `configs/ltx_model/{video_model,policy_model,video_model_infer_fast,video_model_infer_slow}.yaml`, `runner/ge_trainer.py` (학습 모드, 손실), `runner/ge_inferencer.py` (추론 인자), `models/pipeline/custom_pipeline.py` (디노이징 루프), `data/agibotworld_dataset.py`, `data/utils/get_actions.py`, `utils/get_traj_maps.py`, `gesim_video_gen_examples/infer_gesim.py`, `requirements.txt`, `README.md`, `experiments/RUN.md`.

## 0. 저장소 요약

- 저장소 이름이 `Genie-Envisioner`에서 `Genie-Envisioner-V1`로 바뀌었습니다 (리디렉션 확인).
- README 첫 줄은 "Genie Envisioner v1.0"입니다. 2025-08-13 코드 공개, 2025-08-14 GE-Base 가중치 공개.
- 공개 가중치: GE-Base (HF), GE-Sim Cosmos 변형 (ModelScope), GE-Act CALVIN 가중치 (2025-12-18). 실로봇 GE-Act 가중치는 저장소에서 찾지 못했습니다.
- 공식 코드에는 CALVIN과 LIBERO 평가 설정이 있습니다. 이 둘은 논문 본문에 결과가 없습니다 (D12).

## 1. 차이 목록 (D1–D15)

판정 기호: **일치** (논문과 코드가 같음), **불일치** (다름), **미기재** (논문에 없어서 코드만 확인), **없음** (논문 설명 대상인 코드가 저장소에 없음), **확인 필요** (읽기로 확정 못함).

| 번호 | 주제 | 논문 | 코드 (파일:위치) | 판정 |
|---|---|---|---|---|
| D1 | 다시점 교차 주의 | §2.1: 세 시점 (head, left, right), 일부 블록만 시점 교차 | `valid_cam: [head, hand_left, hand_right]`, `transformer_ltx_multiview.py` | 일치 |
| D2 | 행동 표현 서술 | 초록·§1: "54-step **torque** trajectories" / §5.1: 팔마다 [x,y,z,roll,pitch,yaw,o] | `get_actions_eef` (`data/utils/get_actions.py`): 위치 델타와 rpy 델타 + 그리퍼, 14차원. 토크 없음 | **불일치 (서술)**: 토크라는 말이 틀림 |
| D3 | 행동 사전학습의 학습 파라미터 | §3.2: "only the parameters of the action decoding module are updated" | `policy_model.yaml`의 `train_mode: action_full`. `ge_trainer.py` L414–416: 모든 `diffusion_model` 파라미터 `requires_grad=True` | **불일치**: 사전학습에서 영상 모듈도 학습됨 (영상 손실은 0, 행동 손실만 계산). 코드상 영상 가중치는 같이 움직임 |
| D4 | CLIP 동결 인코더와 GE-Sim 기준 영상 | §4.1, §5 (CLIP 이미지 인코더 스타일 앵커), §5.2 (VAE, CLIP 동결) | 저장소 전체에서 CLIP 이미지 인코더 코드를 찾지 못함 (`CLIPVision`, `image_encoder`, `open_clip` 검색 없음). `ge_trainer.py`의 `components_to_disable_grads = []`는 빈 목록 | **없음**: CLIP 기준 영상 조건은 공개 코드에 없음 |
| D5 | 행동 분기 크기 | §3: 약 160M 파라미터, 블록 깊이 동일, 은닉 차원 축소 | `action_expert: true`, `action_num_attention_heads: 16`, `action_attention_head_dim: 32` (512차원), 블록 28개 (영상과 동일) | 일치 (구조). **160M은 확인 못함**: 정적 읽기로 파라미터 수를 셀 수 없음 |
| D6 | 행동 대 영상 비율 | §3.3: 5 Hz 대 30 Hz, 1:6, 54스텝 | `action_chunk: 54`, `chunk: 9`, `video_temporal_stride = action_chunk // chunk = 6` (`agibotworld_dataset.py` L164–165) | 일치 |
| D7 | 희소 메모리 | §2.1, §2.2: 메모리 4프레임, 무작위 샘플 | `n_previous: 4`, `previous_pick_mode: 'random'` | 일치 |
| D8 | 추론 모드 이름 (표준 대 빠른) | §3.3: 표준 (동기), 빠른 (비동기). Figure 8: GE-Act Slow, Fast | `video_model_infer_slow.yaml`: `chunk: 57`, `action_chunk: 57`. `video_model_infer_fast.yaml`: `chunk: 9`, `action_chunk: 54`. 두 설정이 명명상 대응하는지는 논문에 명시 없음 | **확인 필요**: 이름 대응은 추정. 빠른 모드 설정은 논문과 같음 |
| D9 | 자세 표현 (GE-Sim) | §5.1: 자세 rpy를 회전 행렬로 변환, 그리퍼 개방도를 단위 원 음영으로 | `get_traj_maps.py`: 입력은 **쿼터니언** (`pose[:,0:7]`, 팔당 8차원: xyz, quat, 그리퍼 인덱스 7). `gesim_video_gen_examples/sample_0/actions.npy`의 형태는 (300, 16). 그리퍼는 값/120으로 색 농도 결정, 반지름은 카메라 거리 함수 | **불일치 (표현)**: GE-Act 학습은 rpy 14차원, GE-Sim 입력은 쿼터니언 16차원. 그리퍼 정규화 (0–120)는 논문에 없음 |
| D10 | 학습 하이퍼파라미터 | 논문은 GPU 수와 기간만 씀. 배치, 학습률, 스텝 수 없음 | `lr: 3e-5`, `batch_size: 8` (GPU당), AdamW (0.9, 0.95), `constant_with_warmup` 1000, `caption_dropout_p: 0.06`, `noise_to_first_frame: 0.1`, `train_steps: 1e6`(상한), `use_color_jitter: true` | **미기재** |
| D11 | 비동기 추론 | §3.3: 영상 DiT 한 번 디노이징, 캐시, 행동 5스텝 | `custom_pipeline.py` L836–843: `compute_video = i == 0 or return_video`. `return_video=False`이면 영상 연산은 첫 스텝에만, 나머지는 캐시 사용. `num_inference_step: 5` | 일치 |
| D12 | CALVIN, LIBERO 결과 | 논문 본문에 없음 (v1–v3) | README: CALVIN 평균 4.260 (Len-1..5 0.95–0.75), LIBERO 평균 0.965. `experiments/RUN.md`에도 같은 값 | **미기재 (논문)**: 결과는 저장소에만 있음 |
| D13 | 라이선스 | arXiv: CC BY 4.0 | README: 기타 데이터와 코드 CC BY-NC-SA 4.0, 일부 디렉터리 Apache-2.0. LICENSE 파일 없음 (API license null) | **불일치 (표기)**: 코드 라이선스는 비상업·동일조건 조건이 붙음 |
| D14 | 실로봇과 RoboTwin 설정 | §4: Agilex, Dual Franka, RoboTwin 적응 설정 | `configs/`에 `ltx_model/{calvin, libero}`와 AgiBot 정책 설정만 있음. Agilex, Franka, RoboTwin 설정 없음 | **없음**: 교차 구현체 결과 재현 설정 미공개 |
| D15 | 분류기 자유 안내 (CFG) | 논문에 언급 없음 | `custom_pipeline.py` 기본 `guidance_scale=3`. 추론 스크립트는 `guidance_scale=1.0` (CFG 꺼짐) | **미기재**: 실험은 꺼진 설정으로 보임 (추정) |

## 2. 차이 중 중요한 것 (요약)

1. **D3 (학습 파라미터).** 논문 §3.2의 "행동 사전학습에서 영상 모듈은 고정"과 공개 설정의 `action_full`은 다릅니다. 공개 설정이 논문 실험의 실제 설정인지 확인되지 않았습니다. 논문 결과 해석에 영향을 줍니다. 영상 사전학습 가중치가 행동 학습 중에 바뀌었을 수 있습니다.
2. **D4 (CLIP).** GE-Sim의 기준 영상 스타일 앵커는 공개 코드에 없습니다. 따라서 Figure 16 (두 변형 비교)을 공개 코드로 재현할 수 없습니다.
3. **D9 (자세 표현).** 학습 행동 (rpy)과 시뮬레이터 입력 (쿼터니언)이 다릅니다. 논문은 둘 다 rpy라고 씁니다.
4. **D2 (토크 서술).** 본문 서술 오류입니다. 실제 행동은 EEF 자세 델타입니다.
5. **D12, D14.** 논문에 없는 CALVIN, LIBERO 결과는 저장소에만 있고, 논문 실로봇·교차 구현체 설정은 공개되지 않았습니다.

## 3. 라이선스 (코드)

- 확인 방법: GitHub API `license: null`, 저장소 파일 목록에 LICENSE 없음, README 마지막 절 확인.
- README 문장: "Codes in the directory `models/ltx_models`, `models/cosmos_models`, `models/pipeline` and `web_infer_utils/openpi_client` are modified from Diffusers, LTX-Video, Cosmos and openpi, which means these codes under Apache License 2.0." / "Other data and codes within this repo are under CC BY-NC-SA 4.0."
- 판정: 코드 자체는 상업 이용이 금지되고 파생물은 같은 조건을 따라야 합니다. 이 저장소에서 코드를 가져다 쓰는 연구는 비상업 조건을 확인해야 합니다. 논문 PDF (CC BY 4.0)와는 다른 조건입니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 전체 재현은 불가능합니다. 추론 일부는 조건부로 가능하고, 학습과 실로봇은 불가능합니다.**

| 작업 | 필요 자원 | 이 맥에서 | 판정 | 노력 추정 |
|---|---|---|---|---|
| GE-Base 학습 (AgiBot 100만 에피소드) | A100 32장 x 7일 + 3일 | 불가 | 불가 | - |
| GE-Act 행동 사전학습, 특화 | A100 8–16장 x 3–36시간, 데이터 수 TB급 | 불가 | 불가 | - |
| 공식 GE-Base 추론 (LTX 2B, 3시점, 9프레임, 5 디노이징) | 가중치 약 4 GB (LTX), T5-XXL 인코더, 자원 몇 GB | 조건부. MPS에서 bf16은 공식 코드가 막음 (`ge_trainer.py`의 MPS 예외). fp16 또는 fp32로 바꿔야 함 (메모리 근접, 속도 미측정) | 조건부 | 포팅 1–2주 (추정, 측정 전) |
| GE-Act 추론 (행동 54스텝) | GE-Base 추론 + 행동 분기 | 조건부. 위와 같음. 200 ms 주장은 RTX 4090 기준이라 맥에서는 느릴 가능성이 큼 | 조건부 | 위 포팅에 포함 |
| CALVIN, LIBERO 평가 (저장소 README 결과 재현) | 시뮬레이터 설치, 가중치 (CALVIN 학습 GE-Act), GPU 권장 | 시뮬 설치는 승인 필요. 추론 자체는 조건부 | 조건부 (설치 승인 필요, 현재 보류) | 2–3주 (추정) |
| GE-Sim (Cosmos2 변형) 추론 | Cosmos-Predict2 2B, T5, VAE, 샘플 데이터 (예제 포함) | 조건부. CLIP 기준 영상이 없어서 논문과 같은 조건이 되지 않음 (D4) | 조건부 | 2주 (추정) |
| 실로봇 (AgiBot G1, Agilex, Dual Franka) | 로봇, 텔레옵 | 불가 | 불가 | - |
| 표 재계산 (이 분석) | CPU | 가능, 완료 | 완료 | 수 초 |

- **가치가 있는 재현.** GE-Sim 폐루프 (행동 입력에 따른 영상 생성)는 공개 예제 (`sample_0`)로 한 번 돌려볼 가치가 있습니다. 다만 학습 없이 추론만이고 GPU 전제 대비 맥에서 느립니다. 재현 여부는 사용자 확인 후 결정합니다.
- **하지 말아야 할 것.** 공식 코드 설치와 실행, 2B 모델 가중치 다운로드는 이 분석의 범위가 아닙니다 (사용자 지시). 설치 전 승인 필요 (`CLAUDE.md`).
