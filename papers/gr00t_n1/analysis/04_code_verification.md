# GR00T N1 논문과 공식 코드의 대조 (04)

## 1. 대상과 방법

- 대조 대상: 논문 arXiv v2, 공식 코드 `NVIDIA/Isaac-GR00T` 태그 `n1-release` (커밋 `755876a9afdb`, 2025-06-11). 기본 브랜치(N1.7)는 쓰지 않았다.
- 읽은 파일: `gr00t/model/action_head/flow_matching_action_head.py`, `gr00t/model/backbone/eagle_backbone.py`, `gr00t/model/gr00t_n1.py`, `gr00t/experiment/data_config.py`, `gr00t/data/dataset.py`, `scripts/gr00t_finetune.py`, `gr00t/utils/eval.py`, `README.md`, `LICENSE`.
- 방법: 정적 읽기와 `grep`만 했다. 코드를 실행하거나, 가져오거나(import), 설치하지 않았다. 가중치와 데이터도 받지 않았다.
- 모델 설정의 기본값(예: `num_inference_timesteps`, `select_layer`의 실제 값)은 HF의 `config.json`에 있어서 이 저장소에서 확인되지 않는 것이 있다. 이런 항목은 "미확인"으로 표시했다.
- 라벨: D는 논문과 코드의 차이이고, V는 수치 점검 (`03_analysis.md`)이다.

## 2. 일치 확인 (논문 주장 ↔ 코드)

| 항목 | 논문 | 코드 (위치) | 판정 |
|---|---|---|---|
| 행동 청크 H = 16 | §2.1 | `data_config.py` 65, 146 (`action_indices = list(range(16))`), `utils/eval.py` 43 | 일치 |
| 흐름 매칭 손실 (MSE) | 식 (1) | `flow_matching_action_head.py` 315 (`F.mse_loss(pred_actions, velocity)`) | 일치 (목표 부호는 D1) |
| 노이즈 노출 시간 분포 Beta(1.5, 1), s = 0.999 | §2.1 | `flow_matching_action_head.py` 155–158 (`noise_beta_alpha=1.5`, `noise_beta_beta=1.0`, `noise_s=0.999`) | 일치 (수식 형태는 D2) |
| 추론: 오일러 적분, dt = 1/K | §2.1 | `flow_matching_action_head.py` 341 (`dt = 1.0 / num_steps`), 375 (`actions = actions + dt * pred_velocity`) | 일치 (갱신식) |
| 추론 단계 수 K = 4 | §2.1 | 코드는 `num_inference_timesteps`를 읽는다 (`policy.py` 114–118, `flow_matching_action_head.py` 163). 기본값은 설정 파일에 있다. | **미확인** (아래 §5) |
| VLM 12번째 층 사용 | §2.1 | `eagle_backbone.py` 82 (`select_layer: int = 12`), 113 (`while len(layers) > select_layer`: 뒤쪽 층을 잘라 냄) | 일치 (구현은 층을 잘라 낸다) |
| 언어 모델 고정, 비전 인코더와 DiT 학습 | §2.3, 표 6 | `eagle_backbone.py` 84 (`tune_llm=False`), `gr00t_n1.py` 201–204 (`tune_visual=True`, `tune_llm=False`, `tune_projector=True`, `tune_diffusion_model=True`) | 일치 |
| 임베디먼트별 인코더·디코더 | §2.1 | `flow_matching_action_head.py` 167 (`max_num_embodiments=32`), 195–204 (`CategorySpecificMLP`, `MultiEmbodimentActionEncoder`) | 일치 (논문은 "MLP", 구현은 임베디먼트 인덱스별 가중치 뱅크) |
| 행동 인코더에 확산 시간 포함 | §2.1 | `flow_matching_action_head.py` 287–290 (`action_encoder(noisy_trajectory, t_discretized, embodiment_id)`) | 일치 |
| 입력 224×224 | §2.1 | `data_config.py` 102, 183 (`VideoResize(height=224, width=224)`) | 일치 |
| 64 이미지 토큰 프레임당 | §2.1 | 설정 파일에서 확인 필요. 패치 크기 14 가정 시 16×16 패치 → 픽셀 셔플(0.5) → 8×8 = 64로 계산이 맞는다 (계산만, 설정 미확인). | **미확인** (계산상 일치) |
| 상태 min-max 정규화, 상태 회전 6D | 부록 E.2 | `data_config.py` 118 (`normalization_modes = min_max`), 394·503 (`state.*_eef_quat: rotation_6d`) | 일치 (상태 회전만) |
| 미세조정 AdamW, β = (0.95, 0.999), ε = 1e-8, 학습률 1e-4, 가중치 감쇠 1e-5, 웜업 0.05, 코사인 | 표 6 | `scripts/gr00t_finetune.py` 81–87 (lr 1e-4, wd 1e-5, warmup 0.05), 172–179 (`adam_beta1=0.95`, `adam_beta2=0.999`, `adam_epsilon=1e-8`, `lr_scheduler_type="cosine"`) | 일치 |
| 하이퍼파라미터 기본값 (배치, 스텝) | 표 6: 배치 128 또는 1024, 20k–60k 스텝 | `scripts/gr00t_finetune.py` 49, 52 (`batch_size=16`, `max_steps=10000`) | **차이** (D5) |
| 라이선스 | 논문은 CC BY 4.0 (arXiv) | `LICENSE` Apache-2.0 | 코드 라이선스는 Apache-2.0 |
| 평가 행동 청크 16 | §4 | `utils/eval.py` 43 (`action_horizon=16`) | 일치 |

## 3. 차이 (D1–D9)

### D1. 목표 부호: 식 (1)과 코드가 반대다 (표기 오류로 보임)

- 논문 식 (1): 목표 V ≈ ε − A. 같은 절의 오일러 갱신 A^{τ+1/K} = A^τ + (1/K)V_θ와 함께 읽으면 부호가 맞지 않는다.
- 보간식 A^τ = τA + (1−τ)ε에서 dA/dτ = A − ε이므로 학습 목표는 A − ε여야 하고, 그래야 "+V" 갱신과 맞는다.
- 코드: `flow_matching_action_head.py` 287 `velocity = actions - noise` (목표 A − ε), 315 MSE 손실, 375 `actions + dt * pred_velocity`. 코드는 일관된다.
- 토이 확인 (`03_analysis.md` §5): 식 (1)을 문자 그대로 학습하고 "+V"로 적분하면 평균 −3.7, 표준편차 4.7이 나온다 (목표 1.0, 0.1). 코드 규약은 목표에 맞는다.
- 판정: 논문 식의 부호 표기 오류. 구현은 코드가 맞다.

### D2. 시간 분포의 수식 형태가 다르다 (영향 작음)

- 논문: p(τ) = Beta((s−τ)/s; 1.5, 1). 이 식은 τ = s(1−x), x ~ Beta(1.5, 1)로 읽힌다. 기대값 E[τ] ≈ 0.3996.
- 코드 (`flow_matching_action_head.py` 259–261): x ~ Beta(1.5, 1), t = (s − x)/s. 기대값 E[t] ≈ 0.3994. 분포 모양도 거의 같다.
- 차이: 코드 t의 범위는 [(s−1)/s, 1] = [−0.001, 1]이라 0보다 아주 조금 작은 값이 나올 수 있다. 실용상 영향은 없다. 논문 표기가 모호한 것으로 본다.
- 판정: 표기 차이. 영향 작음.

### D3. 보조 검출 손실 L_det이 코드에 없다

- 논문 부록 F: OWL-v2 상자 중심 좌표를 목표로 하는 선형 층과 제곱 손실을 더한다 (L = L_fm + L_det).
- 코드: `owl`, `bbox`, `bounding`, `detection loss`, `localiz` 키워드를 `gr00t/` 전체 `.py`에서 검색했으나 일치하는 항목이 없다.
- 손실 항은 `flow_matching_action_head.py` 315–318 한 곳뿐이다. 보조 손실은 공개 코드에 없다.
- 판정: 논문에만 있다. 재현할 때 차이의 원인이 될 수 있다. 본문 4.4의 결과가 이 손실에 의존하는지는 알 수 없다 (본문에 ablation 없음).

### D4. 데이터 생성 파이프라인이 공개되지 않았다

- 논문에 있지만 `n1-release`에는 없는 것: 잠재 행동 VQ-VAE 학습 (LAPA), IDM 학습과 가짜 행동 라벨링, Wan2.1-I2V-14B LoRA 미세조정과 신경 궤적 생성, 다중모달 LLM 필터링과 재캡션, DexMimicGen 시연 생성, 상자 라벨 (OWL-v2).
- 코드가 읽는 쪽: `gr00t/data/dataset.py` 444 (`lapa_action`, `dream_actions` 키를 특별 취급), `gr00t/data/transform/concat.py` 163 (행동 없는 데이터, LAPA). 즉 **미리 만든 라벨을 학습에 쓰는 코드만** 있다.
- 판정: 신경 궤적 데이터를 논문처럼 다시 만들 수 없다. 공개 저장소의 DreamGen 코드(`GR00T-Dreams`)와 DexMimicGen 코드가 일부를 대체할 수 있지만 같은 파이프라인은 아니다.

### D5. 미세조정 스크립트의 기본값과 논문 설정이 다르다

- 논문 표 6: 미세조정 배치 128 또는 1024, 스텝 20k–60k. 시뮬 DexMG 과제는 배치 128.
- 스크립트: `batch_size=16` (49행), `max_steps=10000` (52행). 예제용 기본값으로 보인다.
- 판정: 스크립트 기본값은 논문 설정이 아니다. 논문 재현에는 설정을 따로 맞춰야 한다. 학습률·옵티마이저·스케줄은 일치한다.

### D6. 보고된 평가 프로토콜이 이 저장소에 없다

- 논문: 100회 시행 평균, 마지막 5개 체크포인트 중 최댓값, 500 스텝마다 저장 (§4.3).
- `n1-release`의 `utils/eval.py`는 오프라인 평가와 행동 청크 실행 루프를 가지고 있지만, "마지막 5개 체크포인트 최댓값" 선택 로직은 없다. 시뮬 벤치마크 평가 스크립트는 이 저장소에 없다.
- 판정: 평가 절차는 코드로 확인되지 않는다 (미확인). 그래서 체크포인트 선택 편향의 크기는 알 수 없다 (`03_analysis.md` §1).

### D7. 행동 회전 표현 (축-각)이 데이터 설정에서 보이지 않는다

- 논문 부록 E.2: 행동의 엔드이펙터 회전은 축-각(axis-angle)으로 표준화한다.
- 코드: 상태 회전은 `rotation_6d`로 바꾸는 설정이 있다 (`data_config.py` 394, 503). 행동 회전에 대한 명시적 변환은 `data_config.py`에서 찾지 못했다. 변환은 데이터 전처리 쪽에서 이루어졌을 수 있다.
- 판정: 행동 회전 표현은 미확인이다. 상태 쪽은 일치한다.

### D8. 논문에 없는 구현 세부 (참고)

- 이산 시간 버킷 1000개 (`num_timestep_buckets`, 160행) 와 위치 임베딩 (`add_pos_embed`, 조건부). 논문에 없다.
- 입력 증강: 영상 크롭 0.95 (`VideoCrop(scale=0.95)`, `data_config.py` 101). 논문에 없다.
- 판정: 구현 세부. 재현 결과에 영향을 줄 수 있으나 논문 주장과 모순되지는 않는다.

### D9. 공개 저장소의 버전 혼동 가능성

- 공식 저장소 기본 브랜치의 README와 코드는 N1.7이다 (2026-11-17 push). N1 수치를 재현하려면 `n1-release` 태그를 써야 한다.
- HF `nvidia/GR00T-N1-2B` 모델 카드는 "Isaac GR00T N1-1B"라는 문장을 포함한다 (모델 카드 본문). 이름과 파라미터 수가 논문(2.2B, 1.34B VLM)과 맞는지는 설정 파일 확인이 필요하다 (미확인).
- 판정: 재현 시 태그와 체크포인트 버전을 기록해야 한다.

## 4. 구현 가능성 판정

질문: 이 맥(Apple M4 Pro, 24 GiB 통합 메모리, MPS, CUDA 없음)에서 GR00T N1을 구현할 수 있는가?

판정: **논문 주장의 재현은 불가능하다.** 추론과 소규모 미세조정은 포팅 후 조건부로 가능할 수 있다. 사전학습과 데이터 생성은 불가능하다.

| 작업 | 판정 | 근거 |
|---|---|---|
| 사전학습 재현 (1,024 GPU, 약 50,000 H100 시간) | **불가** | 자원 규모. 맥은 CUDA가 없다. |
| 신경 궤적 생성 (L40 약 105k 시간) | **불가** | 자원 규모. D4에서 보듯 생성 파이프라인이 공개되지 않았다. |
| 공식 체크포인트 추론 (맥) | **조건부** (포팅 필요) | 가중치 2.2B bf16은 약 4.4 GB로 메모리에는 들어간다. 그러나 README가 CUDA 12.4와 flash-attn 2.7.1을 요구하고, `deployment_scripts`는 TensorRT를 쓴다. MPS 포팅과 속도 측정은 하지 않았다. |
| 전체 미세조정 (2.2B) | **불가** (메모리 계산) | AdamW 상태를 포함하면 fp32 기준 약 35 GB가 필요하다 (파라미터 4바이트 + 그래디언트 4바이트 + Adam 8바이트, 대략 2.2B). 24 GiB를 넘는다. |
| DiT·어댑터만 미세조정 | **조건부** | DiT 파라미터 수를 설정 파일에서 확인하지 못했다 (미확인). 확인 후 메모리를 다시 계산해야 한다. |
| 시뮬 평가 (RoboCasa 등) | **조건부** | 시뮬 환경 설치(RoboCasa, DexMimicGen, robosuite 계열)가 필요하고, 과제당 100회 × 과제 57개 × 체크포인트 5개 = 약 28,500 에피소드다. 맥에서 시간이 현실적인지 측정하지 않았다. |
| 토이 재현 (목표 규약, 부호, K 효과) | **시작 가능** (CPU) | `toy_flow_sign.py`로 이미 수행. 1초 안팎. |

노력 추정 (추정치, 측정 아님):
- 추론 포팅 (flash-attn 대체, MPS 확인, bf16 점검): 1–2주.
- 소규모 미세조정 (DiT 일부): 포팅 이후 2–4주. 데이터는 공개 시뮬 데이터셋 일부가 필요하며, 이때 용량을 먼저 확인해야 한다.
- 논문 수준 재현: 이 맥에서는 범위 밖이다.

## 5. 미확인 항목 (이 저장소만으로는 확인 불가)

- 추론 단계 수 K의 기본값 (HF `config.json`)과 `num_timestep_buckets`의 실제 값.
- 시각 토큰 수 64의 실제 패치 크기 (설정 파일).
- VLM 12번째 층 잘라내기의 정확한 방식 (구현은 층을 잘라 낸다고 확인했지만, 가중치 파일과 맞는지는 모른다).
- 파라미터 수 2.2B, VLM 1.34B (가중치 파일이 없어서 확인하지 못함).
- 시뮬 평가 프로토콜 (D6).
- 행동 회전 표현 (D7).
- DexMimicGen 등 외부 저장소의 라이선스 (NOASSERTION, `02_prior_work.md` §4 참고).
