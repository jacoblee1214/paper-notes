# 04. 논문 ↔ 공식 코드 대조 (정적 읽기만)

대상 저장소: `github.com/yuantianyuan01/FastWAM`, 얕은 복제 `repo/` (커밋 `7faa711`, 2026-08-20 "Optimize IDM action-only inference"). 논문 v2(2026-03-23)보다 약 5개월 뒤의 코드입니다. 이 문서는 **논문 v2 설명과 현재 코드**를 대조하며, 발표 당시 코드와의 차이는 확인하지 못했습니다. 코드는 읽기만 했습니다. 실행, import, 설치, 가중치 다운로드는 하지 않았습니다.

라이선스: `LICENSE`는 MIT입니다("Except for third-party code… released under the MIT License"). 동봉된 `third_party/RoboTwin`은 원 저장소 라이선스를 따릅니다. GitHub API는 저장소 라이선스를 `Other`로 표시하는데, 이는 LICENSE 파일 형식 때문으로 보입니다. 저장소 메타: 별 1665, 포크 214, 마지막 push 2026-08-20.

## 1. 차이 목록 (D1–D16)

"일치"는 논문과 코드가 같은 것, "미기재"는 논문에 없는 구현 세부, "차이"는 설명이 다른 것입니다.

| 번호 | 항목 | 논문 (v2) | 코드 (7faa711) | 구분 | 근거 |
|---|---|---|---|---|---|
| D1 | 정책 입력 | 식 (3)·§3.1: p(a \| o, l). 관측과 지시문만 | LIBERO 설정은 상태(proprio, 8차원)를 선형층으로 임베딩해 T5 문맥 뒤에 토큰으로 붙입니다 (`proprio_output_dim: 8`, `proprio_dim`) | 차이 | `src/fastwam/models/wan22/fastwam.py` 224–244, 363–376; `configs/data/libero_2cam.yaml` |
| D2 | 시간 t 샘플링 | §4.1: "Following [36], logit-normal distribution over t" | logit-normal 없음. 시프트 기반 역함수 φ(u)=s·u/(1+(s−1)u)에서 균등 u를 뽑음. 비디오 시프트 5.0, 행동 시프트 1.0 (학습) | 차이 | `schedulers/scheduler_continuous.py` `sample_training_t`, `_phi`; `configs/model/fastwam.yaml` 48–56 |
| D3 | 학습 손실 가중 | 식 (6): 균일 MSE | 시간 t에 따른 가우시안 모양 가중치(`training_weight`)를 곱함 | 차이 | `fastwam.py` 585–601; `scheduler_continuous.py` `training_weight` |
| D4 | 손실 가중치 λ | 식 (9): λ "balances", 값 미보고 | 기본 λ_video = λ_action = 1.0 (`runtime.py` 157; `loss.lambda_action: 1.0`). λ_video는 설정 파일에 없어 기본값 | 미기재 (값) | `src/fastwam/runtime.py` 157–158; `configs/model/fastwam.yaml` 60 |
| D5 | 첫 관측 프레임 | "clean latent tokens of the first observation frame" | 학습 중 첫 프레임 latent를 잡음 없이 교체하고, 비디오 손실에서 첫 프레임을 뺌 | 일치 (세부 미기재) | `fastwam.py` 563–586 |
| D6 | 행동 전문가 초기화 | §4.1: 1B 행동 전문가. 초기화 방법 미기재 | Wan2.2 DiT 가중치를 선형 보간해 ActionDiT를 미리 만듦 (`scripts/preprocess_action_dit_backbone.py`, README "Model Preparation") | 미기재 (중요) | README; `scripts/preprocess_action_dit_backbone.py` |
| D7 | 추론 스텝, CFG | §4.1: 10 스텝, CFG 1.0 | `eval_num_inference_steps: 10`, `text_cfg_scale: 1.0`. 일치 | 일치 | `configs/train.yaml` 24; `configs/sim_libero.yaml` 33–34 |
| D8 | 추론 시간 이동(shift) | 미기재 | 행동 추론 시프트 기본 1.0, README는 "원 체크포인트는 `EVALUATION.sigma_shift=5.0`으로 재현" | 미기재 (재현에 영향) | `configs/model/fastwam.yaml` 56; `sim_libero.yaml` 33; README "Other improvements" |
| D9 | 재계획 간격 (chunk 실행) | 미기재 | LIBERO `replan_steps: 10`, `num_steps_wait: 30`. RoboTwin `replan_steps: 24` (32 중 24개 실행) | 미기재 | `configs/sim_libero.yaml` 36–41; `configs/sim_robotwin.yaml` 24–25 |
| D10 | "단일 순전파" | §3.2, §1: "single forward pass", 잠재 표현 한 번 | 비디오 전문가 1회 (타임스텝 0, 키-값 캐시), 이후 행동 전문가를 추론 스텝마다 실행 (10회). "단일"은 비디오 백본에 대해서만 맞음 | 해석 차이 | `fastwam.py` 1060–1160 (`infer_action`) |
| D11 | 학습 스텝 | §4.2: LIBERO 20k, RoboTwin 30k, 실로봇 30k | LIBERO 설정은 `num_epochs: 10`, `max_steps: null`. RoboTwin은 `num_epochs: 5`. 스텝 수는 데이터 크기에 따라 정해져 직접 대응하지 않음 | 확인 불가 | `configs/task/libero_uncond_2cam224_1e-4.yaml`; `robotwin_uncond_3cam_384_1e-4.yaml` |
| D12 | RoboTwin 평가 | §4.2: 과제당 100회 | `eval_num_episodes: 100`, `task_config: demo_randomized`, `instruction_type: unseen` (README: Motus 프로토콜). 논문은 지시문 설정을 적지 않음 | 일치 (지시문 미기재) | `configs/sim_robotwin.yaml` 10–13 |
| D13 | Joint 변형 | §3.3: 비디오와 행동 토큰을 함께 denoise, 행동이 생성된 미래 영상 전체를 봄 | 행동→전체 영상 허용, 비디오→비디오는 first_frame_causal | 일치 | `fastwam_joint.py` 29–50 |
| D14 | IDM 변형 | §4.1: 잡음 증강 p=0.5 | `video_cond_noise_prob=0.5` 기본. 학습은 강제 교사 마스크(teacher-forcing), 추론은 영상 생성 후 행동 | 일치 | `fastwam_idm.py` 19–68, 175–240 |
| D15 | 공개 체크포인트 수치 | 표 2: Fast-WAM LIBERO 97.6 (Spatial 98.2, Object 100.0, Goal 97.0, Long 95.2), IDM 98.0 | README 표(sigma_shift 1.0 체크포인트): First-frame 97.75 (98.2, Object 99.2, Goal 97.8, Long 95.8), IDM 98.55 (99.0, 99.6, 98.6, 97.0) | 차이 (체크포인트·설정) | README "Optional IDM" 표 |
| D16 | 지연 측정 하드웨어 | §4.1, 초록: RTX 5090D V2 32GB, 190 ms | README 최신 개선: "RTX 4090 190 ms → 110 ms" (190 ms를 4090 기준으로 적음). 프로젝트 페이지는 5090D | 차이 (하드웨어 표기 불일치) | README "Approximately 2x faster inference"; project page |

## 2. 보조 확인

- **attention 마스크 (식 없음, 그림 2b와 §3.2 설명)**: `_build_mot_attention_mask`는 행동→행동 전체, 행동→첫 프레임만, 첫 프레임→자기 자신만, 미래 영상→전체를 허용합니다(`fastwam.py` 397–417, `wan_video_dit.py` 517–521 `first_frame_causal`). 논문 §3.2의 "행동은 미래 영상을 못 본다", "첫 프레임은 다른 토큰을 보지 않는다"와 일치합니다.
- **추론 경로**: 첫 프레임만 한 번 인코딩하고 KV 캐시를 만든 뒤, 행동 전문가 10 스텝을 돌립니다. 논문 그림 1(C)과 같습니다. 행동 생성에 쓰는 행동 전문가 1B의 10회 순전파가 지연의 대부분입니다.
- **옵션 IDM 체크포인트**: 같은 모델에서 `action_infer_mode=idm` 또는 `first_frame`을 고를 수 있습니다(README). 논문의 "같은 조건 비교"를 같은 가중치로 할 수 있다는 뜻이므로, 재현에서 가장 공정한 대조군입니다. 다만 이 체크포인트의 학습은 논문 Fast-WAM과 다릅니다(D15).
- **평가 프로토콜**: `num_trials: 50` per task, 40 과제 (LIBERO). 논문의 "2000 trials"와 일치합니다.
- **시뮬레이터 버전**: LIBERO에 `mujoco==3.3.2`를 요구합니다(README). 논문은 시뮬 버전을 적지 않습니다.

## 3. 구현 가능성 (이 맥 기준, 이 절이 판정)

이 맥: Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 사용 가능, CUDA 없음.

| 작업 | 판정 | 이유 |
|---|---|---|
| 공식 코드로 6B 모델 학습 (LIBERO 8 GPU, RoboTwin 64 GPU) | **불가** | 대규모 GPU 전제. 이 맥에서 학습은 현실적이지 않음 |
| 공식 체크포인트로 LIBERO 평가 | **불가 (GPU)** | 코드가 `device: cuda`, DeepSpeed, `torch==2.7.1+cu128`을 전제. 가중치 약 수 GB, 시뮬 설치 필요 |
| 5B 비디오 백본의 MPS 추론 포팅 후 지연 측정 | **조건부** | 포팅 작업이 필요하고, bf16 가중치 약 12 GB(추정, 6B × 2 bytes). 가중치 다운로드는 승인 필요. 포팅·측정 1–2주(추정) |
| 축소 재구현 (수 M–수십 M 파라미터, MPS 또는 CPU) | **조건부** | 설계 요소(공동 마스크, 첫 프레임 KV 캐시, 행동 흐름 매칭)는 검증 가능. 논문 수치는 재현되지 않음. 노력 1–2주(추정) |
| 과제 1개 (실로봇) | **불가** | 로봇과 60시간 원격 조작 데이터 필요 |
| 토이 (합성, numpy) | **가능, 완료** | `analysis/toy_imagination.py` |

**구현 판정: 부분 가능.** 공식 결과 재현은 GPU가 필요하므로 이 맥에서는 할 수 없습니다. 아이디어(비디오 보조 손실이 표현을 바꾸는가)와 설계 요소는 축소 재구현과 토이로 확인할 수 있습니다. 추론 지연 주장은 MPS에서 같은 방식으로 재현되지 않으므로 측정하려면 포팅이 필요합니다. 지금 재현을 시작하려면 사용자 승인이 먼저 필요합니다(`CLAUDE.md`).

## 4. 발견 요약

1. 논문의 핵심 설계(첫 프레임만 KV 캐시, 행동 전문가가 미래 영상을 보지 않음)는 코드에서 확인됩니다. D10만 해석을 조정해야 합니다.
2. 논문 식과 다른 구현이 세 곳 있습니다. 입력에 상태가 들어가고(D1), 시간 샘플링이 logit-normal이 아니며(D2), 손실 가중이 식 (6)과 다릅니다(D3). 어느 것이 수치에 영향을 주는지는 확인하지 못했습니다(ablation이 논문에 없음).
3. 행동 전문가의 초기화가 Wan2.2 DiT 보간이라는 점(D6)은 "임베디드 사전학습 없음"의 해석을 바꿉니다. 비디오 생성 사전학습의 영향이 크게 남아 있습니다.
4. 추론 설정 세 개(D8 시프트, D9 재계획 간격, 체크포인트 버전 D15)는 논문에 없고, 재현 수치를 크게 바꿀 수 있습니다. 특히 시프트는 README가 명시적으로 다르다고 적습니다.
5. 논문과 최신 코드의 수치 차이는 작지만 방향이 같습니다. Object 100.0 대 99.2, Long 95.2 대 95.8. 평균 차이는 0.15포인트 안입니다.
