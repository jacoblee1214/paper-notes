# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 대상: [OpenHLM-project/OpenHLM](https://github.com/OpenHLM-project/OpenHLM) 커밋 `4a8d4cd0ecd8313bb8dbb10d2afac2c2ec9ed1b4` (2026-07-10, 커밋 세 개). 얕은 clone(약 101 MB)을 정적으로만 읽었습니다. 실행, import, 설치는 하지 않았습니다.
- 저장소는 세 모듈입니다. `src/openpi4OpenHLM`(OpenPI의 수정본, VLA 학습·추론), `src/GR00T-WholeBodyControl4OpenHLM`(NVIDIA GR00T Whole-Body Control / GEAR-SONIC의 수정본: 원격조작, 데이터 수집, 배포 클라이언트), `src/HuMI4OpenHLM`(HuMI 수집·처리·IK).
- 논문은 arXiv v2입니다. 읽은 파일: `README.md`, `src/openpi4OpenHLM/src/openpi/training/config.py`, `scripts/train_pytorch.py`, `src/openpi/models_pytorch/{pi0_pytorch,preprocessing_pytorch}.py`, `src/openpi/transforms.py`, `src/openpi/policies/g1_policy.py`, `src/openpi/models/{pi0_config,tokenizer}.py`, `examples/unitree_g1/convert_g1_data_to_lerobot.py`, `src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/main.py`, `gear_sonic/scripts/run_openhlm_data_record.py`, `src/HuMI4OpenHLM/src/scripts/processing_pipeline/generate_final.py`와 README들.
- 번호는 D1–D17입니다. "일치"는 논문 값과 코드 값이 같다는 뜻이고, "차이", "미기재"는 코드에서 확인된 값이 논문에 없거나 다르다는 뜻입니다. 코드의 기본값이 논문 실험에 쓰였는지는 정적 읽기로 확정할 수 없습니다.

## 1. 설치 요구사항과 라이선스

| 항목 | 내용 |
|---|---|
| 학습 환경 | Python 3.11, `uv`, PyTorch 학습(`train_pytorch.py`), OpenPI의 `transformers_replace` 파일을 설치된 transformers에 덮어씀. 체크포인트 변환에 `gs://openpi-assets/.../pi05_base` (JAX 원본) 필요 |
| 하드웨어 | README: Ubuntu(25.04 시험), NVIDIA GPU(RTX 5080로 수집 시험, VRAM 8 GB 이상), G1, 그리퍼 2, RealSense D405 2, SV1-25, PICO4U. 학습은 README 예시가 GPU 4장 `torchrun` |
| 라이선스 | 최상위 Apache-2.0. 제3자: OpenPI Apache-2.0 + Gemma 이용약관(`LICENSE_GEMMA.txt`), GR00T-WBC 이중 라이선스(코드 Apache-2.0, 모델 가중치 NVIDIA Open Model License). 포함된 모델 산출물: ONNX 둘, `G1-clip.ckpt`(NOTICE) |
| HF 체크포인트 | 모델 카드 라이선스 표기 MIT(저장소 Apache-2.0과 표기가 다름). 데이터셋은 라이선스 표기 없음 |
| 크기 | 체크포인트 6개 × 7.47 GB = 약 44.8 GB, 데이터셋 약 297.5 GB(API 메타데이터 크기). 받지 않음 |

## 2. 논문 ↔ 코드 대조

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 행동 투영 초기화 (§3.2, 부록 C.2) | 앞 32차원은 복사, 새 차원만 Xavier | `train_pytorch.py`의 weight surgery가 `action_in_proj`와 `action_out_proj`의 앞 32열·행을 복사하고 나머지(34 − 32 = 2)를 Xavier uniform으로 초기화. 편향은 그대로 복사. `TrainConfig.action_proj_init_mode`가 `weight_surgery`(기본)와 `random` 중 선택 | 일치. 덧붙임: 기본 배열(D8)에서 사전학습 32슬롯 중 양팔 배열은 16개이므로, "보존"이 의미를 갖는 것은 앞 16차원으로 보입니다(추론, `03` V9) |
| D2 | 학습 하이퍼파라미터 (표 4) | AdamW(β 0.9, 0.95), 최고 LR 1e-4, 코사인, 워밍업 1k, 30k 스텝, 배치 128, 행동 구간 50, 224 × 224, A800 4장 24 h | `openhlm_example` 설정: 같은 값(최고 LR 1e-4, 워밍업 1,000, `decay_steps` 30,000, 스텝 30,000, `action_horizon` 50). `AdamW` b1 0.9, b2 0.95. README 학습 명령 `--batch_size 128`, `nproc_per_node=4`(배치는 GPU 수로 나눔: 장당 32) | 일치. 미기재: 코사인 종료 LR 1e-5(`decay_lr`), 가중치 감쇠 1e-10(사실상 0), 기울기 클리핑 1.0, EMA 없음, bf16 학습 |
| D3 | 배치 크기 | 128 | 설정 파일 기본값 `batch_size=256`, README 명령은 128, `pipe.sh`의 개발 메모는 32 | 일치(README 명령 기준). 세 곳의 값이 다름 |
| D4 | 상대 행동 목표 (§3.2 (3), 부록 C.2) | 본문: "chunk의 첫 번째 행동 기준", 부록: "chunk의 첫 번째 상태(a_t − s_0)" 기준 | `DeltaActions`에 두 변형. `use_first_action_delta=False`(기본)는 현재 상태 s를 기준(a − s), `True`는 chunk의 첫 행동 기준. 델타 마스크는 관절만 (`make_bool_mask(7, -1, 7, -1, 15, -3)`: 그리퍼와 루트 3 제외). `openhlm_example`은 `use_delta_joint_actions=False`(절대) | 일치(절대가 기본). 본문과 부록의 설명이 코드의 서로 다른 변형에 해당하며, 절제에 어느 쪽을 썼는지는 코드로 확인되지 않음 |
| D5 | 고유감각 입력 (§3.2 (4), 부록 C.2) | "현재 관절 위치 벡터 전체를 입력" | `discrete_state_input=True`: 34-D 상태가 분위수 정규화 후 256구간으로 이산화되어 프롬프트 텍스트("Task: ..., State: 1 2 ...; Action: ")로 들어감(`tokenizer.py`, `max_token_len` 200). `use_quantile_norm` 기본 True | 일치(상태는 입력됨). 미기재: 상태가 이산 텍스트 토큰이라는 점, 분위수 정규화, 256구간 해상도. 고유감각 제거 절제를 켜는 스위치는 저장소에 없음(D10) |
| D6 | 제어 주기와 실행 (§3.1, 부록 C.4) | 본문 "high-level 보통 10 Hz, low-level 50 Hz". 부록: 30 Hz, 50개 생성, 25개 실행, 5/6 s마다 추론 | 배포 클라이언트 `openpi-eval/main.py`: `control_hz=30`, `open_loop_horizon=25`, 50개 청크, 서버 호출은 동기. 기본 `num_steps=10`. 훈련 중 RTC 옵션(`use_training_time_rtc`)은 기본 False. 추론 쪽 RTC 호출은 확인되지 않음(이슈 #6이 이 점을 묻고 있음) | 일치(부록과). 본문 10 Hz와 코드 30 Hz가 다름 |
| D7 | 카메라 입력 (부록 C.1) | 머리 스테레오 RGB와 손목 스테레오 RGB | 수집 스크립트는 스테레오를 좌우로 나눠 저장하지만, 변환기와 정책 입력은 머리 왼쪽 영상 하나와 손목 영상 둘을 224 × 224로 변환해 사용(`G1Inputs`: `head_image_left`, `left_wrist_image`, `right_wrist_image`) | 차이. 머리 오른쪽 영상은 정책에 쓰이지 않음(저장은 됨) |
| D8 | 행동·상태 배열 (부록 C.2) | [왼팔 7, 왼 그리퍼 1, 오른팔 7, 오른 그리퍼 1, 왼다리 6, 오른다리 6, 허리 3, 루트 3] = 34 | `g1_policy.py` 주석과 변환기의 조립 순서가 같음 | 일치 |
| D9 | 데이터 증강 (부록 C.4) | 머리 카메라에 무작위 자르기와 회전, 모든 이미지에 밝기, 대비, 채도 | `preprocessing_pytorch.py`: 머리(`wrist` 아닌 키)에만 95% 무작위 자르기 후 원래 크기로 보간, ±5° 회전, 모든 카메라에 밝기 0.7–1.3, 대비 0.6–1.4, 채도 0.5–1.5. 난수는 `torch.rand(1)` 한 번으로 배치 전체가 같은 값을 씀 | 일치. 미기재: 증강 범위. 배치마다 한 번 뽑는 구현 |
| D10 | 절제 스위치 (§3.1–§3.2) | 구성 9가지 + 인터페이스 4종 | 있음: `action_proj_init_mode`(D1), 델타 행동(D4), `use_drifting`(+ `drifting_temperatures`, `drifting_gen_per_label`)과 `forward_drifting`/`sample_actions_drifting`, 서버 `--num-steps`, `random_init_action_expert`. 없음: 행동 순서 변경 스위치(G1Inputs의 단일 배열), 고유감각 제거 스위치, PaliGemma 초기화 로더(`random_init_action_expert`는 π0.5의 VLM을 유지하고 행동 전문가만 무작위로 바꿈), 처음부터 학습, 분리·VR 3점·SMPL용 VLA 변환기 | 부분. 1단계 생성과 투영 초기화 절제는 코드로 재구성 가능, 나머지 절제는 저장소만으로 재구성할 수 없음 |
| D11 | 1단계 생성 (§3.2, 부록 C.2) | 1단계 flow: 재학습 없음. drifting: 행동 전문가를 교체해 학습 | flow 1단계는 `sample_actions(num_steps=1)`, drifting은 별도 학습 손실(`drift_loss`, 다중 온도) 포함 | 일치. 학습 중 검증 로그에 10단계와 1단계의 MSE를 함께 기록하는 코드가 있음(`evaluate_sample_actions`) |
| D12 | 평가 | 5회 롤아웃, 루브릭 점수, 장기 과제 10회 | 저장소에 루브릭 채점 코드나 롤아웃 점수 기록은 확인되지 않음. 배포 클라이언트는 영상 저장과 행동 청크 저장 옵션을 가짐 | 미확인. 표 1, 3과 그림 4, 5의 롤아웃별 점수는 저장소 밖 |
| D13 | HuMI 처리 (§3.3, 부록 C.3) | 행동을 0.2 s 앞으로 당김, 어안 영상 보정, 오프라인 IK | `generate_final.py`: `teleop_delay_sec` 기본 0.2, 지연 프레임 `round(0.2 × fps)`로 `action_k = k + delay_frames`. 어안 → 직선 변환기(`fisheye_converter`)로 영상 보정 | 일치 |
| D14 | 공개 체크포인트와 데이터 | 코드, 데이터, 체크포인트 공개 (초록) | README는 체크포인트 5개를 나열(`12tasks_12full-teleop`, `..._8full-teleop_4humi`, `..._4stationary_teleop`, 장기 과제 둘). HF에는 `8tasks_8full-teleop`이 하나 더 있음. 데이터셋 디렉터리 5개(`g1_HLM-12_*` 셋, `long_g1_5_fruits_*` 둘) + 예시. 절제 모델(SMPL, PaliGemma, 1단계 등)과 기준선(GR00T N1.6, Ψ0) 체크포인트와 학습 설정은 없음 | 부분 일치. 8과제 기준 데이터셋 디렉터리는 목록에 없음(12과제 데이터의 부분집합일 수 있음, 미확인) |
| D15 | 프로젝트 주소 | `openhlm-project.github.io` | README 배지는 `openhlm-corl.github.io`, HF 모델 카드는 `openhlm-project.github.io` | 차이(같은 페이지인지 미확인) |
| D16 | 학습량과 장치 | A800 4장, 약 24 h | 설정과 README는 GPU 4장 `torchrun`. 시간은 코드로 확인 불가 | 확인 불가 |
| D17 | 저수준 제어기 | SONIC 범용 모션 트래커가 관절 목표를 추종 | 저장소에 GR00T-WBC/GEAR-SONIC 수정본과 ONNX 모델 포함. 배포는 C++ `gear_sonic_deploy`와 ZMQ 스트리밍. 미리보기 프레임 수는 SONIC 설정 파일(`num_future_frames`: 기본 5, release 설정 10)에 있음 | 부분. 논문의 미리보기 지연 Δt ∈ {0, 0.2, 0.4, 0.6} s를 바꾸는 설정은 확인되지 않음(트래커 학습과 묶인 값일 수 있음, 미확인) |

### 2.1 논문 주장의 코드 쪽 확인

- "공개된 레시피": 데이터 수집 → LeRobot 변환 → 분위수 정규화 통계 → 학습 → 정책 서버 → 로봇 클라이언트 흐름이 README에 단계별로 있고 코드가 대응합니다(D2, D6, D12).
- "행동 순서를 π0.5 배열 뒤에 붙인다": 코드의 34-D 순서와 같습니다(D8).
- "1단계 생성이 열세": 코드는 두 변형 모두 제공하지만 절제 실행 기록은 없습니다(D11).
- 저장소 이슈 세 건은 열려 있고 답이 없습니다: SONIC 잠재 토큰과의 비교(#5), RTC 사용 여부(#6), 시뮬 폐루프 벤치마크 요청(#7).

## 3. 공개 자료의 범위

| 항목 | 공개 여부 | 메모 |
|---|---|---|
| 수집·변환·학습·서빙·배포 코드 | 있음 | D2, D6 |
| 최종 체크포인트 | 6개 | 조건 대응은 D14 |
| 시연 데이터 | 5개 데이터셋 + 예시 | 약 297.5 GB |
| 절제 모델, 기준선 학습 설정 | 없음 | 부록 D.2 서술만 있음 |
| 평가 롤아웃 점수 | 확인 안 됨 | D12 |

## 4. 구현 가능성 (Apple M4 Pro, 통합 메모리 약 24 GiB, MPS, CUDA 없음)

**판정: 실로봇 평가와 학습은 이 맥에서 불가능합니다. 정적 읽기와 소규모 점검(데이터 변환 구조, 체크포인트 오프라인 점검)은 가능하고, 일부는 승인과 설치가 필요합니다.**

| 수준 | 가능성 | 근거와 조건 |
|---|---|---|
| 코드 정적 대조 | **가능 (완료)** | 이 문서 §2 |
| 표 수치 재계산, 검정 | **가능 (완료)** | `03`, `analysis/recompute.py` |
| 예시 시연으로 변환 파이프라인 점검 (LeRobot 변환, 정규화 통계) | **조건부** | 예시 데이터 약 0.49 GB 다운로드, `uv` 환경과 lerobot 설치 승인 필요. CUDA 불필요 |
| 공개 체크포인트로 오프라인 행동 예측 점검 (검증 MSE 10단계 대 1단계) | **조건부** | 7.47 GB 체크포인트와 데이터 일부 다운로드 승인, 설치와 공식 코드 실행 승인 필요. PyTorch 경로이므로 MPS에서 돌 가능성이 있으나 `torch.compile`, bf16, `transformers_replace` 의존으로 시험 전에는 알 수 없음(미측정) |
| HuMI 처리와 오프라인 IK 재계산 | **조건부** | `mink`+MuJoCo 설치와 데이터 승인. GoPro 동기 등은 원시 데이터 필요 |
| MuJoCo 시뮬 폐루프 | **조건부(어려움)** | 저장소 시뮬 루프는 있으나 저수준 배포가 C++ `gear_sonic_deploy`와 ZMQ이고 TensorRT 등 Linux/NVIDIA 전제로 보임(미확인). 시뮬 벤치마크는 없음(이슈 #7) |
| 학습 (4 × A800, 약 24 h, 30k 스텝) | **불가** | 맥 MPS에서 수 B 파라미터 VLA를 30k 스텝 학습하는 것은 시간, 메모리 모두 비현실적 |
| 표 1, 3, 그림 4, 5 실로봇 재현 | **불가** | G1, PICO4U, 카메라, 그리퍼, 장면 세트 필요 |
| 기준선(GR00T N1.6, Ψ0) 재학습·평가 | **불가** | 실로봇 + GPU, 설정은 저장소 밖 |

**노력 추정(추정, 측정 전).**

- 변환 파이프라인과 체크포인트 오프라인 점검까지: 승인 후 2–4일. MPS 호환 문제가 나오면 늘어납니다.
- 이 논문의 핵심 질문 일부(행동 순서, weight surgery의 슬롯 효과, 1단계 대 10단계)를 소형 모델과 시뮬 환경으로 재현하는 설계: 수 주(별도 설계 필요).
- 자신의 로봇에서 레시피 전체를 따라 하는 일: 하드웨어 구성부터 수 개월.

## 5. 재실행

이 문서의 검사는 파일 읽기로 끝납니다. 공식 저장소를 다시 얻으려면 아래 명령으로 얕은 복제만 하고, 실행하거나 설치하지 마십시오.

```bash
git clone --depth 1 https://github.com/OpenHLM-project/OpenHLM.git repo
git -C repo checkout 4a8d4cd0ecd8313bb8dbb10d2afac2c2ec9ed1b4
```
