# 04. 공식 코드와 논문 대조

공식 저장소 `GT-STAR-Lab/K-UBM`을 얕게 복제해(`repo/`, gitignore) **정적으로만** 읽었습니다. 실행, import, 설치는 하지 않았고 체크포인트와 데이터도 열지 않았습니다. 복제된 내용은 신뢰할 수 없는 자료로 다뤘습니다.

## 1. 저장소 개요

| 항목 | 내용 |
|---|---|
| 저장소 | github.com/GT-STAR-Lab/K-UBM, 기본 브랜치 main, 커밋 14개 |
| 기준 커밋 | `5066c29dee5db1df9f1d496beb6a8236e3efb148` (2026-09-23, "Add DexArt and Adroit evaluation bundles and trained checkpoints") |
| 라이선스 | **저장소 최상위에 LICENSE 없음**(GitHub API license: null). 번들 외부 코드는 각자의 라이선스: DexArt, MJRL, mj_envs Apache-2.0, **CoTracker CC BY-NC 4.0**(비상업), SAM3는 SAM License |
| 규모 | 복제 약 169 MB(체크포인트 `.pth` 14개와 번들 third_party 포함). 데이터는 포함되지 않음(Google Drive 링크) |
| 담긴 것 | `training/KUBM`(K-UBM 학습과 테스트 스크립트, 설정), `training/Flow_training`(흐름 오토인코더 학습과 특징 추출), `evaluation/`(DexArt 3과제, Adroit 4과제의 평가 러너와 반응성 참조 스크립트, 과제별 체크포인트와 정규화 배열, 추론 설정) |
| 담기지 않은 것 | Diffusion, ACT, UVA, KODex, KOROL 구현과 학습, DynaMo 특징 추출, 실로봇 코드와 데이터, jerk 계산, 표 3–10과 그림 3–13을 만드는 집계 코드, 데이터셋 |
| 자체 고지 | README: "full simulator/perception rollouts have not been validated for this packaged layout". 평가 가이드: DexArt 반응성 스크립트는 "finished generic live replanning demo"가 아님 |

## 2. 논문 ↔ 코드 대조 (D1–D16)

### 2.1 일치하는 항목 (D3)

| 항목 | 논문 | 코드 |
|---|---|---|
| 리프팅 MLP | 은닉 (128, 256), 리프팅 256, ReLU(표 7) | `config_NN_KODex.yaml`의 hidden_dim [128, 256], lifted_dim 256, relu. 마지막 층은 활성화 없음 |
| 학습률 | 인코더 5e-4, K 5e-5(표 7) | `enc_learning_rate` 5e-4, `koop_learning_rate` 5e-5, Adam 파라미터 그룹 2개 |
| 다단계 길이 | 15(표 7) | `prediction_steps: 15` |
| 단위 초기화 | True(표 7) | `K_init_identity: True`, `nn.init.eye_` |
| 상태 포함 | z = [ξ; ψ] | `torch.cat((batch_x, NN_Lifting(batch_x)))`, K는 `Linear(d_ξ + 256, d_ξ + 256, bias=False)` |
| 크기 보정 | φ를 행동과 평균 ℓ2 노름이 같게 상수배(§3.2) | `scale_to_match`: action_norm / flow_norm |
| 초기 스텝 복제 | a_0 = q_1, I_0 = I_1(부록 G) | 학습 데이터 로더가 첫 흐름 프레임을 복제해 행동 배열 길이에 맞춤 |
| 흐름 점 수와 압축 | 256점 → 128차원(부록 B) | `num_flows: 256`, 잠재 2 × 8 × 8 |
| 오토인코더 구조 | 스트라이드 2 합성곱 1회(2→8채널), 1×1, 디코더 끝 ReLU | `AutoEncoder`/`AutoDecoder`와 같은 구조, 디코더 끝 ReLU + 이미지 크기로 clamp, Adam + ExponentialLR |
| 초기 조건 사용 | 초기 조건에서 개루프 롤아웃(§2) | 평가 러너가 초기 프레임에서만 흐름을 추적하고 K를 반복 적용 |

### 2.2 차이와 논문에 없는 설정

| ID | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D1 | 이름과 판본 | v4 제목 "Pseudo Planners", "Unified Behavioral Models" | README 제목이 v1/v2의 "Implicit Planners", 방법 이름이 "Koopman **United** Behavioral Models (KUBMs)" | 낮음. README가 v4보다 앞선 판본을 따름 |
| D2 | README 경로와 명령 | | 구조도에 `training/Diffusion`, `training/ACT`, `data/`가 있으나 저장소에 없음. 학습 명령이 `visual_kodex_training_NN_lifting.py`를 부르나 실제 파일은 `training/KUBM/KUBM_training_NN_lifting.py`. 오토인코더 명령의 `--config_path ./training/KUBM/config_autoencoder.yaml`은 실제로 `training/Flow_training/`에 있음 | 낮음, 그대로 실행하면 경로 오류 |
| D4 | 논문에 없는 학습 설정 | 표 7은 6개 값만 | Adam, 배치 256, **10,000 에폭**, 기울기 클리핑 최대 노름 1.0, 기본 시드 42, 손실은 `MSELoss`(평균 감소)를 15 스텝에 **합산**, 체크포인트 "Best" = 에폭 평균 **학습 손실** 최저 | 중간. 재현 시 필요한 값이 논문에 없음. 평균 감소 손실이라 클리핑 임계 1.0의 의미가 손실 크기에 묶임 |
| D5 | 안정화 요령의 구현 | 부록 J: (i) 항등 초기화, (ii) 클리핑 + 낮은 K 학습률(그림 11 캡션은 "separate learning rates") | `--no_lr_adjustment` **하나가** 클리핑을 끄고 K 학습률을 인코더 학습률로 맞춤. `--no_identity_init`은 별도 | 낮음. 그림 11의 "Identity init only", "Neither" 조건은 클리핑도 꺼진 조건 |
| D6 | 손실 구조 | L = Σ_l ‖K^l z_t − z_{t+l}‖² (식, §3.2) | 타깃 z_{t+l} = [x, ψ(x)]를 **같은 인코더로, stop-gradient 없이** 계산해 손실에 넣음. 앞 d_ξ 블록(원 상태)이 인코더 출력의 값을 고정하는 구조 | 중간. 타깃 쪽 ψ도 학습되므로 ψ가 상수에 가깝게 무너질 여지를 앞 블록(원 상태) 항이 막는 구조입니다. 실제로 무너지는지는 검증하지 않았고, 논문은 이 선택을 설명하지 않음 |
| D7 | 학습 구간과 롤아웃 길이 | 다단계 15(실로봇 50). 평가는 "전체 기술 구간" 롤아웃(§2) | 학습 샘플은 모든 시작 t에서 15 스텝까지(`traj_len − 15`개). 평가 러너는 **최대 100 스텝** 개루프 롤아웃(`maximum_steps = 100`) | 중간. 15 스텝으로 학습한 K^l을 100 스텝까지 외삽. 실로봇 H = 50 설정은 저장소에 없음 |
| D8 | 성공률 분모 | 테스트 30개(§4.1) | `성공 수 / (시연 수 − detection_failure)`. SAM이 객체를 못 찾은 시연은 분모에서 제외하고 그 수를 따로 출력 | 중간. 03 V2, V3의 비정수, 29회 분모와 맞는 구조. 논문에 서술 없음 |
| D9 | 롤아웃 상한과 시간 계측 | 부록 C "specified time duration" | DexArt 러너가 100 스텝에서 멈춤(환경 horizon은 250, `j >= 200` 분기는 도달 불가). 정책 시간 = (초기화 + K 스텝 합) / 성공 스텝 j | 낮음–중간. 03 V12 참고 |
| D10 | 재계획 | 부록 K: 점프가 이전 점프 평균의 **3배**를 넘으면 트리거, "절대 임계 없음"(부록 I: 과제 불변 임계) | `last_jump_is_large_simple(..., factor=3.0)`이 기본 정의이나 호출은 DexArt **5.0**, Adroit **30.0**. DexArt 스크립트는 직후 `replan_trigger = False`로 **재계획을 끔**, `reactive_time_step = 1000000`. 재계획은 기록된 쌍 초기 흐름과 **원래 로봇 초기 상태**로 새 롤아웃을 만들고 j 스텝 전진. Adroit 스크립트만 30 스텝 보간이 있고 `reactive_time_step = 15`(주석 "orig: 32") | 높음. 표 9와 부록 K의 서술이 공개 코드와 같은 설정이 아님. 공개 스크립트만으로는 표 9를 만들 수 없음 |
| D11 | 체크포인트 선택 | 5시드 평균 ± SD(그림 3) | 과제당 **시드 하나**(bucket 15, laptop 25, toilet 25, door 15, hammer 20, pen 10, relocate 25)가 들어 있음. 선택 기준은 "학습 절제 폴더가 있는 시드", 시뮬 평가로 순위를 다시 매기지 않았다고 적음. 부록 J는 "best-performing random seed"로 절제 | 중간. 공개 체크포인트로는 평균 ± SD를 재현할 수 없고, 절제의 시드 선택 방식이 성능 기준이었는지는 코드에서 확인되지 않음 |
| D12 | 오토인코더 학습 | 압축 오토인코더(부록 B)만 서술 | lr 5e-3, 3000 에폭, 배치 1024, weight decay 1e-2, γ = 0.993, 좌표 정규화 없음. 학습 데이터는 훈련 궤적의 **70%**(`train_test_split(test_size=0.3)`), 나머지는 검증과 테스트 | 낮음. K-UBM은 전체 훈련 시연으로 학습하므로 오토인코더와 K-UBM의 훈련 범위가 다름 |
| D13 | 상태와 행동 | a_t = q_target 관절 위치 목표(§2) | DexArt 28차원 = 관절 위치 22(팔 6 + 손 16) + **팔 속도 6**. 손 속도 목표는 0으로 고정 | 낮음. 부록 C.1은 위치와 속도라고 적음 |
| D14 | 평가 인스턴스 | "unseen test sets"(그림 3) | 러너는 `--test_set`(Test_set 초기 관측)과 `--use_test_set`(unseen 객체 인스턴스)를 따로 받음. README 예시 명령은 `--test_set`만 적고 후자를 적지 않음. 환경은 `--use_test_set`이 없으면 seen 인스턴스 목록으로 생성 | 중간. 어느 쪽으로 보고했는지는 데이터 없이 확인 불가 |
| D15 | 공개되지 않은 부분 | 표 3–6, 9, 10, 그림 3, 5, 13, 부록 L | 기준선, 실로봇, jerk, 센서 열화, 집계 코드, 데이터 없음 | 높음. 논문 수치의 대부분을 코드로 대조할 수 없음 |
| D16 | 실행 환경 | | SAPIEN 2.2.1, 레거시 `mujoco_py` + Gym 0.17.2, 외부 메시 자산, SAM3 가중치(Hugging Face 접근), CoTracker 체크포인트 | 높음. 이 맥에서 시뮬 평가는 어려움(§4) |

## 3. 대조 요약

- K-UBM의 핵심(리프팅 MLP, 상태 포함 z, 선형 K, 15 스텝 손실, 항등 초기화)은 논문과 코드가 일치합니다(D3). 코드가 논문에 없는 설정을 많이 가집니다(D4, D7, D12).
- 평가 보고 방식은 코드로 확인되는 부분(성공률 분모 D8, 롤아웃 상한 D9, 반응성 임계 D10)이 논문 서술과 다릅니다. 각 차이가 보고된 수치에 얼마나 영향을 줬는지는 데이터 없이 판단할 수 없습니다.
- 가장 많은 표(3–6, 9, 10)를 만든 코드가 저장소에 없습니다(D15).

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**전체 재현은 불가능합니다. 방법의 핵심은 작은 규모로 구현할 수 있고, 일부는 이미 토이로 확인했습니다.**

| 대상 | 이 맥 | 필요한 것 |
|---|---|---|
| 실로봇(표 10, 그림 5, 6) | 불가 | Franka + LEAP 로봇, 키네스테틱 교시 데이터, 저장소 밖 코드 |
| 시뮬 평가(표 3–6, 9의 K-UBM 열) | 사실상 불가 | SAPIEN 2.2.1 렌더링(GPU), 레거시 mujoco_py + Gym 0.17.2, 외부 자산, 데이터 |
| 기준선 재학습(Diffusion, ACT, UVA) | 불가 | 저장소에 없음. UVA는 다중 GPU 규모 |
| K-UBM 학습(사전 추출된 흐름 특징 + 행동 궤적) | 조건부 가능 | 데이터 다운로드 승인(크기 미확인). 모델은 약 29–43만 파라미터, 배치 256, 15스텝 손실이라 CPU/MPS에서 가능해 보이나 10,000 에폭 시간은 측정하지 않음 |
| 흐름 오토인코더 학습 | 조건부 가능 | 같은 데이터. 입력이 2×16×16이라 작음 |
| K-UBM 롤아웃 시간(토이) | **가능, 완료** | `analysis/toy_rollout_timing.py`(수 초) |
| K-UBM 알고리즘 자체 구현(합성 동역학 + 시각 특징 대신 저차원 상태) | 가능 | 1–2일(추정, 측정 전). 항등 초기화, 별도 학습률, 클리핑, 100 스텝 외삽의 안정성을 볼 수 있음 |
| 공식 체크포인트 추론의 시뮬 성공률 | 불가 | 위 시뮬 환경 |

- 노력 추정: 작은 구현은 1–2일, 공식 데이터로 K-UBM만 학습하고 개루프 예측 오차(시뮬 없이)를 재는 데 3–5일(추정, 측정 전), 시뮬 평가는 Linux GPU 환경이 있어야 하므로 이 맥 밖의 작업입니다.
- 재현은 하지 않았습니다.
