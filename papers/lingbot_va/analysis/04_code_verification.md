# LingBot-VA 논문 ↔ 공식 코드 대조 (04)

## 0. 대조 범위

- 저장소: `github.com/robbyant/lingbot-va`, 커밋 `7c6ffa9` (2026-07-10), 얕은 복제본 `repo/` (gitignore). Apache-2.0.
- 방법: 정적 읽기만 했습니다. 실행, import, 설치, 가중치 다운로드는 하지 않았습니다. 코드와 README는 신뢰하지 않는 데이터로 다뤘습니다.
- 읽은 파일: `wan_va/train.py`, `wan_va/modules/model.py`, `wan_va/wan_va_server.py`, `wan_va/utils/scheduler.py`, `wan_va/utils/utils.py`, `wan_va/utils/sever_utils.py`, `wan_va/dataset/lerobot_latent_dataset.py`, `wan_va/configs/*.py`, `evaluation/robotwin/eval_polict_client_openpi.py`, `evaluation/libero/client.py`과 두 launch 스크립트, `README.md`.
- 아직 읽지 않은 것: `wan_va/modules/model.py`의 FlexAttn 상세 구현, `wan_va/utils/Simple_Remote_Infer/*`(원격 추론 래퍼), `evaluation/robotwin/geometry.py`. 필요하면 나중에 확인합니다.
- 조사 파일의 줄 번호는 위 커밋 기준입니다.

## 1. 논문과 코드가 일치하는 것

| 항목 | 논문 | 코드 | 위치 |
|---|---|---|---|
| 행동 차원 30 | §4.1 | `action_dim = 30` | `configs/va_robotwin_cfg.py:20` |
| 분위수 정규화 | §4.2 | `action_norm_method = 'quantiles'`, q01/q99 → [-1,1] | `configs/va_*_cfg.py`, `dataset/..py:279` |
| 잡음 증강 확률 0.5, s_aug ∈ [0.5,1] | 식 (10) | `noisy_cond_prob=0.5`, `min_timestep_bd=0.5`, `max=1.0` | `train.py:192–199, 229` |
| 교사 강요 + 인과 마스크 | §3.3 | block-causal 마스크: clean→clean 인과, noisy→clean 엄격 과거, noisy→noisy 같은 chunk | `modules/model.py:154–200` |
| 손실은 두 항의 합 (λ=1) | 식 (11)–(12), §4.2 | `loss = latent_loss + action_loss` | `train.py:311` |
| 영상 CFG 5.0, 행동 CFG 1.0 | §4.2 | `guidance_scale = 5`, `action_guidance_scale = 1` | `configs/va_robotwin_cfg.py:22–23`, `server.py` |
| 텍스트 드롭 0.1 | §4.2 | `cfg_prob = 0.1` | `configs/va_robotwin_train_cfg.py:11` |
| 기울기 노름 자르기 2.0 | §4.2 | `clip_grad_norm_(…, 2.0)` | `train.py:319` |
| 백본 Wan2.2-5B, 영상 폭 3072, 30층 | §4.2 | `inner_dim = 24×128 = 3072`, `num_layers=30` | `modules/model.py:601–608` |
| RoboTwin 50K 스텝, lr 1e-5 | §4.3.2 | `num_steps = 50000`, `learning_rate = 1e-5` | `configs/va_robotwin_train_cfg.py` |
| KV 캐시에 실제 관측 갱신 | §3.4, Alg. 1 | 추론 후 예측 캐시 비우고 실제 관측을 다시 캐시 | `wan_va_server.py` `_compute_kv_cache` |
| 행동 채널 가림 | §4.1 | `used_action_channel_ids`, 사용하지 않는 채널 0 | `configs/va_*_cfg.py`, `wan_va_server.py:224–252, 563` |

## 2. 차이 (D1–D20)

표의 "영향"은 논문 주장에 미치는 정도입니다. 높음은 핵심 주장과 직접 관련된 차이입니다.

| ID | 논문 | 코드 | 위치 | 영향 |
|---|---|---|---|---|
| D1 | 행동 30차원 = 양팔 EEF 7 + 관절 7 + 그리퍼 1 (§4.1) | 배포 설정에서 실제로 쓰는 채널은 RoboTwin 16개(EEF 두 팔 + 그리퍼 두 개), LIBERO 7개(EEF+그리퍼). 관절 채널 14–27은 0 | `configs/va_robotwin_cfg.py:33–36`, `va_libero_cfg.py` | 중간. "관절 각도 포함 통합 표현"은 배포 설정에서 성립하지 않음 |
| D2 | 행동은 세계 좌표 절대 자세 (§3.2 "absolute pose") | RoboTwin 학습 때 EEF를 세그먼트 첫 프레임 기준 상대 자세로 변환 (`get_relative_pose`). 추론 클라이언트는 초기 자세를 다시 더함 (`add_init_pose`) | `dataset/lerobot_latent_dataset.py:55–68, 261–262`, `eval_polict_client_openpi.py:439–442` | 높음. 논문에 없는 핵심 표현 선택이며 Table 1 결과의 일부일 수 있음 |
| D3 | 두 스트림 MoT: 영상 폭 3072, 행동 폭 768, 별도 QKV, 행동 스트림 추가 파라미터 약 350M, 총 5.3B (§3.3, §4.2) | 행동 토큰도 같은 블록(`self.blocks`, 폭 3072, FFN 14336)을 공유. 차이는 입력 선형층 `action_embedder: 30→3072`, 시간 임베딩 `condition_embedder_action`, 출력 `action_proj_out`뿐 | `modules/model.py:598–649, 680–682` | **높음**. 공개 모델은 "shared backbone" 버전이고 분리 버전은 "stay tuned" (README 2026-01-29). 논문 아키텍처는 공개되지 않음 |
| D4 | 행동 스트림 초기화: 영상 가중치를 보간하고 스케일 α (§3.3, 그림 7) | 초기화 코드 없음. 체크포인트 생성 방법 미공개 | 저장소 전체 검색 (`interpolate`는 추론 중 이미지 보간뿐) | 중간. 그림 7의 안정성 주장은 재현 불가 |
| D5 | 학습 chunk K ∈ [1,8] (§3.3), 반면 §4.2는 [1,4] | `torch.randint(1, 5)` → {1,2,3,4} | `train.py:245` | 낮음 (본문 내부 불일치). 코드는 §4.2와 맞음 |
| D6 | 배포 K=4 (§3.3) | RoboTwin `frame_chunk_size = 2`, LIBERO 4 | `configs/va_robotwin_cfg.py:12`, `va_libero_cfg.py:13` | 낮음. RoboTwin 설정은 논문과 다름 |
| D7 | 영상 3 Euler 단계 → s=0.6 (§4.2), Alg. 1은 s=0.5 (잡음 증강 효과 "절반") | 기본 `num_inference_steps = 25` (RoboTwin) / 20 (LIBERO), `video_exec_step = -1` → 적분 끝까지. 부분 적분은 `video_exec_step`을 줄여야 함 | `configs/va_robotwin_cfg.py:26–27`, `wan_va_server.py:464–478` | **높음**. 논문의 속도 주장(절반 단계)이 기본 설정에서 적용되지 않음 |
| D8 | 행동 10 단계 (§4.2) | `action_num_inference_steps = 50` | `configs/va_robotwin_cfg.py:28` | 중간. 지연 추정과 직접 연결 |
| D9 | 시간 샘플링 "uniform SNR sampler" (§4.2), 식 (11)–(12)는 가중치 없는 MSE | 인덱스 균등 샘플 후 이동 스케줄(`snr_shift 5.0`) 사용. 손실에 종 모양 가중치 `training_weight` 곱셈. LIBERO 행동 이동은 0.05 | `train.py:138–141, 267–272`, `scheduler.py:57–73`, `configs/va_libero_cfg.py:30–31` | 중간. 논문 식과 다른 손실 |
| D10 | 행동 값 범위 제한 없음 (식 12) | 학습 때 `np.clip(…, -1.5, 1.5)` | `dataset/lerobot_latent_dataset.py:281` | 낮음 |
| D11 | FDM 손실 식 (13), 비동기 Alg. 2 (FDM 기반 재정렬) (§3.4) | FDM 손실 없음 (검색어 `fdm`, `dynamics` 없음). 공개 클라이언트는 동기 방식: 추론 → 청크 전체 실행 → KV 갱신 | `eval_polict_client_openpi.py:561–608`, `evaluation/libero/client.py:95–125` | **높음**. 비동기 속도 주장(§4.4, 2배)과 Table 3 FDM 행을 재현할 수 없음 |
| D12 | 학습 시퀀스는 한 열의 교사 강요 시퀀스 (Fig. 3) | 두 복제 스트림: 노이즈 토큰과 깨끗한 조건 토큰을 같이 넣고 마스크로 구분 | `modules/model.py:703–765` | 낮음 (구현 방식 차이, 동등하다고 볼 수 있음) |
| D13 | LIBERO 4K 스텝, 4개 스위트 × 3시드 × 500회 (§4.2–4.3.2) | `num_steps = 5000`. 평가 스크립트는 `libero_10`(Long)만, `--test-num 50`(과제당), 체크포인트도 Long만 공개 | `configs/va_libero_train_cfg.py:25`, `libero/launch_client.sh` | 중간. Spatial, Object, Goal 수치는 재현 자료 없음 |
| D14 | RoboTwin 시험 횟수 미기재 | `--test_num 100` (과제당) | `robotwin/launch_client.sh:37` | 낮음 (논문 누락) |
| D15 | 실세계 사후학습 두 설정 (§4.2: 3K 스텝, lr 1e-5 / §4.3.1: 500 스텝, lr 1e-4) | 실세계 설정 파일 없음 | - | 낮음 (두 설정이 본문에 함께 있음) |
| D16 | 토큰 수 N = 192 (프레임당, §4.2) | RoboTwin(3 카메라, T자 배치): 프레임당 120 (`latent_h=24, latent_w=20`, 패치 2×2). LIBERO(2 카메라): 프레임당 32 | `wan_va_server.py:385–390`, `configs/va_*` | 중간. 토큰 수와 계산량 주장과 맞지 않음 |
| D17 | 영상 τ=4 솎기, 프레임마다 τ개 행동 (§3.3) | 행동 수 = 잠재 프레임당 `frame_stride × 4` (RoboTwin 16, LIBERO 4). 영상은 12.5Hz로 줄인 뒤 VAE 시간 압축 4 적용 | `dataset/lerobot_latent_dataset.py:262–266`, `configs/va_*_cfg.py` `action_per_frame` | 낮음 (해석 차이. 정의 명시 필요) |
| D18 | 영상 가중치 `dv` 3072, `da` 768 (§3.3) | 행동 폭 768 설정 없음 (D3 참조) | `configs/`, `model.py` | D3과 같음 |
| D19 | 잡음 증강은 "확률 p" 샘플마다 (식 10) | 배치 전체에 한 번 뽑는 `torch.rand(1)` 동전 던지기 | `train.py:192` | 낮음 (배치 크기가 1보다 크면 샘플 간 상관) |
| D20 | 사전학습 1.4T 토큰, 16K시간, 배치와 lr 1e-4 (§4.2) | 사전학습 코드 없음. 공개 체크포인트 `lingbot-va-base`는 사후학습 시작점 | README, `train.py`는 사후학습 중심 | 중간. 사전학습 주장은 확인 불가 |

## 3. 대조 결과 요약

- **일치 (§1):** 행동 차원, 정규화, 증강 확률과 범위, 인과 마스크, 손실 합, CFG 값, 기울기 자르기, 백본 치수, RoboTwin 학습 스텝과 lr, KV 캐시 갱신.
- **가장 큰 차이 (D3, D7, D11, D2):** (a) 논문의 분리된 MoT가 공개 모델에 없고, (b) 기본 추론은 부분 적분을 쓰지 않으며, (c) 비동기 파이프라인과 FDM 손실이 없고, (d) RoboTwin 행동은 상대 자세입니다.
- 이 차이들은 논문의 속도 주장(§4.4)과 아키텍처 설명을 공개 코드로 확인하지 못하게 만듭니다. 성능 주장(표 1, 2)은 체크포인트 없이는 확인할 수 없습니다.

## 4. 구현 가능성 판정

**판정: 재현 불가 (정식 구성). 부분 구현은 가능 (축소판과 토이).**

이 맥(Apple M4 Pro, 24 GiB 통합 메모리, MPS, CUDA 없음) 기준입니다.

| 작업 | 필요 자원 | 이 맥 | 판정 |
|---|---|---|---|
| 사후학습 (RoboTwin, LIBERO) | 8 GPU, FSDP, flex-attention, 데이터 수백 GB | CUDA 필요, 메모리 부족 | 불가 |
| 사전학습 (1.4T 토큰) | 수백 GPU, 수 TB 데이터 | 불가 | 불가 |
| 공식 체크포인트 추론 (RoboTwin/LIBERO 정책 서버) | 약 18–24 GB VRAM (README, 오프로드 시) + 시뮬레이터 | 메모리 빠듯. 코드가 `torch.device("cuda:…")`, `torch.cuda.empty_cache()`, `flex` 어텐션 전제. MPS 포팅 필요. 체크포인트 다운로드(수십 GB) 승인 필요 | 조건부 (포팅 1–2주 추정, 승인 필요) |
| 시뮬레이터 평가 (RoboTwin 2.0) | Vulkan, pytorch3d, sapien 설치 | 맥에서 Vulkan 실행 어려움 | 불가 (사실상) |
| LIBERO 평가 (Long만 공개) | LIBERO 설치, 서버 GPU | 서버 쪽 CUDA 필요 | 불가 (서버 측) |
| 논문 표 1, 2 재현 | D13 (스위트 누락), GPU, 시뮬레이터 | - | 불가 |
| 축소판 (작은 블록 인과 자기회귀 + 잡음 증강, 합성 데이터) | numpy/torch CPU 또는 MPS, 수 M 파라미터 | 가능 | 가능 (1–2주 추정) |
| 토이 (잡음 증강의 정밀도 교환) | numpy | 가능 | 완료 (`03` §7) |

노력 추정은 개인 작업 기준입니다. 확실하지 않은 숫자는 추정이라고 표시했습니다.

## 5. 출처

- 공식 저장소: `github.com/robbyant/lingbot-va`, 커밋 `7c6ffa9` (2026-07-10). `LICENSE.txt` Apache-2.0.
- 줄 번호는 위 커밋의 `wan_va/`, `evaluation/`, `README.md` 기준입니다.
