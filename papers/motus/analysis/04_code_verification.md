# Motus: 논문과 공식 코드 대조 (04)

- 대상 저장소: `github.com/thu-ml/Motus`, 커밋 `88f6fe3` (2026-10-10, PR #49 병합), Apache-2.0. 얕은 복제는 `repo/`에 두었고, **정적으로 읽기만 했습니다** (실행, import, 설치, 가중치 다운로드 없음). 복제물은 신뢰하지 않는 데이터로 취급했습니다.
- 대조 기준: 논문 본문과 Supp. §7–9 (`paper/paper.txt`), 표 11–13.
- 저장소 파일 경로는 `repo/` 기준입니다.

## 1. 요약

- 논문의 **학습 골격과 행동 전문가, 모델 크기, 손실 가중치**는 코드와 맞습니다 (일치 항목 목록 §2).
- 논문 핵심 기여 중 **잠재 행동을 만드는 파이프라인 (광학 흐름 → DC-AE → 14차원 VAE, 식 (2))은 코드에 없습니다.** 학습 데이터 로더는 이미 계산된 `latent_action_dim14/*.pt`를 읽습니다 (`data/latent_action/latent_action_dataset.py` 36, 138, 153행).
- 논문 §3과 Supp. §7이 설명하는 **다섯 가지 추론 모드(WM, IDM, VGM, VLA, 결합)를 고르는 API가 없습니다.** 공식 추론은 영상과 행동을 함께 복원하는 결합 모드 하나입니다.
- 학습 시간 분포는 논문 (`U(0,T_τ)`)과 설정 파일 (`logit_normal`) 어느 쪽도 아니고, `FlowMatchScheduler(shift=5.0)`의 이산 균등 샘플입니다. 설정 파일의 `logit_normal`은 학습 경로에서 호출되지 않습니다.
- 토큰 균형 주장(§4.1)은 공식 설정 숫자로 맞지 않습니다.

## 2. 일치 항목

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 세 전문가 (VGM Wan 2.2 5B, VLM Qwen3-VL-2B, 행동 전문가, 이해 전문가) | §4.1, Table 11 | `models/motus.py`, `models/action_expert.py`, `models/und_expert.py`; 행동 hidden 1024, 이해 hidden 512 (`configs/robotwin.yaml`), 30층 (`models/motus.py` 57행, `action_expert.py` 62행), 24헤드 (`und_expert.py` 142행 기본값) | 일치 |
| Tri-model Joint Attention | §4.1 | `process_joint_attention` (models/motus.py 212행, 학습은 `training_step` 안의 층 반복 루프) | 일치 |
| 손실 = 영상 MSE + 행동 MSE (가중치 1, 1) | Supp. §7.1, Alg. 1 | `total_loss = video_loss_weight*video + action_loss_weight*action`, 가중치 1.0 (`configs/robotwin.yaml` `loss_weights`) | 일치 |
| 속도 목표 ε − x₀ (rectified flow) | Alg. 1 | `video_target = video_noise − clean`, `action_target = action_noise − actions` | 일치 |
| 첫 프레임 조건 (Teacher Forcing) | Alg. 2–6의 o_t | 첫 잠재 프레임 고정, 영상 손실에서 첫 프레임 0 마스크 | 일치 (마스크는 논문에 없음) |
| 추론 단계 10 | Table 11 "Inference steps 10" | `num_inference_timesteps: 10` (설정) | 일치 |
| VLM 동결 (Stage 2) | Table 1 | `vlm: frozen: true` | 일치 |
| 행동 시퀀스 길이 (실세계) | Table 11 "8 @ 5Hz, 48 @ 30Hz" | `num_video_frames 8` × `video_action_freq_ratio 6` = 48 (`configs/ac_one.yaml`, `aloha_agilex_2.yaml`) | 일치 |
| RoboTwin 행동 청크 16 | Supp. §9.2 (IDM 설명) | `video_action_freq_ratio: 2` → 8 × 2 = 16 (`configs/robotwin.yaml`) | 일치 (아래 D6 참고) |
| 행동 차원 14 | §4.2, Table 11 | `action_dim: 14`, `state_dim: 14` | 일치 |
| 데이터: RoboTwin 청정 2500 + 무작위 25000 | §5.2 | `configs/latent_action.yaml`의 데이터 경로에 `robotwin2_copy/clean`, `randomized` 포함 | 일치 (데이터 수는 경로로 확인 불가) |
| 파라미터 합 8B | Table 11, README | 5.00 + 2.13 + 0.64 + 0.25B | 일치 |

## 3. 차이 (D1–D10)

| ID | 논문 | 코드 (위치) | 중요도 | 영향 |
|---|---|---|---|---|
| **D1** | §4.2, 식 (2): DPFlow 광학 흐름 → RGB → DC-AE 기반 VAE (512×4 → 14차원), 손실 `L_recon + λ_a‖a_real−a_pred‖² + βL_KL` (λ_a=1, β=1e-6) | **없음.** 흐름 계산, VAE 학습, 정렬 손실 코드 없음. 로더는 사전 계산된 `latent_action_dim14/<stem>.pt`를 읽고, 분위수 통계 `latent_action_q01_q99.json`으로 정규화 (`latent_action_dataset.py` 108–109, 138, 153, 232행). 설정 `configs/latent_action.yaml`은 `dataset.type: latent_action` | **핵심** | 잠재 행동 라벨을 재현하려면 논문 기여의 절반인 VAE를 새로 짜야 함. 가중치 `latent_action_*` 다운로드 여부는 README에 없음 |
| **D2** | §3, §4.1, Supp. §7 Alg. 2–6: WM, IDM, VGM, VLA, 결합 모드 각각의 깨끗한/노이즈 타임스텝 조합 | 추론 함수는 같은 클래스 안에 `inference_step`이 **세 번** 정의됨 (`models/motus.py` 911, 1037, 1230행). Python에서는 마지막 정의(1230행, FlowUniPC 두 스케줄러로 영상·행동 동시 복원)만 유효. 모드 선택 인자 없음. 모드별 조건부 샘플링 코드 없음 | 높음 | 논문의 "다섯 모드 전환"은 코드만으로는 재현할 수 없음. 같은 모델에 τ를 다르게 주면 가능하겠지만 공식 구현은 아님 |
| **D3** | Supp. §7.1: τ_o, τ_a를 U(0,T_τ)에서 독립 샘플 | 학습: `FlowMatchScheduler(shift=5.0, num_train_timesteps=1000)`의 **이산 인덱스** `randint` (models/motus.py 648–665, 학습 루프). `shift=5`가 시간을 고잡음 쪽으로 기울임 | 중간 | 분포가 논문 설명과 다름. 논문의 "U(0,T)"는 shift 전 기준으로 읽어야 함 |
| **D4** | 학습 시간 분포 설정 항목 | `configs/*.yaml`의 `timestep_sample_method: logit_normal`, `sigmoid_scale: 1.0`. `get_t_distribution`은 `models/motus.py` 18행에서 import만 되고 호출되지 않음 (`utils/common.py` 119행 정의) | 중간 | 설정 파일에 적힌 분포는 학습에 쓰이지 않음. 설정을 바꿔도 결과가 달라지지 않는 "죽은 설정" |
| **D5** | §4.1: 영상 토큰과 행동 토큰 수를 "균형"으로 맞춤 (Action-Dense Video-Sparse) | 계산 (아래 §5): 영상 토큰 ≈ 360 (잠재 3프레임 × 120패치, VAE 16×16 공간·4 시간, 패치 2×2), 행동 토큰 ≈ 53 (실세계 48+1+4 레지스터), RoboTwin ≈ 21 (16+1+4). | 중간 | 비율 약 7:1 (실세계), 17:1 (RoboTwin). 논문 문장과 다름. 주의 집중이 영상 쪽으로 치우친 상태가 남음 |
| **D6** | §4.1: 예시로 "영상 프레임 속도 = 행동의 1/6" | `robotwin.yaml`은 비율 2 (영상 8, 행동 16), `latent_action.yaml`은 비율 1, 실세계 설정은 6 | 낮음 | 논문 예시는 실세계 설정에만 맞음. RoboTwin 결과는 다른 비율로 학습됨 |
| **D7** | Table 13: Stage 1 학습률 8e-5, Stage 2 5e-5, Stage 3 1–5e-5 | 설정 파일 학습률 5e-5 (`training.learning_rate`) 하나. 단계별 설정 파일은 없음 | 낮음 | 단계별 학습률 재현 불가 |
| **D8** | RoboTwin 평가: 과제당 100회 (§5.2) | `inference/robotwin/Motus/deploy_policy.yml`: `task_config: null`, `seed: null`, `instruction_type: unseen`. `eval.sh`와 `auto_eval.sh`에서 시행 수를 정하는 인자를 찾지 못함 (grep `num`, `episode`, `trial` 결과 없음) | 중간 | 공식 설정으로는 논문 평가 조건을 그대로 재현할 수 없음 (빈 값을 채워야 함) |
| **D9** | §5.2: 청정·무작위 모두 40k 스텝 미세조정 | 설정 `training.max_steps: 1000000` (robotwin.yaml). 40k 스텝 설정 흔적 없음 | 낮음 | 미세조정 길이는 실행 명령에서 정해야 함. 설정만으로는 확인 불가 |
| **D10** | 표 11의 "VGM 5.00B" 등 | 설정은 파라미터 수를 적지 않음. 합계는 모델 생성 코드 `log_parameter_counts`로 계산 가능 | 낮음 | 확인은 CPU에서 모델을 만들어야 해서 하지 않음 (설치·실행 금지 조건) |

## 4. 구현 가능성 판정 (이 맥 기준)

**판정: 논문 결과의 재현은 불가능하고, 구조를 축소해 재구현하는 것은 가능합니다.**

- 맥 환경: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 사용 가능, CUDA 없음. 저장소 README 요구 사양은 추론 24 GB 이상 VRAM, 학습 80 GB 이상.
- **불가능한 것 (GPU·데이터 필요):** 8B 모델 학습 (GPU 약 1.8만 시간, Table 13), 공식 설정의 RoboTwin 평가 (Wan 2.2 5B 가중치 약 10 GB (bf16 기준 추정), 시뮬레이터, 시행 100회 × 50과제), 실로봇 재현. 공식 가중치와 데이터는 받지 않았습니다.
- **축소 재현 가능 (CPU·MPS, 몇 주):**
  1. 공식 흐름 과정의 **토이** (2차원, 이미 수행, `03` §6).
  2. 설계 요소의 소형 재구현: Tri-model 주의 (작은 차원, 두 전문가), 독립 타임스텝 학습, 모드 전환 샘플링. 데이터는 합성 궤적. 노력 **1–2주** (추정).
  3. D5 토큰 비율 계산과 D3 시간 분포 비교 (CPU, 몇 시간).
  4. 잠재 행동 VAE 자체 구현 (광학 흐름은 소형 합성 영상으로). 노력 **2–3주** (추정), 공식 DPFlow와 DC-AE 가중치가 필요하면 승인 필요.
- **보류 (승인 필요):** 공식 체크포인트 (`motus-robotics/Motus*`, HF) 다운로드와 MPS 추론 지연 측정. 공식 코드 실행. 이 항목은 `CLAUDE.md` 규칙에 따라 사용자 확인 후 진행.

## 5. 토큰 계산 (D5)

- 입력: 실세계 설정 `num_video_frames=8`, `video_height=384`, `video_width=320`.
- 잠재 프레임 수 = 1 + ⌊8/4⌋ = 3 (`models/motus.py` 947행).
- 공간 압축: `vae_stride = (4, 16, 16)` (`bak/wan/configs/wan_ti2v_5B.py` 17행). 384/16 = 24, 320/16 = 20.
- 패치 `(1, 2, 2)` (같은 파일 20행) → 12 × 10 = 120 토큰/잠재 프레임.
- 영상 토큰 = 3 × 120 = **360**.
- 행동 토큰 = 1 (상태) + 48 (청크) + 4 (레지스터) = 53 (실세계). RoboTwin: 1 + 16 + 4 = 21.
- 비율 360:53 ≈ 6.8 (실세계), 360:21 ≈ 17 (RoboTwin). 논문 §4.1의 "균형"과 다릅니다.
- 주의: 이 계산은 설정과 코드 상수에서 한 것입니다. 실제 텐서 형태는 실행으로 확인하지 않았습니다.

## 6. 기타 관찰

- `models/motus.py` 클래스가 1355행짜리 파일이고 `test_motus()`가 파일 안에 있습니다. 추론 코드 중복은 유지보수 문제입니다 (D2와 같은 원인).
- `bak/` 디렉터리에 옛 코드가 남아 있습니다 (`bak/wan/`, `bak/device_utils.py`). 실행 경로에 쓰이지 않는 것으로 보이지만 확인은 정적 검색만 했습니다.
- 논문 v2 (2025-12-25)와 현재 커밋 (2026-10)의 대응은 확인하지 못했습니다. 이 대조는 커밋 `88f6fe3` 기준입니다.
