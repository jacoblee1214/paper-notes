# 04. 공식 코드와 논문 대조

공식 저장소 `github.com/RLinf/LaWAM`을 커밋 **`c12168d9077edea67fe7c1b70d08b80b8e65a9c8`** (2026-09-17, "docs: add project news", main)로 얕게 복제해 **정적으로만** 읽었습니다. 실행, import, 설치는 하지 않았습니다. 체크포인트, 데이터는 받지 않았고 설정 YAML 네 개(각 2 KB 이하)와 파일 크기 메타데이터만 읽었습니다. 복제본과 HF 설정은 신뢰할 수 없는 외부 자료로 취급했습니다.

대조 번호는 일치 K1–K15, 차이·누락 D1–D17입니다. 코드 참조는 복제본 기준 `파일:줄`입니다.

## 1. 저장소와 라이선스

| 항목 | 내용 |
|---|---|
| 코드 규모 | Python 96개 파일, 4.1 MB. `starVLA/`(모델, 데이터로더, 학습), `latent_action_model/`(LAM), `deployment/`, `examples/LIBERO`, `examples/Robotwin` |
| 라이선스 | **LICENSE 파일 없음.** GitHub API 라이선스는 null. `pyproject.toml`은 `license = {text = "MIT"}`, README 끝은 "StarVLA의 MIT 라이선스를 유지"라고 적음 (D17). 기반 StarVLA 저장소의 API 라이선스는 NOASSERTION |
| 체크포인트, 데이터 | HF 메타데이터는 모두 MIT. 크기: LAM 2.90 GB, 정책 7.17 GB (사전학습, LIBERO, RoboTwin 각각), LIBERO 데이터 1.97 GB, RoboTwin 데이터 78.99 GB |
| 의존성 | `requirements.txt`: torch 2.6.0, transformers 5.2.0, diffusers 0.36.0, lightning 2.4.0, flash-attn 2.8.3 (별도 설치), LIBERO는 mujoco 3.3.2 고정 (README) |
| 학습 진입점 | `train_lawam.sh`, `train_lawam_distributed.sh`, 설정 `starVLA/config/training/train_libero.yaml`, `train_robotwin.yaml` (사후학습 전용), LAM 학습 `latent_action_model/main.py`, `config/dino_large_vae.yaml` |
| 최근 이력 | 커밋 15개, 2026-06-14~09-17. "security: remove exposed W&B API key" 커밋이 2026-08-29에 있음(내용은 보지 않음) |

## 2. 논문과 코드의 일치 (K1–K15)

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| K1 | 2단계 구조, LaWM = LAM 순방향 디코더 | 3.2–3.3절 | `lawam.py`가 `self.lam`(동결, 디코더만 학습)을 불러 `lam.decoder(h_t, pred_action_emb)`로 û_T 생성 (`lawam.py:267`, `:504`) | 일치 |
| K2 | 단일 순전파, 비반복 | 3.1절 식 3 | 추론에서 디코더 1회 호출 후 행동 전문가만 10스텝 (`lawam.py:821`) | 일치 |
| K3 | KI (û_T에 stop-gradient) | 3.3절 | 후학습 설정 `detach_future_feature: true` → `cond_future.detach()` (`lawam.py:572`, `train_libero.yaml:28`). **코드 기본값은 false** (`lawam.py:70`) | 설정에서 일치 |
| K4 | λ_distill = λ_wm = 0.1 | E.6 | `lam_encoder_distill_weight: 0.1`, `perceptual_weight: 0.1` (`train_libero.yaml:31`, 코드 기본값 1.0 `lawam.py:60`) | 설정에서 일치 |
| K5 | 손실 구성 L_act + 0.1 L_distill + 0.1 L_wm | 식 5 | `loss_flow + 0.1·loss_perceptual + 0.1·loss_distill` (`lawam.py:777-781`) | 일치 |
| K6 | Qwen3-VL-2B 첫 16층, 은닉 1,024, 행동 전문가 16층 | E.3 | `keep_llm_first_n_layers: 16`, `hidden_dim: 1024`, `num_layers: 16`, `attention_heads: 16`, `vlm_dim: 2048` (`train_libero.yaml:15-16, 38-43`) | 일치 |
| K7 | Alternate-DiT: 자기 어텐션과 교차 어텐션 교대, VLM과 이미지 번갈아 | 그림 2, E.3 | 홀수 층 자기 어텐션, 짝수 층 교차 어텐션, `idx % 4 == 0`은 VLM, 나머지 이미지 (`cross_attention_dit.py:402-417`). 16층 = 4 × (자기, 의미, 자기, 역동역학) | 일치 |
| K8 | 10 디노이징 스텝, CFG 없음 | E.6 | `num_inference_steps: 10`, `cfg_guidance_scale: 1.0`, `cfg_drop_prob: 0.0` → 단일 패스 (`flowmatching_expert.py:655`) | 일치 |
| K9 | 고유감각 입력 없음, RGB 256 × 256 | E.3 | `use_state: false`, `image_resolution: 256` | 일치 |
| K10 | LIBERO 후학습 25k 스텝, 전역 배치 256, RoboTwin 100k 스텝, 1,024 | F.2, F.3 | LIBERO `max_train_steps: 25000`, 32 × 8 GPU = 256. RoboTwin `100000`, 기기당 16 × 누적 2 × GPU 32개 = 1,024 (README는 GPU 수에 맞춰 조정하라고만 안내) | 일치 |
| K11 | 물리 시간 정렬과 사인 부호화 | E.4 식 6–7 | `ContinuousTimeEncoder`, ν_k = exp(−log P · k/(K−1)), P = 10,000 (`flowmatching_expert.py:22-40`), `t_i = i / hz` (`:68-79`) | 일치 (D8의 floor와 round 차이 제외) |
| K12 | 로봇 데이터에만 행동 손실 | 4절, E.6 | `valid = valid * robot_valid` (embodiment id ≠ 0) (`flowmatching_expert.py:564`) | 일치 |
| K13 | DINOv3 ViT-B/16 동결, 연속 32차원 잠재 행동, 단일 질의 | E.2, E.6 | HF `dino_large_vae.yaml`: `vision_model_id: facebook/dinov3-vitb16-pretrain-lvd1689m`, `vq_type: vae`, `code_dim: 32`, `num_queries: 1` | 일치 |
| K14 | 인코더가 두 관측 토큰을 한 시퀀스로 처리 | E.2 | `QFormer_att`가 현재·지평선 토큰과 질의 1개를 이어 24층 `TransformerEncoderLayer` 통과, `enc_modal_mask: true`는 프레임 토큰 사이 전체 어텐션을 허용 (`modules.py:731-740`, `:67-82`) | 일치 |
| K15 | 디코더가 z를 adaLN으로 주입, 게이트 영초기화 | E.2 | `AdaLNBlock` (`lam_decoder.py:16-78`) | 일치 |

## 3. 논문과 코드의 차이, 누락 (D1–D17)

### 3.1 LaWM(1단계) 설정

| 번호 | 항목 | 논문 | 코드·공개 설정 | 영향 |
|---|---|---|---|---|
| D1 | 로봇 영상 지평선 | 1.2 s (E.6) | `frame_dt_sec: 1.6`, 사람 영상 `0.4` (HF `dino_large_vae.yaml`, 저장소 YAML도 같음). 정책 사전학습 설정은 `sec_chunk: 1.2` (HF `pretrain config.yaml`) | LaWM이 학습한 지평선(1.6 s)과 정책 단계 청크(1.2 s)가 다름. 논문은 "고정 물리 지평선 τ"를 강조 |
| D2 | 손실과 KL | MSE ‖ũ_T − u_T‖², β = 10⁻⁵ (식 4, E.6) | `loss_type: smooth_l1` (β=0.1, `lam_lightinng.py:643-644`), KL 계수는 `VAEQuantizer`의 기본값 `beta: 5e-5` (`vq.py:27`)이고 YAML에서 덮어쓰지 않음 (`vq_kwargs`는 `layer_norm: true`뿐) | 두 값이 모두 논문과 다름. 공개 LAM 체크포인트가 이 설정으로 학습되었다면 논문 서술과 다른 모델 |
| D3 | 학습 설정 | 16 H100, 100k 스텝, lr 3e-4, 배치 1,024 (E.6) | 저장소 YAML: lr 3e-4, 10 에폭, `max_state_dim: 32`. **HF 공개 YAML: lr 1e-4, 40 에폭, `max_state_dim: 14`.** 두 파일이 서로 다름. 배치는 기기당 64 | 어느 쪽이 공개 체크포인트의 설정인지 불명. 논문 수치와도 일치하지 않음 |
| D4 | 디코더 깊이 | 인코더와 디코더 모두 24층 (E.2) | `enc_layers: 24`, **`dec_layers: 12`** (두 YAML). 12층이면 파라미터 228.3M ≈ 논문의 230M, 공개 체크포인트 크기(2.90 GB)와도 맞음 (`03` §3) | 논문 서술 오류 가능성. 코드 기준이 230M과 일치 |
| D5 | 보조 상태 예측 | g(s, z) → s_T (식 4) | `StatePredictor`가 **상태 변화량**을 `tanh`로 예측하고 (`lam_decoder.py:214`), 손실은 `eef_reconstruction_loss(state_delta=...)` (`lam_lightinng.py:697`). 로봇 샘플(embodiment ≠ 0)에만 적용, 가중치 `lambda_aux: 1.0` | 보조 대상이 s_T가 아니라 Δs. 논문 식과 형태가 다름 |
| D6 | 특징 층과 정규화 | "고정 인코더 특징" (3.1절) | `latent_layer_to_use: -2`(끝에서 두 번째 층), `norm_latents: true, type ln` (특징에 층 정규화), 디코더 출력에도 `last_ln` | 부분 목표 공간이 정규화된 중간 층 특징. 논문에는 없음 (재사용 시 맞춰야 함) |

### 3.2 정책(2단계)과 행동 전문가

| 번호 | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D7 | 증류 목표와 손실 | z ~ q (사후분포 표본), ‖ẑ − z‖² (식 5) | 교사는 `quantized`로 사후분포의 **평균 μ**를 사용 (`vq.py:103`, `lawam.py:460`), 손실은 원소 평균 `F.mse_loss` (`lawam.py:475`). 같은 방식으로 `L_wm^(2)`도 원소 평균 (`lawam.py:729`) | 훈련 때 샘플 z, 증류 때 평균 μ. 합 대신 평균이라 λ = 0.1의 실질 크기가 논문 식과 다름 |
| D8 | 지평선과 청크 길이 | 청크 길이 T = round(τ h) (식 6), 지평선 관측은 시간 τ 뒤 | `chunk_len = int(sec_chunk * fps)`, 토큰 수 `floor(horizon_sec * hz)` (`lerobot_datasets.py:200`, `flowmatching_expert.py:72`). 미래 프레임은 **마지막 행동 인덱스** (`lerobot_datasets.py:70`), 즉 (chunk_len − 1)/fps 뒤. **LIBERO는 `sec_chunk: 0.4` = 20 Hz에서 8 행동**, RoboTwin은 1.2 s | LIBERO 청크는 LaWM 사전학습 지평선(1.2–1.6 s)보다 짧음. round 대신 floor. 지연 187 ms는 8 행동분 (`03` §2) |
| D9 | LIBERO 평가 최대 스텝 | 미기재 (F.2는 OpenVLA 설정을 따라 실패 시연을 제거한다고만 적고 평가 스텝 한도는 없음) | `libero_spatial 250, object 300, goal 320, libero_10 550` (`libero_benchmark_adapters.py:147-157`). OpenVLA-OFT 평가 코드의 `TASK_MAX_STEPS`는 220/280/300/520 (OpenVLA-OFT 코드 대조에서 확인) | 스위트별 +30/+20/+20/+30 스텝의 여유. 성공률이 시간 제한에 민감한 정도는 모르며 논문은 이 값을 적지 않음 |
| D10 | 학습률 | 사전학습: 행동 전문가 1e-4, 나머지 3e-5. 후학습: 1e-4 코사인 (E.6) | LIBERO 후학습: 세 그룹 모두 1e-4 (`train_libero.yaml:78-86`). **RoboTwin: LaWM 디코더만 3e-4** (`train_robotwin.yaml:78`). 사전학습 학습률은 공개 설정에 없음 (HF `config.yaml`에도 없음) | 논문의 "후학습 lr 1e-4" 서술은 RoboTwin LaWM에는 맞지 않음 |
| D11 | 논문에 없는 학습 세부 | 미기재 | `repeated_diffusion_steps: 2` (한 배치를 2번 반복해 플로 손실 계산, `lawam.py:741`), 행동 전문가 `dropout=0.2` (DiT 기본값, `cross_attention_dit.py:232`), 가중치 감쇠 1e-8, 그래디언트 클리핑 1.0, 워밍업 1,500/2,000 스텝, 시간 샘플링 Beta(1.5, 1.0) × 0.999, 마지막 LLM 층과 임베딩 동결, 행동 차원 32 패딩, 질의 토큰 8 + 8 | 재현에 필요한 값들이 논문 본문에 없음 |


### 3.3 재현 경로와 평가

| 번호 | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D12 | 부분 목표 주입 방식 | 그림 2의 "Inverse Dynamics Attn" | 키는 `cat(h_t, û_T, VLM)`이고 위치·유형 임베딩 없음 (`flowmatching_expert.py:469`, `:640`). 이미지 교차 어텐션은 16층 중 4층 (`cross_attention_dit.py:413`) | 구조 서술은 맞으나 구분 신호가 없음 (`03` T1) |
| D13 | 사전학습 데이터 혼합 | 로봇 3,000 h, 사람 1,500 h (4절, E.6) | LAM 설정 `data_mix: lam_plus_human`과 정책 사전학습 설정 `data_mix: lam`이 **`mixtures.py`에 정의되어 있지 않음.** 이슈 7에서 저자는 혼합 비율을 공개하지 않겠다고 답함. 이슈 4에서 정리 후 공개 예정이라 답했으나 2026-11-22 현재 그 공개는 확인하지 못함 | 1단계와 2단계 사전학습은 이 저장소로 재현 불가 |
| D14 | 절제 변형 | 표 4, 6 | `future_prediction: true`는 항상 LAM 디코더를 쓰고 (`lawam.py:650-657`: `future_prediction: false`이면 û_T 자리에 현재 특징 h_t가 들어감), 직접 미래 예측, 용량 일치 어댑터, 부분 목표 조건 제거 변형을 만드는 설정이 없음. 이슈 11에서 저자가 "공개 브랜치에 별도 구현이 없고 통합하기 어렵다"고 답함 | 표 4, 6의 변형 재현 불가 |
| D15 | RoboTwin 평가 기본값 | 과제당 100회 (F.3) | `auto_eval_robotwin.sh`의 `ROBOTWIN_TEST_NUM` 기본값 **14** (줄 18), README의 "Full RoboTwin benchmark" 명령은 이를 덮어쓰지 않음. 배치 러너 내부 기본값은 100 (`batched_eval_runner.py:459`). `ROBOTWIN_REPLAN_STEPS` 기본 36, 평가 지시문 `instruction_type: seen` | README 명령 그대로면 100회가 아니라 14회로 평가됨. 재계산 시 `ROBOTWIN_TEST_NUM=100`을 명시해야 함. 논문은 `replan`, 지시문 종류를 적지 않음 |
| D16 | 설정과 체크포인트의 불일치 | — | 공개 RoboTwin 체크포인트의 `config.yaml`은 `data_mix: robotwin_eef_30hz` (`robotwin_eef_all_v30_merged_slow30fps`)인데 저장소 `train_robotwin.yaml`은 `robotwin_merged`. 사전학습 `config.yaml`의 `framework.name`은 `LatentWorldVLAIndependent`(등록명은 `LaWAM`). README의 SFT 설정 파일명은 이슈 3에서 저자가 낡은 이름이라고 인정 | 공개 체크포인트가 어떤 데이터·설정에서 나왔는지 코드만으로는 확정 불가 |
| D17 | 실로봇과 라이선스 | Franka 두 과제, Quanta X1 두 과제 (4.3절) | 실로봇 서버(`server_policy_tcp_fold_towel.py`, `server_policy_http_franka.py`)와 데이터 혼합 `real_pick_place`, `real_open_drawer`, `real_fold_towel`은 있으나 실로봇 학습 설정, 데이터, 체크포인트, Clean Table 관련 설정은 없음. LICENSE 파일 없음(README는 MIT 유지라고 주장) | 표 3, 8 재현 불가. 법적 사용 조건은 파일로 확인할 수 없음 |

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**전체 재현은 불가능하고, 구성 요소 수준의 축소 구현과 정적 검증은 가능합니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 표 수치 재계산, 검정, 일관성 검사 | 가능, 완료 | `analysis/recompute.py`, 수 초 (`03`) |
| 구조 토이 (키 순서 불변성) | 가능, 완료 | `analysis/toy_key_order_invariance.py`, 1초 미만 |
| 공식 코드로 사전학습 (Stage-1 LaWM 400 H100-시간, Stage-2 3,000 H100-시간) | **불가** | 1,024 배치, 16/64 H100, 데이터 혼합 정의 없음(D13), 로봇·사람 영상 수 TB급 |
| 공식 코드로 후학습 (LIBERO 40 H100-시간, RoboTwin 1,200 H100-시간) | **불가** | flash-attn, CUDA 전제 (README), 7.17 GB 체크포인트, RoboTwin 데이터 79 GB. MPS에서 2.6B급 학습은 현실적이지 않음 |
| 공식 체크포인트 추론과 시뮬 평가 | **조건부, 사실상 어려움** | 가중치 7.17 GB + LAM 2.90 GB + Qwen3-VL-2B + DINOv3 다운로드 승인 필요, flash-attn 대체와 MPS 포팅, LIBERO·RoboTwin 시뮬(RoboTwin은 curobo, sapien 소스 빌드가 필요하다고 이슈에 보고) 설치. 노력 1–3주(추정), 속도 미측정 |
| 부분 목표 인터페이스 축소 재구현 (작은 DINO 특징 토이 + 소형 행동 전문가, torch) | 가능 | 수 M 파라미터, 합성 또는 소규모 시뮬 데이터. 노력 3–7일(추정) |
| LAM 디코더 구조(adaLN 블록, 12층)만 가져와 작은 규모로 LIBERO 한 스위트에서 비교 | 조건부 | LIBERO 설치와 데이터(1.97 GB) 다운로드 승인. 학습은 소형 모델로만 |
| 실로봇 (Franka, Quanta X1) | **불가** | 로봇, 시연 150–302개, 설정 모두 저장소 밖 (D17) |

핵심 한계: 2단계 사전학습이 +0.6 pp(`03` §6)이고 사전학습 코드가 없으므로, 재현 가치가 있는 부분은 "공개 LIBERO·RoboTwin 체크포인트로 평가를 재현"하는 쪽이고 이는 이 맥 밖(CUDA GPU)에서 해야 합니다. 이 맥에서의 실용적 가치는 인터페이스 설계의 축소 실험입니다.

## 5. 제3자 재현 보고 (저장소 이슈, 미검증)

GitHub 이슈는 제3자의 글이므로 **사실로 취급하지 않고** 내용만 요약합니다. 이 폴더에서 어떤 것도 재현하지 않았습니다.

| 이슈 | 날짜 | 내용 | 결과 (이슈 안) |
|---|---|---|---|
| #2 | 2026-06~07 | 공개 RoboTwin 체크포인트로 청정 50과제 × 100회 평가에서 **72.3%** (3,616/5,000), 논문 92.64%와 약 20 pp 차이. 관절·정밀 과제에서 낮음 | 저자는 환경(curobo, scipy, toppra 등 버전) 불일치를 의심. 최신 커밋 `49f76a6`으로 정렬한 뒤 보고자가 "보고값에 가깝게 재현"했다고 후속 글 작성 (수치는 이 폴더에서 확인하지 못함) |
| #6 | 2026-07 | LIBERO 재현 평균 97.1% (Spatial 94.6, Object 99.4, Goal 98.6, Long 95.8) 대 논문 98.6% | 저자는 MuJoCo 버전 영향을 지목. mujoco 3.3.2로 맞춘 뒤 Long 95.2%, Spatial은 크게 개선되었다고 보고자가 후속 작성 |
| #7 | 2026-08 | LAM 1단계 데이터 혼합 비율 요청 | 저자: 비율은 체계적 연구를 하지 않아 공개하지 않음 |
| #4 | 2026-07 | 전체 사전학습 파이프라인 공개 계획 문의 | 저자: LeRobot 3.0 변환과 혼합 정의 코드를 정리해 공개 예정 |
| #11 | 2026-11 | 절제 재현 세부 요청 | 저자: 직접 미래 예측 구현은 공개 브랜치에 없음. 통합이 어려워 전체 절제 코드는 제공 불가 |
| #12 | 2026-11 | 행동 전문가의 교차 어텐션 대 연결 비교 | 저자: 효율성과 표현 격차를 이유로 교차 어텐션 선택. 비교 결과는 제시하지 않음(읽은 범위). 후속으로 RoboTwin 청정 50개만으로 학습하면 청정 약 78%, 무작위 약 0%라는 제3자 보고와 색 증강·낮은 학습률 권고가 오감 |
| #10 | 2026-10 | RoboTwin 리더보드 C2R 프로토콜 평가 | 보고자: C2C 72.64%, C2R 한 자릿수. 저자: 해당 설정은 미평가 |

이 이슈들이 시사하는 점: (1) 보고 수치는 시뮬레이터 버전에 민감하다 (LIBERO, RoboTwin 모두), (2) 강한 외형 변화(무작위 장면)에서의 일반화는 제한적일 수 있다는 제3자 관찰 하나, (3) 공개 코드로 절제와 사전학습은 확인할 수 없다.
