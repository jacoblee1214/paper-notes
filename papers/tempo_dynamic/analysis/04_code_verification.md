# 04. 공식 코드 대조와 구현 가능성

대상: `tempo-robot/TEMPO` main `dd8342a1234e7166b856ffec2d7917b51cce05e4` (2026-10-14, Apache-2.0), `tempo-robot/TempoBench` main `7d1d1d4ff7dc3c6ab17590187e108fe663676346` (2026-08-07). 두 저장소를 얕게 clone해서 **정적으로만 읽었습니다.** 코드를 실행, import, 설치하지 않았고, 가중치와 데이터(영상)는 받지 않았습니다. 데이터 폴더의 parquet만 읽기 전용으로 통계를 냈습니다(`03` T2).

## 1. 저장소 구성

| 경로 | 내용 |
|---|---|
| `src/openpi/models_pytorch/pi0_pytorch.py` (726줄) | TEMPO의 모델. `SAM2CrossAttnFusion`, `ActionHistoryCondMLP`, `ActionHistoryTokens`, `embed_prefix`, `embed_suffix`, 흐름 매칭 손실 |
| `src/openpi/models_pytorch/transformers_replace/.../modeling_siglip.py` | SigLIP의 시공간 어텐션(`_temporal_attention`). 설치 때 `transformers` 안에 **덮어써야** 모델이 만들어집니다(README) |
| `src/openpi/models/pi0_config.py`, `training/config.py` | `Pi0Config`의 TEMPO 필드, YAM 학습 구성 |
| `src/openpi/policies/yam_dual_policy.py` | 이력과 SAM2 토큰 로더, 어안 보정 |
| `tools/sam2_encoder.py`, `tools/precompute_sam2_tokens.py` | SAM 2.1-tiny 메모리 어텐션 잠재의 사전 계산 |
| `tools/ut/*`, `models_pytorch/ut_decoder.py` | UVT용 MVAE와 디코더 |
| 없는 것 | VLASH/RTC 비동기 실행, 로봇 배포 루프, 평가·성공 판정·질량 측정, 프로브 코드, 학습 데이터, 체크포인트 |

README의 할 일 목록: 학습 데이터 공개, 체크포인트 공개, I2RT YAM 배포가 모두 미완료입니다.

## 2. 논문 ↔ 코드 차이 (D)

일치하는 부분을 먼저 적고, 차이 번호는 아래 표에 있습니다.

**일치:** 흐름 매칭 손실이 MSE(`F.mse_loss(u_t, v_t)`)이고 시간 샘플이 Beta(1.5, 1)×0.999 + 0.001인 것(π0 표준). 디노이징 10단계(`num_steps=10`, 그림 3의 ×10). ACT 버킷이 K = 10, L = 30(`action_history_seconds` 10 × 30 fps ÷ 10), 시작 이전은 0으로 채우고 버킷 전체가 시작 이전이면 pad 플래그(`LoadActionHistory`, 식 2와 같음). ACT가 접두 토큰과 AdaRMS 잔차 두 경로로 들어가는 것(§4.3). 유효 배치 128.

| 번호 | 항목 | 논문 | 공개 코드 | 비고 |
|---|---|---|---|---|
| D1 | 백본·비동기 | TEMPO는 VLASH의 비동기 파이프라인 위에 얹음(§4.3, 표 4 "TEMPO")이고 RTC는 논문에서 재구현(부록 B.3) | `pi0_pytorch.py`는 π0.5(openpi)의 동기 추론 구조이고 VLASH/RTC 코드가 없음. `tools/ut/encode_ut_windows.py` 주석이 "vlash_uvt dataset"을 언급하나 저장소에 없음 | 표 4, 5의 TEMPO 행은 공개 코드만으로 구성할 수 없습니다 |
| D2 | TEMPO_MOT의 구조 | 현재 프레임 특징을 질의로 직전 W−1 프레임 특징 캐시에 교차 어텐션(식 1). m_t를 VLM 토큰 차원으로 투영해 **접두에 덧붙임**(§4.1, §4.3, 그림 3) | SAM2 메모리 어텐션 출력을 8×8로 풀링한 64토큰(256차원)을 만들어 **현재 head 카메라 SigLIP 토큰에 영초기화 게이트 교차 어텐션으로 더함**(`SAM2CrossAttnFusion`). 토큰 수는 늘지 않음. 손목 카메라에는 적용되지 않음(`cam_idx == 0`만) | 입력을 어디에 어떻게 넣는지가 논문 서술과 다릅니다 |
| D3 | 백본 무수정 | "백본 수정 없음"(초록), "largely intact"(§4.3) | 별도 TEMPO_MOT 경로로 SigLIP에 **인과 시간 어텐션**을 4번째 층마다 추가(`temporal_attn_period=4`), 이미지 3프레임(0.333초 간격, `obs_history=3`)을 모든 카메라에 대해 인코딩. 새 파라미터는 없고 기존 q/k/v/out을 재사용하지만 `transformers` 파일을 덮어써야 함 | 논문 식 1에 없는 구성 요소입니다 |
| D4 | 창 길이 W | 논문에 값이 없음 | 코드의 시간 맥락은 SigLIP 3프레임(약 0.67초)과 SAM2 메모리(`sam2_encoder.py`는 에피소드 전체를 순서대로 스트리밍하며, 캐시 길이는 저장소 밖의 SAM2 설정에 달림). 모듈 설명은 "16-frame windows"라고 쓰고 함수는 전체 T프레임을 받음 | W의 값을 이 코드에서 확정할 수 없습니다 |
| D5 | SAM2 입력 | 카메라 224p(부록 B.1), 프레임당 약 11 ms(§4.4) | head 프레임을 **1024×1024로 리사이즈**하고(`preprocess_video_to_sam2_input`), 매 스텝 전 영역 +10 로짓 가짜 마스크로 메모리를 갱신(객체 위치 가정 없음) | 11 ms의 입력 해상도 조건이 논문에 없습니다 |
| D6 | 배포 경로 | 움직임 인코딩이 별도 스레드에서 센서 속도로 실행(§4.4) | 학습은 에피소드 전체를 **오프라인에서 사전 계산**(`precompute_sam2_tokens.py`)한 토큰을 읽고, 서버 API는 클라이언트가 `sam2_tokens`(64, 256)와 `action_history`(10, A)를 계산해 보내도록 되어 있음. 스트리밍 스레드와 클라이언트 쪽 계산 코드는 없음 | 학습과 배포의 토큰 생성 일치는 코드만으로 확인할 수 없습니다 |
| D7 | ACT의 입력 | 고유수용 "명령" q_τ | `action` 열(시연 명령). 학습은 시연 행동 이력, 배포는 클라이언트가 만든 이력 | 폐루프에서 이력이 정책 자신의 출력으로 바뀌는 효과는 학습에서 다루지 않습니다(`03` T1은 이 차이를 포함해 시험) |
| D8 | 영초기화 | 추가 경로는 영초기화되어 시작 시 사전학습 동작이 보존됨(§4.3) | adaRMS 잔차(출력층 영초기화), SAM2 융합(게이트와 출력 영초기화)은 해당. 그러나 **접두 토큰 10개는 영초기화가 아닙니다**: 투영층은 기본 초기화이고 위치 임베딩만 0입니다. 구성 파일 주석이 "접두 토큰은 no-op이 아니라 step 0 != warm-start 체크포인트"라고 직접 적음 | 논문 문장이 코드의 일부 경로에만 해당합니다 |
| D9 | 추가 파라미터 | 2M (0.08%)(§1, §4.3) | `03` V19의 산술: ACT 접두+조건 MLP 약 2.29M, SAM2 융합 블록 약 17.45M, 합 약 19.7M(25억 기준 0.79%). 시간 어텐션은 새 파라미터가 없음 | ACT만 2M과 같은 규모입니다 |
| D10 | 예측 대상 | TEMPO 행은 행동 청크, TEMPO+UVT 행은 UVT 목표 | `Pi0Config.predict_ut`의 **기본값이 True**(UVT 잠재 u_t 예측)이고, `pi05_yam_*`·`tempo_*` 구성만 명시적으로 `predict_ut=False`를 줌. UVT 학습 명령은 `--beta 0`, `--latent-dim 64`(스크립트 기본값은 β = 1e-3, 잠재 256) | 구성을 새로 만들 때 플래그를 빠뜨리면 `ut_target`이 필요한 경로로 갑니다 |
| D11 | 학습 설정 | 유효 배치 128, 약 2시간, 반복당 0.46–0.50 s(부록 B.3). 스텝 수, 학습률은 없음 | AdamW, 코사인 감쇠 피크 5e-5(워밍업 500, 5,000스텝, 최저 5e-6), EMA 0.999, 기울기 클립 1.0. 기준 구성 `pi05_yam_*`는 10,000스텝, TEMPO 구성은 5,000스텝 + warm start. 10,000 + 5,000 = 15,000스텝 × 0.50 s = 2.08 h로 논문의 약 2시간과 산술이 맞음 | 학습 스텝과 학습률이 논문에는 없고 코드가 유일한 출처입니다 |
| D12 | 학습 사슬(warm start) | 서술 없음 | `pi05_yam_tempo_mot_*`의 `pytorch_weight_path`가 **같은 구성 이름의 step 10,000 체크포인트**(`trained_ckpt(..., step=10_000)`)이고 이 구성은 5,000스텝만 학습합니다. 주석은 "비디오 경로만 있는 체크포인트에서 시작"이라 하나 그 구성(`obs_history=3`, SAM2 융합 없음)이 저장소에 없음(`pi05_yam_*`는 `obs_history=1`). `train_pytorch.py`는 새 모듈(`sam2_fusion.`, `action_history_`)이 warm start에서 비어 있는 것을 예상된 경우로 처리 | 저장소의 구성만으로는 학습 사슬의 첫 고리를 만들 수 없습니다 |
| D13 | 데이터와 과제 구성 | 4과제, 시연 수 미기재 | 데이터 이름은 `dynamic_handover_v2_all`, `dynamic_pour_trimmed`, `spartan_trimmed`, `ball_drop`, `ball_catch_all`. TEMPO 구성이 있는 것은 handover(14차원, 카메라 3대)와 `ball_catch_all`(7차원, head만)뿐이고, pour, spartan, ball_drop은 기준 구성만 있음. 공개 TempoBench의 이름은 `dynamic_pour`, `spartan`(절단 표기 없음)이고 handover는 90에피소드·18,536프레임(약 10.3분) | 표 4의 Drop, Flick, Wine Pour 행의 TEMPO 구성은 공개되지 않았습니다. `ball_catch_all`은 논문의 4과제에 없음 |
| D14 | 로봇 | YAM 팔 두 대, 각 "7자유도"(부록 B.1) | 데이터 설명이 팔당 "6 관절 + 그리퍼 1"(14차원 = 7 + 7)이라고 적음 | 7에 그리퍼를 포함한 표기일 가능성이 큽니다 |
| D15 | 프로브(표 7)와 TEMPO-Bench 평가 | 속도 R² MLP 프로브, 객관식 변형(§6) | TempoBench에 로더, 시각화, `verify.py`가 있고, 프로브 학습 코드, 분할, 객관식 질문 파일은 없음 | 표 7과 객관식 평가는 공개 자료로 재현할 수 없습니다 |
| D16 | 제목 | arXiv 제목은 "TEMPO: Learning Temporal Context for Dynamic Robot Manipulation" | README는 논문을 "Closing the Representational Gap for VLAs in Dynamic Settings"라고 부름 | 이전 제목일 가능성이 있고 BibTeX는 arXiv 제목을 씁니다 |
| D17 | 추론 안내 | 같음 | README 추론 예시는 `images`를 단일 프레임(HWC)으로 보냄. 구성은 `obs_history=3`이고 모델은 `num_frames > 1`일 때만 시간 어텐션을 적용 | 3프레임 이력 전달 방법이 문서에 없고, 단일 프레임을 주면 시간 경로가 꺼집니다 |
| D18 | 추론 비용 | 각 프레임 특징을 한 번만 계산해 재사용(§4.1), 지연 +1.5 ms(표 3) | 시간 경로는 매 스텝 카메라×3프레임을 SigLIP에 다시 통과시킴(캐시 없음). 이 경로를 켠 구성의 지연은 논문에 없음 | 표 3의 구성이 공개 구성인지 알 수 없습니다 |
| D19 | 라이선스 | - | TEMPO: Apache-2.0 + Gemma 약관. TempoBench `LICENSE`는 "All rights reserved ... internal research use ... Replace this file ... before any external distribution"이고 README가 "currently a placeholder"라 적음. GitHub API는 NOASSERTION | 데이터 사용 조건이 확정되어 있지 않습니다 |

## 3. 저자에게 물을 것

1. 표 4의 TEMPO 행은 SAM2 융합과 SigLIP 시간 어텐션을 둘 다 쓴 모델입니까, 식 1의 교차 어텐션만 쓴 모델입니까.
2. W의 값, SAM2 입력 해상도, 11 ms가 측정된 구성은 무엇입니까.
3. 표 6의 "MOT만"과 "ACT만"에서 SigLIP 시간 어텐션은 켜져 있었습니까.
4. 미절단 Flick Catch의 MOT만, ACT만 결과가 있습니까.
5. 프로브의 학습·평가 분할 단위(프레임, 에피소드)와 시드, 후보 인코더의 입력은 무엇입니까.
6. 학습 스텝 수, 학습률, 시연 수는 공개 구성과 같습니까.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 수치의 재현은 불가능하고, 구조의 축소판과 공개 데이터 통계는 이미 이 맥에서 끝냈습니다.**

| 일 | 판정 | 근거 |
|---|---|---|
| 실로봇 평가(표 1, 4, 5, 6) | **불가** | I2RT YAM 쌍팔, 카메라 3대, 사람 시험자가 필요하고 배포 코드가 공개되지 않았습니다 |
| 공식 코드로 학습 | **불가** | README가 "NVIDIA GPU"를 요구하고, 8×RTX PRO 6000에서 약 2시간 학습입니다. 학습 데이터와 π0.5 가중치(약 수 GB), SAM2 체크포인트가 필요합니다. 학습 데이터는 미공개이고, warm start 사슬의 첫 고리도 없습니다(D12) |
| 공식 코드로 추론(CPU/MPS) | **조건부(사실상 불가)** | 새 가상환경(uv sync), `transformers` 덮어쓰기, π0.5 변환과 체크포인트가 필요한데 TEMPO 체크포인트가 공개되지 않았습니다 |
| TempoBench 데이터 읽기와 통계 | **가능, 일부 완료** | parquet(전체 약 15 MB)은 clone에 들어 있고, 영상 514개(약 750 MB)는 LFS입니다. `03` T2는 parquet만 썼습니다 |
| 식 2와 별칭 메커니즘의 축소판(토이) | **가능, 완료** | `analysis/toy_aliasing.py`(약 20초, torch CPU) |
| 두 모듈의 소형 재구현(영초기화 교차 어텐션 + 버킷 이력)을 MuJoCo 추종 과제에서 시험 | **가능, 미착수** | mujoco 4.0과 torch 2.15가 설치되어 있어 새 설치가 필요 없습니다. 처리량은 측정 전입니다 |
| TempoBench 프로브(표 7 유사) | **조건부** | 영상 750 MB(LFS)와 영상 인코더 가중치가 필요합니다. 객체 속도 라벨은 parquet에 있어 라벨 쪽은 가능합니다 |

- **노력(추정, 측정 전):** 토이는 반나절(완료). MuJoCo 축소판은 2–3일. TempoBench 영상 특징으로 프로브를 다시 해 보는 일은 가중치 승인 이후 2–4일. 공식 학습 재현은 GPU와 데이터가 있어도 환경 1–2일 + 데이터 수집이 병목입니다.
