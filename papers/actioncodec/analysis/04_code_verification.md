# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `ZibinDong/actioncodec` (논문 저자 계정의 공식 코드, README가 논문 arXiv 번호를 인용). 커밋 `f034eb9a9da6d6f37b9522b4548779e8a9886b21`(2026-03-01, 저장소의 유일한 커밋, 메시지 "Initial commit: ActionCodec - Neural Codec for Robot Action Sequences"). 얕은 복제(깊이 1)를 `repo/`에 두었고 파일 33개, 약 11,000줄입니다. 재현 명령: `git clone https://github.com/ZibinDong/actioncodec repo && git -C repo checkout f034eb9a9da6d6f37b9522b4548779e8a9886b21`.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 가중치와 데이터는 받지 않았고, Hugging Face의 모델 메타데이터(크기, 해시)와 작은 `config.json`만 확인했습니다.
- 읽은 파일: `README.md`, `config/train/{ar,pd,ki,bar}.yaml`, `config/eval/{pd,bar}.yaml`, `actioncodec/{configuration,modeling,modular}_actioncodec.py`, `actioncodec/rvq.py`(구조), `scripts/{train_vla,eval_libero,utils}.py`, `scripts/eval_libero_{pd,bar,ar}.sh`, `utils/vla_tokenizer.py`(구조), `smolvla/{bar,ki}.py`(손실과 위치 부여), `smolvla/pd.py`(구조), HF의 `ActionCodec-Base`, `-RVQft` 구성과 `SmolVLM2-2.2B-ActionCodec-BAR-LIBERO/config.json`.
- 판정: **일치**는 논문과 코드가 같음, **차이**는 다름이나 논문에 없는 설정, **미확인**은 저장소에서 찾지 못함입니다.

## 1. 설치 요구사항, 라이선스, 범위

- **의존성.** `requirements.txt`: torch ≥ 2.0, numpy < 2, transformers ≥ 4.40, einops, `vector-quantize-pytorch==1.27.15`, lightning, peft, omegaconf, mujoco 3.2.3, robosuite 1.4.1, `libero==0.1.0`, `libero2gym==0.1.0`, gym 0.22.0. `environment.yaml`은 Python 3.10, CUDA 11.8입니다. 학습 설정은 `flash_attention_2`와 `accelerator="cuda"`, 4장의 GPU(`devices: [0, 1, 2, 3]`)를 전제합니다.
- **라이선스.** README는 "released under the MIT License"라고 적지만 저장소에 LICENSE 파일이 없고 GitHub API의 license 값은 null입니다. Hugging Face 모델 카드에도 라이선스 표기가 없습니다(카드는 "Model Card for Model ID", Model Sources "TODO"). 재사용 전에 저자에게 확인이 필요합니다.
- **범위.** 저장소는 (1) 토크나이저 모델 정의와 추론(`actioncodec/`), (2) VLA 변형 4개(AR, PD, KI, BAR)의 학습 래퍼와 모델(`smolvla/`, `scripts/train_vla.py`), (3) LIBERO 평가(`scripts/eval_libero*.py`)로 이루어집니다. **토크나이저 학습 코드, TCL/CLIP 손실, OR 측정, 섭동 실험, 기준선(Binning, String, FAST, VQ-VLA, MiniVLA) 구성, SimplerEnv, SO100, xArm 코드는 저장소에 없습니다**(D1).
- **공개 가중치(메타데이터만 확인).** 토크나이저 2개(ActionCodec-Base 180.4 MB, -RVQft 197.3 MB)와 LIBERO 정책 3개(AR 4.7 GB, PD 5.5 GB, BAR 5.6 GB)입니다. KI, 256M, 500M, Qwen/InternVL 변형, SimplerEnv, 실로봇, 15K시간 모델의 가중치는 없습니다.

## 2. 논문 ↔ 코드 대조 (D)

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | 공개 범위 | 설계 원리(OR, 용량, TCL/CLIP, 잔여 문법)를 학습된 토크나이저들로 검증하고 ActionCodec을 학습(§3, §4, 부록 B.2, B.3) | 토크나이저 학습 스크립트, 대조 손실(식 5, 6, InfoNCE), OR/PPL 평가, 섭동 실험, 기준선 토크나이저 코드가 없음. 모델 클래스의 `forward`와 `_quantize`(PPL 계산 포함)만 있음 | **차이(범위)** | 논문의 핵심 실험(그림 3 ~ 6, 표 5)은 코드로 재현할 수 없음 |
| D2 | VQ 학습 방식 | 식 (1): 재구성 + 코드북 + 약속 손실 | `vector_quantize_pytorch.VectorQuantize`(EMA 코드북, 감쇠 0.99, kmeans 초기화, 죽은 코드 임계 2, 약속 0.25, `straight_through=True`, 회전 트릭 끔) | **차이(구현 방식)** | EMA 코드북은 식 (1)의 코드북 항을 기울기로 학습하지 않음. 표준 구현 차이이지만 논문에 명시 없음 |
| D3 | 토크나이저 크기 | 미기재(부록 B.2, B.3은 Perceiver 구조만) | 클래스 기본값(`encoder_dim` 256, 6층, 8헤드)과 공개 가중치의 구성(dim 384, 12층, 6헤드, z_dim 512, 토큰 16, 코드북 2048)이 다름. 구 클래스(`ActionCodecConfigOld`)는 dim 384, 12층, 4헤드. 공개 Base는 약 45.1M 파라미터(구성에서 센 값과 일치, 03 V13) | **차이(기본값 대 공개 값)** | 논문에 크기가 없어 코드 구성이 유일한 출처. 기본값으로 만들면 공개 가중치와 호환되지 않음 |
| D4 | 소프트 프롬프트와 시각 인코딩 | 로봇별 학습 임베딩과 제어 주파수에서 계산한 푸리에 시각 임베딩(부록 C, 그림 15) | `EmbodimentEmbedding`(로봇 수 × (dim × 출력 길이) 임베딩이 곧 질의), 인코더 key/value와 디코더 질의에 주파수 기반 푸리에 위치 인코딩 | **일치** | — |
| D5 | 사전학습 로봇의 시간 창 | "1초 창"(부록 B.3), Bridge 시퀀스도 1초(그림 9 캡션) | 공개 가중치의 `embodiment_config`: LIBERO 20 Hz 1 s, **Bridge 5 Hz 3 s**(15스텝), DROID 15 Hz 1 s. 같은 설정의 설명 문구는 "5Hz ... for 1s"라고 적음 | **차이** | Bridge 창이 1초가 아니라 3초로 보임. 설명 문구와 값이 서로 다름 |
| D6 | LIBERO 평가의 실행 방식 | 평가는 과제당 50회, 사전 정의된 초기 상태(부록 B.1). 청크 20스텝(표 2 "Horizon 20") | `eval_libero.py` 기본값: 청크 20스텝 중 **4스텝만 실행(`--horizon 4`)하고 모든 겹치는 예측을 같은 가중치로 평균하는 시간 앙상블(`--ensemble True`)**. 초기 대기 10스텝. 시드 1234, 초기 상태 0–49 사용(일치) | **차이(논문에 없음)** | 평가 성공률이 시간 앙상블과 재계획 주기에 의존할 수 있음. 기준선(FAST 등)이 같은 방식으로 평가됐는지는 논문과 코드에 없음. 표 2의 처리량(액션/초)은 20스텝 전체 실행 기준으로 읽힘 |
| D7 | 과제별 위치 오프셋 | 언급 없음 | PD와 BAR 평가 스크립트가 과제마다 `--pos_offset` 3 또는 4를 지정(PD는 40과제 중 4개가 3, BAR는 13개가 3). 기본값은 4, 학습 중 검증은 3 | **차이(논문에 없음)** | 선택 기준이 코드와 논문에 없음. 평가 결과로 골랐다면 보고 성공률에 영향. 이 값은 BAR/PD의 행동 위치 id의 간격(offset + 1)임 |
| D8 | BAR 학습의 위치 증강 | 부록 B.4: 블록 단위 인과 마스크, 층별 블록 | 학습에서 행동 토큰의 위치 id를 간격 1–5 사이 무작위로 누적해 부여(`random_position_offset=True`), 입력은 정답 앞 블록 + [BOS] | **차이(논문에 없음)** | 위치 증강이 BAR와 PD 성능에 기여할 수 있음. 논문 서술만으로는 재구현 불가 |
| D9 | KI(지식 격리) 구현 | 부록 B.4: VLM의 자기회귀 손실과 전문가의 flow-matching 손실을 함께 최적화하되 전문가에서 VLM으로 가는 기울기는 차단 | `smolvla/ki.py`: 손실은 속도장의 MSE 하나(`mse_loss(v_t, labels - x_0)`), t는 균등 샘플. VLM은 일반 forward로 호출되고 기울기를 끊는 코드가 없음(주석은 "기울기 전파를 보장"). 자기회귀 토큰 손실 없음. KI용 평가 스크립트, 설정, 공개 체크포인트도 없음 | **차이** | 공개 KI 코드에는 부록 B.4의 기울기 차단과 자기회귀 손실이 없음. 표 1(c)의 KI 94.3이 이 코드로 얻은 값인지는 확인할 수 없음 |
| D10 | VQ와 RVQ 토큰 배치 | 부록 D: RVQ 3단, n = 16, 토큰 48 | `n_quantizers=3`, `n_tokens=48`(HF RVQft 구성). 인코딩은 `b n s -> b (s n)`(층별 연속)이고 BAR 블록 크기 16과 맞음. `decode` 독스트링은 "interleaved"라고 적음 | **일치(독스트링과 코드의 불일치)** | 코드는 층 단위로 묶고 독스트링은 다르게 설명 |
| D11 | AR/PD가 쓰는 토크나이저 가중치 | 부록 D: "다른 모델은 사후 학습된 RVQ 디코더로 강화한 VQ 버전을 쓴다" | 공개 AR와 PD 정책의 `action_processor/model.safetensors`는 ActionCodec-Base와 같은 파일(크기 180,387,949 bytes, SHA-256 앞 16자 같음), BAR는 RVQft와 같음. Base 카드는 훈련 내용 미기재 | **미확인** | Base에 사후 학습 디코더가 들어 있는지 알 수 없음 |
| D12 | LIBERO 데이터와 정규화 | "official π0 recipe"로 미세조정(부록 B.1), 이미지 224 × 448, 랜덤 크롭 87.5%, 밝기/대비/채도 0.2, 색조 0.05, 고유감각 사용(표 6, 7) | LeRobot `physical-intelligence/libero`, 행동 20스텝(`delta_timestamps`가 `i/10`), 행동 차원별 고정 분위수 상수(`ACTION_Q99`/`Q01`)로 나눔, 그리퍼 부호 반전, 상태는 q01–q99 정규화 후 문자열로 프롬프트에 넣음, 두 이미지를 가로로 연결, 크롭 `256 × 0.875`, ColorJitter(0.2, 0.2, 0.2, 0.05) | **일치(세부는 논문에 없음)** | 행동 정규화와 그리퍼 부호 처리는 논문에 없음. `i/10`이 20 Hz 1초인지는 데이터 fps에 따름(확인하지 않음) |
| D13 | 학습 하이퍼파라미터 | 표 7: 학습률 1e-4(표 6은 2e-4), 워밍업 1k, 30k 스텝, 전역 배치 128(GPU당 32 × 4) | `config/train/*.yaml`: 학습률 2e-4, 워밍업 100, 50,000 스텝, GPU당 배치 16 × 4장 = 64, AdamW(weight decay 1e-10), 기울기 클리핑 1.0, 코사인으로 0.1배까지(일치), 기본 백본 SmolVLM2-256M | **차이(기본값)** | 공개 설정으로 학습하면 논문과 다른 학습. 어떤 설정이 공개 체크포인트를 만들었는지 미기재 |
| D14 | 학습 재현성 | 미기재 | `seed_everything`은 평가 스크립트에서만 호출. 학습 스크립트는 시드를 고정하지 않고, 검증 집합은 무작위 분할로 16개 샘플 | **차이(재현성)** | 같은 설정을 다시 돌려도 결과가 다를 수 있음 |
| D15 | 평가 환경 의존 | LIBERO 표준 평가 | 평가는 `libero2gym`(별도 패키지)로 환경을 만듦. 저장소에 환경 래퍼가 없어 종료 조건, 최대 스텝, 보상 처리를 읽을 수 없음. 성공은 `reward >= 1`(누적 clip) | **미확인** | 에피소드 길이 제한 등이 표준 LIBERO와 같은지 확인 불가 |
| D16 | 절대 경로 | — | 학습과 평가 스크립트가 저자 홈의 libero 경로를 `sys.path`에 추가하고, 평가 설정의 `model_name`이 개인 저장소 경로(SmolVLM2-500M) | **차이(이식성)** | 설정을 바로 쓸 수 없음. 체크포인트는 `ckpt_dir`로 읽음 |
| D17 | PD/BAR 전문가 규모 | 부록 B.4: 300M 파라미터 전문가 | 공개 BAR 구성: hidden 768, 24층, 중간 2048, 32 × 64 헤드. 구성에서 센 값은 264.2M, 가중치 파일 1,063,429,688 bytes(fp32로 265.9M) | **차이(약 13%)** | 서술의 "300M"은 어림값으로 보임 |
| D18 | OR 외 지표 | PPL = 토큰 사용 분포 엔트로피(nat)의 지수(표 5) | `_quantize`가 위치를 합쳐 코드 빈도에서 `exp(-Σ p log p)`를 계산(자연로그) | **일치** | — |
| D19 | 무효 시퀀스 처리 | 무효 토큰 시퀀스는 0 행동 청크로 대체(부록 B.2) | AR 평가에서 디코딩 실패 시 `np.zeros((20, 7))`, 이후 `[-1, 1]`로 클립 | **일치** | — |
| D20 | 기준선 코드 | FAST, Binning, String, VQ-VLA, MiniVLA를 같은 VLA에 연결(부록 B.3, F.2) | `utils/vla_tokenizer.py`에 문자열형 `numeric`/`identity` 모드는 있으나 FAST, VQ-VLA, MiniVLA 연결과 훈련 설정은 없음 | **미확인** | 표 1(c) 기준선은 이 저장소로 만들 수 없음 |

### 일치한 것

모델 구조의 핵심(교차 어텐션 질의, 로봇별 임베딩, 푸리에 시각 인코딩), RVQ 3단과 BAR 블록 구성, 이미지 입력과 증강, 평가의 50회 초기 상태, 가변 길이 토큰 시퀀스 처리, PPL 정의는 논문과 맞습니다.

## 3. 해석 상의 주의

- **공개 가중치로 확인할 수 있는 표 1 칸은 AR(95.5), PD(95.6), BAR(97.4) 세 개뿐입니다.** 그마저 GPU, 4 GB급 다운로드, LIBERO 시뮬레이션이 필요합니다. 평가 기본값(D6, D7)은 논문 서술과 다른 점이 있어, 같은 체크포인트로 논문 값이 나오는지가 곧 논문 서술의 재현은 아닙니다.
- **KI는 공개 코드로 논문 서술을 확인할 수 없습니다**(D9).
- **토크나이저 쪽 핵심 주장(OR, TCL/CLIP, 독립 토큰)은 코드로 검증되지 않습니다**(D1). 공개된 것은 학습이 끝난 토크나이저의 추론용 모델뿐입니다.

## 4. 구현 가능성 (Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 수치의 재현은 이 맥에서 불가능합니다. 토크나이저 아이디어의 축소판(작은 Perceiver VQ와 OR 측정)은 구현할 수 있고, 공개 토크나이저 추론은 설치와 다운로드 승인 후 조건부로 가능합니다.**

| 일 | 이 맥 | 이유와 노력(추정, 측정 전) |
|---|---|---|
| 표 재계산, 정적 대조, OR 토이 (이미 완료) | 가능, 완료 | `analysis/recompute.py`, `analysis/toy_or_tradeoff.py` |
| 공개 ActionCodec-Base로 왕복(encode, decode) 오차와 OR 측정 | 조건부 | 새 설치(transformers, vector-quantize-pytorch 1.27.15, einops), 180.4 MB 다운로드, `trust_remote_code`로 저장소 코드 실행 승인이 필요. 45M 파라미터라 CPU 또는 MPS로 가능. 반나절 |
| 작은 Perceiver VQ 토크나이저를 직접 학습해 OR 대 재구성 오차 측정 | 가능(새 설치 없음) | torch 2.15가 있음. LIBERO 행동만 있으면 되고 데이터 다운로드(승인) 필요. 논문의 TCL/CLIP 손실은 언어 임베딩 모델 선택이 필요(논문에 없음). 1–2주 |
| VLA 미세조정으로 500스텝 학습 곡선 비교(256M) | 사실상 불가 | 이미지 224 × 448, 배치 128, flash attention 전제. MPS에서 속도를 측정하지 않음. 폐루프 평가에는 LIBERO 시뮬레이션이 필요 |
| 공개 AR/PD/BAR 체크포인트로 LIBERO 평가 | 불가(실용) | 4.7–5.6 GB 가중치, `cuda:0`와 flash-attention 전제, LIBERO와 `libero2gym`, 구 mujoco/robosuite 고정 버전 필요 |
| 표 1 전체(2.2B, 4 × A100, 30k 스텝), SimplerEnv(8 × H100), 설계 검증(8 × RTX 3090) | 불가 | GPU 자원 |
| SO100, xArm 실로봇, 15K시간 사전학습 | 불가 | 로봇과 데이터 미공개 |
| 직접 해 볼 수 있는 모조 실험: 합성 또는 공개 행동 데이터로 토큰 길이, 어휘, 양자화와 OR의 관계 | 가능 | 이 분석의 토이 T1이 시작점 |

- 공식 코드를 실행하지 않았고 설치도 하지 않았습니다. 위의 노력은 근거 없는 어림입니다.
