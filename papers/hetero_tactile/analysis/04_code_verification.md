# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 대상: github.com/jxbi1010/HTT, 커밋 `2e5b0b10d9e527c188bd78c66db3c500d1bc8877` (2026-09-05, 얕은 복제). 논문은 arXiv v2.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았고 체크포인트와 데이터는 받지 않았습니다. 복제 내용은 신뢰할 수 없는 데이터로 다뤘습니다.
- 읽은 파일: `config/model/pretrain.yaml`, `config/algo/pretrain_joint.yaml`, `config/algo/probe.yaml`, `config/algo/spl.yaml`, `config/model/{vit,taxel_tf}.yaml`, `config/sensor/*`, `train/run_pretrain_joint.py`, `train/run_probe.py`(검증과 시험 부분), `model/predictor.py`, `model/shared_trunk.py`, `utils/sliding_labels.py`, `utils/cusum.py`, 데이터 로더 일부, `docs/*`, `train/README.md`, `baselines/README.md`, `NOTICE`.
- 코드 README는 `config/algo/pretrain_joint.yaml`을 "공개 체크포인트(step 34,000)를 만든 정확한 레시피"라고 명시합니다. 따라서 아래 차이는 "논문 본문 설정 대 공개 체크포인트 레시피"의 차이입니다.

## 1. 설치 요구사항과 라이선스

- `requirements.txt`: torch 2, torchvision, timm 1, einops, omegaconf, webdataset, scikit-learn, opencv-python, tensorboard 등. "Tested with PyTorch 2.x + CUDA".
- **사전학습은 CUDA가 없으면 오류를 냅니다**: `train/run_pretrain_joint.py:76` `if not torch.cuda.is_available(): raise RuntimeError("CUDA required for joint pretraining.")`, 같은 파일의 `autocast("cuda")`와 `torch.cuda.set_per_process_memory_fraction(0.9)`, `utils/base_trainer.py`의 `GradScaler("cuda")`. 추론 경로(`htt.py`)는 CPU로 떨어집니다(`torch.cuda.is_available()`로 장치 선택).
- 라이선스: LICENSE와 README는 CC BY-NC 4.0, NOTICE는 V-JEPA 파생 파일(`utils/__init__.py`, CC BY-NC 4.0)과 DINOv2·V-JEPA 블록(Apache-2.0), MIT 파일을 밝히고 합친 저작물을 CC BY-NC 4.0으로 둡니다. **비상업적 연구 용도로만 쓸 수 있습니다.** 공유 트렁크 파일 `model/shared_trunk.py`는 T3(MIT, MIT CSAIL)에서 가져왔는데 NOTICE는 이 파일을 "original"로 분류합니다(T3 헤더와 상충, 경미).
- 데이터셋 카드와 모델 카드도 CC BY-NC 4.0입니다.
- 체크포인트: `htt_4sensors_best.pth` 약 69 MB, SHA-256 `024f4c3a…c673f0`. 학습 step 34,000(총 60,000 중)의 슬림 저장본(옵티마이저·예측기 제외, 디코더 포함 17.1M 파라미터).

## 2. 논문 대 코드 (D 항목)

| 번호 | 항목 | 논문 | 코드 (파일) | 영향 |
|---|---|---|---|---|
| D1 | 정렬 가중치와 스케줄 | 정렬 α_t: 초기 워밍업 동안 0, 이후 선형 증가, 최대 0.1, 50,000스텝 중 첫 20,000스텝 동안 증가 (§4.2, 부록 A.1) | 정렬을 step 0부터 켜고(`alignment_warmup_steps: 0`) 10,000스텝 선형 증가(`alignment_coef_warmup_steps`). 계수는 방향별 **0.01 / 0.02**(`9dtact_to_xela` 0.01, `xela_to_9dtact` 0.02, `gsmini_to_tac02` 0.01, `tac02_to_gsmini` 0.02) (`config/algo/pretrain_joint.yaml`) | 큼. 논문의 "MAE만 먼저 학습해 센서 특징을 만든 뒤 정렬"과 다름. 최대 가중치도 0.1이 아니라 0.01–0.02 |
| D2 | 비대칭 방향 가중 | 평균 (식 2, 모든 순서쌍 평균) | taxel → 광학 방향이 광학 → taxel의 2배 | 중. 논문에 없음 |
| D3 | MAE 센서 가중 | 센서 평균(식 1) | `mae_weights`: Xela 1.5, TAC-02 1.0, 9DTact 1.0, GSMini 0.7 | 중. 주석: "taxel을 올리고 gsmini는 낮춤(손실을 지배하므로)" |
| D4 | 정렬 소스 마스킹 | 소스는 마스크 없는 전체 임베딩 (식 2, §4.2 "full source embeddings") | 소스도 마스킹: `alignment_mask_ratios.source` 9DTact 0.5, Xela 0.4, GSMini 0.75, TAC-02 0.6 (`_alignment_forward`). 목표 마스크는 일치(광학 0.9, taxel 0.8) | 중. 식 2의 정의와 다름 |
| D5 | 학습률, 배치, 길이 | lr 3e-4, 배치 256, 50,000스텝 (부록 A.1) | lr 4e-4, weight decay 0.05(논문에 없음), 배치 512, 총 60,000스텝, **공개 체크포인트는 34,000스텝** | 중. 같은 설정이 논문 표의 모델과 같은지 확인할 수 없음 |
| D6 | 인코더 깊이 | 광학 3, taxel **2** (부록 A.1) | `encoder_depth: 3`을 xela·tac02에도 적용 (`config/model/pretrain.yaml`) | 소. 파라미터 수 차이 |
| D7 | taxel 입력 길이 | Xela 20프레임(토큰 5), **TAC-02 40프레임(토큰 10)** (부록 A.1) | 두 센서 모두 `tactile_chunk_size: 20`, 패치 4 → 토큰 5 (`config/sensor/tac_config.yaml`, `docs/PREPROCESSING.md`의 T = 20) | 중. TAC-02 창이 0.2 s 대 절반. 논문 τ = 0.2 s와 불일치 가능 |
| D8 | taxel 전처리 | "비접촉 기준 프레임을 뺀다" (§4.1) | 기준 빼기 + **절댓값** + 차원별 표준편차 나눔 + ±10 클램프 (`docs/PREPROCESSING.md`) | 소(절차 누락) |
| D9 | 최선 상태 복원 | (논문에 없음) 미세조정 후 시험 | `best_probe_model_state = probe_mlp.state_dict().copy()` (`train/run_probe.py:709`). 얕은 복사라 헤드는 복원되지 않음(`03` 토이 1). 인코더·트렁크는 CPU 복제로 올바르게 복원 | 소–중. 헤드와 백본이 서로 다른 시점 |
| D10 | 미세조정 기본값 | 논문에 없음 | `probe_lr 5e-5`, `probe_weight_decay 1e-3`, 2,000스텝, 100스텝마다 검증, 인내 5, 배치 256, **시드 [10] 하나** (`config/algo/probe.yaml`). 논문은 시드 10, 20, 30 | 중. 표의 SD는 여러 시드로 돌려야 나옴 |
| D11 | 힘 라벨 클리핑 | 힘 범위 수직 40 N, 전단 14 N (그림 2 B1) | 힘 3축을 **±20 N으로 클리핑** (`force_clip_min/max`, 로더·`spl.yaml`·`run_probe.py`) | 중. 수직 20 N 초과 라벨이 잘림. 표 5의 수직 MAE(Xela 1.376 N 등)가 클리핑된 라벨 기준인지 불명 |
| D12 | 쌍 섞기 대조 | 논문에 결과 없음 | `alignment_shuffle_pairs`(주석 "Rebuttal C3 control")가 있으나 기본 꺼짐, 결과 미공개. 정렬을 대안(InfoNCE, align-token, mean 예측기)으로 바꾸는 옵션도 있으며 주석은 일부가 "붕괴했다(collapse)"고 적음 | 정보. 쌍 정렬이 실제로 쌍 정보를 쓰는지 가르는 대조가 논문에 없음 |
| D13 | 실로봇 정책, ManiFeel | ACT 정책 학습, Sharpa 입력 처리, ManiFeel TacFF 인코더(§5.5) | 저장소에 정책 코드, Sharpa 코드, ManiFeel 코드 없음. TacFF 인코더와 비슷한 `train/finetune_mae.py`(새 taxel 인코더·디코더 + 상속 트렁크, `--freeze_trunk` 권장, 트렁크 lr 0.1배)는 있음 | 표 3, 4는 코드로 재현 불가 |
| D14 | 그림 3 검색 | Procrustes 검색 (§5.4) | 코드 없음 (`procrustes`, `recall`, `retriev` 검색 결과 없음) | 그림 3은 재현 불가 |
| D15 | 광학 입력 프레임 수 | 한 쌍 = 동기화된 쌍 프레임 (§1), 튜블릿 2개라는 표현 (부록 A.1) | 모델은 광학 **2프레임 입력, 튜블릿 크기 2 → 시간 슬롯 1개, 토큰 196**(`num_frames: 2`, `tubelet_size: 2`). 데이터셋 카드는 "쌍 = 광학 1프레임 + taxel 10프레임", docs는 "중복 프레임이면 품질 저하" | 소. 문장 불일치 |
| D16 | 슬립 라벨 | μ(t)의 기준은 에피소드 앞 10%; 양방향 CUSUM; 큰 슬립은 상한 통계가 임계값을 **여러 프레임 연속** 넘을 때; 초기 슬립은 직전 몇 프레임 (부록 D) | 기준 구간은 앞 25%(`baseline_frac 0.25`, 최소 8프레임); 학습에 쓰는 `labels_bracket`은 **C⁺ > h_trigger(5.0) 첫 프레임부터 마지막 접촉 프레임(|fz| > 1.5)까지 전부 2**, 초기 슬립은 직전 5프레임(`incipient_window 5`), 연속 조건 없음, 하한 통계 C⁻는 미사용 (`utils/sliding_labels.py`) | 큼. "slide" 85%는 이 괄호 규칙의 결과. 라벨이 "미끄러짐 상태"가 아니라 "첫 변화 이후 접촉이 끝날 때까지"에 가까움 |
| D17 | 슬립 손실 | 가중치 언급 없음 | 역빈도 클래스 가중치를 CE에 사용 (`compute_sliding_class_weights`, `run_probe.py`와 `run_spl.py`) | 소–중. 논문에 없음, 불균형 처리는 macro-F1에 영향 |
| D18 | 임베딩 읽기 | CLS 없음, 트렁크 출력은 토큰 열 (부록 A.1) | 추론 임베딩 = 토큰 평균 풀링 `[B, 192]` (`htt.py`, `docs/ARCHITECTURE.md`) | 정보. 하류 헤드의 입력 정의 |
| D19 | Scratch 구조 | "센서별 동일 인코더, 층 더 많이" (부록 B) | 광학 Scratch(`config/model/vit.yaml`) 9층 × 256차원 4헤드(HTT 3층 × 192), taxel Scratch(`taxel_tf.yaml`)는 3층 192로 HTT 인코더와 같음 | 정보. 광학은 더 크고 taxel은 같음 |
| D20 | MAE 단독 기준 | "MAE (ours)" 별도 사전학습 | MAE 단독 설정 파일은 없음(`pretrain_base.py`가 MAE 학습기, 정렬을 끈 설정 파일은 미공개) | 소. 정렬 효과 대조의 설정 미공개 |

### 2.1 코드가 논문과 일치한 것

- 목표 마스크 비율(taxel 0.8, 광학 0.9)과 MAE 마스크 비율(광학 0.75, taxel 0.6), D = 192, 헤드 3, 트렁크 깊이 9, 예측기 깊이 3(교차 어텐션, 학습형 마스크 토큰), 광학 ViT 깊이 3, 패치 크기 4(taxel)·16(광학), AdamW, 워밍업 2,000스텝과 시작·종료 학습률 3e-6, 기울기 클립 1.0.
- 정렬 기울기는 인코더를 막고(`alignment_stop_grad_encoders: true`) 목표에 stop-gradient(`alignment_detach_target: true`, 주석: "없으면 목표를 자명하게 만들어 하류 품질이 붕괴")를 건다. EMA 목표 네트워크는 없고 같은 트렁크의 분리된 출력이 목표입니다(논문도 "stop-gradient"만 언급).
- 정렬 손실은 가려진 목표 토큰에서만 MSE(`_alignment_loss`)이며 식 2와 같은 형태입니다.
- 쌍 샘플링: 학습 스텝마다 쌍 하나(Xela–9DTact 또는 TAC-02–GSMini)를 데이터 크기 비율로 뽑습니다(확률 약 67.5% 대 32.5%). 논문은 언급하지 않습니다.

## 3. 평가 프로토콜 점검

- 검증 손실이 가장 낮은 지점의 백본으로 시험을 한 번 평가합니다(시험 정확도를 보고 고르지 않음). 이 점은 같은 계열의 다른 코드와 달리 깨끗합니다. 단 D9의 헤드 복원 문제가 있습니다.
- 분할은 에피소드 단위입니다(분류 쌍당 40 시험 에피소드, 힘·슬립은 80/10/10, `03` V8).
- 베이스라인(T3, SITR)은 `baselines/`로 같은 프로브 파이프라인을 통과시킵니다. AnyTouch 로더도 `baselines/load_baselines.py`에 있습니다(README에는 SITR와 T3만 기재). AnyTouch는 CLIP ViT-L/14 기반(코드 주석상 약 305M 규모)이라 약 17M인 HTT와 모델 크기가 크게 다릅니다. 센서 슬롯은 GSMini 3, 9DTact 6으로 지정됩니다. 논문은 모델 크기를 비교 조건으로 적지 않았습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 전체를 이 맥에서 재현하는 것은 불가능합니다. 공개 체크포인트로 특징을 뽑는 추론과 소규모 하류 평가는 조건부로 가능하고, 사전학습 재현은 이식과 장시간 학습이 필요해 사실상 어렵습니다. 실로봇과 ManiFeel은 코드가 없어 불가입니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 표 3 실로봇 (Sharpa 손, Franka, ACT) | 불가 | 로봇과 센서, 정책 코드 없음 (D13) |
| 표 4 ManiFeel | 불가 (코드 없음) | 평가 코드 미공개, ManiFeel 환경은 GPU 시뮬 |
| 그림 3 검색 | 불가 (코드 없음) | D14. 같은 지표를 처음부터 다시 구현하는 것은 가능 |
| 사전학습 전체 (60,000스텝, 배치 512) | 사실상 불가 | 4090 한 장 약 9시간, 약 22 GiB 사용(공식 README). 이 맥은 메모리 약 24 GiB 중 사용 가능분이 작고, 코드는 CUDA 필수(§1)라 이식(장치, AMP, 메모리 분할)이 필요. 순전파·역전파 FLOP를 손으로 어림하면(광학 한 쌍 약 10–15 GFLOP) 총 약 4e17 FLOP이고, MPS의 실효 1–3 TFLOPS 가정으로 이틀 안팎에서 일주일 사이(추정, 측정 전). 배치를 줄이면 하이퍼파라미터가 달라짐 |
| 공개 체크포인트(69 MB)로 임베딩 추출 | 조건부 | 다운로드 승인과 torch, timm, einops, omegaconf 등 설치 승인이 필요. 추론 경로는 CPU 폴백이 있음. 속도는 측정 전 |
| 공개 데이터(4.41 GB)로 하류 평가 (분류, 힘, 슬립) | 조건부 | 데이터 다운로드와 webdataset, scikit-learn, opencv 설치 승인이 필요. 미세조정은 CUDA 가정(AMP) 코드를 MPS/CPU로 바꿔야 함. 모델이 약 17M 파라미터라 CPU 미세조정은 작은 분할에서 현실적 |
| 구조 축소 재구현 (인코더 + 공유 트렁크 + 정렬 예측기, 합성·소규모 쌍 데이터) | 가능 | 모델이 작음 (D = 192, 트렁크 9층). 소규모 토이(`03` 토이 2)처럼 CPU/MPS에서 가능 |
| 통계 재계산, 정적 대조 | 완료 | `03`, 이 문서 |

노력 추정(측정 전): 체크포인트 추론 + 하류 평가 파이프라인 1–2주(설치, 승인, CUDA 가정 수정 포함), 사전학습 축소판(작은 배치, 소규모 쌍 데이터) 2–3주. 정확한 속도는 측정하지 않았습니다.
