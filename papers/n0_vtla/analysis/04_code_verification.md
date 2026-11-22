# 04. 공식 코드 대조와 구현 가능성

대상: `neoteai/N0-VTLA`, 커밋 `03a0ce4`(2026-08-18), 얕은 복제 `repo/`(정적 읽기만). 설치, 실행, 임포트를 하지 않았다. 대조 근거는 파일과 행 번호로 적는다. 논문 근거는 절·식·표 번호다.

## 1. 저장소 개요와 범위

- **포함:** 사후 학습 설정(`n0vtla/training/config.py`), 촉각 예측기와 인코더(`models_pytorch/tactile_predictor.py`, `tactile_encoder.py`), 정책 본체(`n0vtla_policy.py`, 817행), 직렬화된 추론 서버(`serve_policy.py`, `serve_zmq.py`), 촉각 인과 탐침(`probe_z_tactile_dependence.py`, `docs/TACTILE_CAUSAL_PROBE.md`), 변환·정규화 스크립트, Docker 파일.
- **제외:** README가 말하는 대규모 사전 학습 파이프라인, NeoData 변환과 데이터 보고서, ALTER(진행도 모델과 advantage 라벨 생성). 저장소에서 `progress`, `advantage`, `infonce`, `contrastive`, `z_star` 같은 이름을 검색했지만 해당 구현은 없다(코드 주석의 "미포함" 설명만 있다).
- **기반:** `pi0` 계열 PyTorch 구현과 JAX 구현을 함께 둔다. 기반 체크포인트는 π0.5 계열이다(README "Backbone: PaliGemma (gemma_2b) prefix + Gemma (300m) action expert").

## 2. 차이점 (D1–D12)

| 번호 | 논문 | 공식 코드 | 평가 |
|---|---|---|---|
| **D1** | 1단계: z를 z*에 InfoNCE + 재구성으로 맞춤(§4.2, Eq. 2–5). 기반 동결 | 파일 머리말 1–4행: "z는 action-MSE 기울기만으로 정해진다. 논문의 감독형 예측기 기반 단계는 이 사후 학습 저장소에 없다." `_forward_predictor` 독스트링은 `predictor_pretrain` 경로를 언급하지만 구현을 찾지 못했다(grep). 배포 설정 `predictor_loss_weight=0.0` | **가장 큰 차이.** 공식 코드는 논문의 **자유 잠재(free-latent) 변형**과 같다. 논문의 주 방법(1단계 grounding)은 공개되지 않았다 |
| **D2** | 잠재 토큰 수 10(z ∈ R^{10×d}, Eq. 1–2, §4.2 "ten latent tokens") | `n_latent=5` 모든 설정(config.py 792, 836, 906, 960). README "5 latent tokens". 예측기 기본값은 10(`tactile_predictor.py` 55행) | 설정이 기본값을 덮는다. 배포 모델은 5토큰이다 |
| **D3** | 게이트 없음: 잠재를 action 접두에 바로 붙임(§2.2 Step 2) | `vtla_tactile_posttrain` 설정은 `z_gate_zero_init=True`(0 초기화 스칼라). 공개 시뮬 체크포인트는 `z_gate_zero_init=False`로 "게이트 파라미터가 없다"고 주석. README "Zero-initialized gate"는 전자만 설명 | 공개 체크포인트와 README 설명이 서로 다르다. 논문에는 게이트가 없다 |
| **D4** | 행동 청크 H = 50(§2.1, 모든 실험) | `sim_dual_arm_tactile`은 `action_horizon=16`(config.py 약 957행). 나머지는 50 | 이중 팔 시뮬 설정은 논문과 다르다. 논문은 이 설정을 명시하지 않는다 |
| **D5** | 단계 2: 기반 VL prefix 키·값을 마스크하고 expert만 학습(§4.2) | 단계 전환 설정이 없다. `vl_dropout_prob=0.0`이며, 이는 확률적 VL 차단이지 단계별 마스크가 아니다(n0vtla_policy.py 97행) | 2단계는 구현되지 않았다 |
| **D6** | 촉각 입력 방식 비교: vlm_concat, expert_concat은 서론에서 비판(§1). 실험 결과 없음 | `tactile_mode`에 `latent`, `vlm_concat`, `expert_concat`이 있고 "baseline tactile-injection modes (ablation)"라고 주석(n0vtla_policy.py 201행) | 기준선 코드는 있으나 논문 결과가 없다(03 K9) |
| **D7** | 인코더: 동결 DINOv2, 10토큰(CLS + 3×3 풀링), 224×224, 기준선 차이 `tac_t − tac_0`(Eq. 1) | `facebook/dinov2-base`(`tactile_encoder.py` 56행), `N_TOKENS=10`, `pool_grid=3`, `tac_t − tac_0` | **일치** |
| **D8** | 시작 시 기준선 재설정, 기준선 미가용 시 자기 기준선 사용(App. D) | `serve_zmq.py` 36–43행의 reset 명령과 첫 예측 프레임을 기준선으로 쓰는 동작 | **일치** |
| **D9** | 전체 청크 실행, 생성 Euler 10스텝(App. D) | `sample_actions(num_steps=10)`(pi0_pytorch.py 454행). 청크 전부 실행 요구는 README에 있다 | **일치** |
| **D10** | 진행도 모델(π0.5 기반 쌍 관찰 구조), 손실 식 7, 라벨 식 8 | 구현 없음(D1과 같은 이유) | **미공개** |
| **D11** | 기반 정책은 π0.5 가중치에서 시작(§5) | README `vtla_tactile_posttrain`, `VTLA_PRETRAINED_CHECKPOINT`. 기반 체크포인트 `n0-vtla-base`는 "사후 학습용 사전 학습 기반"이라고 적음 | 기반 체크포인트가 NeoData 사전 학습을 거쳤는지는 README에 명시되지 않는다. 논문 §4.1의 사전 학습이 이것이라고 추정할 뿐이다 |
| **D12** | 정규화 통계는 청크 상대 표현에서 계산(§3, App. C) | `compute_norm_stats.py`, `compute_canonical_norm.py`, `convert_canonical_data.py`가 있다 | 대조 가능(정적). 실행하지 않았다 |

**요약.** 논문의 주 방법은 두 덩어리로 나뉜다. 예측기 감독(D1)과 단계 전환(D5)은 공개되지 않았다. 공개된 것은 논문의 자유 잠재 변형에 가깝다(D1, 논문 §4.2 마지막 단락의 정의와 일치). 설정 값(D2, D3, D4)은 논문 본문과 다르다. 논문의 반사실 실험(Fig. 10)과 교란 탐침(§5.5)에 대응하는 공개 도구는 있다(`probe_z_tactile_dependence.py`, `gate_c_check.py`). 그러나 결과 수치는 없다.

## 3. 라이선스

- **논문:** CC BY-NC-SA 4.0(01 §0).
- **저장소 LICENSE:** CC BY-SA 4.0. 저장소 README는 "원본 자료"에 적용한다고 하며, 서드파티와 Apache-2.0 구성요소는 NOTICE에 따른다.
- **GitHub API 라이선스 필드:** `NOASSERTION`(other). 자동 판별이 실패했다. 코드 저장소에 CC BY-SA를 쓴 것은 코드 라이선스로는 드문 선택이다. 저장소 안에서 사용 조건을 정할 때 이 점을 확인해야 한다.
- **가중치:** Gemma 이용 약관과 금지 사용 정책을 따른다(README "License").
- **이 분석의 처리:** 저장소 내용은 복제했지만 공개 산출물에는 코드를 넣지 않았다. 공개본에는 저장소 링크와 대조 결과만 적는다.

## 4. 구현 가능성 판정

**판정: 전체 재현은 이 맥에서 불가능하다. 부분 확인(체크포인트 로드와 추론 한 번)은 조건부로 가능하다. 논문의 주 결과는 이 맥에서 재현할 수 없다.**

근거와 자원은 다음과 같다.

| 항목 | 필요 자원 | 이 맥(M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음) | 판정 |
|---|---|---|---|
| 논문 1단계(예측기 grounding) 재현 | NeoData 규모 데이터, 감독 코드 | 코드 없음. 데이터 공개 여부 미확인 | **불가** |
| 대규모 사전 학습 | 클러스터 GPU, 데이터 | 불가. 자원 수치조차 논문에 없다(App. B) | **불가** |
| 사후 학습(SFT, 설정 기본 20,000 스텝, 배치 64) | CUDA GPU, LeRobot 형식 데이터 | 설치 문서가 CUDA 학습을 전제. 3B급 전체 미세조정은 옵티마이저 상태만 수십 GB | **불가** |
| 추론 한 번(체크포인트 로드, 행동 샘플링) | 체크포인트 **8.25 GB**(bf16 추정), DINOv2 캐시, transformers 패치, JAX 또는 PyTorch CUDA 경로 | 메모리는 맞을 수 있음. MPS 포팅과 코드 수정 필요. 미검증 | **조건부** (추정 1–3일, 측정 전) |
| 시뮬 평가(UniVTAC, NeoSim) | UniVTAC 시뮬레이터, ZMQ 서버, 시뮬 체크포인트 | 시뮬레이터 설치 요구는 확인하지 않음 | **미확인** |
| 실로봇 평가 | Flexiv, ARX 등 팔, 센서, 조작 장비 | 불가 | **불가** |
| 촉각 탐침(`probe_z_tactile_dependence.py`) | 체크포인트와 실제 촉각 배치 | 추론 조건이 맞으면 가능. 미실행 | **조건부** |
| ALTER 재현 | 진행도 모델, 배포 데이터, 라벨 생성 코드 | 코드 없음 | **불가** |

**노력 추정(측정 전).**
- 추론 확인: 1–3일(MPS에서 bf16 동작 여부에 따라 크게 달라진다). 체크포인트 승인과 8.25 GB 다운로드가 먼저다.
- 사후 학습 재현: CUDA GPU 한 장 이상이 필요하다. 저장소 밖 GPU 자원과 데이터가 있을 때 1주 이상.
- 논문 전체 재현: 불가. 필요한 것은 NeoData 공개, 감독 1단계 코드, ALTER 코드다. 셋 다 공개되지 않았다.

**이 맥에서 할 수 있는 것.** 재계산과 검정(03), 정적 대조(이 문서), 주장 판정(03 §4). 토이는 만들지 않았다(03 §5).

## 5. 확인하지 못한 것

- 평가 시도 수, 시드, 분산: 논문에 없다(03 K10).
- NeoData의 공개 여부와 규모: 데이터 보고서 [51]을 찾지 못했다. 저장소와 HF 모델 페이지만 확인했다.
- 기반 체크포인트의 학습 이력: README에 NeoData 사전 학습을 명시하지 않는다(D11).
- 두 HF 시뮬 체크포인트 크기: 확인하지 않았다(받지 않음).
- 실행 결과: 공식 코드는 실행하지 않았다(지시사항).
