# 04. 공식 코드 정적 대조 (논문 ↔ 코드)

**대상:** https://github.com/anhdao69/LoopedWAM, 커밋 `51aac25` (2026-11-19), 얕은 복제본 `repo/`. **실행, import, 설치, 가중치 다운로드는 하지 않았다.** 모든 판단은 파일 읽기에 근거한다. 코드 경로는 `repo/` 기준.

## 0. 요약

- 핵심 구조(루프 전문가, 늦은 정렬, 엘라스틱 증류 목적, 아홉 설정, 동결 교사, 쓰기 설정)는 **논문과 코드가 일치**한다 (K1–K12).
- 논문 본문에 **없는 학습 세부**가 코드에 있다: λ 가중치, 마지막 m=2 스텝 역전파, 플로우 shift, 옵티마이저, 에폭, 실행 행동 수 Hx (D1–D10). 저자는 "재현 세부는 코드와 함께 공개"라고 썼다.
- 논문에 있고 **저장소에 없는 것**: 밀집 기준선 (Dense-12, Dense-30, 밀집 증류), 아블레이션 설정 (KV-mix, 교사 궤적 목적 변형, 분할 P/C/E 변형), 지연 측정 코드, 실로봇 코드와 데이터, Fast-WAM 크기 조정 규칙의 출처 (D11–D15).
- 한 가지 주의: 행동 전문가 크기 조정 규칙을 논문은 "Fast-WAM의 규칙"이라고 하고, 코드는 선형 보간과 sqrt(fan-in) 보정을 쓴다 (D6).
- 라이선스: 코드는 MIT (LICENSE 파일). 체크포인트 메타데이터는 apache-2.0 (D16).

## 1. 확인된 항목 (K)

| 번호 | 논문 | 코드 위치 | 결과 |
|---|---|---|---|
| K1 | 프렐루드 3, 코어 6, 코다 3 (§3.2, §4.1) | `src/loopedwam/models/loop_mot.py` `PRE, CORE, CODA, LOOPS = 3, 6, 3, 4` | 일치. 가상 스케줄 30층 |
| K2 | 코어를 같은 가중치로 루프, 루프 인덱스 임베딩·입력 주입·외부 잔차 없음 (§3.2) | `loop_mot.py` `_run_pair(index, …)`는 같은 물리 인덱스 재사용, 루프별 파라미터 없음 | 일치 |
| K3 | 드오너 블록 1–3, 10–15, 28–30 (1-based, §4.1, §3.4) | `src/loopedwam/models/donors.py` `DONOR_LAYERS = (0,1,2,9,…,14,27,28,29)` (0-based) | 일치 |
| K4 | 비디오 FFN 8960 → 6144, 등간격 짝 선택 (§3.4) | `donors.py` `ffn_indices()` linspace, 행·열 짝 | 일치 |
| K5 | 비디오 폭 1536, 헤드 12 × 128; 행동 폭 512 (§4.1) | `donors.py` `expert_configs()`: hidden 1536 / 512, heads 12, head dim 128 | 일치 |
| K6 | 늦은 정렬 σ(r) = max(1, Kv−Ka+r) (식 (3)) | `src/loopedwam/models/elastic.py` `action_core()`: 마지막 Ka개 코어 루프가 비디오의 같은 블록 위치 캐시 사용 | 일치 (1-based r에서 max는 Ka ≤ Kv이면 불필요) |
| K7 | 교사 Kv=Ka=K=4, N=10, 질의당 비디오 1회 + 행동 N회 (§3.1, §3.3) | `loopedwam.py` `prefill()`에서 비디오 KV 캐시, `infer_action()` 기본 `num_inference_steps=10`, `policy.py` 기본 `action_loops=4, steps=10` | 일치 |
| K8 | 플로우 매칭 식 (1), 추론 시 CFG 없음 (§3.1) | `loopedwam.py` `training_loss`, docstring "CFG 1" | 일치 |
| K9 | 학생 = 교사 행동 전문가 복사, 비디오·VAE·텍스트·고유수용 동결 공유 (§3.5) | `elastic.py` `LoopedWAME.__init__`: `copy.deepcopy(parent.action_expert)`, 비디오 전문가 공유 | 일치 |
| K10 | 단계 조건 MLP, d=1/N, 0 초기화, 시간 변조에 더함 (§3.5) | `elastic.py` `StepCondition` (마지막 층 0 초기화), `velocity()`에서 `tmod + step_condition(d)` | 일치. 파라미터 약 400K (`03` V5) |
| K11 | 설정 S = {1,2,4} × {2,5,10}, 교사 10 스텝, 아홉 설정 균형 순환 (§3.5, §4.1) | `elastic.py` `SCHEDULES`, `balanced_schedule_sequence()` (9개 블록마다 순열) | 일치 |
| K12 | 로컬 목표: 학생 자기 상태 z_j에서 교사 평균 속도 (식 (5), 알고리즘 1) | `elastic.py` `forward()`: `target = teacher_target(teacher, z, tau, next_tau)`, `z`는 학생의 현재 상태. 교사 궤적 변형은 없음 | 일치. 주의: 교사 궤적 변형(표 8)은 저장소에 없음 (D12) |
| K13 | 끝점 손실은 집행 Hx개 MSE, 그리퍼 손실은 LIBERO만 (§3.5, §4.1) | `elastic.py` `endpoint_losses()`, `configs/*/elastic.yaml`의 `gripper_loss: true` (LIBERO), `false` (RoboTwin, RoboCasa) | 일치 |
| K14 | 앵커는 Ka=K, d=0의 원래 플로우 매칭 (§3.5) | `elastic.py` `forward()`: `self.velocity(noisy, anchor_tau, …, loops=4, d=0.)` | 일치 |
| K15 | 행동 청크 H=32 (§3.1, §4.1) | `loopedwam.py` `ACTION_HORIZON = 32` | 일치 |
| K16 | Wan2.1-T2V-1.3B에서 시작, VAE와 T5 동결 (§4.1) | `loopedwam.py` `build_loopedwam(wan_dir=…)`, `donors.py` `read_native_wan()` (30블록 형태 검사) | 일치 |
| K17 | 학습은 H100 (§4.1) | `README.md`: 노드당 H100 4장 (80 GB) | 일치 |
| K18 | 평가 프로토콜 LIBERO 50 초기상태 (§4.1) | `experiments/libero/eval_libero.py`: `--trials` 기본 50, task당 10개, suite 4개 | 일치. 3 seed는 스크립트 밖에서 `--seed`를 바꿔 돌려야 함 |
| K19 | 지연은 원시 관측에서 행동까지 (§4.3) | `policy.py` `__call__`: 전처리, 텍스트, `infer_action` 전체 | 개념은 일치. **지연 측정 스크립트는 없음** (D13) |

## 2. 논문과 다르거나 논문에 없는 항목 (D)

각 항목에 "논문 위치"와 "코드 위치"를 적었다. 논문 값이 없는 것은 "정보 누락"이다.

| 번호 | 항목 | 논문 | 코드 | 종류 |
|---|---|---|---|---|
| D1 | 손실 가중치 λ_end, λ_grip, λ_anchor | 식 (6)에 λ만 있음, 값 없음 | `elastic.py` `LOSS_WEIGHTS = dict(local=1.0, end=0.5, grip=0.1, anchor=0.5)` | 정보 누락 |
| D2 | 역전파 스텝 수 m | "마지막 m개 스텝만" (§3.5), m 값 없음 | `elastic.py` `forward()`: `if j <= n - 3: z = z.detach()` → m = 2 | 정보 누락 |
| D3 | 실행 행동 수 Hx | "첫 Hx개 실행" (§3.1), 값 없음 | LIBERO 10 (`configs/libero/elastic.yaml`, `eval_libero.py` 설명), RoboTwin 24 (`configs/robotwin/elastic.yaml`, `README.md`), RoboCasa 12 (`configs/robocasa_gr1/elastic.yaml`, README). 평가에도 같은 값 | 정보 누락 |
| D4 | 옵티마이저와 스케줄 | 없음 | AdamW (β 0.9, 0.95), 가중치 감쇠 0.01, 그래디언트 클리핑 1.0, 코사인 감쇠 (README). 학생 행동 학습률 3e-5, 단계 조건 1e-3 | 정보 누락 |
| D5 | 학습 에폭과 배치 | "재현 세부는 코드와 함께" (§4.1), 값 없음 | LIBERO 단계 1·2·엘라스틱 각 10 에폭, RoboTwin 5, RoboCasa 5 (`configs/*`). 배치 128 (RoboCasa 단계 2는 256) | 정보 누락 |
| D6 | 행동 전문가 크기 조정 | §3.4: "Fast-WAM[63]의 크기 조정 규칙" | `donors.py`: 모든 백본 텐서를 순차 선형 보간 (정렬 끝점) 후 `sqrt(fan-in 비)` 곱. 독스트링에 Fast-WAM 인용 없음 | 출처 불명. Fast-WAM 규칙과 대조하지 않음 |
| D7 | 행동 전문가 FFN 폭 | 없음 (폭 512만 있음) | `donors.py` `expert_configs`: `ffn_dim=2048` | 정보 누락 |
| D8 | 플로우 shift와 시간 샘플링 | 식 (1)의 w(τ)만 있음 | `scheduler.py` `shift`: 비디오 5, 행동 1. `plans.py`: 균등 u에서 `shift·u/(1+(shift−1)u)`로 τ 샘플 | 정보 누락 |
| D9 | 학습 데이터 규모 (RoboTwin 50 깨끗 + 500 무작위) | §4.1에 있음 | `README.md`: FastWAM 데이터셋 사용. 데이터 자체는 받지 않아 확인 안 함 | 확인 안 함 |
| D10 | 소스와 체크포인트 형식 | 없음 | `loopedwam.py` `checkpoint_payload`: 형식 이름 `loopwam-s-v1` 등, 행동 루프 4 고정. `elastic.py` `elastic_payload`: 교사 `mot`에 학생 행동 가중치를 덮어씀 | 정보 누락 (형식) |
| D11 | 밀집 기준선 Dense-12, Dense-30, 밀집 증류 Dense-30-E | 표 1, 3, 4, 7 | 저장소에 설정이나 모델 없음 | 논문에만 있음 |
| D12 | 교사 궤적 목적 변형, 끝점·앵커 제거 변형 (표 8) | 표 8 | 변형 옵션 없음 (`elastic.py`에 단일 목적) | 논문에만 있음 |
| D13 | 지연 측정 하네스 (H100, 500 질의) | §4.3 | 측정 스크립트 없음 | 논문에만 있음 |
| D14 | 실로봇 (AgileX PiPER, 80 시험, Fast-WAM 미세조정) | §4.6 | 실로봇 코드, 데이터, 설정 없음 | 논문에만 있음 |
| D15 | KV-mix (학습된 비디오 루프 혼합), 분할 P/C/E 변형, 코어 초기화 변형 (표 7, C2, C3) | 부록 | 각 변형의 구현 없음. 기본 3/6/3 K=4 설정만 있음 | 논문에만 있음 |
| D16 | 라이선스 | 논문: CC BY 4.0 | 코드 LICENSE: MIT. HF 메타데이터: apache-2.0. 제3자 파일은 원 라이선스 유지 | 불일치 (코드와 가중치) |

### 2.1 D 항목 메모
- D2–D5, D8: 논문이 "재현 세부는 코드와 함께 공개한다"고 한 것이 실제로 이 항목들이다. 따라서 **코드 없이 논문만 읽으면 재현할 수 없는 정보**가 여기에 있다.
- D6: 본문 §3.4의 "evenly spaced hidden units" 규칙은 비디오 FFN에만 맞다. 행동 전문가는 폭을 1536에서 512로 줄이므로 다른 규칙이 필요하다. 논문이 이 규칙을 "Fast-WAM의 규칙"으로 돌리는 것은 확인하지 못했다 (Fast-WAM 코드를 여기서 읽지 않았음).
- D11–D15: 논문 표의 중요한 대비 대부분이 저장소에 없다. 엘라스틱의 "같은 파라미터에서 비디오 루프가 낫다" 주장 (표 7a)은 다른 설정 조합을 실행해 보지 않는 한 확인되지 않는다.

## 3. 코드 품질 메모 (구현 참고용)
- `policy.py`와 `eval_libero.py`에 `torch.autocast('cuda', dtype=torch.bfloat16)`과 `device='cuda'`가 박혀 있다. MPS 포팅에는 이 부분 수정이 필요하다.
- `elastic.py` `LoopedWAME.__init__`은 `copy.deepcopy`로 행동 전문가를 복사한다. 메모리는 교사 1 + 학생 1 + 비디오 KV.
- 학습 스크립트 `train_elastic.py`는 `torch.use_deterministic_algorithms(True)`와 `CUBLAS_WORKSPACE_CONFIG`를 설정하고 `torch.cuda.set_device`를 호출한다. CUDA 없이는 실행되지 않는다.
- 학습 데이터 경로(`data/prepared/…`)와 체크포인트 경로(`runs/…`)는 상대 경로이며 개인 경로는 없다.

## 4. 구현 가능성 판정

**판정: 부분 가능.** 전체 재현은 이 맥에서 불가능하다. 체크포인트 로드와 단일 질의 추론 점검, 코드 검증은 조건부로 가능하다.

### 4.1 이 맥에서 가능한 것 (Apple M4 Pro, 24 GiB, MPS, CUDA 없음)

| 작업 | 판정 | 필요한 것 | 노력 (추정) |
|---|---|---|---|
| 정적 검증 (이 문서) | 완료 | 없음 | 완료 |
| 체크포인트 로드 (LoopedWAM·E, LIBERO) | 조건부 | HF 가중치 약 2.3 GB 다운로드 (승인 필요), `autocast('cuda')`를 MPS로 바꾸는 포팅 | 1–2일 |
| 단일 질의 추론 (입력: 임의 이미지와 텍스트 임베딩) | 조건부 | 위 포팅 + T5 텍스트 인코더 (Wan2.1, 추가 다운로드) | 1–2일 추가 |
| LIBERO 시뮬레이션 평가 (1 suite) | 어려움 | LIBERO·MuJoCo 설치 (승인 필요), MPS 추론의 처리량이 낮음, 10 task × 50 에피소드 | 1주 이상 추정 (설치와 측정) |
| 지연 재현 (H100 기준) | 불가 | CUDA 기기 | — |
| 학습 재현 (단계 1·2, 엘라스틱) | 불가 | H100 4장. LIBERO 잠재 캐시 약 42 GB, RoboTwin 약 1.1 TB, RoboCasa 약 590 GB | — |
| 실로봇 | 불가 | 로봇, 코드 없음 | — |
| 공식 코드의 다른 설정 (Dense, KV-mix 등) | 불가 | 저장소에 없음 | — |

### 4.2 전체 판단
- **저장소를 실제로 실행하는 것은 이 맥에서 조건부.** 코드가 CUDA를 가정한다. 포팅과 승인이 필요하다.
- **학습은 이 맥에서 불가.** 어떤 규모에서도 H100 4장이 필요하다.
- **재현 가치:** 구조와 목적 함수가 논문과 맞으므로, 체크포인트 추론 점검만으로도 "논문 표 1의 LIBERO 96.1" 수치의 한 설정은 확인할 수 있다. 다만 전체 6000 에피소드 평가는 맥에서 현실적이지 않다.

## 5. 저자에게 물을 질문 (공개 저장소에서 답이 없는 것)
1. λ_end, λ_grip, λ_anchor, m의 값과 선택 근거.
2. 행동 전문가 크기 조정이 Fast-WAM과 같은 규칙인지, 아니면 선형 보간과 sqrt 보정인지 (D6).
3. 표 8, 표 7의 변형 실험 설정 (교사 궤적, KV-mix, 분할 변형)을 공개할 수 있는지.
4. 표 5 이득의 짝지음 검정에 쓴 에피소드 단위 성공 기록.
5. 실로봇 데이터와 Fast-WAM 미세조정 설정.
