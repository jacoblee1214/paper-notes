# 공식 코드 대조: Counterfactual World Models (arXiv:2611.24778)

- 대상: [galilai-group/counterfactual-WM](https://github.com/galilai-group/counterfactual-WM), 커밋 `5c84bba` (2026-11-19 "code release"), MIT.
- 방법: `repo/` 얕은 복제(`--depth 1`)를 **정적으로만** 읽었습니다. 설치, 실행, import, 테스트(`pytest`), 체크포인트와 데이터 다운로드를 하지 않았습니다.
- 읽은 파일: `cwm/counterfactual.py`, `cwm/metrics.py`, `cwm/evaluate.py`(일부), `cwm/predictor.py`(구조와 손실), `cwm/readout.py`(일부), `cwm/model.py`(일부), `cwm/encoders.py`(풀링), `cwm/train.py`(장치), `experiments/configs/planning/oneshot.json`, `experiments/configs/runs.yaml`, `experiments/4_evaluate/screens.py`(일부), `tests/test_counterfactual.py`, `THIRD_PARTY_NOTICES.md`, `README.md`, `.github/workflows/ci.yml`.
- 대조 기준: 논문 본문 식 (6), (7), (31), §4.2, §4.4, 부록 B.3, C, 표 3, 4, 5, 15.

## 0. 요약

- **핵심 규칙은 논문과 일치합니다.** `residual_transport`는 식 (6), `readout_transport`는 식 (7)과 같습니다. 단위 테스트가 "관측값 보존"과 "예측 효과 보존"을 확인합니다(코드 읽기로만 확인, 실행 안 함).
- 셔플 대조, 계획 탐색 설정(256, 32, 4), 성공 기준(0.2), 평가 게이트(95%, 60%)는 논문 설명과 맞습니다.
- **차이 하나(D5)**: ρ의 평균 방식이 논문 설명과 다를 수 있습니다(비율의 평균 대 평균의 비율). 값에 영향은 있을 수 있으나 크기는 원자료 없이 알 수 없습니다.
- **구현 장벽(D6, D9)**: 여러 기본 인자가 `device="cuda"`로 고정되어 있고, PassWater는 CUDA 9.2 환경이 필요합니다. 맥에서 그대로는 실행되지 않습니다.
- 공개 시점이 매우 가깝습니다(저장소 생성 2026-11-19). 이슈, 릴리스, 재현 보고가 아직 없습니다.

## 1. 대조 항목

### D1. 잔차 전달 규칙 (정합)

- `cwm/counterfactual.py`: `residual_transport(observed, predicted) = observed[..., None, :] + predicted − predicted[..., :1, :]`. 논문 식 (6)과 같고, 인덱스 0(실행한 행동)에서 관측값을 그대로 돌려줍니다.
- `readout_transport(readout, observed, predicted)`는 같은 식을 읽기 헤드 좌표에 적용합니다. 논문 식 (7)과 같습니다.
- `cwm/evaluate.py` 55행, 61행, 116행: 특징 공간 전달 후 디코딩(`feature`), 읽기 후 전달(`readout`), 앙상블(`*_ensemble`)에 같은 함수를 씁니다. 읽기 후 전달의 관측 항목은 `decode(head, observed)`입니다. 논문 식 (7)과 일치합니다.

### D2. 예측기 구조와 손실 (정합)

- `cwm/predictor.py`: 입력 토큰은 (임베딩, 행동)이고 `TransformerBlock` 여러 개(폭 `hidden[0]`, 헤드 4, MLP 비율 1). `hidden = [1024, 1024]`이면 블록 두 개입니다. 본문 §4.2의 "폭 1,024의 어텐션 블록 두 개"와 맞습니다.
- 목표는 `(z_{t+1} − z_t) / scale`이고, `scale`은 적합 데이터의 변위 RMS입니다. 식 (31)의 `r`과 같은 역할입니다.
- 예측은 `prefix[:, -1] + scale · out`입니다. 식 (32)의 `z_t + r · d_θ`와 같습니다.
- `experiments/configs/runs.yaml`: "final" 레시피는 `hidden [1024, 1024], weight_decay 0.3, anchor_only false, validation_interval 100`이고, "original"은 `[512, 512], 1e-4, anchor_only true, 500`입니다. 논문 표 7의 "orig."와 "sel."과 맞습니다.

### D3. 셔플 대조 (정합)

- `cwm/evaluate.py` 119행: `shuffled = np.roll(p, 1, axis=1)`. 축 1은 외란 추첨입니다(형태 `(parents, draws, actions, dims)`). 논문 "다른 외란 추첨의 결과로 바꿈"과 같습니다.

### D4. 계획 설정 (정합)

- `experiments/configs/planning/oneshot.json`: 교차 엔트로피 후보 256, 엘리트 32, 반복 4, 허용 오차 0.2, 특권 오라클 에피소드 300(PassWater 100), PassWater 특권 탐색은 후보 16, 엘리트 4. 논문 §4.4와 부록 C, 표 15의 n과 맞습니다.
- "오라클 without outcome"은 외란 분포 위의 기대를 Gauss–Hermite 6 노드로 계산합니다(`quadrature_nodes: 6`). 부록 C의 설명과 맞는지는 확인하지 않았습니다.

### D5. 상호작용 비율 ρ의 평균 방식 (차이, 모호)

- `cwm/metrics.py`의 `interaction()`: `rho = effect.mean() / noise.mean()`. `effect`와 `noise`는 부모별 외란 분산(ddof 1)을 부모 평균한 값입니다. 즉 **평균의 비율**입니다.
- `experiments/4_evaluate/screens.py` 80행도 `np.var(yp − y).mean() / np.var(yp).mean()`으로 같은 방식입니다.
- 논문 식 (14)와 §4.1은 쌍(a, a′)과 부모마다 정의된 비율을 쓰고, 표 3 캡션은 "부모와 질의 쌍에 대해 평균"이라고만 씁니다. 평균의 비율과 비율의 평균은 다릅니다. 부모별 ρ의 분산이 크면 차이가 커질 수 있습니다.
- 판단: 방법 설명이 모호합니다. 표 3과 표 4의 ρ가 평균의 비율 기준이라는 점을 논문에 적는 것이 좋습니다. 이 분석에서는 원자료가 없어 차이 크기를 재지 못했습니다.

### D6. 장치 기본값 (구현 장벽)

- `cwm/predictor.py` 70, 114, 133행: `device="cuda"`. `cwm/encoders.py` 34행, `cwm/readout.py` 41행, `cwm/model.py` 18행, `cwm/registry.py` 140행에도 같은 기본값이 있습니다.
- `cwm/train.py`는 `accelerator = "gpu" if device.startswith("cuda") else "cpu"`로 CPU 경로가 있습니다. 그러나 예측기 추론(`predict`)과 인코더 로딩의 기본 장치를 바꾸지 않으면 맥에서 실행이 바로 실패합니다. MPS 경로는 없습니다.
- 평가 수식 자체는 장치와 무관합니다(numpy 기반의 `metrics.py`, `counterfactual.py`).

### D7. 문서 표기 (불일치)

- README: "seven simulators", 논문 §4.2, 그림 5도 "일곱"입니다. 초록은 "네 시뮬레이터와 수중 차량 하나(다섯)"을 주 결과로 말합니다. 주 결과 집합은 다섯(HandReach, Ball-in-Cup, FetchReach, PassWater, BlueROV2)이고 나머지 두 개(Racetrack, Push-T)는 진단용이라 본문과 저장소가 맞습니다. 초록만 다른 말을 씁니다.

### D8. 평가 게이트 (부분 정합)

- `cwm/metrics.py`의 `gates()`: 상태와 외란 회복 ≥ 0.95, 오라클과 복사 < 0.60. 논문 부록 B.3, 표 5·6의 "†" 기준과 맞습니다.
- 표 9의 "오라클 참조 50 ± 8%" 설계 목표는 `gates()`에 없습니다. `screens.py`나 `experiments/4_evaluate/qualify.py`에 있을 것으로 보이나 확인하지 않았습니다.

### D9. 시뮬레이터와 자원 (구현 장벽)

- README: PassWater는 SoftGym/PyFleX(Python 3.7, CUDA 9.2)가 필요합니다. Push-T는 행동 정책을 CUDA에서 샘플링합니다. Racetrack은 NumPy 1.26.4를 권장합니다.
- 논문 §4.2, 부록 C, App. I: 전체 재현에 H100/H200에서 약 150 GPU-시간(데이터 25, 예측기 약 550회 적합 70, 평가 35, App. I 15)이라고 README가 밝힙니다.
- 디스크: 사실 수집 한 번에 약 5 GB, 평가 집단은 최대 70 GB, BlueROV2 128k는 80 GB입니다.

### D10. 부트스트랩과 구간 (정합)

- `cwm/metrics.py`: `_interval`은 2.5, 97.5 분위수, `seed_parent`는 512 다중 추출(부모 × 시드). 논문 "부모 및 시드 부트스트랩 95%"와 맞습니다.
- 이 코드 경로가 표 5의 구간을 만들었는지는 실행하지 않아 확인하지 못했습니다.

### D11. 테스트와 CI (미실행)

- `tests/`에 단위 테스트 7개 파일, `.github/workflows/ci.yml`에 ruff 린트와 pytest. 우리는 실행하지 않았습니다. 테스트 파일의 내용만 읽었고, 잔차 전달 테스트(`test_counterfactual.py`)는 식 (6), (7)을 직접 검사합니다.

## 2. 라이선스와 제3자 자산

| 항목 | 라이선스 | 영향 |
|---|---|---|
| 공식 코드 (`cwm/`, `experiments/`) | MIT (Copyright 2026 Randall Balestriero) | 자유 사용 가능, 고지 유지 |
| `cwm/sims/vendor/pusht_env.py` | MIT (dino_wm 기원) | 고지 파일 포함 |
| `assets/bluerov2_meshes/*.stl` | LGPL-3.0 | 메쉬를 재배포하면 LGPL 조건 |
| MAE ViT-B/16 가중치 (논문 주 인코더) | CC-BY-NC-4.0 | **상업적 사용 불가.** 이 인코더로 학습한 예측기의 사용 범위도 제한 |
| DINOv3 ViT-B | DINOv3 License | 게이티드 접근 (`hf auth login`) |
| Ball-in-Cup 행동 정책 | 모델 카드에 라이선스 없음 | 재배포 불명 |

## 3. 논문 대 코드 차이 목록

| 번호 | 종류 | 내용 | 영향 |
|---|---|---|---|
| D1 | 정합 | 잔차 전달 식 (6), (7) | 없음 |
| D2 | 정합 | 예측기 구조, 손실, 스케일 | 없음 |
| D3 | 정합 | 셔플 대조 | 없음 |
| D4 | 정합 | 계획 설정 (256/32/4, 0.2, 300/100) | 없음 |
| D5 | 차이 (모호) | ρ는 평균의 비율, 논문 설명은 부모별 또는 쌍별 | 표 3·4 값의 작은 변화 가능. 방법 설명 보완 필요 |
| D6 | 구현 장벽 | `device="cuda"` 기본값 다수, MPS 경로 없음 | 맥 실행에 코드 수정 필요 |
| D7 | 문서 | "일곱 시뮬레이터" 대 초록 "네 + 하나" | 표기 차이. 주 결과는 같음 |
| D8 | 부분 확인 | 게이트 기준 일치, 설계 목표 50 ± 8%은 미확인 | 설계 보정 근거 확인 불가 |
| D9 | 자원 | GPU 150시간, 디스크 최대 80 GB, SoftGym 구버전 | 맥에서 전체 재현 불가 |
| D10 | 정합 | 부트스트랩 구간 방식 | 없음 (실행 미확인) |
| D11 | 미실행 | 테스트, CI | 실행 안 함 |

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 전체 재현은 불가능합니다. 핵심 규칙(잔차 전달)과 평가 수식은 이 맥에서 구현할 수 있고, 상태 기반의 축소판은 CPU나 MPS에서 만들 수 있습니다.**

| 작업 | 가능 여부 | 근거 |
|---|---|---|
| 잔차 전달 규칙과 평가 수식 구현 | 가능 (즉시) | D1, D5. 순수 numpy/torch 코드 |
| 표 15–17 재계산과 검정 | 가능, 완료 | `analysis/recompute.py` |
| 단순 토이(진자, 상호작용 비율) | 가능, 완료 | `analysis/toy_interaction.py` |
| 공식 저장소 실행 (설치, 테스트) | 조건부 | 새 가상환경과 패키지 설치는 승인 필요. CUDA 13 전제(README), D6 수정 필요 |
| 픽셀 세계모델의 HandReach 재현 (표 5) | 불가 | 예측기 학습은 GPU 전제. 8k 에피소드 수집은 디스크 약 5 GB, 학습은 GPU 시간 수십 시간 |
| BlueROV2 128k, Push-T, PassWater | 불가 | 디스크 80 GB, SoftGym(CUDA 9.2, Python 3.7), CUDA 샘플링 |
| 상태 기반 잔차 전달 (인코더 없음, MuJoCo 팔) | 조건부 (CPU, MPS 가능) | 설치 승인 후 시작. 학습은 MPS에서 시간 측정 필요 |

**노력 추정 (측정 전, 추정):**
- 공식 코드의 CPU/MPS 포팅과 추론 경로 확인: 1–2주 (D6 수정 포함, 측정 전 추정).
- 상태 기반 축소판(표 3의 ρ 재계산, 읽기 헤드 전달, 오라클 비교): 2–3주.
- 픽셀 재현(표 5, 6): GPU 확보 전에는 시작하지 않음. 확보 후 수 주.

이 추정은 측정 값이 아니라 판단입니다.
