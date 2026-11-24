# 04. 공식 코드 정적 대조

대상: https://github.com/irom-princeton/AnyRank, 커밋 `4d16a97`(2026-10-01, MIT, 단일 커밋). **정적으로만 읽었고 실행·설치·import 하지 않았습니다.** 저장소는 `.gitignore`된 `repo/`에 shallow clone했습니다(280개 파일, 20 MB). 논문은 v2 기준입니다.

## 1. 구조와 범위

| 경로 | 내용 |
|---|---|
| `src/multitest/graphical.py`, `sequential_graphical.py`, `sequential_graphical_active.py` | 배치 그래프 검정, 순차 그래프 검정, 능동 표본 선택(알고리즘 1, 2에 대응) |
| `src/multitest/individual_test.py` | 실험 구동(`main`, `allocate_alpha`, 그래프 생성, 기준선) |
| `src/multitest/transivity.py`, `fixed_sequence*.py` | 추이 폐포(따름정리 1), 고정 순서(γ=∞) |
| `src/multitest/alpha_adaptive_nsm.py` | N-SCORE 래퍼: 재배분된 α로 컷오프 1/α를 갱신 |
| `scripts/`, `slurm/` | 실험 진입점(RoboArena, LBM, DROID-task, 합성, BT 부트스트랩) |
| `data/`, `lbm_real_data/`, `towel_policy_data/` | 실험에 쓰는 per-trial 데이터(RoboArena 14정책, LBM 성공 여부, DROID-task) |
| `tests/` | 추이 폐포, 모드 전달, α 재배분 회귀 테스트 |

**핵심 쌍별 검정(N-SCORE)은 저장소에 없습니다.** `nscore`, `statistical-comparison-core`, `sequentialized-barnard-tests`에 의존하고, 후자 둘은 README가 CC BY-NC 4.0이라고 밝힌 외부 패키지입니다. 따라서 p-process 계산, 정지 규칙의 내부는 정적 대조 범위 밖입니다. 저장소의 `_p_value`는 "실행 중 최솟값 p-value"로 쓰이며(`alpha_adaptive_nsm.py` 주석: 결정 규칙이 실행 중 최소 p-value와 현재 α를 비교), α는 결정 컷오프(1/α)에만 들어갑니다. 이는 논문이 말하는 "p-process는 그래프와 독립"과 같은 구조입니다.

## 2. 논문 대 코드 차이 (D 항목)

| # | 논문 | 코드 | 영향 |
|---|---|---|---|
| D1 | 표본 선택: j ∝ exp(p̂(j)), p̂는 정책이 "시행 수 적은 쪽"인 미기각 가설 중 최대 α (식 3, 4) | `sequential_graphical_active.py`의 `sequential_graphical_multitest`: 먼저 **가설**을 확률 ∝ exp(2α_i) − 1로 뽑고(`beta_selection = 2.0`, 주석 "Arbitrary — set this however you like!", 함수 인자 `beta`는 사용되지 않음), 그 가설에서 시행 수가 적은 정책을 시행. 완료된 가설은 가중치 0 | **큼.** α = 0.1 대 0.01의 가중치 비가 논문 식 1.09, 코드 10.96(`03` V3). 논문식은 거의 균등, 코드는 α에 거의 비례. 논문에 2.0이라는 값은 없음 |
| D2 | 알고리즘 1: 표본 선택에는 G_sim, 최종 판정은 고정한 초기 그래프 G를 배치로 실행 | 그래프 하나를 기각이 날 때마다 갱신하고 남은 쌍별 검정의 α를 그 값으로 다시 맞춤(`nsm_test.alpha = self.alpha[i]`). 최종 배치 재실행 없음 | 실행 중 최소 p-value가 단조라 결과는 같을 수 있으나(가능성) 논문 서술과 구조가 다름. G와 G_sim의 구분이 코드에는 없음 |
| D3 | 정리 1: FWER ≤ α*. Remark 3: 동률이면 2α, 보정 없이 보고 | 기본 `inference_mode = "ranking_no_ties"`(논문 기본), `"ranking"`은 자식 검정 α를 절반으로(README, `tests/test_inference_mode.py`) | 정리 1 문장에 동률 조건이 없고 조건은 Remark 3과 코드 README에만 있음 |
| D4 | 따름정리 1: 최종 판정에 추이 폐포 필터를 적용 | `allow_transitive=True`(기본)이면 **그래프 루프 안에서** 함의된 가설을 기각 처리하고 그 α를 다른 가설에 재배분. 모순 사이클은 폐포에서 제외(`tests/test_transivity.py`) | 인루프 재배분은 따름정리 1(사후 필터)의 증명이 직접 다루지 않는 동작. 유효성은 별도 논증 필요(`03` 참고, 자체 검증하지 않음). CLI `--transitive`는 `type=bool`이라 `--transitive False`도 True로 파싱됨 |
| D5 | γ ∈ {0, 1, 10, ∞} | `beta`가 γ. 스윕 `[0, 1, 10]`(LBM, RoboArena), 합성은 `[0, 1]`. γ=∞는 `fixed_multitest`(첫 비기각에서 중단) 또는 `beta=100`. 표 1, 그림 5, 6의 "AnyRank"가 어느 γ인지 논문에 없음 | 표 1 재현 시 γ 선택 필요 |
| D6 | 표 2: 합성 데이터 M=5–20, Nmax ∈ {500, 5000}, α*=0.1 | `scripts/tests/large_hypotheses.py`의 `synthetic`: 참 평균은 [0,1] 균등 난수 정렬, 점수는 N(µ, 0.15)를 [0,1]로 잘라 사용(이항 아님), 기본 정책 10개, `synthetic_reshuffle`은 `nruns = 1`(시드 = 실행 번호) | 논문은 합성 분포와 반복 수를 적지 않음. 표 2가 단일 실행 값일 가능성(저자 확인 필요) |
| D7 | 부록 F: 이항 합성, Nmax = 2000, 200회, α = 0.05, HO-F/HE-F/HO-I | `scripts/savings/savings_and_fpr_graphical_synthetic_data.py`: `Nmax = 1020`, `n_runs = 200`, `alpha = 0.05`, `n_policies = 5` | Nmax가 2000 대 1020. 다른 스크립트일 수 있어 단정하지 않음 |
| D8 | 부분 진행 점수(연속 유계 지표) | 진행 점수 실험은 `c = np.arange(11)/10`(11개 격자)로 N-SCORE 생성. RoboArena 진행 점수는 서로 다른 값 61개(예: 0.005, 0.33, 0.375) | 격자 밖 값의 처리는 외부 N-SCORE 구현에 달려 있어 확인 불가 |
| D9 | 기본 α* = 0.1 | `ExperimentConfig` 기본 α = 0.05, `graph_type = "soft_masked"`(앞 방향 0.9, 뒤 0.1 가중), `n_prior = 20`. RoboArena·LBM CLI는 α = 0.1, `fully_connected`로 덮어씀. `soft_masked`는 논문에 없음 | 설정 기본값이 논문과 다름. 스크립트 사용 시 덮어쓰기 필요 |
| D10 | "마지막 시행까지" 사용 | 소진 조건 `evals >= len(data) - 1`: 마지막 한 개를 쓰기 전에 완료 처리(off-by-one) | 영향 1시행. 영향 작음 |
| D11 | 시행 예산 5106, 2450, 1200 (표 1) | `Nmax_total = 2553*2`(5106), 2450, 1200 (`roboarena*_bradley_terry_*`) | 일치 |
| D12 | 보류 평가 20회/정책이 사전, 나머지로 검정 | `split_eval(..., nruns = n_prior = 20)`: 앞 20개가 사전, 나머지가 검정 데이터 | 일치 |
| D13 | BT 부트스트랩 5000회 (부록 F.1) | `--n_bootstrap` 기본 5000 | 일치 |
| D14 | 재현 가능성 | `scripts/irom_data.py`에 클러스터 절대 경로 `/n/fs/irom-testing/...` 하드코딩, `slurm/*.sh`도 동일. 데이터는 `data/irom_data`에 있음 | 그대로는 실행 불가, 경로 수정 필요 |
| D15 | 표 1, 그림 5–7은 한 번의 실행인지 반복인지 | 기본 실험은 파일 순서 그대로 재생(`roboarena4` 등). `*_reshuffle`(20–50회)도 있으나 논문이 어느 쪽을 보고하는지 표기 없음 | 판정 수와 시행 수의 변동 폭이 보고되지 않음 |

## 3. 일치하는 부분
- 식 (5): 가설 α 배분(`allocate_alpha`: α·exp(γd)/Σexp(γd))과 가중치(`allocate_weights`는 이웃 가설들의 d로 softmax, 자기 자신 제외)는 식 (5)와 같습니다. 0인 d는 0.01로 대체(논문에 없음).
- 식 (2): `sequential_graphical_test`의 α, G 갱신식은 Bretz et al.의 식과 같습니다.
- 가중 Bonferroni 기준선은 `allocate_alpha(..., beta = −γ)`로 부록 D의 exp(−γd)와 같습니다.
- 사전 상관 재계산과 수치는 `03` V8.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**알고리즘 자체(그래프 재배분, 표본 선택, 쌍별 e-process 검정)는 CPU만으로 구현·시험할 수 있습니다. 이 맥에서 이미 자체 e-process 토이를 만들어 돌렸습니다(`03` §2).** 논문의 실로봇·세계모델 실험은 로봇과 GPU가 필요하지만, 저장소가 per-trial 데이터를 포함하므로 오프라인 재생은 가능합니다.

| 구성 | 이 맥에서 | 필요한 것 | 노력(추정, 측정 없음) |
|---|---|---|---|
| 표·수치 재계산 | 완료 | 없음 | 완료 |
| 그래프 절차 + 추이 폐포 | 구현 가능 | numpy | 0.5–1일 |
| 쌍별 anytime-valid 검정(자체 구현) | 구현 가능(토이 완료) | numpy | 1–3일(분산 적응 베팅까지) |
| 공식 코드로 표 1, 그림 4–7 재현 | 조건부 | `nscore` 등 설치, CC BY-NC 의존(승인·라이선스 확인 필요), 경로 수정 | 1–2일, 이 분석에서는 실행하지 않음 |
| 세계모델 사전(Ctrl-World) 생성 | 불가 | GPU, 가중치 | - |
| 실로봇 평가 수집 | 불가 | 로봇 | - |
