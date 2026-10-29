# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `hgupt3/eigendexplore` @ `5d57180` (2026-10-11 "Initial release", 커밋 1개, MIT). v1(10-06)과 v2(10-13) 사이에 공개되었습니다.
- **함께 확인한 것:** 벤치마크가 고정해 쓰는 외부 설정을 `gh api`로 읽었습니다.
  - IsaacGymEnvs `aeed298`의 DeXtreme PPO 설정
  - `hgupt3/simtoolreal` `fddff74`의 SimToolReal PPO 설정
- **범위:** GPU 실행 없이 코드 읽기, 공개 PCA 자산 계산, CPU 재계산과 toy만 했습니다. 학습 결과·체크포인트·로그는 저장소에 없습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| Eq. 2 Σ = diag(σ_q²) + Eᵀdiag(σ_e²)E | `packages/eigendexplore/src/eigendexplore/distribution.py`: `LowRankMultivariateNormal(mean, Bᵀ·σ_e, σ_q²)` | ✅ 정확히 같음. log-prob·entropy·KL(`policy_kl`)도 이 분포로 계산 |
| σ_q, σ_e 학습, E 고정 (Sec. 3.3) | `log_scale` 파라미터(eigen), 호스트 logstd(IID), `basis`는 buffer | ✅ |
| 관절 반범위 정규화 좌표 | `from_pca(..., coordinate_scale=half_range)`: 기저를 반범위로 나눔 | ✅ |
| σ_e,i ∝ √λ_i 초기화 | `spectrum_power` 인자(component_std의 거듭제곱) | ⚠️ 벤치마크마다 다름(D1, D2) |
| 총 분산을 Joint-Space와 맞추고 반씩 | `eigen_rms`, `iid_sigma` 인자 | ⚠️ DeXtreme과 README 기본값만 일치(D1) |
| 90% 보존 분산 k | `assets/catalog.yaml`(k: Sharpa 9, Wuji 9, Allegro 10, XHand 6, Schunk 5, Inspire 3), `spider/rank.py` | ✅ 공개 자산으로 다시 계산해도 같음(`03` §1) |
| 양손 거울 합치기 | manifest `construction: mirrored_pair` | ✅ |
| rl_games PPO 통합 | `eigendexplore/rl_games.py`, `third_party/patches/*-rl-games.patch` (eigen σ를 버퍼에 저장해 KL 계산) | ✅ |
| SPIDER: 후보 섭동만 교체, 하위 10% 가중 평균, σ 고정 (Eq. 3) | `benchmarks/spider/proposal.py`(SPIDER 기본 IID 표본에 eigen 섭동 더함), `sampling.py` | ✅ 구조 일치. 스케일은 D2 |
| 무작위 방향 대조 (App. B.2) | `sampling.random_orthonormal_rows`(고정 seed 20260820), `--set basis=random` | ✅ SPIDER에만 있음(D8) |
| 네 가지 행동 공간 + Δ 변형 (Eq. 1, 4) | `action_spaces/runtime.py`, `settings.py` | ⚠️ 세부가 다름(D7) |
| 벤치마크 3종 + 손 6종 | README: DeXtreme(Allegro), SimToolReal(Sharpa), SPIDER(여섯 손) | ⚠️ DexMachina 없음(D6) |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **SimToolReal preset의 탐색 총량이 두 배입니다.** `presets/simtoolreal.yaml`은 `iid_sigma: 1.0, eigen_rms: 1.0, spectrum_power: 0.5`이고, 호스트 초기 σ는 1(`sigma_init val 0`)입니다. 그래서 IID 부분은 Joint-Space와 같고, 그 위에 eigen 잡음이 더해져 관절당 초기 분산이 2.0이 됩니다(Joint-Space 1.0). 스펙트럼도 σ_e ∝ λ^¼로 평평합니다. 반면 `presets/dextreme.yaml`(√0.5, √0.5, 1)과 README 기본값은 논문 서술(총량 맞춤, 반씩, √λ)과 같습니다. 팔 7채널은 IID σ 1이고 eigen 잡음이 없습니다. | 논문 실험이 이 preset으로 돌았다면 SimToolReal의 +34%, 연장 학습, 실물 33 → 82%는 "구조"와 "두 배 탐색"이 섞인 결과입니다. toy에서 총량 차이는 결과의 부호까지 바꿨습니다(`03` §3.2, §5). 논문은 이 차이를 언급하지 않습니다. |
| D2 | **SPIDER의 스펙트럼은 σ_e ∝ λ입니다.** `sampling._powered_component_scale`의 주석이 "Spider's recorded convention uses eigenvalue power 1 (not std power 1)"이라고 씁니다. eigen RMS는 손별 "ARCTIC posture PCA의 full-rank joint RMS"(Sharpa 0.235 rad, `configs/hands.yaml`)에 맞춥니다. 기존 IID는 `iid_scale 1.0`으로 그대로 두므로 총 분산이 늘어납니다(가정을 두고 약 1.35배, `03` §2). cold start 자세도 ARCTIC PCA 평균이고, 두 방법이 같이 씁니다. | 서술(√λ, 총량 맞춤)과 다릅니다. 다만 SPIDER에는 같은 스펙트럼의 무작위 방향 대조군이 있어서, "크기만으로는 이득이 안 난다"는 근거가 이 벤치마크에서는 성립합니다. |
| D3 | 벤치마크마다 스펙트럼 지수가 다릅니다. DeXtreme √λ(마지막/첫 σ 비 0.23), SimToolReal λ^¼(0.48), SPIDER λ(0.053). 같은 "EigenDEXplore"라는 이름 아래 세 가지 분포가 쓰였습니다. | 데이터셋 비교(Fig. 7a, SimToolReal)와 ρ sweep(Fig. 7b, SPIDER)이 서로 다른 분포 위에서 이루어졌습니다. "90% 설정"이 다른 벤치마크로 그대로 옮겨진다는 근거가 약합니다. |
| D4 | **공개된 "EgoSuite 기저"는 세 데이터셋의 가중 혼합입니다.** 여섯 손 manifest 모두 `source_corpora`가 [ARCTIC, GRAB, EgoSuite], `dataset_weights`가 [0.6, 0.2, 0.2]입니다(목록 순서대로 대응한다고 가정). 원 시퀀스는 3,596개이고(fit 3,169개), 6 Hz, 계수 범위는 P1/P99입니다. 저장소 규칙(`AGENTS.md`)은 "손마다 fitted EgoSuite basis 하나, 원 데이터·학습 코퍼스·producer workflow는 싣지 않음, dataset selector 없음"입니다. | 논문은 "EgoSuite(10만 시간) 사전 분포가 GRAB, ARCTIC보다 낫다"(Fig. 7a)고 하는데, 공개 기저는 ARCTIC 비중이 가장 큰 혼합으로 보입니다. GRAB·ARCTIC 단독 기저가 없어 Fig. 7a, App. B.1은 재현할 수 없습니다. 다른 손에 쓰려면 리타겟과 PCA fitting을 직접 해야 합니다. |
| D5 | **SimToolReal 공개 프로토콜이 Fig. 3a와 다릅니다.** preset은 `success_tolerance: 0.075`, `curriculum_start_frames: 0`이고, 문서도 "native curriculum이 처음부터 켜져 7.5 cm에서 1 cm·8°로 조인다"고 씁니다. 논문 Fig. 3a는 9B frame 동안 6 cm·50°로 고정입니다. 패치에 `tolerance_curriculum_start_frames` 옵션이 있어서 바꿀 수는 있습니다. 병렬 환경 24,576개, SAPG 6그룹입니다. | 기본 실행으로는 Fig. 3a 수치를 재현할 수 없습니다. |
| D6 | DexMachina 실험 코드가 없습니다. README의 벤치마크 표는 DeXtreme, SimToolReal, SPIDER 셋입니다("the four benchmarks"라고 쓰지만 표에는 셋뿐). | Fig. 5와 서론의 "+28%"는 재현할 수 없습니다. |
| D7 | 기준선 행동 공간에 논문에 없는 세부가 있습니다. Eigen-Space는 q⋆ = PCA 평균 + Eᵀc이고, 행동 [−1, 1]을 계수의 P1/P99 범위로 비대칭 사상합니다. Eigen-Residual은 관절 목표에 "0.3 × (복원 자세 − 중심)"을 더합니다. Δ 변형은 초당 3 범위입니다. | Eq. 1과 다릅니다. Eigen-Space는 부분공간 안에서도 사람 범위 상자로 제한되고, Eigen-Residual의 eigen 기여는 0.3배로 줄어 있습니다. 기준선 결과를 해석할 때 필요합니다. |
| D8 | 무작위 방향 대조는 고정된 무작위 기저 **하나**이고, SPIDER에만 있습니다. | "사람 방향이 이득의 원천"이라는 결론의 근거가 무작위 기저 한 개와 SPIDER 한 벤치마크입니다. RL 벤치마크에서는 검증되지 않았습니다. |
| D9 | README가 시간 상관 변형(AR(1), `correlation=0.94`, 지속 약 16 step)을 "기본보다 나을 수 있다"고 권합니다. 논문에는 결과가 없고, 공개 벤치마크는 모두 0(논문 설정)을 씁니다. | 근거 없는 권고입니다. 쓰려면 직접 비교해야 합니다. |

## 3. 수치 재계산 (`table_check.py`, `pca_check.py`, `fig_digitize.py`)

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | SimToolReal +34% (Sec. 1, Fig. 3a) | 10.6 / 7.9 = +34.2% (seed 5) | ✅ |
| C2 | DeXtreme +40% (Sec. 1) | 12.3 / 8.8 = +39.8% (seed 5) | ✅ |
| C3 | 연장 학습 14.1 대 12.0 대 4.9 (Fig. 3b) | Joint 대비 2.88배, Eigen-Residual 대비 +17.5%. 방법마다 실행 1개. v1의 "larger gains in extended runs"는 v2 서론에서 빠짐 | ⚠️ 표본 1 |
| C4 | 실물 DeXtreme +41%, 최장 +28% (Sec. 4.5.2) | 8.45 → 11.90 = +40.8%, 32 → 41 = +28.1%. 방법마다 40회, 정책 1개, SD 미보고 | ✅ (검정 불가) |
| C5 | DexMachina +28% (Sec. 1) | Fig. 5 판독, 과제 보상만: 평균 0.504 → 0.648 = +28.6%(평균의 비) / +26.2%(비의 평균). 커리큘럼 있음: +5.9% | ✅ 과제 보상만에 한정 |
| C6 | SPIDER 평균 −16% (Sec. 1, Fig. 6) | 여섯 손 평균 −16.05%. DoF와 Spearman 0.94 (p = 0.005) | ✅ |
| C7 | Fig. 7b: ρ = 0.9에서 Joint 대비 −25% | 0.066 / 0.089 = −25.8% (Fig. 6 Sharpa −25.4%와 반올림 범위에서 일치). ρ = 0.5 eigen만은 +41.6% | ✅ |
| C8 | 실물 SimToolReal 33% → 82% (Sec. 4.5.1) | Fig. 8의 과제별 평균(0.42/0.71, 0.34/0.92, 0.20/0.76, 0.36/0.90)의 평균이 0.33 → 0.8225. 물체-궤적 조합당 10회, 정책 1개, SD 미보고 | ✅ (검정 불가) |
| C9 | k = 9(Sharpa), 6(XHand), 10(Allegro), Inspire 세 성분 90% | 공개 자산에서 같은 k. Inspire k = 3 누적 93.4% | ✅ |
| C10 | "이득은 상관 자체가 아니라 사람 방향에서" (Sec. 4.4, App. B.2) | SPIDER, 무작위 기저 1개로만 검증. toy에서도 같은 스펙트럼의 무작위 방향은 IID보다 나쁨(`03` §3.2) | ⚠️ 범위 한정 |
| C11 | 계산 규모 | RL 실험 합계 약 9.4×10¹¹ frame(그림에 나온 조건만). 연장 실행 1주 = 약 99k frame/s | 참고 |

## 4. 구현 가능성 판단

### 4.1 공식 코드를 그대로 쓰는 경우

| 대상 | 요구 사항 | 판단 |
|---|---|---|
| 방법 패키지 (`packages/eigendexplore`, 약 430줄) | Python 3.10+, torch(README는 CPU wheel로 설치) | **바로 사용 가능.** 공개 손 6종은 PCA 자산 포함(MIT). 다른 손은 `from_pca`에 직접 만든 PCA를 넣으면 됨 |
| CPU 예제 (`examples/ppo.py`, `sampling.py`) | torch CPU | 가능 |
| DeXtreme | Linux, NVIDIA GPU, IsaacGym(Python 3.8), 환경 16,384개, 실행당 2B frame | GPU 서버 필요 |
| SimToolReal | Linux, NVIDIA GPU, Isaac Sim/IsaacLab 2.3.2, CUDA 12.6, 환경 24,576개. 9B frame은 약 25시간(1주 = 60B로 환산한 추정), 60B는 약 1주 | GPU 서버 필요. Fig. 3a를 재현하려면 프로토콜 수정 필요(D5) |
| SPIDER | Linux, NVIDIA GPU, MuJoCo Warp 3.7, CUDA. 과제 데이터(OakInk2 10과제, 32 MB)는 저장소에 포함 | GPU 서버 필요. 데이터 준비 부담은 작음 |
| DexMachina, 데이터셋 비교, 다른 손의 기저 | 코드·코퍼스·fitting 파이프라인 없음 | 불가(D4, D6) |

**판단:** 방법 자체는 바로 쓸 수 있는 형태로 공개되어 있습니다. 세 벤치마크도 고정 커밋과 패치로 재현 환경을 만들 수 있게 되어 있어서, 재현 정보는 좋은 편입니다. 다만 모두 NVIDIA GPU가 필요하고, SimToolReal은 기본 설정이 논문 서술과 달라서(D1, D5) 같은 수치를 기대하려면 설정부터 맞춰야 합니다.

### 4.2 이 맥(Apple M4 Pro, CUDA 없음)에서

| 단계 | 가능 여부 | 근거 |
|---|---|---|
| 공개 PCA 자산 검산, 초기 공분산·toy·그림 판독 | ✅ 완료 | numpy, 설치 없음, 각 수 초 |
| 방법 패키지와 CPU 예제 | 가능할 것으로 보임(torch 설치 승인 필요) | README가 CPU wheel 설치를 안내함. MPS에서 `LowRankMultivariateNormal`(Cholesky 등)이 도는지는 확인하지 않음 |
| 내 MuJoCo 과제에 EigenDEXplore 잡음을 붙인 표본 최적화 | 가능할 것으로 보임(mujoco, torch 설치 승인 필요) | 공개 Allegro·XHand 등 기저를 그대로 쓸 수 있음. SPIDER 벤치마크 자체는 MuJoCo Warp·CUDA 그래프를 써서 불가 |
| DeXtreme, SimToolReal 학습 | ❌ | IsaacGym·Isaac Sim은 CUDA 전용 |

### 4.3 아이디어만 다시 구현하는 경우

- **난이도: 매우 쉬움.** 핵심은 low-rank + 대각 Gaussian 하나입니다. torch에 `LowRankMultivariateNormal`이 있어서 수십 줄이면 되고, PPO의 log-prob·entropy·KL만 이 분포로 바꾸면 됩니다.
- **진짜 비용은 사전 분포입니다.** 내 손에 맞는 기저를 만들려면 사람 손 데이터(GRAB·ARCTIC 라이선스, EgoSuite 이용 조건 확인 필요)와 손별 리타겟이 필요합니다. 공개 코드에는 이 파이프라인이 없습니다(D4). 대안으로 텔레옵 로그의 관절 PCA를 쓸 수 있습니다.
- **비교할 때 반드시 맞출 것:** 탐색 총량(trΣ/d), 스펙트럼 지수, 그리고 같은 스펙트럼의 무작위 방향 대조군입니다. 공개 preset끼리도 이 셋이 달라서(D1–D3), 맞추지 않으면 결과를 구조의 효과로 읽을 수 없습니다.
