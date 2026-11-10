# 04. 코드 대조: 논문 ↔ dmiller12/TemporalPolicy (e1e7ee5)

공식 저장소를 얕게 클론했습니다(`repo/TemporalPolicy/`, git 기록은 1개 커밋, `e1e7ee5`, 2026-08-18 "add bib", MIT). 실물 제어 코드와 데이터, 체크포인트는 없습니다. 학습과 평가는 실행하지 않았습니다(GPU와 데이터가 필요하고 설치를 하지 않기로 했습니다). 대조는 **읽기만으로** 했고, 수치 검증은 같은 수식을 옮긴 토이(`analysis/transport_toy.py`)로 했습니다.

## 1. 확인한 파일

| 항목 | 파일 (저장소 기준) |
|---|---|
| 정책과 손실 | `temporal_policy/policy/temporal_policy.py` (`conditional_sample` 97–109, `predict_action` 147–177, `compute_loss` 185–240) |
| 보간과 점수 | `temporal_policy/interpolants/interpolants.py` (`compute_score` 23–45, `compute_updated_drift` 47–50, `LinearForecastInterpolant` 52–87) |
| 짝짓기와 샘플링 | `temporal_policy/couplings/coupling.py`, `temporal_policy/dataset/robomimic_replay_dataset.py` (166–173, 295–331, `_obs_to_action` 374–413), `temporal_policy/common/sampler.py` (`SequenceOffsetSampler` 173–277) |
| 정규화 | `temporal_policy/common/normalize_util.py` (`get_range_normalizer_from_stat` 6–, `robomimic_abs_action_only_normalizer_from_stat` 110–) |
| 학습과 설정 | `train.py`, `temporal_policy/workspace/train_workspace.py`(224–225 손실, 87–90 디버그 덮어쓰기), `temporal_policy/config/train_temporal_policy_unet_image_workspace.yaml`, `temporal_policy/config/policy/temporal_policy.yaml`, `temporal_policy/config/training/training.yaml`, `temporal_policy/config/task/{robomimic_default,square_image_abs}.yaml` |
| 평가와 전달 비용 | `eval.py`(24–55), `temporal_policy/env_runner/robomimic_runner.py` (`compute_transport` 81–155, 롤아웃 루프 약 560–640) |
| 실물 | `temporal_policy/real_world/` (`video_recorder.py`만 있음) |
| 테스트 | `tests/` (8개, 대부분 유틸리티. 방법 대조에는 쓰지 않음) |

## 2. 논문 ↔ 코드 대조 (D-항목)

판정 기호: **일치**, **부분 일치**, **불일치**, **미공개**(저장소에 없음), **미검증**(읽기로는 확인 불가, 실행 필요).

| ID | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| **D1** | 학습 목표 | 식 3: `u = (x1 − x0) − ε√λ z`, 보간 `xλ = (1−λ)x0 + λx1 + ε(1−λ)√λ z` (식 1, Alg. 1) | `temporal_policy.py:210–222`: `it = (1−t)x0 + t x1`, `gamma = √t · ε(1−t)`, 목표 `it_dot + gamma_dot · z` 와 `gamma_dot = −ε√t` → `u = x1 − x0 − ε√t z` | **일치** | — |
| **D2** | 원천과 목표의 인덱스 정렬 | 본문 식: 원천 `s[t−H+1 : t]`, 목표 `s[t−H+1+d : t+d]`, `d = H − 2 = 14`(학위논문 부록 A.1). 본문 목표는 "미래 행동"이라고 씀 | 원천 창의 시작 오프셋 `horizon − n_obs = 14`(`dataset.py:166`), 패딩 `pad_before = n_obs − 1 + 14 = 15`(`dataset.py:173`). 따라서 원천 = `s[t−15 … t]`(16개 상태), 목표 = `a[t−1 … t+14]`(16개 행동). 같은 인덱스 `j`의 쌍은 **14 스텝** 떨어진 시점 | **일치**(d = 14). 표기는 모호(식은 상태 시점, 코드는 행동 시점) | 전달 비용이 "14 스텝 동작량"이 됨(`03` §3, §8) |
| **D3** | 상태와 행동의 같은 표현 | Sec. III-A: 상태와 행동은 같은 표현(관절 또는 말단 자세) | `_obs_to_action`(`dataset.py:374–413`): 위치 3 + 회전(사원수 → 6D, 보정 상수 곱) + 그리퍼 `gripper_qpos[..., 7:8]`(손가락 위치). 행동의 그리퍼는 명령(±1). 정규화만 같은 통계로 맞춤 | **부분 일치** | 그리퍼 차원은 상태(위치)와 행동(명령)이 다른 양. 실물(§IV-E)은 "그리퍼 속도"를 결합에 씀(D12) |
| **D4** | 행동 정규화 | Sec. III-B: 자유도별 표준화, `E‖x‖² = HD` | `normalize_util.py:6–20, 110–130`: 위치, 6D 회전, 그리퍼를 **[−1, 1] 범위 정규화**(학위논문 A.1과 같음) | **불일치**(가정) | `J_indep = 2HD`가 `HD(1+m)`로 바뀜(`03` §2) |
| **D5** | 추론 설정 | Table I: NFE 10, 결정적 샘플링(Sec. IV-C 결론) | 학습 설정(`train_..._yaml:30, 43`): `inference_sde_epsilon_coefficient: 0.0`, `num_inference_steps: 10` → **학습 중 롤아웃은 일치**. 반면 `eval.py:43–44`는 `inference_sde_epsilon_coefficient = 1.0`, `num_inference_steps = 101`로 덮어쓴다. 정책 기본값은 0.1(`temporal_policy.yaml:18`) | **부분 일치**(공개 평가 스크립트는 다름) | 공개 `eval.py`를 돌리면 SDE, 100 단계. 표 I의 NFE 10·결정적과 다른 결과가 나온다 |
| **D6** | 학습 epoch | Sec. IV-A: 1500 epoch (학위논문도 1500) | `training.yaml:8`: `num_epochs: 4500`. 과제 설정과 README 명령에 덮어쓰기 없음 | **불일치**(기본값) | 설정대로 학습하면 3배 길다. 논문 수치의 학습 길이를 확인할 수 없음 |
| **D7** | 학습 잡음 ε | 본문 주 결과에 없음(Fig. 3 오른쪽은 추론 SDE만) | `train_..._yaml:33`: `sde_epsilon_coefficient: 1.0`. 학위논문 5.1: ε = 1.0 고정 | **미공개**(논문 본문) / 일치(학위논문, 코드) | 논문만 읽으면 ε를 알 수 없음 |
| **D8** | 전달 비용과 직진성 정의 | 식 7: `Σ(‖Δp‖² + r·θ)`, 식 8: 위 값 ÷ `‖p_K − p_0‖² + r·θ` | `robomimic_runner.py:81–155`: 스텝 거리 **합(제곱 없음)** + `0.1 × 회전 각` 합, 직진성 = 총합 ÷ 직선 거리(제곱 없음). 값은 `unnormalize`한 행동 단위(m, rad), 정규화 공간 버전(`_standardized`)도 기록 | **불일치**(정의) | 직선 경로에서 식은 1/K, 코드는 1. 표 II의 1.02·1.08은 코드 정의와 맞음(`03` §4). 어느 정의를 썼는지 논문에 없음 |
| **D9** | 점수와 추론 드리프트 | 식 5: `∇log p = [λu − (x−x0)]/(ε²λ(1−λ))`, Alg. 2 줄 5: `b̃ = b + (g²−γ²)/2 · ∇log p` | `interpolants.py:23–50`: `As = 1/(tσ(β̇σ−βσ̇))`, `cs = β̇x + (βα̇−β̇α)x0`, `score = As(βb − cs)`, `drift = b + ½(g²−σ²)·score`, `σ = ε(1−t)`. `temporal_policy.py:103–104`에서 `g = inference_sde · (1−t)` | **일치** | 토이로 식 5 수치 일치 확인(최대 차 2e-9, `03` §7). `g = 0`에서도 보정항이 남는다(`03` §6) |
| **D10** | 적분 격자와 Euler 갱신 | Alg. 2: `x ← x + b̃Δλ (+ g√Δλ z)` | `temporal_policy.py:97–109`: `times = linspace(1, 0, num+1)`을 뒤집어 0→1, 단계 `num − 1 = 10`(설정 10). `mu = x + b̃·dt`, `x = mu + g·z·√dt` | **일치** | — |
| **D11** | 실행 수평 | Sec. III-E 끝, IV-E: 16개 예측, 8개 실행 | `n_action_steps: 8`(`train_..._yaml:41`), 실행 구간 `[n_obs−1 : n_obs−1+8]` = 인덱스 1–8(`temporal_policy.py:174–177`). 인덱스 0(`a_{t−1}`)은 버림 | **일치** | — |
| **D12** | 실물 제어와 결합 | Sec. IV-E: 7자유도 WAM, 10 Hz, 그리퍼 속도 결합, 500 Hz 보간(학위논문) | `real_world/`에는 `video_recorder.py`뿐. 제어 루프·보간·결합 생성 코드 없음 | **미공개** | 실물 결과는 코드로 확인 불가 |
| **D13** | 평가 횟수와 시드 | Table I: 체크포인트당 50회, 시드 3개 | `robomimic_default.yaml:20, 22`: `n_test: 50`, `test_start_seed: 100000`(모든 실행 공통, 같은 초기 조건 블록). 시드 3개는 실행 스크립트에 없음. `eval.py`는 100회, 시드 49 | **일치**(50회·공통 초기 조건), **미검증**(시드 3개) | 공통 초기 조건은 쌍 설계지만 에피소드별 결과가 공개되지 않아 쌍 검정 불가 |
| **D14** | 평가 주기와 체크포인트 집계 | 50 epoch마다 평가, Max·Average(마지막 10개) | `training.yaml:16`의 `rollout_every: 50`. Max·Average 집계 코드는 없음(wandb 로그에 의존) | **일치**(주기), **미검증**(집계) | — |
| **D15** | 네트워크 구조 | Sec. III-D, IV-A: 1D U-Net, 17M, FiLM, Spatial Softmax | `train_temporal_policy_unet_image_workspace.yaml:26, 28`: `use_initial: True`, `down_dims: [256, 512]`. 파라미터 수는 읽기로 셀 수 없음(torch 미설치) | **부분 일치**(구조), **미검증**(17M) | — |
| **D16** | 시간 표본 분포 | Alg. 1: λ ~ U(0, 1) | `config/policy/sigma_distribution/rand_uniform.yaml`(`RandUniform(0, 1)`, `common/distributions.py:82–90`) | **일치** | — |
| **D17** | 보조 손실 | 식 3(드리프트 손실만) | `compute_loss`가 `x1_loss`, `vel_loss`, `noise_loss`, `accel/vel_penalty`를 계산하지만 `train_workspace.py:224`는 `loss_dict["loss"]`(드리프트)만 역전파 | **일치**(로그용 보조 손실) | 논문에 보조 손실이 없어도 무방 |
| **D18** | 에피소드 경계 패딩 | 논문에 없음 | `sampler.py:270`: 시작 전 이력은 **첫 상태 복제**, 끝 뒤는 마지막 값 복제 | **미공개**(논문) | 시작 구간의 원천은 정지한 이력(`03` §9.3) |
| **D19** | 전달 비용을 잰 샘플 | Table II: Square ph, 50 에피소드 | 측정 코드는 있음(`robomimic_runner.py`, 롤아웃 중 `collector`에 경로 저장). 측정 시 설정(ODE 10 단계 또는 SDE 100 단계)은 확인 불가 | **미검증** | 표 II가 어느 샘플러에서 나왔는지 알 수 없음. DP 190은 확률적 샘플러 효과일 수 있음(`03` §4.3) |
| **D20** | 추론 시 `infill_overlap` | 논문에 없음 | `predict_action(..., infill_overlap=False)`(기본 꺼짐). 켜면 첫 두 스텝을 관측으로 고정 | **미공개**(논문), 기본 꺼짐 | 논문 결과에는 꺼진 설정이 쓰였다고 보임(미검증) |

## 3. 재현에 필요한데 논문에 없는 정보

1. **원천과 목표의 시간 정렬**(D2). 식 본문과 그림 2가 서로 다르게 적혀 있습니다. 코드를 봐야 d = 14와 인덱스 구조를 알 수 있습니다.
2. **학습 epoch**(D6). 1500(논문)과 4500(기본 설정) 중 무엇을 썼는지.
3. **공개 평가 설정과 논문 표 설정의 차이**(D5). SDE 1.0·100 단계(`eval.py`)와 결정적·10 단계(표 I) 중 어느 것을 썼는지.
4. **전달 비용 정의와 측정 샘플러**(D8, D19). 식과 코드 중 어느 것, 어떤 단계 수에서 쟀는지.
5. **학습 잡음 ε의 본문 값**(D7). 학위논문에는 1.0.
6. **행동 정규화 방식과 그리퍼 처리**(D3, D4). 분산 가정과 실제 정규화가 다릅니다.
7. **실물 결합 표현과 제어 루프**(D12). "그리퍼 속도"가 무엇인지, 500 Hz 보간은 어떻게 했는지.
8. **시드 수와 집계 방법**(D13, D14). Max와 Average 계산 스크립트, 에피소드별 결과.
9. **기준선 설정**. DP와 CFM의 학습 설정은 "Diffusion Policy 설정을 따른다"고만 씁니다. 두 방법의 파라미터 수(255M)를 확인하지 못했습니다.
10. **직진성 계산 대상**. 학습 중 롤아웃 경로인지, 평가 경로인지.

## 4. 구현할 수 있는지

### 4.1 공식 재현

**시뮬레이션(Robomimic):**
- 필요한 것: Robomimic 데이터(저장소 README 기준 전체 256 GB, 과제 하나는 이미지 데이터 수 GB 수준으로 추정. 실제 크기는 받아 보지 않아 미검증), `robomimic` 패키지와 MuJoCo 설치, 학습용 GPU.
- 학습 시간: 학위논문 부록에서 과제당 2.5–53 시간(A100 1장, 배치 128, 1500 epoch). 기본 설정(4500 epoch)이면 더 길다.
- 평가: 50회 × 체크포인트 수(50 epoch마다). 시뮬 렌더링은 CPU로 가능하지만 느리다.
- **판정: 공식 재현 가능**(GPU가 있으면), 이 맥에서는 학습이 불가능합니다(§4.2).

**실물:**
- 코드가 없고 하드웨어(Barrett WAM 2대, BarrettHand, FLIR 카메라)가 필요합니다. **재현 불가**.

### 4.2 이 맥(Apple M4 Pro, 통합 메모리 약 24 GiB, MPS, CUDA 없음, torch 미설치)에서

| 항목 | 판정 | 이유 |
|---|---|---|
| 저장소 코드 실행 | **어려움** | 코드가 `device: cuda:0`을 기본으로 하고 `torch.autocast`가 CUDA 전용입니다(`robomimic_runner.py` 롤아웃 루프). MPS로 바꾸는 패치는 가능해 보이지만 확인하지 못했습니다. 현재 torch가 없어 설치가 필요하고, 설치는 이번 작업에서 금지되었습니다 |
| 17M 모델 학습 | **조건부 가능**(패치와 설치 승인 후) | 파라미터가 작아 메모리는 충분합니다(24 GiB). 속도는 A100보다 느릴 것으로 예상되지만 측정하지 않았습니다 |
| 255M 기준선(DP, CFM) 학습 | **불가** | 메모리와 시간 모두 과합니다(추정) |
| Robomimic 평가 | **조건부 가능** | MuJoCo와 robomimic 설치가 필요. CPU 렌더링은 느립니다 |
| 실물 | **불가** | 하드웨어 없음 |
| 지연 측정(RTX 4080 대비) | **비교 불가** | MPS 지연은 CUDA와 다르고, 논문 지연은 RTX 4080 기준입니다 |

### 4.3 아이디어만 다시 구현

- **난이도: 중간**. 핵심 변경은 원천을 `x0 = 이력`으로 바꾸는 것입니다. 나머지(보간, 손실, 샘플러)는 표준 플로우 매칭 코드에 점수 보정 두 줄을 더하면 됩니다. 토이(`transport_toy.py`)에 닫힌 식 드리프트와 Euler 샘플러가 이미 있어, 같은 식을 신경망 드리프트에 연결하기만 하면 됩니다.
- 필요한 설계 결정: d(쌍 간격), 정규화(분산 1 또는 범위), 점수 보정 사용 여부, 추론 SDE 계수, 원천을 어떤 상태 표현으로 만들지(그리퍼 포함).
- 논문에 없는 것: 학습 ε 값, 17M 구조 세부(깊이, 커널은 학위논문에 있음), 기준선 설정의 세부.

### 4.4 요약 표

| 경로 | 가능 여부 | 필요 자원 | 이 맥에서 |
|---|---|---|---|
| 공식 시뮬 재현(Robomimic, 한 과제, 1 시드) | 가능(코드 공개) | GPU 1장(A100 급 권장), 데이터 수 GB, MuJoCo | 학습 불가, 평가는 조건부 |
| 공식 실물 재현 | 불가(코드 없음) | 로봇 하드웨어 | 불가 |
| 아이디어 재구현 | 가능(중간 난이도) | GPU 1장(시뮬 학습용) | 토이 수준까지 가능, 신경망 학습은 설치 승인 후 조건부 |
| 토이·표 재계산 | 완료 | CPU | 완료(`03` §1–§7) |

## 5. 권리 표기 (license verdict)

- **저장소 코드: MIT**(`LICENSE`, GitHub 메타데이터 확인). README가 Diffusion Policy(MIT)에서 파생되었다고 밝힙니다. 코드는 이 저장소에 클론만 했고 재배포하지 않습니다.
- **논문:** arXiv 비독점 배포(`arxiv.org/licenses/nonexclusive-distrib/1.0/`). **CC BY가 아닙니다.** PDF 1쪽에 IEEE 저작권 문구는 **없습니다**(`pdftotext -f 1 -l 1`로 확인: ©, copyright, IEEE, permission 문구 없음). 그래서 논문 그림과 표는 개인용 리포트와 PPT에만 넣고, 공개 산출물에서는 뺍니다. 공개물에는 직접 만든 그래프와 우리 문장만 씁니다.
- **학위논문:** 표지에 "© Dylan Miller, 2026"이 있습니다. 이 폴더는 학위논문의 수치와 문장을 짧게 인용하고, 그림은 쓰지 않습니다.
- **프로젝트 페이지:** 사이트 HTML에 CC BY-SA 4.0 링크가 있으나 템플릿 출처 표기와 함께 있어, 사이트 글의 재사용 조건인지 불분명합니다. 이 폴더는 페이지 글을 복사하지 않고 링크만 씁니다.

## 6. 주장 판정 요약

| ID | 주장 (논문) | 판정 | 근거 |
|---|---|---|---|
| C1 | 전달 비용을 거의 한 자릿수 줄였다(Abstract, Table II) | **수치는 일관, 해석은 과장** | CFM 대비 8.3배는 ρ₁₄ ≈ 0.76–0.88이면 가능(`03` §3). DP 대비 157배의 대부분은 결합이 아니라 측정·샘플러 효과(`03` §4.3). D2, D8 |
| C2 | 직진성 1.02 (Table II) | **코드 정의로만 일관** | 식 8로는 10단계 직선에서 0.1(`03` §4.1). 코드 길이 기준으로는 1(D8). CFM 1.08과의 차이 7%는 통계적으로 구별되지만 작다(`03` §1.4) |
| C3 | 19.1 ms 지연 (Table I) | **원천 효과와 분리되지 않음** | NFE 10으로 같은 CFM 63.5 ms 대비 3.3배 차이는 모델 크기 255M → 17M와 함께 나왔다(`03` §5). 재현 불가(RTX 4080) |
| C4 | 성공률이 SOTA와 맞먹는다(Abstract, Table I) | **대체로 확인(근사)** | 9셀 CMH 대 DP p = 0.69, 대 CFM p = 0.0014(Transport mh 한 셀이 대부분)(`03` §1.1). n = 150 가정은 근사 |
| C5 | 데이터 효율이 더 좋다(Sec. IV-D, Fig. 4) | **확인 불가** | 그림 한 장, 시드와 오차 막대 없음. 학위논문도 같은 곡선(TP·DP·CFM) |
| C6 | 1 단계에서도 성공률이 높다(Sec. IV-A, Fig. 3, 학위논문 표 5.3) | **수치는 확인, 해석은 회귀와 구별 안 됨** | 1단계 출력 = 조건부 평균(`03` §7, 차 2e-16). 회귀 기준선이 없음 |
| C7 | 결정적 샘플링이 효율적이고 다중 모드를 잃지 않는다(학위논문 부록 A.2) | **토이로 반증(결정적 모드 붕괴)** | 점 원천의 결정적 ODE는 평균 경로에 머묾(`03` §6.2). 다중 모드는 SDE에서만 나옴(§6.3). 부록 A.2는 한 장면의 정성 그림뿐 |
| C8 | 실물 50%(10/20), 집기 95%(19/20) | **미확인(기준선 없음)** | Wilson 95%: [0.30, 0.70], [0.76, 0.99](`03` §1.5). 코드 없음(D12) |
| C9 | 이력 원천이 직선 흐름을 만든다(Sec. IV-B) | **원인 해석 과장** | 점 원천에서 직선은 자동으로 나온다(`03` §6.2). CFM도 토이에서 거의 직선(코드 길이 기준 1.00–1.02) |
| C10 | J_temporal < J_indep(Sec. III-B) | **확인(조건부)** | 단위 분산이면 `J_TP/J_indep = 1 − ρ_d`, 따라서 ρ_d > 0이면 성립. 분산 1/3이면 `2m(1−ρ)/(1+m)`(`03` §2) |
| C11 | 시작 근처 이력 초기화가 민감하다는 한계(Sec. V) | **확인** | 보정항에 의한 시작 근처 오차 증폭 2.5배(`03` §6.4) |

## 7. 기록

- 저장소 커밋: `e1e7ee5ebf689d9f65c7453893dc35df5c40d158` (2026-08-18). 얕은 클론이라 이전 기록이 없습니다.
- 확인한 날짜: 2026-11-10.
- 실행하지 않은 것: 학습, 평가, 설치, 데이터·체크포인트 다운로드, 실물.
