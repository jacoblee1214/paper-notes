# TopoRetarget 논문과 공개 코드 대조 (정적 읽기)

- **대상 코드:** `wuji-technology/toporetarget`, 얕은 clone `repo/` (커밋 `039c54022016a882ef8d3a6d34b4578aee7cb51f`, 2026-10-26 "docs(readme): Updates the project page and paper links"). 재현하려면 `git clone https://github.com/wuji-technology/toporetarget repo && git -C repo checkout 039c54022016a882ef8d3a6d34b4578aee7cb51f`. 첫 공개 커밋은 `9c4c33e`(v0.1.0, 2026-10-23)이고 커밋은 3개입니다.
- **함께 읽은 것:** 강화학습 훈련 저장소 `wuji-technology/wuji-mjlab`의 릴리스 v2026.10.22(`b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9`) 중 `pen_spin` 과제의 설정 파일 12개(`config/wuji_hand2/env_cfgs.py`, `rsl_rl/ppo.py`, `mdp/rewards.py`, `mdp/terminations.py`, `pen_spin_env_cfg.py`, `docs/train.md` 등). `data/wuji_mjlab_b2b98d7/`에 따로 받았고 저장소 전체는 clone하지 않았습니다. toporetarget의 README가 이 릴리스와 호환된다고 적습니다.
- **라이선스:** toporetarget MIT, wuji-mjlab Apache-2.0, 손 모델 서브모듈 MIT (`THIRD_PARTY_NOTICES.md`).
- **방법:** 읽기만 했습니다. 설치, 실행, 시험, 가져오기를 하지 않았습니다. 서브모듈(`wuji-description`)은 clone에 없어 손 모델 파일은 읽지 못했습니다. 코드 분량은 `toporetarget/` 파이썬 약 9,600줄입니다.
- **읽은 파일(toporetarget):** `README.md`, `CHANGELOG.md`, `toporetarget/defaults.json`, `solver/{settings,retargeter,qp,laplacian,session,clip,sampling,selfcol,model,hand,landmarks,run}.py`, `robots/wujihand2_right.json`, `data/pen_spin/manifest.json`, 시험 파일들.

## 1. 저장소가 담은 것과 빠진 것

| 구성 | 저장소 상태 | 근거 |
|---|---|---|
| 상호작용 메시 풀이기(Delaunay, 거리 가중 라플라시안, 뼈 방향 초기화, 관통 제약, SQP) | **있음** (Wuji Hand 2 설정) | `solver/` (laplacian, retargeter, qp) |
| 손 모델 | 서브모듈로 분리 (`wuji-description` v2026.10.12). 얕은 clone에는 비어 있음 | `.gitmodules` |
| 가져오기 | GRAB, TACO, OakInk2, HOT3D, 키포인트 캡처 묶음 | `datasets/`, `docs/datasets/` |
| ContactPose, Ho-cap 가져오기 | **없음** | `grep -ri contactpose\|ho-cap repo` 결과 없음 |
| 기준선 (OmniRetarget, Mink, DexPilot, GeoRT) | **없음** | 같은 검색 결과 없음 |
| 지표 $E_{prec}$, $E_{align}$, 관통 비율(식 10–12) | **없음.** 저장소가 기록하는 지표는 손가락 끝 오차, 손-물체 관통, 접촉 프레임 비율 등 | `solver/run.py`의 진단 항목 |
| 예제 데이터 | 펜 돌리기 클립 1개 (`pen_spin_029`, 231프레임, 50 Hz, 원본 4.6 s, 78.7 fps) | `data/pen_spin/manifest.json` |
| 강화학습 추적 | toporetarget에는 **없음**. `wuji-mjlab`의 `pen_spin` 과제가 담당 | README "Training in wuji-mjlab" |
| 평가 데이터 (표 2의 Ho-cap, Pen-Spin 32개) | 저장소 밖 (wuji-mjlab 릴리스 묶음은 클립 40개, 5그룹으로 설명) | `wuji-mjlab docs/train.md` |
| 하드웨어 배포 | `wuji-mjlab`의 `deploy/pen_spin`, `deploy/reorient` | wuji-mjlab 파일 목록 |
| 변환기 | 풀이 결과 → wuji-mjlab 클립 (`examples/wuji_mjlab/`, 한 펜과 오른손만) | README |

## 2. 논문과 일치하는 것

표 3의 매개변수 12개가 `toporetarget/defaults.json`과 `solver/settings.py`의 기본값에 모두 있습니다(✔).

| 표 3 | 논문 값 | 코드 기본값 | 위치 |
|---|---|---|---|
| $N_o$ | 50 | `object.sample_count` 50 | settings.py `ObjectConfig` |
| $\lambda_{IM}$ | 500 | `mesh.laplacian_weight` 500.0 | `MeshConfig` |
| $\kappa$ | 30 | `mesh.edge_weight.exp_decay_k` 30.0 (모드 `exp`) | `EdgeWeightConfig` |
| $\lambda_{warm}$ | 1 | `warmup.weight` 1.0 | `WarmupConfig` |
| $\lambda_{bone}$ | 0.1 | `pose_prior.bone_dir_weight` 0.1 | `PosePriorConfig` |
| $\lambda_{smooth}$, $\lambda_{reg}$ | 2.5, 2.5 | `smooth.weight` 2.5 (두 단계 공용) | `SmoothConfig` |
| $\lambda_{base,pos}$, $\lambda_{base,rot}$ | 100, 1 | `wrist.zero_translation_weight` 100, `zero_rotation_weight` 1 | `WristConfig` |
| $\tau$ | 0.001 m | `penetration.tolerance` 1e-3 | `PenetrationConfig` |
| $b$ | 0.030 m | `penetration.hybrid.backstop` 0.030 | `HybridPenetrationConfig` |
| $w_s$ | $1.0\times10^5$ | `penetration.hybrid.slack_weight` 1.0e5 | 같은 곳 |

- 슬랙 상한 $b-\tau$는 `qp.py:223`의 `slack_max = backstop - tolerance`, 제약 $\phi_i+s_i\ge-\tau$는 `qp.py:351`의 `build_penetration_constraints`(행 `G=−n̂ᵀJ`, `h=φ+τ`)로 구현되어 식 8과 같습니다.
- 간선 가중치는 사람 쪽 꼭짓점에서 계산해 목표 라플라시안으로 고정하고(`retargeter.py:178`의 `build_laplacian_topology`), 행 정규화(`laplacian.py`의 `_assign_normalized_weights`)가 식 5와 같습니다. Delaunay는 프레임마다 사람 쪽 꼭짓점에서 다시 만들고 간선 길이 문턱은 기본으로 끕니다(`delaunay.use_edge_threshold: false`). 논문도 문턱을 적지 않습니다.
- 프레임 사이 매끄러움은 이전 프레임의 최종 해 쪽으로(`qp.py`의 `build_temporal_smoothness_cost`) 이어지고 식 8의 $E_{reg}$ 첫 항과 같습니다.
- 학습 단계가 없다는 서술("there is no training stage", README)은 풀이기 구조와 일치합니다.

## 3. 차이 (D1–D18)

**풀이기 (D1–D12)**

- **D1. 식 7의 $1/N_v$가 코드에 없습니다.** 논문 식 7은 $E_{IM}=\frac1{N_v}\sum_i\|\cdot\|^2$이고 $\lambda_{IM}$ = 500입니다. `qp.py:537`의 `build_laplacian_cost`는 모든 꼭짓점 행에 `sqrt(500)`을 곱한 잔차의 제곱합을 쓰며 `1/Nv`가 없습니다(`length_scale`은 기본 `None`). 이 QP에서는 모든 항이 같은 1/2 계수를 받으므로 항 사이 비율은 보존되고, 식을 literal로 읽으면 $E_{IM}$의 상대 가중치는 $N_v$ = 71배(유효 7.04 대 500) 작습니다. 표 3의 $\lambda_{IM}$ = 500은 코드 기준에서는 정규화 없는 값으로 읽힙니다. 장난감에서는 이 차이가 결과를 크게 바꾸었습니다(`03` V15).
- **D2. 뼈 방향 항(식 1, 2)의 형태와 기준점이 다릅니다.**
  - 식 1은 **인접한 두 뼈의 방향 차이** $(d^r_k-d^r_l)-(d^s_k-d^s_l)$를 맞춥니다. 코드(`qp.py:265` `build_edge_direction_system`, `landmarks.py`의 `mediapipe_chain_edges`)는 손목 기준 좌표에서 **뼈 20개 각각의 단위 방향** $d^r-d^s$를 맞추고(손목-MCP 5개 포함) 가중치는 `inv_count`($1/\sqrt{20}$)입니다. 차이를 쓰는 형태는 찾지 못했습니다.
  - 식 2는 이전 프레임의 **초기 추정** $\tilde q_{t-1}$에 매끄럽게 붙습니다. 코드는 같은 프레임 안에서 이전 프레임의 **최종 해**를 시작점과 매끄러움 기준으로 쓰고(`retargeter.py:540`, `q_prev.copy()`), 첫 프레임은 20회, 이후 프레임은 반복 1회로 돌립니다(`warmup.iters_first` 20, `iters` 1). 이 초기화 단계에도 관통 제약과 손목 사전이 들어갑니다.
  - 최종 단계의 $\lambda_{bone}$ = 0.1 항은 같은 뼈 방향 항입니다(`pose_prior`, `chain`, `inv_count`).
- **D3. 물체 표면 샘플링이 균일 추출이 아니라 클립 전체의 손가락 끝 위치를 쓰는 가중 추출입니다.** 논문 §3.3은 메시 $M$에서 $N_o$ = 50개를 뽑는다고만 적습니다. 코드 기본은 `sampler.mode: proxcurv`로, 면 면적에 `1 + 곡률 + 30 × 손가락 끝 근접(σ = 1 cm)`을 곱한 가중치로 뽑습니다(`sampling.py:109`). 손가락 끝 근접은 **클립의 모든 프레임**의 손가락 끝을 물체 좌표로 옮긴 점들에서 계산하고(`finger_tips_object_local`), 샘플은 클립마다 한 번 뽑아 모든 프레임이 공유합니다(`run.py:91`). 그래서 샘플 선택이 미래 프레임의 손 위치에 의존하고, 프레임을 순서대로 처리하는 스트리밍 사용(실시간)과는 맞지 않습니다. `FrameSolver`에는 샘플을 외부에서 넣는 경로가 있습니다.
- **D4. 로봇 베이스(손목)는 사람 손목에서 ±5 cm, ±45° 안의 보정입니다.** 논문 §3.1은 "최적 로봇 베이스 포즈 궤적"을 구한다고 적습니다. 코드는 사람 키포인트 0, 5, 9번에서 손목 프레임을 추정해 입력 손목 포즈로 쓰고(`clip.py:142`, `landmarks.py`의 `estimate_frame_from_hand_points`), 풀이기의 6자유도 손목은 그 위의 **보정량**입니다. 한계는 `wrist.translation_limit` 0.05 m, `rotation_limit` π/4이고(`qp.py:381`의 `_intersect_with_native`), 영점 사전 가중치 100/1(표 3)이 함께 있습니다. 한계값은 논문에 없습니다.
- **D5. 풀이 구조가 논문에는 "arg min"으로만 있습니다.** 코드는 프레임마다 순차 이차계획(SQP)을 돌립니다. 반복은 첫 프레임 20회, 이후 10회까지이고 단계 크기 상한 0.1(`solver.step_size`), 수렴 문턱 1e-3, 신뢰영역 축소 0.5배를 최대 3번(`max_trust_shrinks`), 풀이기는 DAQP입니다(`qp.py`, `retargeter.py:235`). 표 1의 4.70 ms가 이 반복 횟수 기준인지는 알 수 없습니다.
- **D6. 라플라시안은 기본으로 손목(palm) 좌표계에서 계산합니다.** 결론은 "물체 좌표계에서 상호작용을 보존한다"(§6)고 적습니다. 코드는 입력을 손목 좌표계로 옮겨 풀고(`session.py:84` `world_to_palm`), 물체 좌표계 옵션 `object.use_frame`은 기본 `False`입니다. 라플라시안 좌표는 회전에 등가이므로 두 좌표계의 결과가 손목 보정량만큼 다를 수 있습니다.
- **D7. 키포인트 대응에서 MCP와 엄지 CMC가 두 번째 링크(`link2`, 근위 외전 몸체)로 대응합니다.** 논문은 "대응하는 로봇 키포인트"라고만 적습니다. 코드는 PIP→`links[2]`, DIP→`links[3]`, 끝→팁 사이트이고 뿌리 랜드마크(1, 5, 9, 13, 17)는 `point_mapping.mcp_surrogate: link2`로 `links[1]`입니다(`hand.py:119`, `retargeter.py`의 `PointMapper`).
- **D8. 논문에 없는 손가락 자기 충돌 제약과 클립 전체 후처리가 있습니다.** 프레임별 QP에 손가락 사이 캡슐 여유 제약 행이 붙고(`selfcol.py:192` `FingerClearance`, `margin` 1 mm), 모든 프레임을 푼 뒤 **창 40프레임, 겹침 10, 최대 8회**의 슬라이딩 윈도 SQP로 손가락 관절 궤적을 다시 다듬습니다(`run.py:217`의 `optimize_trajectory`, `selfcol.py:411`). 이 후처리는 창 안의 미래 프레임을 함께 보고, 링크 위치 이동량(평균, 95분위, 최대 mm)을 진단으로 남깁니다. 논문의 식 8, 표 3에는 이 항목이 없고, 표 1의 풀이 시간과 관통 지표가 이 단계를 포함하는지 적혀 있지 않습니다.
- **D9. 참조 주파수가 다릅니다.** 가져오기가 만드는 표준 클립과 변환기의 출력은 50 Hz입니다(예제 클립 원본 78.7 fps → 50 Hz로 재표본). 논문 부록 표 6의 참조·제어 주파수는 20 Hz입니다. 20 Hz는 RL 단계의 값이고 리타게팅 단계의 주파수는 논문에 없습니다. 풀이기는 50 Hz 기준으로 다듬어진 반면(예: 변환기가 50 Hz가 아니면 거부) 표 6은 20 Hz라서, 어느 주파수에서 표 1, 2가 만들어졌는지 확인이 필요합니다.
- **D10. 평가·기준선·데이터가 저장소에 없습니다.** ContactPose와 Ho-cap 가져오기, OmniRetarget·Mink·DexPilot·GeoRT 구현, 식 10–12의 지표 계산이 없어 표 1과 표 2를 이 저장소로 다시 만들 수 없습니다. 기여 3("리타게팅 궤적, 과제 참조, 학습된 정책의 데이터셋")에 해당하는 것은 저장소에 예제 클립 하나와 릴리스 첨부 `toporetarget-assets.zip`(펜 돌리기 원시 캡처, 6.3 MB)이고, wuji-mjlab 릴리스에 클립 40개(5그룹)와 체크포인트 5세트가 있다고 문서가 설명합니다. 논문 A.4의 Pen-Spin 32개와 40개의 관계는 적혀 있지 않습니다.
- **D11. 관통 제약의 구현 표현이 논문의 지표와 다릅니다.** 제약은 MuJoCo `mj_geomDistance`로 손 충돌 지오메트리와 물체 충돌 지오메트리(펜은 원기둥 1개와 상자 2개의 합성, 일반 메시는 볼록 껍질) 사이 부호 거리를 재고, 5 cm 안의 쌍만 선형화합니다. 접촉 중으로 판정되지 않은 음수 거리는 건너뜁니다(`model.py:257` `query_hand_penetration`, `:285`). 평가 지표(식 12)는 "표본 추출한 손 표면 점과 물체 메시의 부호 거리장"으로 정의되어 있어(A.3) 풀이기가 쓰는 표현과 다르고, 지표 코드는 공개되지 않았습니다. 설정 검사는 `interaction_radius_m`가 $\tau+0.03$ m 이상일 것을 요구합니다(`settings.py:315`).
- **D12. 손 모델이 Wuji Hand 2입니다.** 논문은 "Wuji Hand"(그림 4, §5.2)라고만 쓰고 세대를 적지 않습니다. 공개 코드는 `wujihand2_right/left`와 `hand2/hand2_2` MJCF를 쓰며(20개 손가락 관절 + 6자유도 손목), 논문 실험의 손 버전은 이 저장소로 확인할 수 없습니다. 프로젝트 페이지가 말하는 Leap 손 설정은 `robots/`에 없습니다(Wuji Hand 2 좌우만).

**강화학습 설정 (D13–D18).** 아래는 공개된 `wuji-mjlab` pen_spin 과제의 설정(릴리스 v2026.10.22)과 논문 부록 표 4–6의 대조입니다. 이 과제는 "손목 고정 펜 돌리기, 카메라 피드백으로 실물 배포"를 위한 설정이며, **논문 실험에 쓰인 설정과 같은지는 확인할 수 없습니다.** 논문 이후에 정리된 배포용 과제일 가능성이 있습니다.

- **D13. 시간 설정.** 논문 표 6은 시뮬 스텝 0.01 s, 데시메이션 5, 20 Hz, 에피소드 20 s(400 스텝). 공개 과제는 스텝 0.005 s, 데시메이션 4, **50 Hz**, 에피소드 **60 s**(`pen_spin_env_cfg.py`).
- **D14. 신경망.** 논문 표 6의 actor [512, 256, 128], critic [512, 512, 256, 128]. 공개 과제는 actor (1024, 512, 256), critic (1024, 1024, 512, 256)(`rsl_rl/ppo.py`). 환경 4096, 반복당 40 스텝, 학습률 1e-4, 엔트로피 0.001, γ 0.99, λ 0.95, 에포크 4, 미니배치 32, 소프트플러스 가우시안, ELU, 관측 정규화는 일치합니다(✔).
- **D15. 보상 항.** 논문 표 4는 네 항(물체 8.0 [축 점 σ 0.04], 링크 1.0 [σ 0.025], 관절 1.0 [σ 0.1, 범위로 정규화], 부드러움 −0.01). 공개 과제는 이 네 항의 가중치 8.0, 1.0, 1.0과 σ가 같으나 부드러움이 **−0.1**이고, 논문에 없는 **물체 위치 항(2.0, σ 0.05), 토크 L2(−1.0), 지속 과전류(−4.0), 손가락 자기 충돌(−1.0)**이 있습니다(`env_cfgs.py`의 `cfg.rewards`).
- **D16. 종료 조건.** 논문: 위치 오차 5 cm 초과, 방향 오차 45° 초과, 축 점 하나라도 5 cm 초과, 높이 < 0.06 m, 속도 한계(표 4). 공개 과제: 위치 오차 5 cm, **샤프트 방향 오차 60°**, 수치 불안정(위치 3 m, 속도 80), 참조 끝. 높이 조건과 축 점 조건은 찾지 못했습니다(`terminations.py`, `env_cfgs.py`).
- **D17. 참조 시작과 커리큘럼.** 논문은 참조 상태 초기화로 **균일 무작위** 시작 프레임을 씁니다(§4, A.5.3). 공개 과제는 **적응형 구간 샘플링**(0.5 s 구간, 균일 혼합 0.2), 첫 프레임 정지 확률 0.15, 외란 커리큘럼(1,000 반복까지 0, 6,000 반복에서 최대), 행동 가림(마스크 확률 0.06)이 있습니다.
- **D18. 도메인 무작위화 값.** 일부는 일치합니다(물체 질량 0.4–1.6배, 링크 질량·관성 0.4–1.5배, 엔코더 편향 ±0.01 rad). 다른 것: 마찰(논문 배율 0.7–1.3, 공개 과제 절대 마찰 0.35–1.4 로그균일), PD 강성(0.75–1.5 대 0.9–1.1), PD 감쇠(0.5–2.0 대 0.8–1.1), 관절 전기자(0.75–1.3 대 0.5–2.0), 물체 질량중심(±3 mm 대 축방향 ±40 mm, 반경 ±0.5 mm), 외력(논문: ±0.25 N 단일 스텝 충격, 0.6–1.8 s마다 / 공개 과제: 중력 배수 1–2의 모멘트 외란과 손가락 끝 임의 힘). 관절 감쇠는 공개 과제에 항이 없다고 주석에 있습니다(컴파일된 감쇠가 0).

## 4. 구현 가능성 판정 (구현할 수 있는지)

**판정: 풀이기(리타게팅)는 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 CPU로 구현하고 돌릴 수 있습니다. 논문의 표 2(RL 추적 성공률)와 실로봇은 이 맥에서 재현할 수 없습니다. 표 1은 데이터와 기준선을 받는 것을 승인하면 CPU로 가능합니다.**

### 4.1 이 맥에서 되는 것

| 무엇 | 필요한 것 | 판정 |
|---|---|---|
| 공식 풀이기로 예제 클립 `pen_spin_029` 풀기 | 공식 문서 기준 한 코어 약 30초(231프레임). Linux x86_64만 지원한다고 적힘. 필요한 패키지(`daqp`, `scipy`, `mujoco ≥ 3.4`, `viser` 등)와 서브모듈 clone(손 모델, MIT, 저장소 크기 약 150 MB로 GitHub 메타데이터에 표시). `daqp`는 PyPI에 macOS arm64 휠이 있고 이 맥에는 mujoco 4.0이 이미 설치되어 있음 | **조건부.** 설치·실행·서브모듈 받기를 승인하면 가능할 것으로 예상(맥 지원은 미검증) |
| 식 1–8의 최소 재구현 | numpy, scipy, mujoco(충돌 질의). QP 풀이기는 scipy로 대체(작은 문제), 손 모델은 직접 작성하거나 MuJoCo Menagerie류를 별도 승인 후 | 가능. 장난감은 이 분석에서 이미 돌렸음(`analysis/toy_interaction_mesh.py`) |
| 표 1 지표(식 10–12) 계산 | ContactPose 데이터(받아야 함), 접촉 링크 정의, 기준선 4개 | **조건부.** 데이터·기준선 설치 승인 후. 평가 코드는 직접 작성(공개 없음) |
| 간선 가중치·크기 민감도 실험 | 예제 클립과 장난감(이 분석의 도구) | 가능 |

### 4.2 이 맥에서 안 되는 것

| 무엇 | 이유 |
|---|---|
| 표 2의 RL 추적 정책 학습 (방법당 정책 1개, 4096 환경, 반복 10,000 권장) | `wuji-mjlab` 훈련은 "Linux x86_64, NVIDIA GPU, CUDA 12.8" 필요(`docs/train.md`). 이 맥은 CUDA 없음 |
| 기준선(OmniRetarget 등)으로 만든 참조로 같은 정책 학습 | 같은 이유. 논문의 기준선 구현 설정도 공개되지 않음 |
| Wuji Hand 실물 실행 | 로봇 하드웨어, 카메라, 펜 태그 |
| 논문 표 2의 클립 32개 그대로 | Ho-cap 32개의 추출 목록과 Pen-Spin 32개가 공개되지 않음 (D10) |

### 4.3 공수 추정 (분석자 추정, 측정 전)

- 공식 풀이기 실행과 지표 기록: 반나절(설치 승인 후). 프레임당 시간은 실제로 재 봐야 합니다.
- 식 1–8의 독립 재구현(MuJoCo 관통 질의 포함): 약 1–2주. 공개 코드와 대조하며 하면 줄어듭니다.
- 표 1의 평가 코드와 기준선 포함 재생성: 약 1–2주(데이터 가져오기, 접촉 링크 정의, 기준선 4개 설치).
- 표 2의 RL 부분: GPU가 있는 곳에서 방법당 학습 수 시간–수십 시간 규모로 추정(공식 문서가 10,000 반복을 권장하고 GPU 시간은 적지 않음).

## 5. 주장 대조 (C1–C9)

판정 기호: ✔ 확인, ◐ 부분 또는 조건부, ? 확인 불가.

| ID | 주장 (위치) | 판정 | 근거·메모 |
|---|---|---|---|
| C1 | 한 벌의 매개변수로 모든 실험 (초록, §3.4, 표 3) | ◐ | 코드의 기본값 12개가 표 3과 같고 클립별 설정 없이 한 `defaults.json`으로 돌아가는 구조(§2). 다만 코드에는 논문에 없는 값(손목 한계, 샘플러, 자기 충돌)이 함께 고정되어 있음 (D3, D4, D8). ContactPose 실험의 설정은 확인 불가 |
| C2 | 접촉 정밀도·정렬이 모든 기준선 중 최고 (표 1) | ? | 표 수치는 재계산과 일치(`03` V2). 지표 코드와 기준선이 공개되지 않아 독립 재생성 불가 (D10) |
| C3 | 프레임당 5 ms 미만, 실시간 (§5.1) | ? | 코드에 프레임당 최대 10회 SQP(+초기화), 충돌 질의, 클립 후처리가 있고 README는 231프레임에 약 30초(상한 130 ms/프레임)라고만 적음 (D5, D8, `03` V9). 표의 조건 미기재 |
| C4 | 기여 3: 리타게팅 궤적, 과제 참조, 정책 데이터셋 공개 (§1) | ◐ | 저장소에는 예제 클립 1개. 릴리스 첨부 6.3 MB는 원시 캡처. wuji-mjlab 문서는 클립 40개와 체크포인트 5세트를 설명(받지 않음, 논문의 32개 클립과의 관계 미기재) (D10) |
| C5 | Wuji Hand에 정육면체 재배향과 펜 돌리기 제로샷 전이 (초록, 그림 4) | ◐ | 펜 돌리기 배포 경로(`deploy/pen_spin`, 카메라 AprilTag 포즈)와 정육면체 재배향 배포 폴더(`deploy/reorient`)가 있음. 정육면체 과제가 TopoRetarget 참조를 쓰는지는 읽은 파일에서 확인하지 못함. 성공률·시행 수는 논문에 없음 |
| C6 | 손 형태와 물체 크기를 바꿔도 재조정 없이 증강 (§5.3, 그림 5) | ◐ | 손은 `robots/*.json` 설정과 MJCF로, 물체는 메시로 바꾸는 구조(`HandSpec.from_config`). 저장소에 설정이 있는 손은 Wuji Hand 2 좌우뿐이라 다른 손은 설정을 만들어야 함. 사람 손과 로봇 손 크기 비교는 코드에 없음 (`03` V15) |
| C7 | RL 설정은 부록 표 4–6과 같다 (A.5) | ? | 공개된 훈련 과제는 시간 설정, 신경망, 보상, 종료, 무작위화에서 여러 곳이 다름 (D13–D18). 어느 쪽이 논문 실험인지 확인 불가 |
| C8 | 학습 단계 없이 프레임별 최적화 (§3) | ✔ | README와 풀이기 구조가 일치. 초기화 단계가 이전 최종 해를 이어받는 점은 다름 (D2) |
| C9 | 공개 코드와 프로젝트 페이지 (초록, 프로젝트 페이지) | ✔ | MIT 코드, 풀이기와 예제만. 평가·기준선·RL은 따로 (§1) |

## 6. 저자에게 물을 것 (게시 시 질문 목록)

1. 표 1의 "관통 > 2 mm 프레임 %"의 분모(파지 25개인지 프레임 수인지)와 풀이 시간의 측정 조건(CPU, 스레드, 클립 후처리 포함 여부).
2. 표 2의 성공률 평균 오차가 전체 클립인지 성공한 롤아웃인지, 학습 시드 수, 클립별 성공 목록.
3. 논문 실험의 RL 설정과 공개 `pen_spin` 과제의 차이(D13–D18)가 의도된 것인지.
4. 식 7의 $1/N_v$ 적용 여부(D1)와 식 1의 인접 뼈 차이 형태(D2).
5. Ho-cap 32개 클립의 추출 목록, Pen-Spin 32개와 공개 40개의 관계.
