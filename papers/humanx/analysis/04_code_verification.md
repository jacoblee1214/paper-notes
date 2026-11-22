# 04. 논문 ↔ 공식 코드 대조: HumanX (arXiv:2602.02473)

대상은 공식 저장소 `wyhuai/HumanX`의 커밋 `02634bfd96f95f7f78413aa6a9c0130cdd2db8ce`(2026-10-22, "Update documentation")입니다. 얕은 복제(`repo/`, 226 MB, gitignore)를 **정적으로만** 읽었습니다. 코드를 실행하거나 import하거나 설치하지 않았고, `.pkl`은 pickle이라 열지 않았으며 `.npz`(allow_pickle 없이)만 읽었습니다. 파일 경로는 `repo/` 기준이고, 줄 번호는 이 커밋의 `xmimic/humanoidverse/envs/humanx/hoi.py`(3,558줄) 등입니다.

번호 규칙: **K**는 논문과 코드가 일치한 항목, **D**는 차이·누락입니다.

## 1. 저장소 구성

| 경로 | 내용 | 비고 |
|---|---|---|
| `xgen/xgen_editor/hoi_editor/` | XGen 웹 편집기 (`server.py` 2,628줄 + `app.js`, MuJoCo 백엔드) | 사람이 브라우저에서 접촉 프레임을 찍고 최적화·저장하는 도구 |
| `xgen/data/example/` | XGen npz 예제 3개 (농구 캐치 HOI, 상자 집기 로봇 전용·합성 결과) | `03` §4에서 점검 |
| `xmimic/humanoidverse/` | 학습·평가 (HumanoidVerse 기반 Hydra 설정), PPO·AMP·DAgger 변형, Isaac Gym/Isaac Sim/Genesis/MuJoCo 래퍼 | HOI 환경은 `envs/humanx/hoi.py` |
| `xmimic/data/motions/example/` | 예시 모션 pkl 4개 (농구 캐치-던지기, 페이크 슛, 상자(빈) 집어 놓기, 축구 킥 64개 증강) | 4개 기술만 |
| `xmimic/tools/convert_xgen_hoi_npz.py` | XGen npz → HumanX pkl 변환 | |
| `xmimic/tests/` | 단위 테스트 파일 13개 | 실행하지 않음 |

README의 범위 선언: "이 저장소는 시뮬레이션 쪽 학습·평가 코드를 공개한다. **완전한 하드웨어 배포, ROS/Unitree 제어, sim-to-real 보정 파이프라인은 포함하지 않는다**"(`xmimic/README.md`).

## 2. 일치한 항목 (K)

| 번호 | 논문 | 코드 | 일치 |
|---|---|---|---|
| K1 | PD 이득 k_p = Iω², k_d = 2Iζω, ζ = 2, ω = 10 Hz (식 11–12, 7.7) | `config/robot/g1/g1_29dof_model16.yaml`의 `dof_armature_prop`(I)과 `control.stiffness/damping`. 아래 계산 | **일치.** 그룹 5개(hip_pitch 0.025102, hip_yaw 0.010178, ankle 0.007219, shoulder 0.003610, wrist_pitch/yaw 0.002181)에서 I·(2π·10)²와 2·I·ζ·(2π·10)이 설정값과 상대 오차 10⁻¹⁰ 수준으로 일치(예: 계산 99.0984277823, 설정 99.0984277767). ω는 10 Hz = 62.83 rad/s로 해석돼 있음 |
| K2 | 행동 스케일 α = 0.25·τ_max/k_p (식 13) | `hoi.py:766–773`: `action_scale(0.25) * effort_limit / p_gain` | **일치.** 예: 엉덩이 피치 0.351, 발목 0.307, 어깨 0.439 |
| K3 | 정책 100 Hz (5절) | `config/simulator/isaacgym_hoi.yaml`: `fps: 200`, `control_decimation: 2` | **일치.** (실제 로봇 PD 1 kHz는 배포 코드가 없어 확인 불가) |
| K4 | 보상 합산 r = r^body + r^obj + r^iv + r^c + r^reg, 각 항 γ·exp(−λe) (식 1–3) | `hoi.py:1679–1712` 합산. 항마다 `exp(−MSE/σ)`, 가중치는 `reward_scales` | **구조 일치.** γ = 스케일(몸 위치 1.0, 손목 3점 1.6, 발 2.1, 관절 위치 0.75 등), λ = 1/σ. 값이 논문에 없고 코드에 있음 |
| K5 | 교사는 특권 관측(참조 몸, 물체), 학생은 고유감각 + 선택적 물체 (그림 13) | `config/obs/humanx/hoi_teacher_tracking.yaml`(참조 `dif_local_rigid_body_pos`, `local_ref_rigid_body_pos` 포함), `mocap/hoi_DAggerBC.yaml`(actor는 고유감각 + `local_curr_ball_position` + 이력 4), `NEP/hoi_DAggerBC.yaml`(물체 줄이 주석 처리) | **일치** |
| K6 | 학생 = RL 보상 + 교사 BC 손실 (4절) | `agents/dagger/DAggerBC.py:512–525`: PPO 손실 + `dagger_coef(1.0) × mean((a_stu − a_tea)²)`, 롤아웃은 학생 정책 | **일치** (DAgger식) |
| K7 | 가우시안 정책, 학습 가능 공분산 (4절) | `ppo.yaml`: MLP 512-256-256 ELU, `init_noise_std: 0.8` | **일치** |
| K8 | IV: 몸 점 − 물체 중심 (식 5 아래) | `hoi.py:1367, 1400`: `ig = body_pos_extend[hoi_interaction_bodies] − object_pos`, 월드 좌표 | **일치** |
| K9 | 조기 종료는 확률적, IV 오차 기반 (4절, 7.6) | `hoi.py:1156–1186`: 접촉 조건부 IV 종료, `torch.rand < p` | **일치** (세부는 D11) |
| K10 | MoCap 손실 두 모드, 거리 임계 샘플링, 물체 관측을 0으로 마스킹 (7.8) | `env/humanx/hoi.yaml`의 `packet_loss`: 임계 [0.05, 0.1] m, 먼 곳 p = 0.02·길이 1–2, 가까운 곳 p = 0.20·길이 1–10, `mask_type: "zero"` | **일치.** 구체 값이 논문에 없고 코드에 있음 |
| K11 | 무작위화: 마찰 U(0.5, 1.25), 링크 질량·PD 이득 0.9–1.1, 기본 관절 오프셋 ±0.01, 토크 RFI 0.1 (그림 14) | `domain_rand/domain_rand_low.yaml` | **일치** (차이 D12 제외) |

## 3. 차이와 누락 (D)

### 공개 범위

| 번호 | 논문 | 코드 | 비고 |
|---|---|---|---|
| D1 | 9개 실제 기술, 5개 영역, 실제 로봇 배포(MoCap, 100 Hz 정책, 1 kHz PD) | 시뮬레이션 코드만. **배포·ROS·MoCap·sim2real 없음**. 예시 모션은 농구 2, 축구 1, 상자/빈 1 (4개). **배드민턴·싸움·드리블·레이업·점프슛·줍기 데이터와 설정 없음** | README가 명시. 셔틀콕 URDF와 라켓 달린 G1 URDF는 있으나 배드민턴 설정은 없음 |
| D2 | XGen = 영상에서 로봇 HOI 데이터를 합성·증강하는 **파이프라인** | XGen = **로컬 웹 편집기**(`run_hoi_editor.sh`). 사람이 접촉 시작·끝 프레임을 찍고 앵커·물체 크기·궤적 방식을 고른 뒤 "Optimize contact stage"를 누름. README는 "여러 파일은 LLM 에이전트에 시키라"고 안내. GVHMR, UMR(또는 GMR), SAM 3D는 저장소 밖 | 과제별 수동 설정이 남음 (`03` C8) |
| D3 | 증강: 물체 크기·교체, 접촉 궤적 이동·스케일, 초기 속도 무작위화 → 영상당 50 클립 (3절, 5절) | **증강 생성기 없음.** `xgen/`, `xmimic/`에 데이터 증강 코드가 없음("augment"는 확장 몸 점 `num_augment_joint`에만 등장). 축구 예시 pkl(64 augs)은 결과물만 | 가장 중요한 누락: 일반화(GSR)의 큰 요인이 데이터 증강(`03` V3)인데 생성 코드가 없음 |
| D4 | 비접촉 궤적: **Isaac Gym 시뮬레이터**, 접촉 전은 시간 역전 + **감쇠 계수 반전** (3절) | 해석적 **포물선** 계산, 중력 [0, 0, −9.80665], 접촉 후 속도는 접촉 종료 뒤 5프레임 손목 평균 속도(README_details 4)에서 가져옴, 접촉 전은 "Inverted parabolic" 또는 "Hold still". 예제 데이터는 가속도 −9.8067, 잔차 0.0 mm(`03` §4) | 항력·감쇠 없음. 시뮬레이터 기반 합성 아님 |
| D5 | 접촉 최적화: **접촉과 force-closure 목표** (3절) | 팔 관절만 최적화, 손바닥-물체 표면 거리·손바닥 방향·손 침투·양손 접촉 균형 손실(`server.py` `_contact_loss`, `_optimize_contact_frame`). **force closure 없음**("force_closure" 검색 0건). 물체는 상자·구만 | 논문 서술보다 단순 |

### 학습 코드

| 번호 | 논문 | 코드 | 비고 |
|---|---|---|---|
| D6 | 학생: n개 교사 → 하나의 통합 학생, 입력에 skill label (4절, 그림 13) | README의 학생 명령은 **교사 한 개**(`DAggerBC`, `teacher_ckpt_path` 하나). 다중 교사 `Specialist2Generalist.py`(740줄)는 있으나 **어떤 설정 파일도 가리키지 않음**(`config/algo/`에 없음). 관측 설정 어디에도 `skill_label`이 없음(모션 라이브러리는 라벨을 읽고 없으면 0) | 다중 기술 통합 학생은 공개 설정만으로 재현 불가 |
| D7 | 몸 보상에 **AMP** 항 포함, "모든 실험에서" (7.3) | README 기본 교사 명령은 AMP 없음(`hoi_teacher`). AMP 설정(`hoi_teacher_w_amp`)은 별도 선택. 학생 설정은 주석에 "non-AMP trained" | 논문 설정이 기본값이 아님 |
| D8 | IV 보상: ‖u − û‖₂ (식 5), 확장 점 = 손가락 끝·라켓 모서리 (4절) | `_reward_interaction_graph`와 `_reward_ig_extended_hand`: **제곱 평균 오차**를 exp(−MSE/0.05)로 변환, 가중치 각 2.0. 확장 점은 손 8점(손목, 손 중심 + 0.165 m, 가지 1·2 각 0.165 m 오프셋: `g1_29dof_model16.yaml` `extended_hand_branch_link`). 손가락 끝·라켓 점 설정은 없음 | 식 5는 노름, 코드는 평균 제곱. G1에 손가락이 없어 가지 점이 대신함 |
| D9 | 접촉 보상: J차원 이진 벡터, 간선별 λ_j (식 6–7) | `_reward_contact_graph`: 스칼라 두 개(물체 접촉 불일치, 금지 몸 접촉)에 각각 exp(−|e|/0.05)을 곱함, 가중치 2.1. 금지 몸 14개(`hoi_no_contact_bodies`). **간선별 λ_j 없음**(σ 0.05 공통) | 손별·부위별 접촉 일치를 개별로 보지 않음 |
| D10 | (언급 없음) | 물체 보상(`object_related_rewards`), 접촉 오차 로깅, 외력 크기 전환이 모두 `episode 시각 < stand_ref_episode_length`일 때만 적용(`hoi.py:1695–1702, 1049, 1148`, `legged_robot_base.py:554`). 참조 모션의 "서 있는 프레임" 이후에는 물체 보상이 0 | 논문에 없는 시간 창 장치 |
| D11 | IVT: IV 오차로 확률적 종료 (4절) | **기본값 꺼짐**(`contact_conditioned_ig_termination: False`, `terminate_when_ig_status_far: False`). README 명령이 `contact_conditioned_ig_termination=true`로 켬: 접촉 프레임에서만, **양손 2개 몸**, 임계 0.2 m, 스텝당 p = 0.02. 별도로 물체 상태 이탈 종료(0.5 m, p = 0.01)가 기본 켜짐 | 표 1 "+IVT"에 해당하는 설정은 README 플래그. p와 임계가 논문에 없음 |
| D12 | IP: 로봇과 **물체**에 무작위 외력 (4절, 그림 14: 몸 ±50 N, 물체 ±2 N, 구간 U(1,2) s), 속도 교란 구간 U(1,3) s, 무작위화에 물체 크기·질량·반발 계수 | 외력은 **몸통(torso) 한 몸에만**, 범위 ±20 N(서 있는 구간 ±50 N), 구간은 `torch.randint(1, 2)`라 **항상 1 s**(상한 제외). **물체 외력 코드 없음**("apply_rigid_body_force" 호출은 로봇 하나). 속도 푸시 구간은 randint(1, 5)라 1–4 s 정수(논문 U(1,3) s), 속도 ±0.5는 일치. 링크 질량·마찰·PD 이득은 일치하고 질량중심 x는 ±0.025(논문 ±0.05). **물체 크기·질량 무작위화는 기본 꺼짐**, 공 반발 계수 0.81 고정. 투사체("proj") 교란이 있으나 논문에는 없음 | 논문의 IP 정의가 코드보다 넓음 |
| D13 | 환경 16,384개, 정책당 20,000 iteration (5절) | README는 4,096개. `num_learning_iterations: 1000000`(수동 중단), `num_steps_per_env: 24`. 학습률 1e-3 적응형(KL 0.01), clip 0.2, γ 0.99, λ 0.95, 엔트로피 0.01, 5 epoch × 4 미니배치, MLP 512-256-256는 **논문에 없음** | 논문은 PPO 하이퍼파라미터를 보고하지 않음 |
| D14 | 성공 기준: 골대 20 cm, 타격률, 화물 10 cm; GSR ±0.3 m 샘플링 (5절, 7.2) | 평가 콜백은 "종료되지 않은 비율"과 MPJPE만 계산(`analysis_plot_motion_tracking.py`). **SR/GSR 평가 코드, ±0.3 m 샘플러 없음**(정적 검색) | 표 1은 공개 코드로 재계산 불가 |
| D15 | (언급 없음) | 보상 패널티 커리큘럼(`reward_penalty_curriculum`, README에서 `degree=2e-5`), 관절·속도·토크 소프트 한계 커리큘럼, 모션 이탈 임계 커리큘럼(1.5 → 0.5–2.0) | 논문에 없는 학습 안정화 장치 |
| D16 | 기준선 SkillMimic(곱셈 보상), OmniRetarget(합산), HDMI(접촉 보상) (5절, 표 1) | 기준선 설정 없음. `multiplicative_reward: []`(주석 처리), 보상 세트는 `wyh` 하나 | 기준선 재현 불가 |
| D17 | 라이선스 | 루트 `LICENSE`: **Apache-2.0 + Commons Clause**(판매 금지: 호스팅·컨설팅 등 소프트웨어 가치에 기반한 상품·서비스 제공 포함). `xmimic/LICENSE`도 같은 문구. `xmimic/setup.py`는 `license="BSD-3-Clause"`. GitHub API는 NOASSERTION | 상업적 이용 제한, 표기 불일치. 연락처 이메일이 LICENSE에 있어 문서에 옮기지 않음 |

### 참고 (정적 관찰, 논문과의 차이 아님)

- `_reward_waist_dof`는 `dof_pos[:, 12:14]`(waist_yaw, waist_roll)만 벌점을 줍니다. waist_pitch(14번)는 제외됩니다(`hoi.py:3495–3499`).
- XGen 번들에는 `decoupled_wbc/.../g1_29dof_old.xml`과 G1 메시가 포함돼 있습니다. 디렉터리 이름으로 보아 다른 프로젝트에서 가져온 자산으로 보이나 출처와 라이선스는 확인하지 못했습니다.
- `tests/` 13개는 시각화·경로·SMPL 참조 등 보조 기능을 대상으로 하며 보상·종료 로직 단위 테스트는 파일 이름에서 찾지 못했습니다(실행하지 않음).

## 4. 구현 가능성 판정 ("구현할 수 있는지")

기준 환경: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, **CUDA 없음**. 저장소 루트 가상환경에 torch 2.15, mujoco 4.0, numpy, scipy, joblib. 새 설치와 공식 코드 실행은 하지 않았습니다.

**전체 재현은 불가능합니다. 부품 수준의 재구현과 데이터 점검은 가능합니다.**

| 수준 | 판정 | 필요한 것 / 이유 |
|---|---|---|
| 실제 로봇 재현 (표 2) | **불가** | Unitree G1, Noitom 모캡 14대, 배포 코드(저장소에 없음) |
| 공식 XMimic 학습·평가 (표 1, 3) | **불가(이 맥)** | Linux + Python 3.8 + NVIDIA GPU + Isaac Gym Preview 4 (`numpy==1.23.5` 고정). README는 Isaac Gym 경로만 안내. Isaac Sim과 Genesis, MuJoCo 래퍼가 코드에는 있으나 HOI 환경이 그 백엔드에서 도는지 정적으로 확인하지 못함(미검증, 새 설치 필요) |
| 표 1의 SR/GSR 재계산 | **불가** | 평가 코드와 체크포인트 없음 (D14) |
| XGen 편집기 실행 (예제 npz로 접촉 합성) | **조건부 가능** | numpy, mujoco ≥ 3.15, joblib(가상환경에 이미 있음), WebGL 브라우저. GPU 불필요. 단 **공식 코드 실행 승인**이 필요하고 사람이 UI로 조작 |
| 보고 수치 재계산, 정수 분모 검사, XGen 예제 점검 | **완료** | `03` V1–V15 |
| IV 보상·접촉 보상·IVT를 작은 토이로 재구현 (numpy/torch) | **가능** | 예: 2D 점 질량 팔 + 공, 또는 MuJoCo 공 던지기. 노력 1–3일(추정, 미측정) |
| G1 + 공을 MuJoCo·MPS로 소규모 XMimic 재구현 (기술 한 개, 환경 수백 개) | **조건부 가능** | G1 MJCF는 XGen 번들에 있음. 병렬 환경 없이 CPU MuJoCo로 PPO를 돌리면 20,000 iteration × 16,384 환경 규모는 불가능하고 학습 시간은 측정 전. 노력 3–6주(추정, 미측정), 결과가 논문 수치와 같다는 보장 없음 |
| 영상 → 로봇 동작 (GVHMR + 리타게팅) | **미확인** | GVHMR의 가중치 다운로드와 GPU 필요 여부를 확인하지 않음(승인 범위 밖) |

요약: **무엇을 믿을 수 있는지**(표 수치의 산술, 공개 설정 값, XGen 합성의 성질)는 이 맥에서 확인했습니다. **무엇이 성능을 내는지**(IV 보상, IVT, 증강의 기여)는 GPU와 증강 코드 없이는 검증할 수 없습니다.

## 5. 라이선스와 이용 조건 요약

| 대상 | 내용 |
|---|---|
| 논문 | arXiv 비독점 배포 라이선스(CC BY 아님). 논문 그림은 공개본에 쓰지 않음 |
| 코드 | Apache-2.0 + Commons Clause. 판매·유료 호스팅·컨설팅 제공 금지, 상업 이용은 저자 문의. `setup.py`의 BSD-3-Clause와 불일치 |
| 의존 | NVIDIA Isaac Gym Preview 4(별도 라이선스), GVHMR(라이선스 NOASSERTION), SMPL 계열 모델(별도 등록 필요 여부 미확인) |
| 가중치·데이터 | 정책 체크포인트 없음. 예시 모션은 저장소에 포함(소형) |
