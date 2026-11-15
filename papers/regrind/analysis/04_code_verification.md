# 04. 코드 대조: REGRIND 논문 ↔ 공식 저장소 (arXiv:2607.11874)

## 0. 대상과 방법

- **저장소:** https://github.com/yunhaif/regrind, 커밋 `38347a9e30184620df04e19c63c7c72378cae103` (작성자 시각 2026-07-14, 커밋 1개, MIT).
- **받은 범위:** `git clone --depth 1 --filter=blob:none --sparse`로 `source/`와 `scripts/`만 받았습니다. 이 과정은 `regrind/repo/`에만 생기고, 연구 저장소의 상태는 바뀌지 않았습니다. `data/`(시연 h5, 리타게팅 궤적 h5, 키포인트 npy, 영상)는 받지 않았습니다.
- **읽기만 했습니다.** 설치, import, 실행을 하지 않았습니다. 학습·평가·리타게팅 스크립트를 돌리지 않았습니다(요청 제약).
- **검색 기준:** 정적 검색(grep)과 해당 파일 직접 읽기. 줄 번호는 위 커밋 기준입니다.

## 1. 확인 범위

**읽은 파일:** `dexterous/mdp/rewards.py`, `mdp/terminations.py`, `mdp/actions.py`(일부), `mdp/commands.py`(증강, RSI, 관측 명령), `dexterous_env_cfg.py`(보상·종료·교육과정·시뮬레이션), `config/leaphand/*`, `config/wujihand/*`(환경 설정과 PPO 설정), `robots/free_leap_right_hand.py`(행동 크기, PD), `retargeting/leaphand_constants.py`, `retargeting/wujihand_constants.py`, `retargeting/retargeter.py`(일부), `utils/retargeting.py`(Delaunay와 라플라시안), `scripts/retarget_hand_object.py`, `scripts/rsl_rl/train.py`, `play.py`(일부).

**확인하지 못한 것:** Table 5의 도메인 무작위화 범위와 관측 잡음 상수 전부를 한 줄씩 맞추지는 않았습니다(`events.py` 일부만 봄). 실기 배포 코드, 시스템 식별 코드, 기준선(SPIDER, DexMachina, Mink) 코드는 저장소에 없습니다(§3 D7).

## 2. 대조표

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 객체 키포인트 근접 보상 가중치, 형태 | Table 4: 1.5, exp, σ = 0.02 | `dexterous_env_cfg.py:434–437` 1.5, exp, σ 0.02. **WUJI는 σ 0.015** (`config/wujihand/wuji_base_env_cfg.py:119`) | 불일치 (D1) |
| 객체 속도 보상 | Table 4: σ 1.0 (선속도), 3.14 (각속도) | `dexterous_env_cfg.py:439–447`, `exp(−err/σ²)` | 일치 |
| 손목 위치·자세 보상 | Table 4: 0.05, σ 0.02, 0.2 | `dexterous_env_cfg.py:449–457` | 일치 |
| 행동 크기 보상 | Table 4: 0.5, WUJI만 | `wuji_base_env_cfg.py:120` 0.5, LEAP 기본 0.0 (`dexterous_env_cfg.py:459–462`) | 일치 |
| 행동 변화율 보상 | Table 4: 1.0, σ 0.5 | `dexterous_env_cfg.py:464–467` | 일치 |
| 행동 한계 위반 보상 | Table 4: 1.0, σ 1.0 | `dexterous_env_cfg.py:469–472` | 일치 |
| 조기 종료 벌점 | Table 4: −10 | `dexterous_env_cfg.py:474–476` | 일치 |
| 물체 편차 종료 임계 | B.5: 0.15 m | 기본 0.15 m (`dexterous_env_cfg.py:492`). **4개 과제 설정 모두 1.0 m으로 덮어씀** (`leap_scissors_env_cfg.py:76`, `leap_screwdriver_env_cfg.py:86`, `wuji_scissors_env_cfg.py:107`, `wuji_screwdriver_env_cfg.py:106`) | 불일치 (D2) |
| WUJI 손끝-탁자 힘 종료 | B.5: 10 N | WUJI 기본에서 10 N 설정 (`wuji_base_env_cfg.py:122–124`), 그러나 두 WUJI 과제가 `hit_table = None` (`wuji_scissors_env_cfg.py:109`, `wuji_screwdriver_env_cfg.py:108`) | 불일치 (D3) |
| 물체 이탈·작업 공간 종료 | B.5 i) | `object_out_of_bounds` 있음 | 일치 |
| 손-물체 거리 종료 | B.5 ii) | 0.5 m (`dexterous_env_cfg.py:499–500`) | 논문에 수치 없음 (코드 값 기록) |
| 정책 행동 | §3.3: qtarget = q̄ + α⊙π | 관절 행동 크기 `RELATIVE_ACTION_SCALE = 3.2` × dt × decimation (`robots/free_leap_right_hand.py:14–16`, `wuji` 동일 값) = 3.2 × (1/120) × 4 ≈ 0.107 rad/단위 | 논문에 α 수치 없음. 코드가 값을 준다 (D10) |
| 관절 PD | 논문 수치 없음 | LEAP 손가락 kp 3.0, kd 0.1, 효과 한계 0.5 N·m (`robots/free_leap_right_hand.py:41–47`) | 논문에 없음 (기록) |
| 관측: 위상 φ | §3.3: 배우에 φ | `MotionCommand.command = self.phi` (`mdp/commands.py:460–461`), 관측 `generated_commands`로 포함 | 일치 |
| 관측: 속도 미사용 (배우), 손끝·관절 속도 (비평자) | §3.3 | 배우: 관절 위치 이력 2, 이전 행동 (`dexterous_env_cfg.py:177–197`). 비평자: `fingertips_pos`, `hand_joint_vel` (`:206–207`) | 일치 |
| 물체 자세 표현 | Rot6D | `object_ori` (`mdp/observations.py:55–`), Rot6D 여부는 코드 확인 안 함 | 부분 확인 |
| 시뮬레이션 주기 | B.1: dt 1/120, decimation 4, 9 s | `dexterous_env_cfg.py:557, 561`, `episode_length_s = 9.0` (과제별 설정) | 일치 |
| PPO 설정 | Table 7 | 스텝/환경 24, 학습률 1e-3, 적응 스케줄, KL 0.01, 5 epoch, 4 mini-batch, γ 0.998, λ 0.95, clip 0.2, 엔트로피 0.002, 최대 기울기 1.0, 1024-512-256-128, ELU, 초기 행동 잡음 0.5, 잔차 최종층 0 초기화 (`config/*/agents/rsl_rl_ppo_cfg.py`) | 일치 |
| 학습 반복 수 | 논문에 없음 | `max_iterations = 20000` (`rsl_rl_ppo_cfg.py:16`) | 논문 미기재 (코드 기록) |
| 중력 교육과정 | Table 6 | `curriculum_gravity` 20k–130k 단계, 값 표와 같음 (`dexterous_env_cfg.py:361–383`) | 일치 (단계 스텝은 `common_step_counter`, 벡터화 스텝) |
| 밀어내기 교육과정 | B.4: 130k부터, 170k에서 최대 | 130k에서 시작, 170k 최대 (`dexterous_env_cfg.py:385–`) | 일치 |
| 증강 범위 | §3.3, B.3: 위치 xy ±5 cm, z 0, yaw ±30° | 네 설정 모두 `traj_aug_translation_*` = ±0.05, `traj_aug_yaw_range` = ±0.5236 (`wuji_scissors_env_cfg.py:44–47`, `wuji_screwdriver_env_cfg.py:48–51`, `leap_scissors_env_cfg.py:41–44`, `leap_screwdriver_env_cfg.py:49–52`) | 일치 (증강은 네 과제 모두 켜짐) |
| 증강 블렌드 구간 | Eq. 11: tpickup → tuse | scissors start 0.281, use 0.4375; screwdriver start 0.135, use 0.292 | 논문에 수치 없음 (기록) |
| 증강 방식 | 강체 변환, 리타게팅 재실행 없음 | 손목 기준 강체 변환, `mdp/commands.py:502–554` | 일치 |
| RSI | §3.3: 기준 궤적 임의 위상 | `_sample_traj_aug`와 관절 상태 기록 (`mdp/commands.py:862–`), 적응 샘플링은 기본 꺼짐 (`adaptive_sampling_enabled = False`, `commands.py:1225`) | 일치 (적응 샘플링은 코드에만 있고 설정에서 안 씀) |
| 기준 초기화 섭동 | §3.3: "작은 섭동" | `enable_reset_perturbation` 기본 꺼짐 (`commands.py:1203`), 설정에서 켜지 않음. 물체 위치 ±2 mm, 자세 ±0.02 rad 섭동 코드는 있음 | 확인 필요 (D9) |
| 리타게팅 목적: Delaunay와 라플라시안 | Eq. 3–5, App. A.2 | `scipy.spatial.Delaunay` (`utils/retargeting.py:22`), 라플라시안 (`:78–87`), SQP 단계 (`retargeter.py`), MOSEK 기본 (`retargeter.py:74`) | 일치 |
| 라플라시안 가중치, 매끄러움 가중치 | λ 값 논문에 없음 | 라플라시안 가중치 10, 매끄러움 0.5 (`retargeter.py:207–208`) | 논문 미기재 (코드 기록) |
| 스텝 크기, 반복 | App. A.2: "small number of SQP iterations" | 스텝 크기 0.2 (`scripts/retarget_hand_object.py:225`) | 논문 미기재 (기록) |
| 객체 키포인트 수 | App. A.1: 객체당 50 | 100개 파일 (`retargeting/leaphand_constants.py:53–54, 65`, `wujihand_constants.py:60–61, 72`) | 불일치 (D4) |
| 객체 스케일 | D.1: LEAP 가위 2배, WUJI 1배 | 가위 LEAP 2.0 (`leaphand_constants.py:47`), 드라이버 LEAP 1.5 (`:59`), WUJI 1.0 (`wujihand_constants.py:54, 66`) | 일치 (드라이버 1.5배는 논문 본문에 없음) |
| WUJI 가위 특수 처리 | 논문 없음 | 특정 MANO 관절 가중치 0.1 (ids 1, 2, 3, 17), 관통 허용 0.002 (`scripts/retarget_hand_object.py:207–210`) | 논문 미기재 (D5) |
| 성공 판정 (가위 20°, 드라이버 2π) | D.1, Eq. 15–16 | 판정 코드 없음. `MotionCommand.metrics`는 추적 오차와 보상만 (`mdp/rewards.py:30–`) | 코드에 없음 (D6) |
| 기준선 SPIDER, DexMachina, Mink IK | D.2 | 없음 | 코드에 없음 (D7) |
| 실기 배포, 모션 캡처 인터페이스, 시스템 식별 | §4.1, App. E | 없음 (`ur5`, `deploy`, `mocap`, `sysid` 검색은 시연 파일 이름의 `zed_mocap`만 맞음) | 코드에 없음 (D7) |
| 시연 물체 자세 장비 | App. C: 모션 캡처 | 드라이버 시연 종류 `zed_mocap` (`wujihand_constants.py:69–70`) | 확인 필요 (D8) |
| 물체 상태 입력 | 실기에서 모션 캡처 | 관측은 시뮬레이터 상태 기반. 실기 경로는 없음 | 해당 없음 (D7) |

## 3. 불일치와 빈칸 (D1–D10)

- **D1. 객체 키포인트 보상 σ가 WUJI에서 다릅니다.** 논문 Table 4는 σ = 0.02를 씁니다. WUJI 기본 설정은 0.015로 덮어씁니다(`wuji_base_env_cfg.py:119`). 논문 Table 1과 2의 WUJI 결과가 어느 σ로 학습되었는지 본문에 없습니다.
- **D2. 물체 편차 종료 임계가 논문과 다릅니다.** 논문 §3.3은 "tight object deviation threshold"가 정책 이탈을 막는다고 쓰고, B.5는 0.15 m입니다. 0.15 m은 σ = 0.02 m의 7.5배입니다. 저장소의 네 과제 설정은 1.0 m으로 덮어써 사실상 끕니다. 설정 주석도 "avoid early termination due to deviation"이라고 씁니다. 실기 결과에 직접 영향이 있는지는 알 수 없습니다. 논문 설명과 공개 설정이 다르다는 점을 저자에게 물어야 합니다.
- **D3. WUJI 손끝-탁자 힘 종료가 두 WUJI 과제에서 꺼져 있습니다.** B.5 iv는 10 N 종료를 말합니다. 기본 WUJI 설정은 10 N을 두지만, 두 WUJI 과제 설정이 `hit_table = None`으로 지웁니다. 실기 안전 장치가 따로 있는지 논문에 없습니다.
- **D4. 객체 키포인트 수가 다릅니다.** 논문 App. A.1은 객체당 50개를 쓰지만, 저장소는 100개 파일(`scissors_bottom_100_filtered_handles.npy`, `screwdriver_100.npy`)을 씁니다. 리타게팅 결과가 달라질 수 있습니다.
- **D5. 리타게팅의 세부 조정이 논문에 없습니다.** WUJI 가위에서 MANO 관절 네 개에 가중치 0.1, 관통 허용치를 0.002로 올리고, 라플라시안 가중치 10, 매끄러움 0.5를 씁니다. 논문은 λ와 이 가중치를 밝히지 않습니다. 재현할 때 이 값이 결과에 영향을 줄 수 있습니다.
- **D6. 성공 판정 코드가 없습니다.** App. D.1의 가위·드라이버 판정은 공개 저장소에 구현이 없습니다. 따라서 Table 1–3의 SR을 저장소로 다시 셀 수 없습니다. `play.py`는 영상과 로그를 씁니다.
- **D7. 기준선, 실기 배포, 시스템 식별 코드가 없습니다.** SPIDER(공식 구현 링크만), DexMachina(자체 구현), Mink IK + RL 기준선 코드가 저장소에 없습니다. UR5e 배포, 모션 캡처 입력, 시스템 식별 코드도 없습니다. 따라서 표 2와 3의 실기 결과는 저장소로 재현할 수 없습니다.
- **D8. 드라이버 시연 장비 이름이 다릅니다.** 논문 App. C는 드라이버 물체 자세를 "motion capture system"으로 기록하지만, 저장소는 시연 종류를 `zed_mocap`으로 씁니다. 같은 장비일 수도 있으나 확인이 필요합니다.
- **D9. 기준 초기화 섭동이 설정에서 꺼져 있습니다.** 논문 §3.3은 "작은 섭동"을 말하는데, `enable_reset_perturbation`은 기본 꺼짐이고 설정에서 켜지 않습니다. 증강 범위(±5 cm, ±30°)가 같은 역할을 하는지는 논문이 밝히지 않습니다. 확인 필요입니다.
- **D10. 논문이 빼 둔 값들(기록).** 행동 크기 α(0.107 rad/단위), 관절 PD 이득, 총 학습 반복 20,000, 증강 블렌드 구간, 라플라시안 가중치. 재현하는 사람에게 필요한 값이라 기록합니다.

**일치하는 부분(요약):** 보상 가중치와 σ(WUJI σ 제외), PPO 설정 전부, 교육과정 값, 증강 범위, 위상 관측, 비평자 관측, 시뮬레이션 주기와 에피소드 길이, Delaunay와 라플라시안 목적. 특히 증강은 LEAP 설정에도 켜져 있어서, 증강 효과가 WUJI에만 있다는 의심은 맞지 않습니다.

## 4. 구현 가능성 판정 (이 맥)

**기기:** Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 있음, CUDA 없음. torch와 mujoco는 로컬 가상환경에 설치되어 있습니다.

**판정: 전체 재현은 불가, 부분 구현은 가능(조건부).**

| 경로 | 이 맥에서 | 근거 | 노력 (추정) |
|---|---|---|---|
| 공식 Isaac Lab 학습 (4096 환경, 20k 반복) | **불가** | 저장소 README는 IsaacSim 5.1.0과 IsaacLab 2.3.0의 공식 설치 절차를 따르라고만 씁니다. NVIDIA GPU가 필요하다는 것은 이 세션에서 공식 문서로 확인하지 않았지만, 이 맥에는 CUDA가 없어 어느 경우든 실행할 수 없습니다. | Linux + NVIDIA GPU가 있으면 설치 1–2일. 학습 시간은 GPU에 따라 다르고 측정이 필요함. 설정상 총 전이는 약 19.7억 |
| 공식 리타게팅 스크립트 (`scripts/retarget_hand_object.py`) | **조건부** | pydrake 설치(`[retargeting]` extra)와 MOSEK 라이선스 또는 Clarabel(무료) 대안이 필요합니다. 공식 `data/`의 키포인트 npy와 시연 h5가 있어야 합니다. 리타게팅 결과 h5가 저장소에 이미 있어서 학습 자체는 리타게팅 없이 가능합니다. | 설치와 확인 반나절–하루(추정). 데이터 다운로드 크기는 확인 필요 |
| 리타게팅 목적을 numpy·scipy로 다시 구현 | **가능 (부분)** | 라플라시안 목적과 Delaunay는 scipy로 됩니다. 준뉴턴 SQP 부분은 솔버 선택(Clarabel 등)에 따라 달라집니다. | 1–2주(추정) |
| MuJoCo로 축소 재구현 (한 손, 한 과제, 잔차 RL, 증강) | **조건부** | mujoco는 설치되어 있습니다. WUJI MJCF는 저장소 `assets/mjcf/`에 있고, LEAP URDF는 `assets/leaphand_urdf/`에 있습니다. 이 자산을 복사하려면 공식 저장소 자산 사용이 필요합니다. 접촉 많은 손의 MuJoCo 처리량은 측정하지 않았습니다. | 2–4주(추정). 처리량이 가장 큰 미지수 |
| 토이 수준 (기준 + 잔차 RL, numpy) | **가능** | 이 폴더의 `toy_retarget_prior.py`로 수행했습니다(`03` §3). | 완료 |
| 실기 (UR5e + LEAP/WUJI + 모션 캡처) | **불가** | 하드웨어와 모션 캡처가 필요하고, 배포 코드가 저장소에 없습니다(D7). | 해당 없음 |

**결론(요약):** 이 맥에서 논문 표 1–3의 수치를 재현하는 것은 불가입니다. 논문의 **방법 구조**(상호작용 보존 리타게팅 + 잔차 RL + 증강)는 numpy 수준의 토이와 리타게팅 목적까지는 맥에서 구현할 수 있습니다. 시뮬레이션 학습을 옮기려면 MuJoCo 재구현이 필요하고, 그 처리량은 아직 모릅니다.

## 5. 저자에게 확인할 질문

1. 표 2·3의 WUJI와 LEAP 실기 정책은 몇 개의 학습 시드에서 나왔는지, 10 시도는 정책 하나의 시도인지 적어 주실 수 있나요?
2. 물체 편차 임계는 학습에서 1.0 m이고 0.15 m은 어디에 쓰였는지 (D2)?
3. 두 WUJI 과제에서 10 N 탁자 접촉 종료를 끈 이유(D3)?
4. 객체 키포인트 수 50과 100 중 실제 학습에 쓴 것(D4)?
5. Table 1 SR의 판정 코드(가위 20°, 드라이버 2π)를 공개할 수 있나요(D6)?
6. DexMachina 자체 구현에서 충돌 후처리와 RL 설정은 본 방법과 어디까지 같나요?
7. 증강을 끈 같은 정책과의 실기 비교가 있나요(`03` §1.3)?
