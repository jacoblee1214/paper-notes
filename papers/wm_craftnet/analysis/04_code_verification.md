# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `https://github.com/sharpa-robotics/WM-Craftnet`, 커밋 `a1e825acaf50ade298e30b0b13c6671627af3a1e` (2026-10-29, `main`). `git clone --depth 1`로 `repo/`에 받았고(디스크 약 323 MB, 그중 .git 144 MB), 이 문서의 모든 줄 번호는 이 커밋 기준입니다. `repo/`는 복원하려면 `git clone https://github.com/sharpa-robotics/WM-Craftnet repo && git -C repo checkout a1e825acaf50ade298e30b0b13c6671627af3a1e`로 받습니다.
- **정적 읽기만 했습니다.** 코드를 실행하거나 import하거나 설치하지 않았고, 체크포인트, 데이터, SDK 바이너리를 받지 않았습니다. 저장소 안의 `.so` 바이너리와 SDK는 열지 않았습니다.
- 기준: 논문 v1의 식 1–10, 표 7·8, 부록 A. 공식 코드에는 기준선, 보류 물체, 49물체 구성, 실로봇 평가 스크립트가 없습니다(D16).
- 판정 용어: "일치" 논문과 코드가 같음, "다름" 값 또는 구성이 다름, "논문에 없음" 코드에 있고 논문에 설명이 없음. 코드의 기본값이 곧 논문 실험의 값이라는 뜻은 아닙니다(공개 체크포인트의 설정 파일은 별도로 대조).

## 1. 설치 요구사항과 라이선스

| 항목 | 내용 | 근거 |
|---|---|---|
| 운영체제, 장치 | Linux + NVIDIA GPU (CUDA 드라이버) | README "Requirements" |
| 소프트웨어 | Python 3.8, PyTorch 2.1.0 (CUDA 11.8), NVIDIA Isaac Gym Preview 4 (별도 내려받기, NVIDIA 라이선스), pytorch3d, hydra 등 | README, `install.md` |
| 배포 | Python 3.10, Sharpa Wave SDK(`deploy/sharpa_sdk/`의 .so 바이너리), RealSense, 실제 손 | `deploy/README.md` |
| 라이선스 | `LICENSE`는 Apache-2.0 (Copyright 2026 Sharpa Group). GitHub API 라이선스 칸은 NOASSERTION. `NOTICE`와 `THIRD_PARTY_NOTICES.md`: 제3자 코드(IsaacGymEnvs BSD-3, WMP Apache-2.0, in-hand-rotation MIT, dreamerv3-torch MIT, rl_games MIT), 로봇·물체 자산, SDK는 별도 조건 | 저장소 파일 |
| 체크포인트 | 허깅페이스 `SharpaIT/WM-Craftnet`의 `wm_craftnet_set_z.pth` 하나(약 400.6 MB, API 보고). 받지 않음 | HF API 메타데이터 |

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 논문 | 코드 | 판정 |
|---|---|---|---|
| D1 | 식 7: λ_r L_reward 항, §3.3 "Reward supervises the predictive model", 그림 1, 표 1의 보상 헤드 | 보상 헤드는 만들어지고 예측되지만 `wsm/configs.yaml:38-39`의 `reward_head.loss_scale: 0.0`가 `wsm/models.py:169-173`의 `_scales["reward"]`로 들어가 손실이 0이 됨(`models.py:240-244`). `build_wm_config`의 허용 키(`utils/world_model.py:63-83`)에 보상 가중치가 없어 작업 설정으로 바꿀 수 없음 | **다름.** 공개 코드에서 보상 예측은 학습 신호를 만들지 않습니다. 단 보상 헤드 가중치는 사전학습 이식 대상(`_WM_BODY_RESUME_PREFIXES`의 `heads.reward.`)에 들어 있음 |
| D2 | 식 4, 식 7: 촉각 c_t는 별도 인코더 입력이고 L_tac(λ_c) 재구성 항이 있음 | 월드 모델의 "prop" 입력이 한 스텝 관측 벡터 `last_obs_buf` 전체(`tasks/wm_craftnet_camera_rotation.py:2109`)이고 이 안에 촉각 접촉(`wm_craftnet_rotation.py:1656-1667`)과 명령 축(`:1668`, `spin_axis.repeat(1,8)`)이 들어 있음. 촉각 전용 인코더 가지와 별도 λ_c 재구성 항은 없고, 고유감각 디코더(`L_prop`)가 촉각 입력도 함께 재구성함. "촉각 예측" 헤드(`wm_tac_pred`)는 잠재에서 17개 링크의 접촉 여부를 시뮬레이터 힘(임계 0.01 N)으로 예측하는 별도 보조 헤드 | **다름(구성).** 논문의 c_t와 L_tac는 코드에서 prop 벡터와 보조 헤드로 나뉨. 또 월드 모델이 명령 축을 봄(논문은 월드 모델 입력에 축을 쓰지 않음) |
| D3 | 표 7 "Actor projection 512→64→32→16", §3.3 "16-dim feature" | 배포 설정 `isaacgymenvs/cfg/task/WMCraftnetRotation.yaml:185-186`와 공개 체크포인트 설정 `example_ckpt/wm_craftnet_set_z.yaml:279-282`는 `wm_latent_dim: 32`, `wm_feature_hidden: [256, 128]`, 즉 512→256→128→32. 키가 없을 때의 코드 기본값(`rl_games/algos_torch/network_builder.py:578-579`, `train.py:143`)은 16과 [64, 32] | **다름.** 논문 값은 코드 기본값과 같고 공개 설정 파일 값과 다름 |
| D4 | 식 1, 식 8: 정책 관측 o_t에 현재 잡음 깊이 D_t 포함, 표 8 "Depth image: Actor, WM" | 액터 순전파는 `obs_dict['obs']['obs']`(스택된 고유감각·촉각·목표·축)와 WM 특징만 이어 붙임(`network_builder.py:208-213, 366-370`). 깊이는 `obs_dict['obs']['depth']`에 있지만 액터 네트워크에 CNN이 없고 쓰이지 않음 | **다름.** 공개 코드의 정책은 깊이를 직접 보지 않고 WSM을 통해서만 받음 |
| D5 | §3.3, 표 1: 전체 WSM은 깊이, 고유감각, 촉각, 보상, 자세, 가치, 형상 헤드 | 공개 체크포인트의 설정은 `wm_pose_pred`, `wm_tac_pred`, `wm_value_pred`, `wm_obj_pred`가 모두 `false`(`example_ckpt/wm_craftnet_set_z.yaml:295, 301, 330, 335`). README: "legacy setup: proprioception + depth only (matches released checkpoint)". 저장소 기본 설정은 네 헤드가 켜짐(`WMCraftnetRotation.yaml:207-252`). `scripts/test_wm_craftnet.sh`의 `set_y` 분기가 가리키는 `wm_craftnet_set_y.pth`는 공개되지 않음 | **다름.** 공개 체크포인트는 보조 헤드가 없는 구성이고, 표 10의 "보조 헤드 없음"(Return 688.1)에 해당할 가능성이 있으나 확인하지 못함(체크포인트를 받지 않음) |
| D6 | 표 7: 깊이 시간 dropout 0.02 / 상관 0.95 | `WMCraftnetRotation.yaml:159`, 체크포인트 설정 `:205`가 `temporal_noise_p_dropout: 0.01`, 상관 0.95 | **다름** (0.02 대 0.01) |
| D7 | §3.2: 입력 깊이에 시간 상관 dropout, 가우시안 잡음, 작은 회전. "Depth is clipped and cropped" | `gaussian_noise_enabled: False`(`:162`), `discontinuity_invalidate_enabled: False`(`:155`), 크롭 `enabled: False`(`:146`, 값은 위 20/아래 8/왼 9/오 15 px). 켜진 것은 시간 dropout, 무작위 회전(±0.3°)뿐 | **다름.** 서술된 가우시안 잡음과 크롭이 기본 설정에서는 꺼져 있음 |
| D8 | 표 7: 리셋 위치 [0.035, 0.02, 0.0025] m, 리셋 회전 10° | z축: `[0.03, 0.015, 0.001]`, 5.0° (`WMCraftnetRotation.yaml:28-29`, 체크포인트 설정 동일). x, y축: `[0.01, 0.005, 0.0025]`, 회전은 상속(5.0°) | **다름** |
| D9 | 표 7·부록 A.3: 물체 질량 학습/테스트 [0.2, 0.4] / [0.1, 1.4] kg, "evaluated over a wider test range" | `tasks/wm_craftnet_rotation.py:1745-1756`에서 `is_test` 분기와 학습 분기가 같은 `[m_low, m_up]`을 씀. `WMCraftnetRotation.yaml:261-272`의 `test_m_low/up`는 "not currently used independently" 주석과 함께 쓰이지 않음 | **다름.** 평가에 더 넓은 질량 범위를 쓰려면 `m_low`, `m_up` 덮어쓰기가 필요하고 공개 스크립트는 그렇게 하지 않음 |
| D10 | 표 7: 보상 계수(토크 −0.003, 일 −0.003, 행동 −0.002, 제어 −1.0, 손 자세 1.5, …). 부록: "All axis-specific policies use the same … reward" | z축 설정은 표 7과 같음. x, y축 설정(`WMCraftnetRotationX.yaml`, `Y.yaml`)은 토크 −0.001, 일 −0.001, 행동 −1e-6, 제어 −0.001, 손 자세 0.5로 덮어씀 | **다름(축별).** 표 7은 z축 값이고 x, y축 값은 논문에 없음 |
| D11 | §4.1: 방법당 128 평가 에피소드의 평균과 95% CI. 지표 RotR "회전율", Return | `utils/recorders/metrics.py:16, 90-101`: CI = 1.96·SD/√n, n은 유효 에피소드(낙하로 10 스텝 미만에 끝난 에피소드는 제외), 집계 창 기본 1024(`wm_craftnet_camera_rotation.py:277-279`). RotR은 에피소드 평균 [clip(20·Δθ, −6.28, 2.0)·spin_coef](`:789-806`, `utils/spin_reward_utils.py:7` 스케일 20), Return은 에피소드 보상 합. 128을 지정하는 설정은 찾지 못함(스크립트는 환경 9개 또는 16개) | **논문에 없음/다름.** 에피소드 수, 창 크기, 제외 규칙, RotR 단위(스케일 20, 클립 2.0)는 논문에 없음 |
| D12 | 표 7: PPO LR 1e-4, 1024 환경, 미니배치 4096 | 학습률은 `lr_schedule: adaptive`, `kl_threshold: 0.02`(`WMCraftnetPPO.yaml:65-68`)로 KL에 따라 조정됨. 실행 스크립트 기본은 환경 256, 미니배치 1024(README가 밝힘, 논문 값은 환경 변수로 지정) | **논문에 없음(적응 학습률).** 환경 수는 README가 공개 |
| D13 | 표 7 무작위화: 마찰, PD 배율 [0.8, 1.2], 관측·행동 잡음, 외력, 중력 곡선 | 표의 항목은 일치. 추가로 관절 stiffness 로그균등 [0.75, 1.5] 무작위화(`WMCraftnetRotation.yaml:404-407`), 행동 이동평균 0.8, 관절 한계 0.9배(`trainLimit`), 초기 손 자세 "improved" 등이 있고 논문 표에는 없음 | **논문에 없음** |
| D14 | 부록 A.3: 손목 깊이 카메라 RealSense L515, 손 제어·카메라·촉각 모두 10 Hz | 배포 코드는 "D405 depth stream"(`deploy/examples/wm_craftnet_infer.py:7, 698, 769`), 카메라 30 fps로 640x480을 받아 96x72로 줄임(`:1084-1090`), RealSense 공간·시간 필터(공간 α 0.5, 시간 α 0.4, `:754-759`) 사용. README: 비CUDA SDK는 촉각 추론이 30 Hz | **다름/논문에 없음** (카메라 기종, 필터, 촉각 수신율) |
| D15 | 표 7·8: 시뮬레이션 촉각 임계 후 이진 접촉(시뮬 `TacThresh: 0.3`, 임계 무작위 ±0.05, 탈락 0.02, 지연 0.02) | 실제 배포는 힘 노름에 1/1.5를 곱한 뒤 0.05 임계로 이진화(`:805-858, 1097-1098`), 즉 원 힘 기준 약 0.075 | **다름(설정).** 단위와 센서 응답이 시뮬과 같은 힘 척도인지 논문에 없음 |
| D16 | 표 1, 3–6: 기준선(Open-loop, Blind RL, TD, RL w/o tactile, IHR), 보류 물체, 49물체, 실로봇 20물체, 복구 시험 | 저장소에는 WM-Craftnet 학습·평가(z, x, y축), 물체 URDF 19개(`assets/urdf/objects/`), 배포 예시만 있음. 기준선 구현, 보류 물체 구성, 49물체 목록, 복구 시험, 열린고리 재생은 없음. `set_y`의 학습 물체에 `flashlight_y_axis`가 있고(`wm_craftnet_rotation.py:350-352`) 논문의 y축 보류 물체에도 "large flashlight"가 있음(자산이 다른 크기인지 확인 불가) | **논문에 없음/코드에 없음.** 표 1, 3–6의 비교 대부분은 코드로 재현할 수 없음 |
| D17 | §3.4, 부록 A.1: 사전학습 후 인코더, 다중모달 RSSM, 깊이·고유감각 디코더, 보상 예측기를 이식, 월드 모델은 계속 적응 | `utils/world_model.py:34-39, 416-453`의 `resume_body_from`이 `encoder.`, `dynamics.`, `heads.decoder.`, `heads.reward.` 접두사만 이식하고 보조 헤드와 옵티마이저는 새로 시작 | **일치** |

### 2.1 일치하는 것

- 구조: 월드 모델 최적화는 PPO 경로와 분리되고 정책은 detach한 결정론적 상태 h를 받음(`wm_craftnet_camera_rotation.py:2123-2126`, 월드 모델 호출이 `torch.no_grad`, 월드 모델은 `WMA2CAgent.train_epoch` 뒤 학습, `learning/wm_continuous.py`).
- 월드 모델 하이퍼파라미터: deter/stoch/discrete 512/32/32, 은닉 512, 재귀 깊이 1, 리플레이 256행, 워밍업 10,000, 배치 16, 청크 32, 에폭당 10회 갱신, Adam 1e-4, 클립 1000, KL free 1.0, dyn 0.5, rep 0.1(`WMCraftnetRotation.yaml:185-198`, `wsm/configs.yaml`, `utils/world_model.py`).
- 깊이: 96x72, 최대 0.4 m, 가까움 = 큰 값으로 뒤집은 입력, 복원 목표는 "크롭만 한 깨끗한 깊이"(`wm_craftnet_camera_rotation.py:2085-2097`). 잡음 깊이를 입력, 깨끗한 깊이를 디코더 목표로 쓰는 손실(`wsm/models.py:235-237`).
- PPO: γ 0.99, GAE 0.95, 클립 0.2, 지평 16, 미니 에폭 4, 정책 [512, 256, 256] ELU(`WMCraftnetPPO.yaml`). 비대칭 비평가는 별도 중앙 가치 네트워크.
- 시뮬레이션 주기: dt 0.01667 x 제어 간격 6 = 0.1 s (10 Hz), 에피소드 500, 22 관절.
- 무작위화: 마찰 [0.5, 2.0], PD [0.8, 1.2], 관측 [0, 0.05], 행동 [0, 0.04], 외력 0.5/0.25/0.99, 중력 곡선 1° → 4°(0.05° 단계).
- 보상 계수(z축): 스핀 1.0, 최대 스핀 2.0, 속도 −0.1, 손가락 거리 0.1, 손 자세 1.5, 드리프트 100, 무회전 −50, 낙하 −50, 팁 힘 0.9/2.0/0.01.

## 3. 요약 해석

- 가장 큰 구조 차이는 D1(보상 헤드 손실 0), D2(촉각이 prop 벡터에 들어감), D4(액터가 깊이를 직접 보지 않음), D5(공개 체크포인트가 보조 헤드 없는 구성)입니다. 논문의 "예측 목표", "촉각 입력", "전체 WSM" 서술과 공개 설정이 어떻게 대응하는지는 저자 확인이 필요합니다. 논문의 실험이 다른 설정으로 학습됐을 가능성도 열려 있습니다.
- 평가 쪽에서는 CI와 RotR의 정의(D11)가 논문에 없고, 기준선과 보류 물체는 공개되지 않았습니다(D16).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문의 시뮬 학습·평가·실로봇 재현은 이 맥에서 불가능합니다. WSM 구조의 축소 재구현과 보고 수치 재계산은 가능하고 일부 완료했습니다.**

| 수준 | 판정 | 필요한 것과 근거 |
|---|---|---|
| 실로봇 (표 3, 6) | 불가 | Sharpa Wave 22자유도 손, 촉각 센서, RealSense, SDK(.so, x86 Linux 전용)가 필요. 저장소 밖 장비 |
| 공식 코드로 학습 (표 1, 4, 5) | 불가 | Isaac Gym Preview 4(Linux, NVIDIA GPU, Python 3.8)가 필요하고 맥에서 설치되지 않음. 환경 1024개의 월드 모델 리플레이만 CPU 메모리 약 13.5 GiB(`03` V9) |
| 공식 체크포인트로 시뮬 평가 | 불가 | 같은 Isaac Gym 요구. 체크포인트(약 400 MB)는 받지 않음. 공개 설정은 보조 헤드 없는 구성이라 표 1의 전체 WSM과 같지 않을 수 있음 |
| WSM(RSSM)만 단독 구현 (합성 또는 MuJoCo 깊이) | 가능 | torch 2.15(MPS/CPU), mujoco 4.0 설치됨. dreamerv3-torch식 RSSM은 작은 규모에서 CPU로 가능. 새 설치 없음 |
| 토이 T1 (목표 오염 대 깨끗한 깊이) | 완료 | `analysis/toy_clean_target.py` 약 1분 |
| 표 재계산 | 완료 | `analysis/recompute.py` 수 초 |
| 단순 손(MuJoCo) 회전 과제에서 WSM 문맥 PPO | 조건부 | 직접 짠 MuJoCo 손과 깊이 렌더가 필요. CPU PPO는 느림 |

**노력 추정** (측정 전의 추정):
- 월드 모델 상태 추정기 재구현과 합성 깊이 대조: 1–2주 (T1의 연장).
- MuJoCo 단순 손과 깊이 렌더, 정책 학습 포함 전체 파이프라인의 축소판: 4–8주.
- 논문 수준(22자유도 손, 49물체, 실로봇): 해당 하드웨어와 GPU 서버 없이는 평가 불가능. GPU 서버에서 공식 코드를 쓰면 설치와 z축 학습은 며칠(공식 스크립트 기준, 학습 시간은 측정 못 함).

## 5. 재실행

공식 코드는 실행하지 않았습니다. 이 문서의 대조는 `repo/`를 읽은 것이고, 수치 재계산은 `03`의 스크립트입니다.
