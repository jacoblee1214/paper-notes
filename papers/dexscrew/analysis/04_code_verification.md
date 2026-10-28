# 04. 코드 검증: 논문 ↔ 구현 대조

- **대상:** `dexscrew` @ `3bde4e3`, `skill-teleop` @ `dec60d6`, `minbc` @ `b64c53f`(README가 지정한 논문 기준 버전)
- **범위:** 코드 읽기와 정적 재현만 했습니다. IsaacGym 학습과 실제 로봇 실행은 하지 않았습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| 단순화 물체 (회전 관절, 받침 고정) | `assets/screw/*`, `xhand_hora.py:1180` `fix_base_link=True` | ✅ |
| 행동 a = η·f + a_{t−1}, PD 토크 제어 | `configs/task/*.yaml` `action_scale 0.05 / 0.04167`, `pgain 3`, `dgain 0.01`, `controlFrequencyInv 10`(20 Hz) | ✅ |
| 보상 항과 가중치 (Table IV) | `xhand_hora.py:672-730`, `compute_hand_reward` | ✅ 가중치 일치. 단 D3 참조 |
| 과속 페널티 커리큘럼 7.5→15 rad/s | 드라이버 yaml `angvelPenaltyThres: [7.5, 15.0, 3e7, 6e7]` | ✅ |
| 종료 조건 (거리, 정체, 무접촉) | `check_termination` (`xhand_hora.py:878-915`) | ✅ 거리 임계값은 스크립트에서 덮어씀: oracle 너트 0.07, 드라이버 0.1, student 드라이버 0.15. 그 밖에 나사 관절 한계 근처에서 리셋하는 조건이 추가로 있음 |
| 랜덤화 (Table V) | yaml `randomization` | ✅ 수치 일치 |
| PPO 하이퍼파라미터 (Table VI) | `configs/train/*.yaml` | ✅ (lr 5e-3, horizon 12, minibatch 16384, γ 0.99, λ 0.95, clip 0.2, kl 0.02). 단 `max_agent_steps 1e10`이라 학습 종료 시점은 수동 |
| Sensorimotor 증류 L = ‖a−â‖² + ‖z−ẑ‖² | `algo/ppo/padapt.py:159-170` | ✅ (행동은 [−1,1]로 clip한 뒤 비교) |
| Student env 48개 (Table VII) | `scripts/*_student_padapt.sh` `numEnvs=48` | ✅ |
| 촉각 5×120×3 = 1800 | `minbc/configs/base.py` `xhand_tactile_input_dim 1800` | ✅ |
| BC K=5, H=16, hourglass, 200 epoch | `minbc` 기본값은 `obs_horizon 1`, `act_horizon 8`, `action_decoder mlp`, `num_epoch 30` | ⚠️ 논문 설정을 담은 실행 스크립트나 설정 파일이 공개되지 않음 |

## 2. 불일치와 버그

| # | 내용 | 영향 |
|---|---|---|
| **D1** | **Privileged info 인덱스 겹침 (버그).** `xhand_hora.py:1024-1057`에서 앞의 5개 항목(물체 위치·스케일·질량·마찰·무게중심)을 0–9 칸에 고정한 뒤, 나머지 항목을 `start_index = 0`부터 다시 배치합니다. 상위 코드 penspin(`allegro_hand_hora.py:836`)은 `start_index = 9`입니다. 매 스텝 동적 항목(물체 자세·선속도 등)이 0–9 칸을 덮어쓰므로, oracle은 **질량·마찰·스케일·위치를 보지 못합니다.** | `03` §2. 논문 Table III과 Fig. 7의 해석이 달라질 수 있습니다. 논문 실험에 이 버전이 쓰였는지는 확인할 수 없습니다. |
| D2 | `obj_angvel`, `hand_scale`, `nut_contact`는 차원만 잡혀 있고 `_update_priv_buf` 호출이 없습니다. 다른 항목에 덮어써지지 않은 칸은 항상 0입니다. | Table III에 있는 "너트 접촉 지시자"와 "물체 각속도"는 실제로는 전달되지 않습니다. |
| D3 | **λ_pc-z 항은 상수입니다.** 물체 point cloud z 범위가 3 cm를 넘으면 페널티를 주는데(`xhand_hora.py:714-717`), 물체 받침이 고정되어 있어서 point cloud가 에피소드 안에서 회전하지 않습니다(드라이버는 리셋 때 ±5° 기울기만 무작위). 즉 행동과 무관한 상수 보상입니다. 같은 위치에 계산만 하고 쓰지 않는 `reverse_penalty`도 있습니다. | 학습에 미치는 영향은 작습니다. 상수 음의 보상은 종료를 약하게 유도할 수 있지만 회전 보상(가중치를 곱하면 최대 10–24/step)에 비하면 작습니다. 본문에는 정의가 없습니다. |
| D4 | **Oracle이 물체 point cloud(100점)를 입력으로 받습니다** (`use_point_cloud_info: True`, `point_mlp [32,32,32]`). 본문과 Table III에는 없고, 저장소 README에만 "privileged information and point-clouds"라고 적혀 있습니다. | 물체 모양 정보가 oracle에 직접 들어갑니다. 서로 다른 너트 모양에 따른 gait 차이의 원천일 수 있습니다. |
| D5 | BC 데이터 변환(`skill-teleop/scripts/convert_to_bc_dataset.py`)에는 **성공 시행만 남기는 필터가 없습니다.** 폴더별 앞뒤 구간 자르기와 깨진 파일 제외만 합니다. | 논문이 "BC > replay"의 근거로 든 filtered BC 설명이 공개 파이프라인과 맞지 않습니다(수동 선별 가능성은 배제할 수 없음). |
| D6 | 배포는 겹치는 chunk를 **균등 평균**하는 temporal ensembling을 씁니다(`run_env.py:356`, `zmq_server_client.py:139`). | 본문에 없는 장치입니다. 매끄러움에 기여할 수 있습니다. |
| D7 | Oracle 학습 step 수: 본문은 1.5×10⁹, Table VI와 Fig. 7은 3×10⁹입니다. | 표기 불일치 |
| D8 | Table III은 손끝 2개 기준 97차원이지만, 코드는 `fingers_num = 5`라서 드라이버 84차원, 너트 127차원입니다(`analysis/priv_layout_*.json`). | 표기 불일치 |
| D9 | `screw_joint_friction`은 항상 0.2로 고정해서 기록합니다(`xhand_hora.py:285`). | 정보량이 없는 privileged 항목입니다. |

## 3. 재현할 때의 요구 사항
- **RL:** IsaacGym Preview(구형 CUDA, Python 3.8 계열)와 GPU 1장. 논문 기준으로 하루 이내에 학습됩니다. 자세한 내용은 `dexscrew/docs/install.md`에 있습니다.
- **텔레옵과 BC 배포:** UR5e, XHand(촉각 내장), Quest 2, RealSense가 필요합니다. 하드웨어가 없으면 재현할 수 없습니다.
- **BC 학습:** GPU가 있어야 하고 수집 데이터가 필요한데, 데이터셋은 **공개되지 않았습니다.**
