# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `Holiday-Robot/FlashDexRetarget` @ `6f95e13` (2026-10-20), 서브모듈 `Holiday-Robot/rsl_rl_flashsac` @ `9822b1b`. 비교용으로 `facebookresearch/spider`의 `spider/postprocess/get_success_rate.py` @ `4471700`을 확인했습니다.
- **범위:** GPU 실행 없이 코드 읽기, 설정 대조, CPU 재계산만 했습니다. 공개 저장소에는 학습된 정책, rollout, 학습 로그가 없어서 Table I을 결과 파일로 재계산할 수는 없습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| 정책 하나를 전체 시연에 공동 학습, 병렬 환경에 레퍼런스 배정 (Eq. 1–2) | `src/envs/commands/motion_tracking/`, `sampling.mode: uniform` | ✅ 시작 프레임도 레퍼런스 위에서 균일하게 뽑습니다 |
| 물체 점군 128점, 손목 좌표계 (Sec. III-A) | `observations/motion_tracking.py: obj_point_cloud_wrist` | ✅ 점은 2,048개 꼭짓점에서 FPS로 뽑습니다(D6) |
| SDF 거리·법선, 128³ 격자, 60 cm | `utils/sdf.py`, `robot_links_obj_sdf` (palm + 손끝 5), `sdf.grid_n: 128`, `extent: 0.30` | ✅ |
| K = 10 미래 프레임 → 128차원 latent (Eq. 4) | `agents/modules/traj_encoder.py` (Conv1D, `aggregation: conv1d`), `ref_future_traj_wrist` (`n_future: 10`, `stride: 1`) | ✅ |
| 물체 점군 정합 보상 (Eq. 6): ρ = 0.058, k = 3, σ_obj = 0.1 | `rewards/motion_tracking.py: _obj_pcd_match_error_exp`, `reward_params: {topk: 3, norm_radius: 0.058}`, `scale: 10.0` | ✅ 식과 정확히 일치합니다 |
| 상호작용 보상 (Eq. 7): σ_int = 0.01, 손끝 5개 | `tracking_ref_mano_tip_distance_bound_exp` (`sigma=0.01`) | ✅ |
| 사람 손끝 거리 = 가장 가까운 꼭짓점까지 부호 없는 거리 | `retarget/build_motion_pt.py: tip_contacts` (KD-tree) | ✅ |
| 손 추적 + 관절 BC, 행동·종료 벌칙 | 손목 위치·회전, 손끝, `tracking_joint_pos_bc_exp`, `action_penalty`, `termination_penalty` | ✅ 가중치는 논문에 없습니다(D9) |
| 손별 actor-critic, 전체 관측, 자기 손 행동만 출력 (Eq. 8) | `SideFactoredActor`(손마다 ActorTower), `MultiHeadCritic`(손 × twin 앙상블), `_gate_actions` | ✅ 세부 구조는 D8 |
| FlashSAC, buffer 50M, critic 1024 | `config/runner/flashsac.yaml`: `buffer_max_length: 50_000_000`, `critic_hidden_dim: 1024` | ✅ |
| 성공 기준 SR_MT, SR_MT-Obj, SR_SPIDER | `evaluation/callbacks/tracking_performance.py` | ✅ 정의 세부는 D3, D4 |
| 누적 성공: 한 번이라도 통과한 rollout을 저장 | `evaluation/evaluate.py: _ever_success, _archive_success`, `eval.interval: 20_000_000` | ✅ |
| 50개 모션 (물체 하나 25, 두 물체 25) | `scripts/retarget/50_motion_benchmark.sh` (TACO 10, OakInk2 15, HOT3D 25) | ✅ 클립 이름과 프레임 수까지 고정 |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **물체 보조 힘 커리큘럼이 기본으로 켜져 있습니다.** `commands.object.pin_objects: true`, `pin_mode: xfrc`이고, `curriculum.xfrc_curriculum_adaptive`가 물체를 레퍼런스 자세로 당기는 PD wrench(초기 kp 80, kv 5, 힘 상한 50 N, 회전 강성 ×96)를 겁니다. 과제·모방·BC 보상 평균이 임계값(0.6 / 0.01 / 0.01)을 넘고 완주율이 0.65 이상이면 0.9배씩 줄입니다. 직전 이득의 0.98배로 되돌리는 dial-back 코드도 있지만, 이 설정에서는 실행되지 않습니다. `_set_curriculum`이 감쇠 조건(완주율 ≥ 0.65)을 통과한 뒤에만 dial-back(완주율 < 0.5)을 검사하기 때문입니다. 따라서 보조 힘은 줄어들기만 합니다. 환경마다 [하한, 상한]에서 이득을 무작위로 뽑습니다. 이득 하나를 **모든 모션이 공유**합니다(`src/envs/curriculum/assist.py`). | 논문 Method에는 언급이 없습니다. DexMachina[9]의 핵심 장치와 같은 계열이라 기여 귀속에 중요합니다. 평가는 `disable_object_pin: true`로 보조 없이 하므로 **저장된 궤적은 물리적으로 유효합니다.** 그러나 학습 효율에 대한 기여는 ablation이 없습니다. |
| D2 | **정책 관측에 접촉 wrench와 접촉점이 있습니다.** actor·critic 모두에 `robot_body_contact_wrench_wrist`(손당 링크 13개 × 힘·토크 log-norm 8 = 104차원)와 `robot_body_contact_trans_wrist`(링크별 가장 깊은 접촉점, 39차원)가 들어갑니다. 물체 위치·회전·속도(`obj_trans_wrist` 등)도 actor 관측입니다. | Eq. 3에는 x^cur, P^cur, d^cur만 있습니다. 저자가 한계에서 말한 "privileged 정보"의 실체입니다. "w/o interaction reward"는 SDF 특징만 빼므로 접촉 관측의 기여는 측정되지 않았습니다. 실제 배포용 증류에서 대체할 입력이 필요합니다. |
| D3 | **SR_SPIDER는 위치 오차에서 궤적 평균 오프셋을 뺍니다.** `_spider_errors`가 시뮬레이션−레퍼런스 차이에서 그 시간 평균을 빼고 노름의 평균을 냅니다. 평균 이동량이 1 mm 미만인 쪽 물체는 빼고, 나머지 두 쪽을 평균합니다. SPIDER 공식 스크립트도 양손 모드에서 궤적과 레퍼런스를 각각 평균 중심화하므로 **수학적으로 같은 정의**입니다(한 손 모드에서는 중심화하지 않음). | 기준선과 같은 지표라 비교는 공정합니다. 다만 일정한 위치 오프셋은 아무리 커도 오차로 잡히지 않습니다. 논문의 "10 cm"를 원점 기준 절대 오차로 읽으면 안 됩니다. |
| D4 | SR_MT·SR_MT-Obj는 **궤적 시간 평균 오차**(첫 15스텝 = 0.25 s 제외)에 임계값을 적용하고, 두 손 가운데 나쁜 쪽이 통과해야 합니다. | ManipTrans Sec. 4.1의 정의(오차를 T 프레임 평균으로 정의하고 한 손이라도 실패하면 실패)와 같습니다. 불일치는 아닙니다. 다만 "3 cm 이내"가 매 프레임 조건이 아니라 평균 조건이라는 점은 본문만으로는 드러나지 않습니다. |
| D5 | **행동 공간이 운동학 레퍼런스 기준 잔차입니다.** 손목 위치는 `ref + 0.1·a`(m), 손목 회전은 `ref + 1.0·a`(rad)이고 5 Hz 저역통과 필터를 겁니다. 손가락은 잔차 없이 a ∈ [−1, 1]을 관절 범위에 대응시킵니다(`config/envs/actions/flashdexretarget.yaml`). | 논문에 행동 공간 설명이 없습니다. 손목이 레퍼런스 근처에 묶이므로 SR_MT = SR_MT-Obj(01 §3.2)인 결과를 설명합니다. 다른 손이나 팔에 이식할 때 바꿔야 할 핵심 설계입니다. |
| D6 | 레퍼런스 점군은 원래 좌표가 아니라 점별 **차이(다음 레퍼런스 − 현재 시뮬레이션)**로 줍니다(`ref_future_obj_point_cloud_delta_wrist`). 점은 "균일 샘플"이 아니라 FPS로 뽑습니다. | 정보량은 같습니다. 재구현할 때 입력 표현을 맞추려면 필요합니다. |
| D7 | 조기 종료 규칙은 다음과 같습니다. 5스텝 누적 점군 보상 < 0.6(오차 약 5.1 cm, `03` §3), 손끝이 레퍼런스에서 0.2 m 이상 벗어남(유예 15스텝), 물체가 레퍼런스보다 0.10 m 아래로 떨어짐, 물리 이상(속도 폭주, NaN). 실패 종료 벌칙 −50은 원인이 된 손에만 매깁니다. | 논문은 "미리 정한 임계값"이라고만 적었습니다. 점군 보상 기반 종료가 학습 신호의 상당 부분을 만들 가능성이 큽니다. |
| D8 | 손별 critic은 [twin 2개 × 손 2개] = 4개 멤버를 하나의 앙상블 연산으로 묶었고, 멤버마다 가중치가 따로입니다. critic의 미래 궤적 인코더는 네 멤버가 공유하고, actor는 손마다 인코더를 따로 둡니다. 엔트로피 온도도 손마다 따로입니다. critic h는 손 h의 행동 차원으로만 미분되도록 나머지 행동을 detach합니다(`_gate_actions`). | 논문 설명과 맞습니다. 인코더 공유 같은 세부는 논문에 없습니다. |
| D9 | 보상 가중치와 스케일이 논문에 없습니다(01 §2.3 표). 물체 1.0, 상호작용 0.3, 손목 0.15 + 0.15, 손끝 0.3, BC 0.3입니다. | 재현하려면 설정 파일이 필수입니다. |
| D10 | 도메인 랜덤화와 관측 잡음이 모두 꺼져 있습니다(`events: no_domain_rand`, `noise_scales` 전부 0, `ref_noise.enable: false`). | 리타겟팅이 목적이라 합리적입니다. 다만 실제 로봇 재생(Fig. 5)은 랜덤화 없이 찾은 open-loop 궤적입니다. |
| D11 | 운동학 리타겟은 MuJoCo equality 제약으로 풉니다. 손바닥·물체는 weld, 손끝은 connect이고, 약지·소지 제약은 폭을 따로 조정합니다(`retarget/robot_hand/ik.py`). | 논문은 "CHORD의 전처리를 따름"이라고만 적었습니다. BC 항의 목표가 이 결과이므로 품질에 직접 영향을 줍니다. |
| D12 | 기본 `num_env_steps`는 500M입니다(`config/runner/base.yaml`). 논문의 300M(XHand), 400M(Sharpa)을 재현하려면 덮어써야 합니다. 대신 `eval.stop_when_all_solved: true`라서 모든 클립이 성공하면 일찍 끝납니다. | 사소하지만 GPU-h를 비교할 때 주의해야 합니다. |
| D13 | 리셋할 때 레퍼런스 클립을 **균등하게** 다시 뽑습니다(`_resample_command`의 `torch.randint`). 이미 성공해서 저장된 클립도 같은 확률로 계속 학습 예산을 씁니다. 어려운 클립을 더 자주 뽑는 우선순위 샘플링은 없습니다. | 모음이 커질수록 변환 비율이 떨어지는 Fig. 6 결과와 관련될 수 있습니다. 개선 여지가 큰 지점입니다. |

## 3. 수치 재계산 (`analysis/table_check.py`, `analysis/reward_toy.py`)

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | CHORD 대비 계산량 약 100배 (Abstract) | XHand 2,847 / 29 = **98.2배**, Sharpa 3,314 / 44 = **75.3배** | XHand만 ✅ |
| C2 | 성공률 +44%p (Abstract) | XHand 0.90 − 0.46 = **44%p**, Sharpa 0.86 − 0.50 = 36%p | XHand만 ✅ |
| C3 | 모든 기준선보다 40%p 이상 (Sec. IV-B) | XHand 최소 격차 0.90 − 0.48 = 42%p | ✅ |
| C4 | Do as I Do보다 약 2배 적은 계산 (Sec. IV-B) | XHand 2.28배, Sharpa 1.41배 | XHand만 ✅ |
| C5 | 두 손 모두 기준선보다 적은 계산 | XHand, Sharpa 모두 성립 | ✅ |
| C6 | Sharpa 86% vs 64% / 50%, SR_MT 80% vs 10% / 0% | 표와 일치 | ✅ |
| C7 | 표 내부 일관성 | 모든 SR × 50이 정수이고(예: 45, 43, 43), 모든 행에서 SR_MT ≤ SR_MT-Obj | ✅ |
| C8 | 차이가 유의한가 | FlashDexRetarget SR_SPIDER 45/50, Wilson [0.79, 0.96]. 최강 기준선 DexMachina 24/50, [0.35, 0.62]. Fisher p = 8.6×10⁻⁶. SR_MT-Obj 43 vs 16, p = 5.3×10⁻⁸ | ✅ 매우 유의 (단일 시드, 비대응 근사) |
| C9 | 성공 1건당 GPU-h | SR_MT-Obj 기준 FlashDexRetarget 0.67, DexMachina 27.9, Do as I Do 33.0, CHORD 2,847 | `03` §5 |
| C10 | 하드웨어 요구 | 버퍼 50M 행 = 88 GB, 기본 10M = 17.6 GB. RTX 3090(24 GB)에는 둘 다 GPU에 못 올림 → 호스트 RAM 약 100 GB 필요 | 논문에 없는 요구 |
| C11 | 1,000개 중 600개 이상을 60 GPU-h에 (프로젝트 페이지) | GPU-h당 10개 이상 = 50개 실험의 6.7배. 600M 스텝 ≈ 58 GPU-h라 60 GPU-h는 예산 거의 전체 | ✅ (해석 주의) |
| C12 | MOPE·MORE 비교 | SR_SPIDER 통과 궤적만의 평균이라 표본 크기가 다름(XHand 18 / 24 / 23 / 45) | 직접 비교 주의 |
| C13 | 학습량 | 50개 클립 = 26,387프레임(7.3분), 레퍼런스 프레임당 약 11,400 env step | 참고 |
| V1 | ρ = 0.058 m면 30° ≈ 3 cm (Sec. III-B a) | 2ρ·sin 15° = **3.002 cm** | ✅ (회전축에 수직인 점 기준) |
| V2 | 점군 정합이 위치와 회전을 하나의 단위로 측정 | 형상·축에 따라 30°가 0.62–3.0 cm. 막대 장축 회전에서 거의 보이지 않음 | ⚠️ `03` §2 |

## 4. 구현 가능성 판단

### 4.1 공식 코드를 그대로 돌려 Table I을 재현하는 경우

| 항목 | 요구 사항 | 비고 |
|---|---|---|
| GPU | NVIDIA CUDA GPU. 기본 설정은 GPU 메모리 약 110 GB(버퍼 88 GB 포함). `buffer_device_type=cpu`로 두면 GPU 약 16 GB | 110 GB는 단일 H100 80 GB에도 들어가지 않습니다. 24–48 GB급 GPU라면 버퍼를 호스트 RAM에 둬야 합니다 |
| 호스트 RAM | 버퍼를 CPU에 두면 약 100 GB(pinned 88 GB) | 128 GB 이상 서버급 머신이 현실적입니다 |
| 시간 | XHand 50개 모션 29 GPU-h, Sharpa 44 GPU-h (RTX 3090 기준) | 실행 한 번에 하루 이상 걸립니다 |
| 소프트웨어 | Linux, conda, Python 3.11, Isaac Sim 5.1 / Isaac Lab 2.3, torch 2.7 cu128, warp 1.10. `setup.sh`가 고정 버전을 `--no-deps`로 설치 | macOS는 Isaac Sim을 지원하지 않습니다. 일반 `pip install`을 쓰면 환경이 깨진다고 README가 경고합니다 |
| 데이터 | TACO, OakInk2, HOT3D 원본 + MANO 모델(등록 후 수동 다운로드) | 저장소에 모션 데이터가 없습니다. 각 데이터셋과 MANO의 라이선스를 따로 확인해야 합니다 |
| 라이선스 | 코드 MIT, Sharpa 자산 Apache 2.0, mjlab vendored 코드 Apache 2.0 | 코드 자체의 연구 사용 제약은 거의 없습니다 |

**판단:** 공식 코드는 설정, 벤치마크 클립 목록, 평가 코드까지 갖추고 있어서 **재현에 필요한 정보는 충분히 공개되어 있습니다.** 다만 Linux + 대용량 메모리 GPU 서버가 필요하므로, 이 맥에서는 학습 재현이 불가능합니다. 우리 저장소 규칙(GPU 재현 금지)에 따라 여기서는 하지 않았습니다.

### 4.2 부분 재현: CPU로 가능한 단계

| 단계 | CPU 가능 여부 | 근거 |
|---|---|---|
| 보상·관측 식 검증 | ✅ 완료 | `reward_toy.py` (numpy, 몇 초) |
| 사람 시연 import → 운동학 리타겟 → `motion.pt` | ✅ 가능할 것으로 보임 | 이 단계는 numpy, scipy, trimesh, MuJoCo, torch(MANO)만 씁니다. SDF 굽기에는 `sdf.device=cpu` 옵션이 있습니다. USD 변환은 Isaac이 필요하므로 끄고(`steps.usd=false`) 돌려야 합니다. 클립 몇 개면 데이터 일부만 받아도 됩니다 |
| 사람 손끝-물체 거리 분포 분석 | ✅ 위 단계 결과로 가능 | Eq. 7의 d_ref 분포(떠 있는 손가락, 파고든 손가락 비율)를 정량화할 수 있습니다 |
| RL 학습 (FlashSAC + Isaac Sim) | ❌ | GPU 물리 시뮬레이션과 대용량 버퍼가 필요합니다 |

### 4.3 핵심 아이디어를 다른 환경에 다시 구현하는 경우

공식 코드를 쓰지 않고 아이디어만 옮긴다면, 필요한 구성 요소와 난이도는 다음과 같습니다.

| 구성 요소 | 난이도 | 메모 |
|---|---|---|
| 점군 정합 보상 (Eq. 6), 한쪽 방향 거리 보상 (Eq. 7) | 낮음 | 수십 줄입니다. 레퍼런스 거리의 꼭짓점 여유(`03` §4)와 장축 회전 사각지대(`03` §2)를 같이 고려해야 합니다 |
| 미래 K 프레임 Conv1D 인코더 | 낮음 | `traj_encoder.py` 약 100줄 |
| 손별 actor-critic + 행동 gating | 중간 | 앙상블 critic과 손별 온도까지 맞추려면 FlashSAC 구조를 알아야 합니다 |
| 잔차 행동 공간, 물체 보조 커리큘럼, 조기 종료 규칙 | 중간 | 논문에 없고 코드에만 있습니다(D1, D5, D7). 재구현 성능을 좌우할 가능성이 큽니다 |
| 대규모 off-policy 학습 (50M 버퍼, 2,048 env, 업데이트 2회/스텝) | 높음 | 메모리와 처리량을 맞추는 엔지니어링이 핵심입니다. Fig. 4(d)에서 버퍼와 critic 크기가 성능 차이의 대부분을 만들었으므로 줄이면 성능이 떨어질 것으로 예상합니다 |
| 다른 로봇손 이식 | 중간 | MJCF/URDF에 손끝·손바닥 site 추가, 관절 그룹 정의(`config/envs/robot/*.yaml`), USD 변환. Sharpa 이식 과정이 `assets/robot/sharpa/README.md`에 정리되어 있습니다 |

**요약:** 아이디어 수준의 구현은 어렵지 않습니다. 하지만 논문 성능을 얻으려면 논문에 없는 세 가지(물체 보조 커리큘럼, 잔차 행동 공간, 접촉 관측)와 대용량 off-policy 인프라를 함께 가져와야 합니다. 현실적인 경로는 공식 코드를 GPU 서버에서 그대로 돌리고, 바꾸고 싶은 부분만 수정하는 것입니다.
