# 04. 공식 코드와 논문의 대조 (TacEx)

대상: `github.com/klemensiten/tacex`, 커밋 `d268cdb0d769fc9868fe37af462f81ef59c4412c`(2026-10-15, "adjust readme"). 얕은 복제(`git clone --depth 1`)를 정적으로만 읽었습니다. **코드를 실행하거나 가져오거나 설치하지 않았고**, 서브모듈(LIBERO, openpi, tactile_envs)과 가중치, 데이터를 받지 않았습니다. 이 문서의 "코드는 …"은 모두 읽은 결과이며 실행으로 확인한 것이 아닙니다. 이 저장소를 `repo/`로 다시 받으려면 `git clone --depth 1 https://github.com/klemensiten/tacex repo && git -C repo fetch --depth 1 origin d268cdb0d769fc9868fe37af462f81ef59c4412c && git -C repo checkout FETCH_HEAD`입니다.

## 1. 저장소와 라이선스

| 항목 | 내용 |
|---|---|
| 구성 | `jaxrl2/`(에이전트, 앙상블, 인코더, 재생 버퍼), `examples/exploration/`(무보상 탐색 실험과 런처), `examples/diffusion_steering/`(LIBERO 확산 조종 학습, PPO와 DPPO 기준선, 런처 9개). 파이썬 파일 183개, 약 1.1 MB |
| 언어, 프레임워크 | JAX 0.5.0, flax 0.10.2 (`requirements.txt`는 tensorflow 2.19, torch 2.6, robosuite 1.4.1, mujoco 3.8.1 등을 고정) |
| 서브모듈 | `LIBERO`(nakamotoo 포크), `openpi`(nakamotoo 포크), `tactile_envs`(carlosferrazza) |
| 라이선스 | **LICENSE 파일이 없습니다**(GitHub API도 license null). 저작권 기본 규칙이 적용되므로 재사용 조건이 불명확합니다 |
| README 주장 | 탐색 실험(상태 기반 MaxInfo-SAC 스모크 테스트와 촉각 삽입 환경 예시)과 LIBERO π0 조종 학습 예시 하나. 학회 채택 문구와 프로젝트 페이지 링크가 있음 |
| 의존성 메모 | 설치에 `jaxrl`(sukhijab 포크의 특정 커밋)이 필요. wandb 개체 이름 기본값이 저자 계정으로 되어 있어 로그를 끄는 옵션(`--wandb_log 0`)이 필요 |

## 2. 논문 ↔ 코드 차이 목록 (D1–D19)

판정: **일치**, **차이**(논문 서술과 코드가 다름), **불명**(코드나 논문 어느 쪽에 정보가 없음), **범위 밖**(저장소에 없음).

| # | 항목 | 논문 | 코드 (파일:줄) | 판정 |
|---|---|---|---|---|
| D1 | 구현 범위 | 두 설정: 무보상 탐색, LIBERO 확산 조종 (§4) | 두 설정의 학습기가 있음: `tactile_drq_explorer`(`jaxrl2/agents/tactile_agents/tactile_explorer/drq_explorer_learner.py`), `pixel_maxinfosac`(`jaxrl2/agents/pixel_maxinfosac/`), 기준선 `pixel_sac`, `pixel_ppo`, `pixel_maxinfosac_taxel_sum`, `pixel_sac_taxel_reward`, DPPO | 일치 |
| D2 | 무보상 탐색 환경 | 그리퍼와 물체 하나 또는 네 개가 있는 환경 [22] (§4.1) | `make_env`는 `tactile_envs.envs.exploration`의 `ExplorationEnv`를 불러옴(`jaxrl2/utils/tactile/train_utils.py:28-37, 89-122`). 서브모듈의 **고정 커밋 `f131d49`(2024-07-08) 트리 목록의 `tactile_envs/envs/` 아래에는 `insertion.py`, `robosuite/`, `Gymnasium-Robotics/`뿐이고 `exploration`, `arm`, `hand_exploration` 모듈이 없음**(GitHub API 트리로 확인, 서브모듈은 받지 않음). README의 탐색 예시는 `insertion` 환경만 다룸 | 범위 밖 (고정 서브모듈에 없음) |
| D3 | 하류 오프라인 학습 | 탐색이 끝난 버퍼를 고정하고 tolerance 보상으로 재라벨, SAC를 완전히 오프라인으로 학습, 마지막 평가 체크포인트 10개 평균 (§4.1, 그림 2) | 저장소 전체에서 "offline", "relabel"(재라벨)을 하는 탐색용 코드가 없음(`grep`). 하류 보상 정의(tolerance 마진)와 오프라인 SAC 설정도 없음. DrQ 온라인 기준선(`tactile_drq`)은 구현되어 있으나 탐색 런처의 알고리즘 목록에서 주석 처리(`examples/exploration/examples/tactile/insertion/exploration/launcher.py:69-70`) | 범위 밖 |
| D4 | 탐색 앙상블의 예측 목표 | 모델이 다음 상태 s′를 예측하고 z′, v′, ι′에 대한 불일치를 냄(그림 1, §3) | `update_ensemble`(`drq_explorer_learner.py:54-90`): 입력은 현재 (잠재, 행동). 목표는 `[다음 잠재(z′), 영상 임베딩, 촉각 임베딩]`인데, 마지막 두 블록은 `full_obs = concat([next_obs, obs])`에서 `[batch_size:]`, 곧 **현재 관측**의 풀링 임베딩임(줄 64, 66). 같은 코드에서 잠재 블록만 `state[:batch_size]`(다음)를 씀. DSRL 학습기(`pixel_maxinfosac/ensemble_utils.py:102-122`, `critic_updater.py:44-47`)는 세 모달리티 모두 **다음** 값(`predict_diff=True`면 차이)을 씀 | 차이 (두 구현 사이 불일치). 의도된 것인지는 코드만으로 알 수 없음 |
| D5 | 보너스 식 | 식 4: Σ_k ω_k‖σ_n^k(s,a)‖에 λ를 곱해 Q에 더하고 π ∝ exp(α⁻¹(Q + λ…)) (§3) | `get_info_gain`(`jaxrl2/networks/ensemble_model.py:242-276`): 앙상블 평균 표준편차 `ep_std`를 고정 우연적 표준편차 `al_std = 1e-3`(결정적 앙상블, 줄 180, 245)로 나눈 비 제곱에 가중치를 곱해 `log(1 + ratio)`를 **차원에 대해 평균**한 뒤, 실행 평균과 표준편차로 **표준화**(줄 267-275). DSRL 학습기는 별도의 탐색 비평가 `expl_critic`을 두고 `actor_loss = (α·logπ − Q − λ·Q_expl).mean()`(`pixel_maxinfosac/actor_updater.py:86`)이며 λ는 정보 이득 목표를 따라 자동 조정(`pixel_maxinfosac_learner.py:138-140`) | 차이 (노름 합이 아니라 로그 비 평균, 표준화 추가, 두 비평가 구조). MaxInfoRL 계열의 구현이므로 논문은 [1]의 방식을 간략히 쓴 것으로 보임 |
| D6 | 모달리티 가중치 | ω_k ≥ 0 실수, ω_ι ≫ ω_v, ω_z 가능 (§3) | DSRL 학습기: `disagreement_weights`(`ensemble_utils.py:70-102`)는 선택한 모달리티의 차원에 `전체 차원/선택 차원`을 주고 나머지는 0. **선택 마스크만 지원**하며 임의 비율은 표현되지 않음(토이 T3). 탐색 학습기는 실수 가중치 인자(`latent_disg_weight` 등)를 받지만 런처는 0과 1만 사용(탐색 `launcher.py:54`) | 차이 (식은 연속 가중, 코드는 DSRL에서 선택 마스크) |
| D7 | 모달리티 정의 | z(로봇 상태의 잠재), v(영상 임베딩), ι(촉각 임베딩) 세 개 | 코드는 네 개: `latent`(영상 인코더 뒤 Dense+LayerNorm+tanh 50차원), `image`, `state`(저차원 고유감각), `tactile`(`ensemble_utils.py:10`). **영상과 촉각 예측 목표는 인코더 임베딩이 아니라 합성곱 특징을 64차원으로 최대 풀링한 값**(`pixel_maxinfosac/networks.py:16-30, 72-91, 145`). 그림 6의 "상태", "잠재", "영상"이 서로 다른 세 비교 대상인 것과 코드의 네 모달리티가 대응 | 차이 (논문의 z는 코드의 `latent`, `state` 중 무엇인지 서술이 없음) |
| D8 | 촉각 인코더 | 그림 1: 별개의 CNN 인코더. 표 1: "Tactile encoder MLP (256,256)" | `TactileEncoder`(`pixel_maxinfosac/networks.py:107-155`): 입력이 영상 모양(배치 포함 4D, README의 32×64×3)이면 CNN(채널 (32,32,32,32), 스트라이드 (2,1,1,1)), 영상 모양이 아닐 때만 MLP `tactile_hidden_dims`=(256,256). README 설정은 CNN 경로 | 차이 (표 1이 쓰이지 않는 경로를 적음. 그림 1은 코드와 같음) |
| D9 | 표 1 하이퍼파라미터 | 액터 1e-4, 비평가 3e-4, 온도 3e-4, τ 0.005, 할인 0.999, 배치 256, 워밍업 500, 질의 20, 행동 크기 1.0, 영상 64×64, 촉각 32×64×3, MLP (128,128,128), CNN (32,32,32,32), 잠재 50, λ 학습률 3e-4, 초기 λ 1.0, 동역학 모델 5×(256,256), 과제 번호 58, 69, 61, 77, 55, 21, 56, 20 | `launch_train_sim.py:48-98`와 README 예시, `launcher_ablate.py`에서 **일치**. 예외: 동역학 모델 학습률과 감쇠. 표 1은 1e-3과 1e-4이고 `launcher_ablate.py:58-59`, `launcher.py`, `launcher_taxel_baselines.py:72-73`도 1e-3, 1e-4. 그러나 **`launch_train_sim.py` 기본값(줄 62-63)과 README 예시, `run_libero_maxinfo*.sh`는 3e-4, 0.0** | 일치 (런처 기준). README 예시만 따라 하면 값이 다름 |
| D10 | 갱신 대 환경 스텝 | 표 1 "Update-to-data ratio 20 gradient steps per environment step" | `trajwise_alternating_training_loop`(`train_utils_sim.py:493-520`): `num_gradsteps = len(traj["rewards"]) * multi_grad_step`, 보상은 질의(조종 전이) 한 번에 하나. README 133행: `max_steps`는 경사 갱신 횟수, `multi_grad_step`은 저장된 조종 전이 하나당 갱신 횟수, `query_freq`는 행동 덩어리 최대 길이(20) | 차이 (표기는 환경 스텝, 코드는 질의 단위). `03` V7: 500k 갱신은 25,000 질의, 최대 500k 환경 스텝 |
| D11 | 로봇과 그리퍼 | 표 1: 그리퍼 "PandaGripper from [22]" | README 예시는 `--libero_robot Panda --touch_gripper_type Robotiq85TactileGripper`(`gripper_state_indices 0 3`). `launcher_ablate.py`, `launcher_sac.py`, 기준선 런처는 `Panda` + `PandaGripper`(`[0, 1]`). **`launcher.py:26`은 `Piper` + `PiperGripper`**. 논문은 Piper를 언급하지 않음 | 차이 (예시와 런처가 서로 다름) |
| D12 | 시드와 기준선 | 그림 4, 5, 6 모두 시드 10개 (캡션). 합산 촉각 "보상" 기준선은 "집계 신호를 내재 보상으로" (그림 6 캡션, §4.2 끝 문단) | 촉각 목표 변형 `launcher_ablate.py:76`은 10시드. **촉각 목표 없는 변형(`SWEEP_FLAGS_BASE`, 줄 103)과 합산 촉각 기준선(`launcher_taxel_baselines.py:30`)과 `launcher.py:76`은 기본 5시드**. DSRL-SAC 런처(`launcher_sac.py:76-79`)의 과제 목록은 58번만 남고 나머지는 주석. **`PixelSACTaxelRewardLearner`는 합산 촉각 신호로 보상을 교체하여 SAC를 학습**(`pixel_sac_taxel_reward_learner.py:13-29`): 과제 보상 없이 촉각 크기만 최대화 | 불명/차이 (추가 실행이 있었는지 알 수 없음. "Reward" 기준선은 과제 보상을 더한 것이 아니라 대체) |
| D13 | π0 체크포인트 | "같은 pretrained π0 체크포인트" (§4.2) | `train_sim.py:320-321`: openpi 설정 `pi0_libero`와 `s3://openpi-assets/checkpoints/pi0_libero`. 이름과 openpi README의 `pi05_libero` 설명("fine-tuned for the LIBERO benchmark")으로 미루어 LIBERO용 미세조정 체크포인트로 보이나, 받아서 확인하지 않음 | 불명 (논문은 사전학습 단계의 입력에 촉각이 없다는 점만 적음) |
| D14 | 보상 | 희소 과제 보상 (§4.2), 식 1에서 r: S×A×S → R₊ (v3) | `collect_traj`(`train_utils_sim.py:683-691`): 질의마다 보상 −1, 성공 시 마지막 질의 0과 마스크 0. 평가 지표는 성공률 | 차이 (부호와 단위. 해석에는 영향이 작음) |
| D15 | 탐색 런처 설정 | 단일 물체와 4개 물체, 시드 5개, 3M과 5M 스텝 축(그림 2, 3) | 탐색 `launcher.py`: **단일 물체 설정은 주석 처리**(줄 51, 77). 다중 물체는 물체 수를 1에서 4로 올리는 **커리큘럼**(`n_objects_steps_to_end` 4,000,000, 줄 53), `max_steps` 1,500,000(줄 16), `action_repeat` 4, 재생 버퍼 500,000(줄 25), 은닉 1024(줄 18), `pixels_only` [0, 1] 스윕(줄 29), 갱신은 250 스텝마다 256회(줄 106-107, `training_start` 10,000), 탐색 변형은 `{0,1}³` 전부. 논문은 커리큘럼, 버퍼 크기, 갱신 주기를 적지 않음 | 불명 (논문에 없는 설정이 코드의 기본 런처에 있음) |
| D16 | 탐색 후 활용(explore-then-exploit) | 처음 100k 환경 스텝은 외부 보상 없이 탐색, 이후 과제 최대화 (부록 C) | `pixel_maxinfosac_explorer`와 `launcher_explorer.py:89`의 `explore_until` 100000(기본값은 300000, 줄 54). 단위가 경사 갱신인지 환경 스텝인지는 코드 주석 없음 | 일치 (수치). 단위는 불명 |
| D17 | DPPO 기준선 | "DPPO [33]는 확산 정책 자체를 미세조정" (§4.2), 캡션 "DPPO adaptation updates the action-output head using tactile disagreement" | `pixel_dppo/pi0_policy.py:1-12`: "smallest explicit adaptation of the paper"이라 밝히고, 가우시안 노이즈를 넣은 확률적 디노이징 전이를 구현. 학습 범위는 `action_head`(줄 45-62), `dppo_rollout_steps` 40, 과제 58은 따로 실행하고 나머지 7과제를 `launcher_dppo.py`가 실행(문서 문자열) | 일치 (캡션이 "adaptation"이라고 밝힘). 원 DPPO 코드와의 동등성은 확인 못 함 |
| D18 | 논문에 없는 구성 | — | 비평가 10개 `num_qs=10`과 평균 축소 `critic_reduction='mean'`(`launch_train_sim.py:83, 92`), 영상 증강(`aug_next=1`, 색 변환), 작은 인코더(`encoder_type='small'`), 동역학 모델이 보상도 예측(`predict_reward=True`)하고 차이를 예측(`predict_diff=True`), `mask_expl_critic` 1(런처)과 0(`run_libero_maxinfo.sh`) | 불명 (표 1에 없는 설정들) |
| D19 | 모델 선택과 평가 | 평균 평가 반환 = 성공률 (§4.2). 평가 에피소드 수 미기재 | 평가 10 에피소드를 20,000 갱신마다(`eval_episodes 10`, `eval_interval 20000`). 곡선의 점 수와 간격이 이와 일치(`03` V7). 체크포인트 선택 규칙은 없음(평가 곡선을 그대로 보고) | 일치 (평가 에피소드 수 10은 논문에 없음) |

### 2.1 차이의 영향

- **D4(탐색 앙상블 목표)**는 가장 큰 구현 불일치입니다. 영상과 촉각의 목표가 현재 값이면 이 블록의 불일치는 다음 상태가 아니라 현재 관측의 재현 가능성을 재는 쪽으로 기웁니다. 그림 1의 설명(다음 상태 예측)과 DSRL 학습기의 구현은 다음 값을 씁니다. 코드를 실행해 확인하지 않았으므로 "코드를 읽으면 그렇게 보인다"까지만 말할 수 있고, 그림 2, 3의 결과에 어느 정도 영향이 있는지는 알 수 없습니다.
- **D2, D3**은 그림 2, 3(무보상 탐색과 하류 반환) 전체를 이 저장소만으로 되짚을 수 없다는 뜻입니다. LIBERO 실험(그림 4–6)은 코드, 런처, 서브모듈이 모두 있어 되짚을 수 있습니다.
- **D5, D6**은 논문의 식 4와 코드의 계산 모양 차이입니다. 토이(`03` §3)에서 모달리티 비중의 차이는 작았으나, ω 비율 변화(ω_ι ≫ ω_v)는 DSRL 코드로 표현되지 않습니다.
- **D9, D10, D11**은 README 예시만 따라 하면 표 1과 다른 설정으로 학습된다는 뜻입니다. 표 1은 `launcher_ablate.py` 계열과 맞습니다.
- **D12**의 합산 촉각 보상 기준선은 과제 보상을 쓰지 않으므로, 그림 6의 "Reward" 칸(0.00–0.09)은 "촉각 크기만 최대화한 SAC"의 결과로 읽어야 합니다.

## 3. 일치하는 부분

- 그림 1의 모달리티 분해 구조, 앙상블 5개 머리, 정보 이득 보너스 구조는 코드에 있습니다(D1, D5).
- 표 1의 대부분의 값이 `launcher_ablate.py`와 일치하고, 과제 번호 8개와 평가 간격(20,000)이 그림과 맞습니다.
- 탐색 단계에서 활용 에이전트를 갱신하지 않는 설정(`agent_update_period` 10,000,000, `launcher.py:74`)이 "무보상 탐색 후 별도 학습"과 맞습니다.

## 4. 구현할 수 있는지 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 수치의 재현은 불가능합니다. 핵심 알고리즘의 소형 재구현은 가능합니다.**

| 무엇 | 이 맥에서 | 이유 |
|---|---|---|
| LIBERO 확산 조종 학습 (그림 4–6) | **불가** | π0 가중치(openpi `pi0_libero`)와 JAX CUDA 12가 전제(README `jax[cuda12]==0.5.0`). 과제당 500k 경사 갱신, 25,000회 π0 추론, 시드 10개 × 변형 8–11개 × 과제 8개. 시뮬 렌더링과 EGL이 필요. 가중치는 받지 않았고 크기도 확인하지 않음 |
| 무보상 탐색 (그림 2, 3) | **불가** | 환경(`ExplorationEnv`)이 고정 서브모듈에 없음(D2). 런처가 클러스터 작업(GPU 1, 32 GB, 24시간급)을 가정 |
| 공식 코드의 CPU 스모크 테스트 (상태 기반 MaxInfo-SAC, 20 스텝) | **조건부** | README가 `JAX_PLATFORMS=cpu`로 Pendulum 예시를 제시. 고정된 요구 사항(tensorflow 2.19, torch 2.6, robosuite 1.4.1 등) 설치와 공식 코드 실행에 사용자 승인이 필요. 이 분석에서는 하지 않음 |
| 보너스 계산의 numpy 이식 | **가능, 완료** | `analysis/toy_bonus_weights.py`(수 초) |
| 모달리티 가중 앙상블 불일치 + SAC의 torch 재구현을 MuJoCo 소형 환경(그리퍼와 물체 하나, 접촉 패드 힘 지도 16×16)에서 | **가능** (새 설치 없음) | 저장소 루트 가상환경에 torch 2.15, mujoco 4.0이 있음. 이 맥의 MPS 또는 CPU로 소규모 학습 가능. 환경 정의와 촉각 힘 지도 계산을 직접 짜야 함 |
| 논문의 효과 크기 재현 (잡기 비율 0.4, 오프라인 반환) | **불가에 가까움** | 환경과 하류 설정이 코드 밖(D2, D3). 소형 환경의 결과는 다른 문제의 결과 |

**노력 추정 (측정 전, 추정):** 소형 MuJoCo 환경에서 접촉 탐색의 정성적 재현(식 4의 세 모달리티 비교, 그림 2a의 유사 곡선)은 2–3주, 서로 다른 ω를 바꿔 가며 시드 5개까지 도는 비교 실험은 추가 1–2주입니다. 공식 LIBERO 조종을 GPU 서버에서 재현하는 일은 이 맥 밖의 작업이고, π0 체크포인트와 두 개의 포크 서브모듈을 받는 데서 시작합니다.
