# 04. 코드 대조: RL-Co 논문 ↔ 공식 저장소 RLinf (arXiv:2602.12628 v5)

## 0. 대상과 방법

- **공식 저장소:** https://github.com/RLinf/RLinf (Apache-2.0, 별 5,652, 2026-11-15 확인). 논문 PDF에는 코드 링크가 없습니다. 프로젝트 페이지 번들과 RLinf 문서(`rlinf.readthedocs.io/.../co_training.html`)의 "RL-based Sim-Real Co-Training" 항목이 RLinf 저장소를 가리킵니다. 저장소의 README도 이 논문을 인용하고 2026-02 업데이트로 적어 두었습니다.
- **받은 범위:** `repo/`에 `--depth 1 --filter=blob:none --sparse`로 받았습니다. 커밋 `6a7dcbd88c4b4f166e31144f63fd6959f4cac3a5`(2026-11-13). 필요한 디렉터리만 펼쳤습니다(`examples/embodiment`, `rlinf/workers`, `rlinf/algorithms`, `rlinf/models/embodiment/openpi`, `rlinf/envs`, `tests/e2e_tests/embodied`, `docs/.../embodied`).
- **하지 않은 것:** 실행, import, 설치, 빌드 없음. 가중치(`RLinf/RLinf-Pi05-RLCo-...-SFT`), 데이터셋(`RLinf/RLCo-Example-Real-Data`, `RLinf/RLCo-Example-Mix-Data`), 시뮬 자산(`RLinf/RLCo-maniskill-assets`) 받지 않음. 이 셋은 Hugging Face에 있고, 크기를 확인하지 않았습니다.
- **읽기 방법:** 정적 텍스트 읽기. 줄 번호는 커밋 `6a7dcbd` 기준입니다.

## 1. 확인 범위

| 파일 (repo 기준) | 내용 | 확인 |
|---|---|---|
| `rlinf/workers/actor/embodied_fsdp_actor_worker.py` | 공동 학습 플래그, SFT 데이터 로더, 손실 결합 | 전체 관련 구간 (L95–110, L540–625, L870–890) |
| `examples/embodiment/config/maniskill_ppo_co_training_openpi_pi05.yaml` | π0.5 공동 학습 예시 설정 | 전체 |
| `rlinf/models/embodiment/openpi/__init__.py` | RL 모드에서 VLM 동결 | L282, L302–316 |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | `pi05_maniskill_sim_real_co_training` | L340–361 |
| `rlinf/envs/sim/maniskill/tasks/panda_put_on_in_scene_multi.py` | PutOnPlate 디지털 트윈 보상 | L849–862, L1318–1382 |
| `docs/source-en/rst_source/examples/embodied/co_training.rst` | 공식 사용 설명 | 전체 |
| `tests/e2e_tests/embodied/maniskill_ppo_co_training_openpi_pi05.yaml` | 종단 시험 설정 | `sft_loss_weight`, `sft_data_path` |

- **없는 것:** OpenVLA 공동 학습 설정(논문 §5.1이 Liu 외 [12] 구현을 쓴다고 함). 이 저장소에는 `openvla` 설정이 많지만, RL-Co(SFT 손실 결합)가 들어간 OpenVLA 설정은 없습니다. 실로봇 배포, 평가 판정, MimicGen 파이프라인, 시스템 식별 코드도 없습니다.

## 2. 대조표

| 항목 | 논문 | 저장소 | 판정 |
|---|---|---|---|
| 손실 결합 식 (2) | L_total = L_RL + β L_SFT(θ; D_real) | `embodied_fsdp_actor_worker.py` L605: `total_loss = loss + self.sft_loss_weight * sft_loss` (`loss`는 PPO 손실, `sft_loss`는 실데이터 배치) | **일치** |
| SFT 손실의 데이터 | 실데이터 D_real만(Stage II) | `_build_sft_data_loader`(L546–577)가 `actor.sft_data_path`를 읽고, 이것은 실데이터 LeRobot 경로(예시: `RLCo-Example-Real-Data`) | **일치** (데이터 경로에 시뮬이 없음) |
| SFT 손실 호출 위치 | 매 PPO 갱신 | `_train_sft_epoch`는 L880에서 PPO 손실 계산 뒤, minibatch마다 한 번 호출. 실데이터 배치를 하나씩 꺼내 쓰고, 끝나면 에포크 증가 후 다시 시작 | **일치** (세부 구현) |
| 실데이터 SFT 가중치 β 기본값 | 표 7 참조 | 코드 기본값 0.1(L577). 예시 설정은 0.2 | 아래 D2 |
| π0.5 학습 범위 | v5: action expert만 학습 | `_freeze_after_load`(L302–316): `train_expert_only`이면 `freeze_vlm()`(SigLIP과 LLM 전문가 0 동결) | **일치**(v5). v1·v4의 "full-parameter"와는 다름 (01 §6) |
| 보상 (π0.5, 희소) | 부록 E.2: 파지 0.2, 배치 1, 배치 후 이탈 −0.4 | `panda_put_on_in_scene_multi.py` L1325–1361: 파지 +1, 성공 +5, 성공 후 실패 −2. L849–853에서 `/5`로 정규화 → 0.2, 1, −0.4 | **일치** (정규화 후) |
| 공동 학습 설정 키 | 논문 본문에 키 없음 | `enable_sft_co_train: True`, `sft_loss_weight: 0.2`, `sft_data_path` | 이름만 있음 |
| PPO 설정 | 논문: 학습률만 표 7. PPO "clip", "γ" 등은 없음 | clip 0.2, γ 0.99, GAE λ 0.95, update_epoch 2, entropy 0.005, value_lr 2e-4 | **논문에 없음** (저장소에서만 확인) |
| 액션 청크와 흐름 단계 | 논문에 h 값 없음 | `num_action_chunks: 4`, `action_horizon: 8`, `num_steps: 4`, noise `[0.16, 0.12, 200]`, `joint_logprob: True` | **논문에 없음** |
| 학습률 (π0.5 PnP) | 표 7: actor 4e-6, critic 2e-4 | `optim.lr: 4e-6`, `value_lr: 2e-4` | **일치** |
| 배치 크기 | 논문에 없음 | `micro_batch_size: 40`, `global_batch_size: 2560`. openpi 데이터 설정은 배치 16 | 논문에 없음 |
| 실시연 수 (PnP) | 표 7: 50 | 예시 설정(PutOnPlate)과 문서: 50 실궤적 | **일치**(수) |
| 시뮬 데이터 수 | 과제당 1,000 (B.2) | 문서: 50 실 + 1,499 시뮬(LeRobot 데이터) | **불일치** (D5) |
| 시뮬 환경 | ManiSkill, 과제 "Pick and Place" (그릇) | `PandaPutOnPlateInScene25DigitalTwin-v1`(접시) | **다른 과제** (D5) |
| 시뮬 백엔드 | 논문 본문에 명시 없음 | `sim_backend: "gpu"`, `obs_mode: rgb+segmentation`, 제어 5 Hz, 시뮬 500 Hz | 논문에 없음. GPU 백엔드 (D7) |
| 최대 에피소드 길이 | 논문에 없음 | 80 스텝 | 논문에 없음 |
| 카메라 | 고정 RGB 1대 (§5.1) | 문서: 실로봇 3인칭 RGB 640×480. 설정: `policy_setup: panda_wristcam`(이름은 손목 카메라) | **불일치 가능.** 이름과 문서가 다름. 실제 입력 카메라는 확인 안 됨 (D7) |
| 손실 척도 감시 | 논문 없음 | `train/loss_ratio` = β\|L_SFT\|/\|L_RL\|, 1e5를 넘으면 경고 | 논문에 없음 (D3) |
| 종단 시험 값 | — | `tests/.../maniskill_ppo_co_training_openpi_pi05.yaml`: `sft_loss_weight: 0.2`, `sft_data_path`는 `pick_and_place_real` 경로 | 예시와 같은 값 (D2) |

## 3. 불일치와 빈칸 (D1–D10)

- **D1 (π0.5 학습 범위, 버전 차이):** v1·v4 본문은 "full-parameter updates"라고 씀. 저장소는 `train_expert_only`일 때 VLM을 동결합니다. v5의 "action expert만"이 저장소와 맞습니다. 논문 버전 간 정정이 실제 구현과 맞은 경우입니다.
- **D2 (β 값):** 표 7의 π0.5 β는 PnP 1.0, Push 0.2, OD 1.0, CD 0.1입니다. 공개 예시와 종단 시험은 0.2입니다. 공개 예시는 다른 과제(PutOnPlate, 디지털 트윈)라서 직접 비교는 안 되지만, 같은 모델과 같은 손실 구조입니다. 논문의 β가 과제마다 다른 이유는 본문에 없습니다. 그림 5c에서 OD의 최적은 β=0.2(0.75)인데 표 7은 1.0을 씁니다(01 §5.7).
- **D3 (손실 척도와 β의 의미):** 저장소는 `loss_ratio`를 기록하고 1e5가 넘으면 경고합니다. 이는 β가 척도에 따라 달라진다는 뜻입니다. 논문은 β의 척도 기준을 적지 않았습니다. 같은 β 값을 다른 PPO 설정에서 옮길 수 없습니다. 토이 A의 β도 같은 이유로 척도가 다릅니다(03 §3.3).
- **D4 (OpenVLA 공동 학습 코드 없음):** 저장소에 RL-Co의 OpenVLA 설정이 없습니다. 논문은 Liu 외 [12]의 PPO 구현과 LoRA rank 32를 씁니다(App. I). 이 부분은 이 저장소에서 확인할 수 없습니다.
- **D5 (공개 설정과 논문 과제가 다름):** 공개 예시는 PutOnPlate 디지털 트윈(접시, 환경 이름 `PandaPutOnPlateInScene25DigitalTwin-v1`)이고, 논문 PnP는 그릇 과제(25 물체, App. D.3)입니다. 시뮬 데이터도 공개는 1,499개, 논문은 과제당 1,000개입니다. 따라서 공개 설정은 "논문 방법의 예시"이지 논문 표 1을 재현하는 설정이 아닙니다. 문서의 "~35%→~50% 시뮬 성공"은 논문의 시뮬 곡선(Fig. 11)과 다른 과제의 값이므로 비교하지 않았습니다.
- **D6 (성공 판정과 평가 코드 없음):** 실로봇 성공 판정(App. D.6)과 세션 평가 스크립트가 없습니다. 시뮬 성공 판정은 환경 코드의 `success`(배치 기준)가 있지만, 실로봇 판정과 같다고 볼 수 없습니다.
- **D7 (실로봇 배포와 GPU 시뮬):** 공개 예시는 GPU 시뮬 백엔드(`sim_backend: gpu`)와 CUDA 도커(`--gpus all`)를 씁니다. 실로봇(Franka, RealSense) 배포 코드는 이 설정에 없습니다. 논문 App. E.1의 시스템 식별(강성, 감쇠, 시뮬레이티드 어닐링)도 저장소에 없습니다.
- **D8 (MimicGen 파이프라인 없음):** 시뮬 데이터 생성(시드 재생, 핵심 단계만 남기는 가지치기, B.2)은 저장소에 없습니다. 공개 데이터셋(LeRobot 형식)으로만 공개됩니다. 이 데이터셋을 받으면 학습 데이터는 확인할 수 있지만, 생성 과정은 확인되지 않습니다.
- **D9 (PPO 세부 설정):** 논문은 PPO의 clip, γ, GAE, 에포크, 엔트로피 계수를 적지 않습니다. 저장소 값은 있지만 논문과 맞는지 확인할 수 없습니다(부록 I는 PPO와 학습된 가치 함수만 말함).
- **D10 (논문 값 누락):** 액션 청크 길이, 흐름 단계 수, 배치 크기, 시뮬 최대 길이, 카메라 해상도가 논문에 없습니다. 재현하려면 이 값을 저장소에서 가져와야 합니다.

## 4. 구현 가능성 판정 (이 맥)

**판정: 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 논문 수치의 재현은 불가능합니다. 방법의 핵심(두 단계 손실 결합)은 CPU와 MPS로 축소판을 만들 수 있습니다(조건부).**

| 구성 | 맥에서 가능한가 | 근거 |
|---|---|---|
| 공식 π0.5 Stage II (RLinf, GPU 시뮬) | **불가.** Linux + CUDA 도커 설치가 문서 전제이고, 시뮬이 GPU 백엔드입니다. macOS에서 이 시뮬 백엔드를 쓸 수 있는지는 확인하지 않았습니다(추정: 불가). | 문서 "Installation", 설정 `sim_backend: gpu` |
| 공식 π0.5 Stage I (SFT) | **불가.** 모델 규모(PaliGemma 2B + 전문가 300M급 설정)와 GPU 학습을 전제합니다. | `pi0_config`(gemma_2b, gemma_300m), 문서 Option A |
| OpenVLA 공동 학습 | **불가.** 공개 코드가 없고(D4), 7B급 가중치는 1 GB를 훨씬 넘습니다. | D4 |
| 실로봇 재현 | **불가.** Franka 하드웨어, 카메라, 배포 코드가 필요합니다. | D6, D7 |
| 손실 결합과 β 민감도 (토이) | **가능.** numpy 토이 완료(03 §3). PyTorch 소형 정책은 MPS나 CPU에서 가능합니다(torch 2.15는 설치되어 있고 승인됨). | 03 §3 |
| 시뮬 RL 축소판 (MuJoCo, 접촉 있는 소형 과제, 실데이터 BC 앵커) | **조건부.** mujoco 4.0과 torch 2.15는 이미 설치되어 있어 새 설치는 없습니다. 처리량은 측정하지 않았습니다. | `03` §3.3 (토이의 한계) |
| 설정 파일 이해와 대조 (이 문서) | **가능.** 완료. | §2 |

- **구현 노력(추정):** 방법의 축소판(손실 결합 + 실데이터 앵커 + 소형 정책)은 **2–3주**. 공식 파이프라인을 Linux GPU 기기에서 돌리면 설치 1–2일과 학습 시간이 추가됩니다. 논문 J.2의 예시로, OD 한 과제의 MimicGen 생성 1.5 h(RTX 4090)와 RL 학습 9.43 h(2 x H100)가 걸립니다. Linux 기기에 GPU가 있는지는 확인하지 않았습니다.
- **결론:** 맥에서 할 수 있는 것은 "방법이 어떻게 작동하는가"를 작은 모델로 보는 것입니다. "논문 표 1의 숫자가 맞는가"는 맥에서 답할 수 없습니다. 이 구분은 재현 가능성 표에 그대로 씁니다.

## 5. 저자에게 확인할 질문

1. 표 7의 β(PnP 1.0, OD 1.0)와 공개 설정 β=0.2의 차이는 무엇인가? 과제별 β는 무엇으로 골랐나?
2. 그림 5b의 40.3%는 어떤 시도 수와 세션 구성에서 나왔나? (2 x 32 시도로는 나오지 않음)
3. 그림 6의 SFT 20 시연 값(약 0.05)과 표 6의 10%는 같은 체크포인트인가?
4. 표 1의 Avg SD(OpenVLA RL-Co ±0.7, π0.5 실데이터만 ±1.4)는 어떻게 계산했나? Eq. (9)로는 10.6과 6.1입니다.
5. OpenVLA 공동 학습 설정은 공개할 계획인가? Liu 외 [12]의 구현 버전은?
6. 실로봇 성공 판정 기준과 세션별 초기 조건 목록을 공개할 수 있나?
7. 시뮬 데이터(MimicGen 시드, 가지치기 규칙)의 생성 코드는 공개되나?

## 6. 요약

- 공식 저장소는 **π0.5 공동 학습 예시와 그 보상·학습률 설정**을 제공하고, 그것은 논문의 식 (2)와 부록 E.2의 보상과 **맞습니다**.
- 그러나 **논문 실험의 설정(과제, 데이터 수, β, OpenVLA)은 공개 예시와 다르고**, 실로봇과 평가 코드는 없습니다.
- 따라서 이 저장소로는 방법의 **구조**를 확인할 수 있지만 **논문 수치를 재현할 수는 없습니다**.
