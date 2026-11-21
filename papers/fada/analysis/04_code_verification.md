# 04. 공식 코드 대조와 구현 가능성

대상은 LeCAR-Lab/FADA의 커밋 `290b3a9`(2026-09-09, 커밋 1개)입니다. `repo/`에 얕은 복제로 두었고 **정적으로만 읽었습니다.** 설치하거나 실행하거나 import하지 않았습니다. 내려받은 내용은 신뢰하지 않는 데이터로 다뤘습니다. 대조 기준은 논문 v2의 본문과 부록입니다.

## 1. 라이선스와 출처

- **FADA 저장소:** Apache-2.0. `LICENSE` 파일이 있고 GitHub API도 Apache-2.0으로 답합니다. `NOTICE`와 `THIRD_PARTY_LICENSES`가 있습니다.
- **상위 저장소:** README는 FADA가 [amazon-far/holosoma](https://github.com/amazon-far/holosoma)(Apache-2.0, Amazon FAR)을 확장한 것이라고 밝힙니다. FADA 추가분은 `src/holosoma/holosoma/fada/`와 `src/holosoma_inference/`의 `LocomotionPolicy_FADA`입니다. 나머지는 상위 저장소의 코드입니다.
- **결론:** 라이선스 문제는 보이지 않습니다. 저장소는 Apache-2.0 규정에 따라 복사와 재사용이 가능합니다. 다만 이 분석에서는 코드를 복사하지 않았습니다.

## 2. 논문 ↔ 코드 대조

판정: **일치** (논문 기술과 코드 기본값이 같음), **차이** (기본값이나 동작이 다름), **누락** (논문이 말하지만 코드에 없거나, 코드에 있지만 논문에 없는 중요한 설정).

| 번호 | 항목 | 논문 | 코드 (파일:위치) | 판정 |
|---|---|---|---|---|
| D1 | 목표 적응 학습률·스텝·정규화 | 학습률 3×10⁻⁴, 800 스텝, 배치 4096 (B.3) | `planner_idm/finetune_idm_lora.py:796–823`: 학습률 **1×10⁻⁴**, `max_train_steps` **100**, 가중치 감쇠 1×10⁻³, 기울기 클리핑 0.5. README 5단계 명령은 이를 바꾸지 않습니다. 테스트(`tests/fada/test_finetune_idm_lora.py:355–358`)도 1×10⁻⁴를 기대합니다 | **차이** |
| D2 | 부분 최적 데이터 양 | 최적 데이터의 **2배** (B.2) | `planner_idm/config.py:210`: `suboptimal_data_ratio = 1.0`. `offline_data_utils.py:446`: 목표 수 = ceil(최적 수 × 비율). README 2단계는 비율을 쓰지 않습니다 | **차이** (기본값 1.0, 논문 2.0) |
| D3 | 학습 증강과 보조 단계 | 논문에 없음 (B.2는 IDM 손실과 교차 학습 쌍만 설명) | `config.py`: 관측 노이즈 증강 **켜짐** (항목별 잡음 `base_ang_vel` 0.15, `projected_gravity` 0.02, `dof_vel` 0.4, `dof_pos` 0.02), 행동 노이즈 ±0.01, 히스토리·미래 마스킹 0.2씩, 오프라인 혼합 비율 0.2, 워밍업 2000 스텝, 검증용 궤적 15% 분할, 소스 IDM 배치 비율(온라인 궤적 0.5, 부분 최적 전문가 0.5) | **누락** (논문 설정에 없는 학습 요소) |
| D4 | DAgger 반복과 샘플 수 | 논문에 수치 없음 | `config.py`: DAgger 30회, 반복당 롤아웃 500 스텝, 학습 1000 스텝, 환경 1024개, 명령 재샘플 간격 200 스텝 | **누락** |
| D5 | IDM·플래너 손실 | 식 4.2(첫 행동 MSE), 식 4.3(IDM 기울기 정지, 관측 회귀 없음), 절제 A·B | `config.py`: `weighted_horizon_mse`(`loss_utils.py`)에서 `gamma = 0`이면 첫 스텝만 씀. `planner_obs_loss_coef = 0.0`, `idm_planner_pass_action_loss_coef = 1.0`, `idm_detach_planner_future_obs = True` | **일치** |
| D6 | 구조 치수와 예측 | 플래너 3층·4헤드·128, IDM 인코더 3층·디코더 2층·128, H = 30, K = 6, 플래너는 잔차 예측 (B.1) | `config.py`: `planner_num_layers = 3`, `planner_nhead = 4`, `planner_d_model = 128`, `idm_encoder_num_layers = 3`, `idm_decoder_num_layers = 2`, `history_len = 30`, `pred_horizon = 6`, `planner_predict_delta = True`. `model.py`에서 IDM 디코더는 `nn.TransformerDecoder`, 인과 마스크 없음 | **일치** |
| D7 | 적응 LoRA | 랭크 8, α 16, 드롭아웃 0.05 (B.3). 대상 모듈은 명시하지 않음 | `finetune_idm_lora.py:799–801`: `lora_r = 8`, `lora_alpha = 16`, `lora_dropout = 0.05`. 대상은 `lora_target_scope = "encoder_decoder_qkv"` (논문에 없음) | **일치** (대상 모듈은 논문 누락) |
| D8 | 적응 배치 | 4096 (B.3) | `finetune_idm_lora.py:784`: `batch_size = 4096` | **일치** |
| D9 | 오라클 학습 | 논문에 반복 수와 구조 없음 | `config_values/loco/g1/experiment.py`: `g1_29dof_oracle`은 PPO 25,000 반복, 액터와 크리틱 은닉 [512, 256, 128]. README의 `model_24999.pt`와 맞습니다 | **누락** (논문 부록 없음) |
| D10 | 도메인 무작위화 (표 7, 지형·동역학 열) | 마찰 U(0.1, 2.0), 추가 질량 U(−3, 6) kg, 링크 질량 ×U(0.8, 1.3), 무게중심 ±0.15 m, Kp·Kd ×U(0.8, 1.2), 밀기 속도 U(0.1, 1.5), 밀기 주기 U(5, 10) s, 제어 지연 {0, 1} | `config_values/loco/g1/randomization.py:112–`와 `t1/randomization.py:116–`의 마지막 재정의가 위 값과 일치합니다 | **일치** (로코모션 열) |
| D10b | 같은 표 7의 전신 추종(WBT) 열 | 마찰 U(0.1, 2.0) | `config_values/wbt/g1/randomization.py:9`: 정적 마찰 [0.3, 1.6], 동적 [0.3, 1.2]. 그 밖의 WBT 값은 확인하지 않았습니다 | **차이** (WBT, 마찰) |
| D11 | 전신 추종 학습과 평가 | 표 1·2의 G1 Kungfu, 그림 1(f)의 춤 과제 | README 상태표: "Whole-body tracking" 항목이 **체크되지 않았습니다**. 관련 설정은 상위 저장소와 `wbt/` 디렉터리에만 있습니다 | **누락** (이 저장소에서 재현 불가) |
| D12 | 기준선 (TF–DAgger, TF–CoPred-zs/ft) | 논문 B.4가 구현 내용을 설명 | `tools/check_stale_identifiers.py:189–197`: "Deleted plain compact-obs-transformer baseline entry point and its wrapper scripts. Not part of this release". TF–CoPred 관련 파일은 저장소에 없습니다(검색 0건). `policy_mode="transformer"`는 플래너와 IDM 없는 모드로만 남아 있고, 논문 기준선과 같은지는 확인하지 않았습니다 | **누락** |
| D13 | MLP 절제 (표 8, 부록 C.1) | MLP 조합 MMM, MTM, TTM | `config.py:129–130`: "the fine-grained MLP/hybrid encoder-decoder switches have been folded away — always transformer". MLP 경로는 제거되었습니다 | **누락** |
| D14 | 평가 지표 계산 (정규화 Ēv, Ēmpjpe, 성공률 정의, TF–DAgger 기준 정규화) | 부록 A.4 식 A.1–A.3, 성공률 정의 | README 6단계: "What to measure, and how, is left to you." 지표 계산 코드는 없습니다. `holosoma_inference/utils/compute_mocap_velocity_metrics.py`가 있으나 논문의 정규화와 같은지 확인하지 않았습니다 | **누락** |
| D15 | 실험 데이터와 학생 체크포인트 | 표의 모든 수치 | README는 **오라클 체크포인트(T1, G1)**만 Hugging Face 링크로 밝힙니다(`AngchenXie/fada-checkpoints`, 받지 않음). 학생 체크포인트와 목표 롤아웃(H5)과 실로봇 로그는 공개 목록에 없습니다 | **누락** |
| D16 | 재표지 (오라클 롤아웃) | §4.1, B.2: 방문 상태마다 스냅샷을 저장하고 오라클을 K 스텝 굴림 | `planner_idm/trainer_rollout.py`가 `OracleRelabeledChunks`를 씁니다. RNG 상태 스냅샷 함수가 있습니다(`:77–89`). 시뮬레이터 상태 복원 경로의 세부는 읽지 않았습니다 | **일치** (존재 확인만) |
| D17 | IDM 학습 쌍 선택 | 식 B.2: 궤적 배치는 실제 쌍, 오라클 배치는 그림자 쌍 | `config.py`: `online_trajectory_batch_ratio = 0.5`, `suboptimal_expert_batch_ratio = 0.5`. 논문에는 이 비율이 없습니다 | **누락** (D3과 같은 묶음) |
| D18 | 적응 단계의 장치 | 논문은 GPU 학습을 전제하지 않음 | `finetune_idm_lora.py:1440–1446`: CUDA가 없으면 CPU로 **대체**합니다. `--device` 플래그가 있습니다. 적응 경로(`finetune_idm_lora.py`, `lora_utils.py`, `dataset.py`, `eval_checkpoint.py`)에는 IsaacSim 직접 import가 없습니다 | **확인** (CPU 실행이 코드상 가능) |
| D19 | 적응 손실 | 식 4.5: IDM 단독, 첫 행동 | 릴리스 경로는 손실을 `"idm"`으로 고정합니다(`finetune_idm_lora.py:1629`). 앵커 항(`idm_anchor_lambda`)은 읽히지 않습니다. 로그 이름 `idm_planner_fdm_obs`는 혼동을 줄 수 있습니다 | **일치** (기록용 설정이 있음) |
| D20 | 적응에 소스 데이터를 쓰는가 | 아니오 (B.3) | 주석: "buffer mixing off". 목표 데이터만 씀 | **일치** |
| D21 | 쓰이지 않는 설정 | - | `planner_mlp_hidden_dims`, `idm_mlp_hidden_dims` 등 MLP 치수가 설정에 남아 있으나 사용되지 않습니다 | **누락** (사소함) |

### 2.1 차이가 결과에 주는 영향

- **D1 (미세조정 스텝 100 대 800).** 같은 데이터로 한 번 돌리면 결과가 다를 수 있습니다. 논문 표 1·2의 적응 결과를 재현하려면 800 스텝과 학습률 3×10⁻⁴를 **명시적으로** 넘겨야 합니다. README의 5단계 명령은 논문 설정이 아닙니다.
- **D2와 D3.** 학생 학습 데이터 구성이 다르면 IDM의 분포가 달라집니다. 논문 본문에 없는 노이즈와 마스킹이 결과에 어느 정도 기여하는지는 코드 플래그로 절제해야 알 수 있습니다.
- **D10b.** WBT 마찰 차이는 전신 추종 과제에만 해당합니다. 이 저장소에서 WBT는 재현할 수 없으므로(D11) 실제 영향은 확인하지 못했습니다.

## 3. 구현 가능성 판정

| 항목 | 이 맥에서 가능 여부 | 근거 |
|---|---|---|
| 오라클 PPO 학습 (25,000 반복, 1024 환경) | **불가** | IsaacSim과 CUDA를 전제합니다. 맥은 CUDA가 없고, MPS는 Isaac 계열에 쓸 수 없습니다 (`README` 설치 절, `docker/isaacsim.Dockerfile`) |
| DAgger 학생 학습 (30 반복) | **불가** | 위와 같습니다. 학생 체크포인트가 공개되지 않았습니다 (D15) |
| 목표 롤아웃 수집과 MuJoCo 추론 (4단계) | **조건부** | MuJoCo 설치와 ONNX 추론이 필요합니다. 설치 스크립트는 Linux 중심입니다. 학생 ONNX가 없으면 시작하지 못합니다 |
| 목표 IDM LoRA 적응 (5단계) | **조건부** | 코드는 CPU로 대체할 수 있습니다 (D18). 그러나 학생 체크포인트와 목표 H5가 있어야 합니다 |
| 공개 오라클 체크포인트로 오라클 롤아웃 | **조건부** | 가중치 다운로드 승인이 필요합니다. 오라클은 특권 관측이므로 배포 정책이 아닙니다 |
| 논문 표 재계산 | **가능, 완료** | `analysis/recompute.py` |
| 역할 분리 토이 (논문 §5.7의 팔 실험 재구현) | **가능** | 새 구현이 필요합니다. 이 분석에서는 하지 않았습니다 |
| 실로봇 재현 (표 1) | **불가** | G1, T1 하드웨어와 WBT 코드가 없습니다 (D11) |

**판정 (한 줄):** 이 맥에서는 **학습과 시뮬 평가를 재현할 수 없습니다.** 공개된 것은 오라클 체크포인트와 적응 코드입니다. 학생 체크포인트와 목표 데이터가 없어 적응 실험의 입력도 없습니다. 맥에서 할 수 있는 것은 표 재계산, 코드 대조, 그리고 방법의 축소 재구현입니다.

**노력 추정 (미측정).** 맥에서 토이 재구현(플래너 + IDM + LoRA, 한 팔) 1–2일. 공식 파이프라인의 GPU 학습과 시뮬 평가는 IsaacSim과 CUDA 환경을 쓰는 별도 작업이므로 이 분석에서는 추정하지 않습니다.

## 4. 재현 단계의 체크리스트 (승인 전 메모)

- 설치 전 승인 항목: IsaacSim 5.1 (Ubuntu 22.04 이상, CUDA GPU), MuJoCo 3.x, ONNX 런타임, PyTorch, wandb(기본 `online`; 끄려면 `--wandb-mode disabled`).
- 가중치 다운로드 승인 항목: 오라클 체크포인트 (HF `AngchenXie/fada-checkpoints`). 크기는 확인하지 않았습니다.
- 실행 전 고정할 설정: D1(학습률 3×10⁻⁴, 800 스텝), D2(부분 최적 비율 2.0). D3의 증강 플래그를 켜고 끄는 두 조건을 나누어야 논문 설정에 가깝습니다.
