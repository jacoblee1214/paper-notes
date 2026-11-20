# 04. 코드 대조와 구현 가능성

대상: 공식 저장소 `ShuangLI59/unified_video_action`, 커밋 `fe3dec8` (2026-08-20), MIT. 복제본은 `repo/` (gitignore). **정적으로만 읽었다.** 코드를 실행하거나 가져오거나 설치하지 않았다. 줄 번호는 이 커밋 기준이다.

## 1. 대조 결과 요약

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 추론 때 영상 생성 건너뛰기 | §III-C | `mar_con_unified.py` L1047: `policy_model`이면 행동 확산 샘플 후 반환, 영상 확산 실행 전 | 일치 (D3 참고) |
| 행동 청크 16, 실행 8 | §V, Table I 설명 | `horizon 32`를 과거·미래 절반씩 나눔 (`data_utils.py` L368–386), `n_action_steps 8` | 일치 |
| VAE kl-f16, MAR-B | §III-A, 부록 §X-A | `kl16.ckpt`, `vae_embed_dim 16`, `mar_base` | 일치 |
| 행동 확산 학습 스텝 1000 | §III-C | `act_diff_training_steps 1000` | 일치 |
| 시뮬 추론 확산 100 | 부록 §X-D | `act_diff_testing_steps "100"` | 일치 |
| 실로봇 추론 확산 16 | Table II 설명 | 기본값 100. 16은 설정에 없음 | **불일치 또는 미확인** (D5) |
| 손실 L = L_action + L_video | §III-C 식 | 같은 배치에서 두 손실을 합하는 모드는 `full_dynamic_model` 하나 (L763–779) | **불일치** (D1) |
| 학습 방식 | §I: "두 손실을 동시에 학습" | README: "two-stage training works better"; 스크립트는 영상 단계 후 행동 단계 | **불일치** (D2) |
| 마스크 학습 다섯 과제 | §III-D, 그림 1b | 5개 모드 (`policy_model.py` L124–131) | 일치 |
| 언어 CLIP 토큰 | §III-B | `text_model` CLIP, Libero10 `language_emb_model: clip` | 일치 |
| Libero10 평가 규모 | §V-A 50/과제 | `libero10.yaml` `n_test: 3` (기본) | **불일치** (D7) |
| 실로봇 입력 | 논문 설명 없음 | README 명령에 proprioception, 이력 주파수 옵션 | **논문에 없는 설정** (D6) |

## 2. 차이 목록 (D1–D10)

**D1. 학습 목적 (손실 합 대 무작위 모드 선택) — 중요**
- 논문: 식(본문 §III-C)에서 L = L_action + L_video를 시간 지평에 걸쳐 합해 같은 배치에서 최적화한다.
- 코드: `selected_training_mode`를 지정하지 않으면 `task_modes`가 다섯 개가 된다 (`unified_video_action_policy.py` L122–131). 배치마다 `random.choice`로 한 모드를 고른다 (L408). 모드 `video_model`은 영상 손실만, `policy_model`과 `inverse_model`은 행동 손실만 쓴다 (`mar_con_unified.py` L739–761). **두 손실을 함께 쓰는 모드는 `full_dynamic_model` 하나뿐이다** (L763–779). `policy_model_full_dynamics_model` 옵션은 두 모드 (`policy_model`, `full_dynamic_model`) 중 하나를 고른다 (L133).
- 영향: 기본 설정으로 학습하면 배치의 대략 1/5만 공동 손실을 받는다. 논문이 말하는 "매 배치 두 손실 동시 최적화"와 다르다. 논문이 실제로 어느 설정으로 표를 냈는지는 알 수 없다. 공동 손실의 효과를 주장하는 Table I의 UVA 대 UVA-action 비교가 어느 모드로 학습됐는지 본문이 밝히지 않는다.

**D2. 두 단계 학습 대 동시 학습**
- 논문 §I: "UVA is trained simultaneously with supervision from both video and action data."
- README: "We found that two-stage training works better than training on both video and action tasks directly." 스크립트 `train_uva_libero10.sh` 등은 영상 전용 단계 (`selected_training_mode=video_model`)를 먼저 돌리고, 주석 처리된 두 번째 단계에서 `predict_action=True`와 기존 체크포인트 로드를 쓴다.
- 판정: 논문 본문에는 두 단계 학습이 없다. 재현하려면 이 두 단계 절차를 따라야 한다.

**D3. 추론 때 영상 생략 (일치, 근거 확인)**
- `sample_tokens`는 `predict_video`가 참인 경우 첫 스텝에서 `policy_model`이면 행동 확산 샘플만 뽑고 바로 반환한다 (L1047). 영상 확산 루프의 나머지는 실행되지 않는다. 논문 §III-C와 Table II, VII의 속도 해석을 뒷받침한다.
- 주의: `forward_mae_encoder`와 `forward_mae_decoder`는 한 번 실행된다. 즉 추론에는 Transformer 한 번과 행동 확산 100스텝이 든다 (Table VII의 40 + 40 + 93 ms 구조와 맞다).

**D4. 행동 청크 길이 (일치, 해석 주의)**
- `horizon 32`, `n_obs_steps 16`, `n_action_steps 8`. `get_trajectory`는 32 길이 창을 절반으로 나눈다. 앞 16은 이력, 뒤 16은 미래 행동이다 (shift_action=True이면 한 칸 밀린다). 따라서 미래 행동 청크는 16이고 실행은 8이다. 논문과 맞는다.

**D5. 실로봇 추론 확산 스텝**
- Table II 설명: 실로봇 UVA와 DP-UMI 모두 16 확산 스텝. 본문은 100 스텝이 더 부드럽지만 느리다고 쓴다 (§V-D, 부록 §X-D).
- 코드: `uva.yaml`의 `act_diff_testing_steps`는 "100". 실로봇 설정 파일 (`uva_umi*.yaml`)에서 이 값을 16으로 바꾸는 줄이 없다. `eval_real.py` L81–87은 `num_sampling_steps` (영상 헤드)를 100으로 바꾼다. 행동 헤드 값은 건드리지 않는다.
- 판정: 기본 실행은 100 스텝이다. 16 스텝 결과를 재현하려면 체크포인트 설정을 바꿔야 한다. 공개 체크포인트의 저장된 설정은 확인하지 않았다 (다운로드하지 않음).

**D6. 실로봇 설정이 논문 방법 설명에 없다**
- README 실로봇 명령: `use_proprioception=True`, `predict_proprioception=True`, `shift_action=False`, `different_history_freq=True`. 논문 §III에는 고유감각(proprioception) 입력이나 이력 주파수 옵션이 없다. 그림 6 축 이름에만 "Proprioception"이 나온다.
- 판정: 실로봇 재현에는 논문 본문에 없는 설정이 필요하다.

**D7. Libero10 평가 횟수**
- 논문 §V-A: 과제당 50 환경. 부록 §X-B: 과제당 10회, 총 500 에피소드.
- 코드 `libero10.yaml`: `n_test: 3`, `n_test_vis: 1`. 기본 평가는 과제당 3회다. 표 VIII의 "30회" 칸과 맞는다. 50회 평가 설정은 저장소에 없다.
- 판정: 본문 50회와 부록 10회가 서로 다르고, 코드 기본값은 3회다. 보고 수치를 재현하려면 평가 횟수 설정을 직접 바꿔야 한다.

**D8. PushT-M 데이터 규모**
- 논문: 새 과제로 소개하지만 시연 수를 쓰지 않는다. README: 247개 시연. 판정: 논문 보충 정보로 충분하지 않다.

**D9. 입력 해상도와 학습 길이 (논문 미기재)**
- 해상도: Libero10 128, PushT 96, ToolHang 240, UMI 224 (task 설정). 논문은 해상도를 쓰지 않는다.
- 학습 길이: 모든 설정 `num_epochs: 3050`, 학습률 1e-4, 가중치 감쇠 0.02, EMA 사용. 논문은 학습 예산을 쓰지 않는다.

**D10. 마스킹 비율**
- 논문 부록 Table IX는 마스킹 비율 25/50/75%를 독립 변수로 쓴다.
- 코드: 영상 마스크는 `mask_ratio_min 0.7`의 잘린 정규분포에서 샘플한다 (`mar_con_unified.py` L86–87, L428). 행동 마스크는 `action_mask_ratio 0.5` (`uva.yaml`). Table IX의 비율을 어떤 설정 값으로 바꿨는지 논문과 코드 사이에 대응이 없다.
- 판정: 확인 필요.

**D11. 기타 환경 제약 (재현성)**
- `conda_environment.yml`: Python 3.9, PyTorch 2.1.0, pytorch-cuda 12.1, accelerate 1.0.1, gym 0.21.0, pymunk 6.11.1.
- CUDA 참조가 10개 파이썬 파일에 있다. 혼합 정밀도는 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` (workspace L260)다. 학습 스크립트는 `accelerate launch --num_processes=8`을 쓴다. 평가 설정도 `device: cuda:0`이다.
- 판정: Mac (MPS)에서 돌리려면 장치와 혼합 정밀도 코드를 바꿔야 한다.

**D12. 라이선스**
- 저장소 MIT (GitHub API로 확인). 논문 CC BY 4.0. 두 라이선스는 서로 다른 대상이다. 코드 재사용은 MIT 조건을 따른다.

## 3. 정적 확인에서 본 위험 신호

- 저장소에는 학습 로그, 결과 CSV, 평가 스크립트 출력이 없다. Table I, II를 재현했다는 기록이 없다.
- `policy_model.py` L408의 무작위 모드 선택과 `mar_con_unified.py`의 손실 분기는 학습 결과에 직접 영향을 준다. 논문 식과 다르므로 결과 해석에 주의가 필요하다.
- 이 대조는 코드 읽기로 한 것이다. 실제 학습으로 D1이 성능에 미치는 영향은 확인하지 않았다.

## 4. 구현할 수 있는지 (판정)

**판정: 부분적으로 가능하다. 이 맥에서 원본 학습은 불가능하고, 추론 평가는 조건부로 가능하다.**

| 구분 | 이 맥 (M4 Pro, 24 GiB, MPS, CUDA 없음)에서 | 필요 자원 | 노력 추정 |
|---|---|---|---|
| 코드 정적 대조, 표 재계산, 토이 | 가능 (이 분석에서 완료) | 없음 | 완료 |
| PushT 체크포인트 추론과 평가 | 조건부. 체크포인트와 데이터 다운로드 승인, 장치 코드 수정(cuda → mps), pymunk 환경 설치 승인 필요 | 체크포인트 (구글 드라이브), PushT 데이터 | MPS 포팅 포함 3–5일 |
| Libero10 추론 평가 | 조건부 (느림). LIBERO와 mujoco 환경을 설치해야 함. 50회 설정은 코드 수정 필요 | LIBERO 데이터와 체크포인트, mujoco | 1–2주 |
| 원본 학습 (PushT, Libero10, UMI 다중) | 불가. 공식 설정이 8 GPU 가정이고, 영상 단계와 행동 단계를 따로 돌린다 | 다중 GPU, 수 일 이상 (GPU 시간은 논문에 없음) | 사용자 확인 후 GPU 환경에서만 |
| 실로봇 재현 | 불가 | ARX X5, UMI 하드웨어 | 불가 |
| 자체 소형 구현 (PushT 영상 없는 축소판) | 가능. 작은 Transformer와 확산 헤드를 MPS나 CPU에서 학습 | 자체 코드 | 2–3주 |

- 재현 가치가 있는 질문은 D1 (손실 설계)과 D5 (추론 확산 스텝) 두 가지다. 둘 다 체크포인트 없이 토이나 작은 학습으로 시험할 수 있다. 본격적인 재현은 GPU가 필요하므로 사용자와 상의해야 한다 (`CLAUDE.md`).
- 추론 지연 비교는 이 맥에서 측정해도 Table VII과 같은 조건이 아니다 (CUDA GPU, flash attention 없음). 절대값 비교는 하지 않는다.
