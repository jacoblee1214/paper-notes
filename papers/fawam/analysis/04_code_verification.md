# 04. 논문 ↔ 공식 코드 대조 (FAWAM, arXiv:2606.08555 v3)

공식 저장소: https://github.com/HaotianHehaha/FAWAM (MIT, 커밋 `7d90af0`, 2026-06-21, 추적 파일 83개, 얕은 clone을 `repo/`에 둠). 실행, import, 설치, 가중치 다운로드는 하지 않았고 파일을 정적으로만 읽었습니다. 저장소 내용은 외부에서 받은 자료이므로 코드 주석과 설정은 데이터로만 취급했습니다.

## 1. 한눈에 보기

- **공개 범위.** README는 "이번 공개는 Force-Envisioned Action Model의 학습과 추론 코드이고, Force-Guided Residual Correction 코드는 정리 중이라 나중에 공개한다"고 씁니다. 논문의 **두 번째 기여인 잔차 보정 모듈(식 5–10, 게이트, τ = 0.8, 10 Hz 루프, 교정 데이터 수집)은 코드에 없습니다**(D1). 표 1의 FAWAM 85%와 표 2–4의 보정 관련 수치는 이 저장소로 재현할 수 없습니다.
- **일치하는 부분.** 힘 인코더 입력(0.5 s, 60프레임 x 6축 = 360), AdaLN-Zero 오프셋의 영 초기화, 행동과 힘의 결합 투영(14 = 8 + 6), 학습률·옵티마이저·스케줄·warmup·gradient clip·bf16·ZeRO-2 설정, λF = 1, 영상 단계 30k 스텝과 학습률 3e-5, 행동 단계 학습률 1.5e-4가 설정 파일과 맞습니다.
- **불일치 또는 확인 불가.** 행동 단계 배치 크기(D3), 힘 인코더 은닉층 폭(D2), 손실의 평균 정규화 방식(D4), 10 Hz 저역통과 필터와 1 Hz 행동 주기의 구현 위치(D5, D6)가 논문과 다릅니다 또는 확인되지 않습니다.

## 2. 대조표

판정: **일치** = 설정과 코드가 논문 값과 같음, **불일치** = 값이나 구조가 다름, **미공개** = 저장소에 없음, **확인 불가** = 저장소만으로 판단할 수 없음.

| ID | 논문 위치 | 논문 내용 | 코드 위치 | 판정 |
|---|---|---|---|---|
| — | 3.2, 식 (1) | 힘 특징 c_t = Enc_f(w_{t−H_f+1:t}), 6축 힘·토크 | `configs/wipe_vase/policy_model_wipe_vase_ft.yaml`: `ft_input_dim: 360` (= 60 x 6), `ft_encoder_type: mlp`, `ft_output_dim: 256`, `models/ft_encoder/ft_encoder_torch.py` | 일치 |
| D2 | A.1 | 힘 인코더는 "은닉 폭 256의 3층 MLP" | `FTMLPEncoder`의 기본값 `hidden_dims=(256, 128)`, 뒤에 출력 투영 Linear 256. 설정 파일은 `mlp_hidden_dim`을 바꾸지 않음 | 불일치 (선형층 3개는 같지만 두 번째 은닉층이 128) |
| — | 3.2, 식 (2) | AdaLN-Zero 파라미터에 힘 조건 오프셋 MLP_f(c_t)를 더함, 마지막 층 영 초기화 | `models/action_patches/patches_ft.py`: `ft_adaLN_modulation`(Linear → 6·dim, 마지막 층 가중치와 편향을 0으로), `ada_values = ada_values + ft_ada_values` | 일치 (구조). 코드는 힘 토큰 하나를 RMSNorm 뒤 MLP에 넣는 전역 조건 |
| — | 3.2 | 행동과 손목 힘을 한 선형 투영으로 함께 출력 (Z_t = [A_t, W_t]) | `action_in_channels: 14` (= action_dim 8 + force 6), `force_out_channels: 6`, `joint_action_force: true` | 일치 |
| D4 | 식 (4), 3.2 | L_stage2 = L_action + λ_F L_force, λ_F = 1 | `runner/ge_trainer_ft.py` 약 826–840행: 두 항 모두 `(pred − target)²`의 **요소별 평균**, `loss = action_loss_scale·loss_action + force_loss_scale·loss_force` (둘 다 1.0). 식 (4)의 제곱 노름이 차원 합이라면 코드는 힘 항을 상대적으로 약 4/3배 크게 셈 (1/6 대 1/8) | 불일치 가능 (표기 해석에 따라 다름) |
| — | 3.2 식 (3), A.1 | 영상 단계: GE-Base 기반 latent flow matching, 30k 스텝, 학습률 3e-5 | `configs/wipe_vase/video_model_wipe_vase.yaml`: `train_steps: 30001`, `lr: 3e-5`, `batch_size: 8` | 일치 (스텝은 30000 대 30001) |
| — | 3.2, A.1 | 행동 단계: 30k 스텝, 학습률 1.5e-4, AdamW(β1 0.9, β2 0.95, ε 1e-8, wd 1e-5), warmup 1000, constant_with_warmup, clip 1.0, bf16, ZeRO-2 | `configs/wipe_vase/policy_model_wipe_vase_ft.yaml`: 같은 값, `use_deepspeed: true`, `deepspeed.zero_optimization.stage: 2` | 일치 |
| D3 | A.1 | 8 x A100, 전체 배치 64 (두 단계 공통) | 행동 단계 설정 `batch_size: 16`(장치당), `runner/ge_trainer_ft.py` 약 210행: `train_batch_size = batch_size × WORLD_SIZE × grad_accum`. 8 GPU면 128. 영상 단계 `batch_size: 8`이면 8 x 8 = 64로 맞음 | 불일치 (행동 단계만 128) 또는 GPU 수 불명 |
| D5 | A.1 | 힘은 120 Hz로 기록하고 2차 Butterworth 저역통과(10 Hz 차단)를 적용 | 데이터 키 `observation.state.wrench_filtered_chunk`를 그대로 읽음. 필터 코드는 없음 | 확인 불가 (필터는 데이터 생성 단계에 있는 것으로 보임) |
| D6 | 3.1 | 행동 모델 1 Hz, 잔차 보정 10 Hz | 학습 코드의 영상 fps 기본값 `basic_fps` 30. 1 Hz 설정은 없음. 잔차 보정이 없어 10 Hz 루프도 없음 | 확인 불가 (행동 주기는 추론 서버 쪽에서 봐야 함) |
| D1 | 3.3, 식 (5)–(8), 3.3.2, 식 (9)–(10), A.2 | 잔차 보정기(이중 헤드 MLP, 게이트, τ = 0.8, λ_gate = 1, 500 epoch, 학습률 3e-4) | 저장소에 없음. README: "still being organized" | 미공개 |
| D1' | 3.3.2 | 교정 데이터는 FACTR 힘 피드백 텔레오퍼레이션으로 수집 | 없음 (FACTR은 별도 저장소) | 미공개 |
| — | 4.1 | 실험 로봇 Franka FR3, ATI Axia80-M8, RealSense D435i x 3 | `web_infer_utils/`(WebSocket 서버와 클라이언트)만 있음. 로봇 드라이버 없음 | 확인 불가 (하드웨어 코드 없음) |
| — | 4.1, A.1 | 과제 네 개: 칠판 지우기, 오이 깎기, 상자 피벗, 꽃병 닦기 | HF 데이터셋 `hahate/FAWAM`: `wipe_board`, `peel_cucumber_mix`, `pivot_box`, `wipe_vase` (Zarr, zip 네 개) | 일치 (이름이 다름: 칠판 = `wipe_board`, 오이 = `peel_cucumber_mix`) |
| — | 3.1 | 베이스는 GE-Act(GE-Base 기반) | README: LTX-Video의 토크나이저, 텍스트 인코더, VAE와 GE-Base 체크포인트 사용 | 일치 |
| D7 | 3.2 식 (4) | 디코더가 "joint velocity"를 예측 | `action_type: absolute`, `action_space: eef`. 식의 "velocity"는 플로 매칭의 속도장으로 보임 | 표현 모호 (논문의 표현을 바로잡을 것) |
| — | 4.1 | 학습 시 힘 목표는 미래 손목 힘 | `data/lerobot_like_dataset.py` `extract_force_targets`: 행동 창에 맞춘 힘의 마지막 6채널, ee_ft 통계로 정규화 | 일치 (정의와 정규화 방식은 코드로 확인) |

## 3. 코드에서만 보이는 것 (논문에 없음)

- **학습 손실의 영상 항.** `train_mode == action_only`이면 영상 손실은 0으로 처리됩니다(`ge_trainer_ft.py` 약 786–800행). 2단계에서 영상 모델이 얼어 있다는 논문 설명과 맞습니다.
- **힘 목표 정렬.** 힘 목표는 행동 창과 같은 길이로 잘린 뒤 마지막 6채널을 씁니다. 식 (4)의 W_t가 행동과 같은 시간축이라는 뜻입니다.
- **추론 서버.** `web_infer_scripts/main_server.py`는 WebSocket으로 다중 시점 영상, 상태, 지시문, 선택적 힘 이력을 받습니다. 힘 이력이 없을 때의 동작은 확인하지 않았습니다.
- **설정 값의 출처.** 설정 파일에는 저자 실험 환경의 절대 경로가 그대로 들어 있습니다. 실험 환경을 옮긴 흔적으로 보이며, 이 분석에서는 경로를 옮기지 않았습니다.

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 논문 결과는 재현할 수 없습니다. 힘 조건 행동 모델의 학습은 불가능하고, 잔차 보정은 코드가 없어 구현 자체가 필요합니다.**

| 항목 | 필요 자원 | 이 맥에서 | 근거 |
|---|---|---|---|
| 영상 단계 학습(30k 스텝, 배치 8 x 장치 수) | 설정 기준 A100급 GPU 다수, 수일 | 불가 | 설정 파일, 논문 A.1 |
| 행동 단계 학습(30k 스텝, 배치 16 x 장치 수) | 같음 | 불가 | 설정 파일 |
| 추론 서버 실행 | 대형 DiT(28층, 2048 폭, 추정 약 2B 파라미터) + T5 텍스트 인코더 + VAE. 크기는 설정으로 계산한 추정이고 측정하지 않음. 코드는 MPS에서 혼합 정밀도를 끄므로 메모리가 더 듦 | 미확인 (메모리 부족 가능성 큼) | `ge_trainer_ft.py` 205행, 설정 |
| 데이터 | 네 과제 zip 합계 약 9.7 GB (HF 메타데이터, 받지 않음) | 디스크와 시간 문제, 받을 수는 있음 | HF API 파일 크기 |
| 실로봇 재현 | FR3, ATI 힘 센서, RealSense x 3, FACTR 텔레오퍼레이션 | 불가 (하드웨어) | 논문 4.1 |
| 잔차 보정 재현 | 코드 없음. 식 (5)–(8)을 새로 구현하고 교정 데이터를 모아야 함 | 구현은 가능하나 데이터는 로봇 필요 | D1 |
| CPU 축소 실험 | 1자유도 또는 2자유도 접촉 시뮬레이션에서 예측 힘 기준 보정이 측정 힘만 쓰는 보정보다 나은지 비교 | 가능 (torch·numpy로 새 설치 없이 가능한지는 미확인) | 이 분석에서는 수행하지 않음 |

**노력 추정.** (a) 공개 코드로 FEAM(보정 없음)을 GPU에서 학습하고 시뮬레이션이나 데이터셋으로 검증하는 데 1–2주(GPU 자원 포함). (b) 잔차 보정기를 식 (5)–(8)대로 새로 구현하는 데 1주. (c) 로봇 실험을 재현하는 것은 하드웨어 확보에 달려 있으며 이 저장소로는 시작할 수 없습니다.

## 5. 저자에게 물을 것

1. 잔차 보정 코드와 교정 데이터 공개 시점(D1).
2. 행동 단계 전체 배치 64인지, 장치 수와 배치 설정(D3).
3. 힘 인코더의 은닉 폭(256 대 256·128)(D2).
4. 식 (4)의 제곱 노름이 차원 합인지 평균인지(D4).
5. 10 Hz 저역통과 필터와 1 Hz 행동 주기를 구현한 위치(D5, D6).
6. 표 4의 "Slice cucumber" 열이 본문의 "pivot box"를 가리키는지(`03` §5 참고).

## 6. 확인 방법

- 파일 목록: `git -C repo ls-files` (83개).
- 설정 값: `repo/configs/wipe_vase/policy_model_wipe_vase_ft.yaml`, `repo/configs/wipe_vase/video_model_wipe_vase.yaml`.
- 손실과 배치: `repo/runner/ge_trainer_ft.py`의 `loss_action`, `loss_force`, `train_batch_size`.
- 힘 조건: `repo/models/action_patches/patches_ft.py`의 `ft_adaLN_modulation`.
- 힘 인코더: `repo/models/ft_encoder/ft_encoder_torch.py`의 `FTMLPEncoder`, `FTEncoder`.
