# 04. 공식 코드 대조 (정적 읽기)

공식 저장소: `github.com/yifan-hou/adaptive_compliance_policy`, 커밋 `74d68ec`(2026-04-18 "Updated bibtex"), MIT 라이선스, 별 145, 포크 14. 얕은 복제본은 `repo/`에 두었고 `.gitignore`에 넣었습니다. 코드를 실행하거나 설치하지 않았고, 모든 대조는 파일을 읽어서 했습니다. 데이터, 체크포인트, 가상환경도 받지 않았습니다.

## 1. 확인 대상 파일

| 역할 | 파일 |
|---|---|
| 가상 목표·강성 레이블 생성 | `PyriteEnvSuites/scripts/postprocess_add_virtual_target_label.py` |
| 강성 추정기 (`VirtualTargetEstimator`) | `PyriteUtility/planning_control/compliance_helpers.py` |
| 강성 행렬 재구성, 실행 | `PyriteEnvSuites/env_runners/virtual_target_real_env_runner.py` |
| 힘 스펙트로그램 인코더 | `PyriteML/diffusion_policy/model/vision/force_spec_encoder.py` |
| 작업 설정 (관측·행동 구간) | `PyriteML/diffusion_policy/config/task/flip_up_spec.yaml`, `flip_up_conv.yaml` |
| 학습 설정 (비전 백본) | `PyriteML/diffusion_policy/config/train_conv_workspace.yaml`, `train_spec_workspace.yaml` |

## 2. 대조 결과 (D1–D9)

판정 기호: 일치 = 논문과 코드가 같음, 차이 = 논문에 없거나 다름, 미확인 = 정적 읽기로 판단 불가.

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 강성 함수 형태 (식 7) | `k_max`, `k_min`, `f_min`, `f_max`의 구간별 선형 감소. 값은 "하드웨어에 따라" 정함 (값 없음) | `k_max = 5000`, `k_min = 200` N/m, `f_low = 0.5`, `f_high = 5` N (postprocess 설정). 같은 구간별 선형 식 | 형태 일치, **값은 논문에 없음** |
| D2 | 속도 기반 보정 | 없음 | `vel_tol` 이상이면 강성 방향 벡터에서 운동 방향 성분을 뺌 (`update`의 `flag_adjusted`). 기본 `vel_tol = 0.001` | **차이 (논문에 없음)** |
| D3 | 속도 추정 | 언급 없음 | 앞뒤 ±10 샘플 차분으로 twist를 계산 (`half_window_size = 10`) | **차이 (논문에 없음)** |
| D4 | 힘 평활 | 1초 창 이동 평균, 미래 접촉 정보 제공 | 7000 샘플(약 1초, 7 kHz) 대칭 창 `np.convolve(..., mode="same")`. 비인과 | 일치 |
| D5 | 강성 방향 | 식 6: 힘 방향은 낮은 강성, 나머지는 높은 강성 | 추정기는 스칼라 `k`와 힘 방향 오프셋(`f/k`)을 낸다. 행렬은 실행기에서 `S·diag(k, k_high, k_high)·S⁻¹`로 만든다. 방향은 기준 자세에서 가상 목표 쪽 | 일치 (방향 정의는 가상 목표 기준) |
| D6 | 높은 강성과 회전 강성 | `k_high`는 "경험적으로" 설정 (값 없음) | 병진 높은 강성 **5000 N/m**, 회전 강성 **100** 고정 (`default_stiffness`, `default_stiffness_rot`) | **차이 (논문에 없음)** |
| D7 | 힘 인코더 (스펙트로그램) | 6×30×17 (1초) | `signal.spectrogram`, fs = 7000, nperseg = 512, noverlap = 128, nfft = 1024, 0–200 Hz에서 자름. 1초당 30 주파수 칸 × 17 시간 칸 | 일치 (산술 검산 통과) |
| D8 | 인코더 백본 | "CLIP 사전학습 ViT-B/16" | `vit_base_patch32_clip_224.openai` (B/**32**) | **차이** |
| D9 | 행동 차원과 관측 구성 | 19차원 = 9(기준 자세) + 9(가상 목표) + 1(강성). 이미지 2프레임, 자세 3프레임 | `shape: [19]`, rgb 횟수 2, 저차원 3. 가상 목표는 저장 시 7차원 쿼터니언 자세 | 일치 |

## 3. 추가 차이: 힘 관측 시간 창 (D10)

논문 IV절은 "ACP w.o. FFT는 250 Hz에서 샘플링한 힘 32프레임"이라고 씁니다. 설정 파일 `flip_up_conv.yaml`은 힘 관측에 `sparse_obs_wrench_horizon: 32`, `down_sample_steps: 4`를 씁니다. 스펙트럼 설정(`flip_up_spec.yaml`)은 힘 원시 흐름을 7000 Hz로 가정합니다(`fs = 7000`, horizon 7000). 두 설정이 같은 원시 흐름이라면 4칸 건너뛰기는 약 1750 Hz이고, 32프레임은 약 18 ms입니다. 논문의 250 Hz, 128 ms와 맞지 않습니다. 원시 샘플링을 확인할 데이터가 없어 **미확인**으로 남깁니다.

## 4. 기타 확인 사항

- 레이블 평활 창(1초)과 강성 식은 논문 설명과 맞습니다. 다만 식 7 주석은 "저강성은 0이어야 하지만"이라고 쓰면서 실제 코드는 `k_min = 200`을 씁니다. 논문 본문에도 같은 모순이 있습니다.
- 논문은 "사용하는 가상 목표는 힘이 기준 힘을 낸다"라고 씁니다(IIIe절). 코드도 `pos_TC = f_reg / k`로 같은 구조입니다.
- 기준선(Compliant, Stiff)의 학습 설정은 정적 검색으로 찾지 못했습니다. **미확인**입니다.
- 실행기가 읽는 `ch.VirtualTargetEstimator`(C++ 바인딩 측)와 Python 구현 중 어느 쪽이 실제 학습 레이블을 만드는지는 이 대조에서 완전히 확인하지 못했습니다. 논문과 비교한 것은 Python 구현입니다.

## 5. 구현 가능성 판정

- **논문 그대로의 재현은 이 맥에서 불가능합니다.** 실로봇(UR5e, ATI mini-45 F/T, GoPro, C++ 컴플라이언스 제어기)이 필요합니다.
- **정책 학습은 조건부입니다.** 맥에는 CUDA가 없으니 MPS나 CPU로 학습해야 합니다. 설정이 300 epoch 학습을 전제하고, 데이터 크기는 확인하지 못했습니다. 공식 설치는 Ubuntu 22.04와 mamba 환경을 가정합니다. 설치 전에 사용자 승인이 필요합니다.
- **D1, D2, D6의 숫자는 논문에 없습니다.** 공식 코드로만 재현할 수 있습니다.
- **노력 추정.** 실로봇 재현은 하드웨어를 갖춘 뒤 수 주가 필요합니다. 맥에서 할 수 있는 것은 레이블 생성 함수를 합성 궤적에 적용하는 검사(반나절 이내)와 강성 행렬 재구성 검사입니다.

## 6. 보안과 취급

- 저장소 내용은 신뢰하지 않는 데이터로 다뤘습니다. 스크립트는 실행하지 않았고, `repo/`는 읽기 전용으로만 사용했습니다.
- 공식 저장소의 `hardware_interfaces`(외부 저장소)는 복제하지 않았습니다.
