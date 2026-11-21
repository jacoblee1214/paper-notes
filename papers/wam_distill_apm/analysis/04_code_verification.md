# 04. 공식 코드 대조 (논문 ↔ 코드, 차이 D1–Dn)

대상 논문: *Distilling World Action Models for Efficient Bimanual Manipulation via Anchored Path Matching* (arXiv:2611.22592 v1)

## 0. 대조 범위와 방법

- 저장소: `https://github.com/JIAlonglong/Anchored-Path-Matching`, 커밋 `b84d9c3` (2026-11-17, "Link paper to arXiv and update citation metadata"), 라이선스 Apache-2.0(LICENSE 파일 확인). 별 8개, 포크 0개(2026-11-22 조회).
- 방법: `git clone --depth 1`로 `repo/`에 받아 **정적으로만** 읽었습니다. 실행, import, 설치, 테스트 실행, 가중치 다운로드는 하지 않았습니다.
- 대조 대상: `anchorstep/core/anchored_path.py`(목표와 손실), `anchorstep/core/anchor_priority.py`(투영), `anchorstep/training/trainer.py`(기울기 합성, EMA), `anchorstep/core/schedules.py`(시간 격자), `configs/train/*.yaml`, `configs/ablation/*.yaml`, `configs/model/lingbot_robotwin.yaml`, `docs/method.md`, `docs/training.md`, `README.md`, `MODEL_ZOO.md`.
- 저장소 자체가 **논문 목표와 현재 구현이 다르다고 밝힙니다.** `README.md`는 "These commands run the current instantaneous-field implementation ... not the finite-interval paper target described above"라고 적습니다. 아래 D1은 이 자체 고백을 확인한 것입니다.
- 가중치는 Hugging Face에 있지만 모델 저장소는 "uploading/incomplete"로 표시되어 있습니다(README, MODEL_ZOO). 체크포인트 대조는 하지 않았습니다.

## 1. 일치하는 부분

| 논문 | 코드 위치 | 판정 |
|---|---|---|
| 식 (5) 복원 앵커 `y0restart = sg[Φ^{r→0}_T(z_r, c)]`, 교사 8단계 | `anchored_path.py` L134–L143 (`interval_grid(state_t, zero, teacher_steps)`, `restart_endpoint = continuation[-1].detach()`) | 일치 |
| 식 (4) 접두 상태 분리(detach) | `anchored_path.py` L119–L131 (`correction_state.detach()`) | 일치 |
| 보정 상태 인덱스(식 11 기준 Beta(5,2), 4단계 `i = min(1+⌊3b⌋, 3)`) | `anchored_path.py` L44–L64 (`_beta_five_two`, `_correction_index`) | 일치 (CDF 역변환으로 구현) |
| 교사 8단계 강제 | `anchored_path.py` L114–L116 (`teacher_reference_steps != 8`이면 오류) | 일치 |
| 원래 앵커(식 12) `Φ^{1→0}_T(x1, c)` | `anchored_path.py` L146–L156 (`anchor_source == "original"`) | 일치 |
| 손실 가중치: 영상 1.0, 행동 0.1 (보정 손실) | `configs/train/stage2.yaml` `loss:` | 일치 |
| 기본 손실 `L_I` 유지, 그래디언트 합 `g_total = g_base + g_a + g_f⊥` | `trainer.py` L209–L216 (`_add_gradients(base, anchor, corrected_field)`) | 일치 |
| 모달리티별 투영, 앵커 기준 한쪽 투영 | `trainer.py` L176–L186 (모달리티 `video`/`action`별 `project_conflicting_gradient`) | 일치 |
| 투영식 `g_f⊥ = g_f − min(⟨g_f,g_a⟩,0)/‖g_a‖² · g_a` | `anchor_priority.py` L50–L56 (`coefficient = dot.clamp(max=0)/anchor_sq`) | 일치 (`g_a = 0`이면 `g_f` 유지도 일치) |
| 어댑터 랭크: FFN 64, 시간축 128 | `configs/model/lingbot_robotwin.yaml` (`ffn_video_rank: 64`, `ffn_action_rank: 64`, `temporal_rank: 128`) | 일치 |
| 배치 32, Stage I 4,000 스텝, Stage II 추가 5,000 스텝 | `configs/train/stage1.yaml`, `stage2.yaml` | 일치 |
| 교사 CFG 5/1, 배포 CFG 5/1 | `configs/model/lingbot_robotwin.yaml` (`teacher_target_cfg`, `deployment_cfg`) | 일치 |
| 1/2/4 단계 한 체크포인트 | `schedules.py` `build_time_grid({1,2,4})`, `README` | 일치 |
| 원래 앵커 vs 복원 앵커, 투영 유무의 절제 구성 | `configs/ablation/*.yaml`, `MODEL_ZOO.md` 표 | 일치 (구성 이름까지 대응) |

## 2. 차이 D1–Dn

| 번호 | 논문 | 공식 코드·문서 | 영향 |
|---|---|---|---|
| **D1 (핵심)** | 식 (7)–(8): 짧은 전이 `s = r − 1/K_T`에서 교사 첫 단계 `z_s^T = sg[Φ^{r→s}_T(z_r,c)]`를 만들고, 학생 `ẑ_s = Ψ_θ(z_r; r→s, c)`가 그 전이 상태를 맞춘다(유한 구간 질의). 초록도 "short-transition predictions"라고 씀. | `anchored_path.py` L112–L113: `field_target`은 `"instantaneous"`만 허용하고 다른 값이면 오류. 손실은 같은 시각 `t`에서 `student_direction(state, t)`와 `teacher_direction(state, t)`를 MSE로 맞춘다(L174–L186). `docs/method.md`는 "instantaneous direction ... at that same detached state and correction time"이라고 씀. 프로젝트 페이지 초록도 "instantaneous direction fields". | 논문이 설명한 **목적 함수와 공식 구현이 다릅니다.** 구현은 속도장(순간 방향) 일치이고, 논문은 유한 구간 전이 일치입니다. 두 목표는 같은 기울기 충돌 구조를 가지지만 학습 신호가 다릅니다. 논문 절제(표 2의 "Field")와 그림 3(local transition 0.5→0.375, 0.5→0.25)이 어느 쪽 결과인지 불분명합니다. README가 이 차이를 인정합니다. |
| D2 | 논문 본문에 EMA 언급 없음. 평가 체크포인트 선택도 명시 없음. | `stage1.yaml`, `stage2.yaml`: `ema_decay: 0.999`. `trainer.py` L229–L233에서 EMA 갱신. `MODEL_ZOO.md`는 "target/EMA inference tensors"만 공개 대상으로 명시. 프로젝트 페이지 그림 캡션은 "step10000 EMA"라고 씀. | 평가에 쓴 가중치가 온라인 가중치인지 EMA인지 논문만으로는 알 수 없습니다. 프로젝트 페이지의 "step10000"은 논문의 총 9,000 스텝(4,000 + 5,000)과 맞지 않습니다(`03` V6). |
| D3 | 기울기 클리핑 언급 없음. | `stage1.yaml`, `stage2.yaml`: `gradient_clip_norm: 1.0`. `trainer.py` L219–L226에서 클리핑 후 최적화. | 기울기 크기 규약이 논문 식 (10)의 `g_total` 합과 정확히 같지 않습니다. 투영 뒤 합의 노름이 1로 잘립니다. |
| D4 | 옵티마이저, 학습률, λ_a 값, 정밀도, 학습 시간, 학습 시드 모두 미기재. Stage I의 `λ_a`는 "action term 균형"이라고만 씀. | 공개 저장소에 옵티마이저 생성 코드 없음(`trainer.py`가 외부에서 주입받음). `stage1.yaml`의 `action_weight: 0.1`이 `λ_a`일 것으로 보이나 논문 문장으로는 확인되지 않음. `README`: 개인 설정 오버레이(`configs/local.example.yaml`)로 분리. | 재현에 필요한 핵심 하이퍼파라미터가 공개되지 않았습니다. 학습 재현은 저자 설정을 알아야 합니다. |
| D5 | 텍스트 드롭아웃 언급 없음(CFG만 언급). | `configs/model/lingbot_robotwin.yaml`: `text_dropout_probability: 0.1`. `README`의 "Guidance contract"도 이 값을 기록. | 논문 본문만 보면 학습 조건부 드롭아웃이 빠져 있습니다. 교사 CFG 5/1의 의미에 영향을 줄 수 있습니다. |
| D6 | 논문은 실로봇 실험(표 3)을 보고. | `anchorstep/robotwin/`(시뮬레이터 평가)만 있고 Astribot 배포 코드는 없음. README의 "Astribot S1 deployment" 절은 표 3과 같은 수치와 그림을 실었을 뿐 실행 코드는 없음. | 표 3은 저장소로 재현할 수 없습니다. |
| D7 | 논문은 영상 지표(PSNR, SSIM, LPIPS)를 보고. | 지표 계산 코드 없음(`grep psnr|ssim|lpips` 결과 README의 수치 문장뿐). | 그림 4 수치는 저장소로 재계산할 수 없습니다. 재계산은 논문 그림과 표에 적힌 값으로만 가능합니다. |
| D8 | 논문 Stage I 손실은 `L_m = MSE(x̂_r, x_r^T)`(식 3)로 유한 구간 학습. | Stage I 구현은 `flow_map.py`(`stage1_objective`)에 있음. 이번 대조에서 세부 수식은 정독하지 않았습니다. | 미확인 항목으로 둡니다(§5). |

## 3. 차이의 해석

- D1은 논문의 핵심 주장("student-reached state에서 짧은 전이와 끝점을 함께 맞춘다")은 공식 코드에도 있지만, **짧은 전이라는 구체적 형태**는 구현되지 않았다는 뜻입니다. 저장소 README가 그것을 명시하므로, 논문 그림 3과 표 2의 "Field" 항목을 저장소 결과와 같은 것으로 볼 수 없습니다.
- D2와 D4 때문에 논문 수치를 재현하려면 저자의 학습 설정과 평가 체크포인트 규칙을 알아야 합니다.
- 기여 세 가지(한 교사 연속으로 두 목표, 학생 상태 기반 앵커, 투영)는 공식 코드에서 모두 확인됩니다. 다만 두 번째 목표(지역 전이)의 형태가 다릅니다.

## 4. 구현 가능성 판정

**판정: 이 맥에서 논문 결과의 재현은 불가능합니다. 코드 구조를 이해하고 투영과 목표를 작은 모델에서 옮겨 보는 것은 가능합니다.**

- 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음) 기준:
  - **불가능:** 공식 설정 학습. 교사와 동결 백본은 LingBot-VA(논문 [17])이고, 학습 코드는 CUDA 전제입니다. 논문은 학습 GPU 수와 시간을 적지 않았고, 실로봇 추론에 A800 80 GB를 썼다고만 적었습니다.
  - **불가능:** RoboTwin 2.0 Clean/Random 평가. 시뮬레이터, 평가 세트 설치와 GPU 추론이 필요합니다.
  - **불가능:** 실로봇 재현(Astribot S1).
  - **조건부:** 공식 체크포인트 MPS 추론 지연 측정. 모델 저장소가 미완성이고, 백본과 어댑터 가중치를 내려받는 용량과 MPS 포팅 작업이 필요합니다. 승인 필요.
  - **가능(CPU):** `anchor_priority.py`의 투영 수학을 임의 벡터로 확인하는 검사, 그리고 목표 형태 비교 같은 설계 실험. 공식 코드 실행은 하지 않고 수식만 옮겨 쓰는 방식입니다.
- 노력 추정(추정치, 측정 전): 공식 코드 구조를 따라 축소 모델에 투영과 두 목표를 옮기는 작업 1–2주. 공식 설정 재현은 맥에서 불가능하므로 추정하지 않습니다.

## 5. 미확인 항목

- Stage I(`flow_map.py`)의 세부 식은 정독하지 않았습니다. 논문 식 (3)과의 대응은 확인하지 않았습니다.
- 평가 스크립트(`anchorstep/robotwin/evaluate.py`, `results.py`)의 시드와 에피소드 수 규칙은 확인하지 않았습니다. `docs/evaluation.md`에 결정적 평가와 시드 기록이 있다는 것만 확인했습니다.
- `tests/`(예: `test_anchor_priority.py`)는 읽지 않았고 실행하지 않았습니다.
- 모델 저장소(Hugging Face)의 `manifest.json`과 `audit.json`은 내려받지 않았습니다. 목록만 보았습니다(`bases/`, `variants/apm/` 등).
- 별 수(8)와 최종 푸시 날짜는 웹 페이지 메타데이터로 확인했습니다. 이 값은 시점에 따라 바뀝니다.
