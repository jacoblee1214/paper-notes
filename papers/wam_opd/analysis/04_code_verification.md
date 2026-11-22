# 04. 논문 ↔ 공식 코드 정적 대조와 구현 가능성

## 1. 대조 범위

- **저장소:** https://github.com/UCL-ERL/WAM-OPD, Apache-2.0, 커밋 `191b2e392f6ffbd987f9abcc8f7f1ec3b0853ff6`(2026-10-19), 얕은 clone을 `repo/`에 두고 **정적으로만** 읽었습니다. 실행, import, 설치, 가중치·데이터 다운로드는 하지 않았습니다.
- **읽은 곳:** `README.md`, `docs/REPRODUCIBILITY.md`, `docs/ARTIFACT_POLICY.md`, `docs/SERVER_SOURCE_SYNC.md`, `docs/deployment/REAL_ROBOT_DEPLOYMENT.md`, `docs/method/*`, `QUALIFIED_SUCCESS_PATH_PIPELINE_V1.md`, `experiments/bind_qualified_success_path_task.py`, `bind_scaled_qualified_success_path_task.py`, `opd_task_specs.py`, `stage_h_task_progress.py`(과제 목록), `joint_lora.py`, `train_video_trajectory_opd.py`(마스크), `waopd_v0_video_opd.py`(손실, 속도), `train_iterative_on_policy_flow_opd.py`(학습 본체 7,581줄 중 손실·배치·설정 정규화·표지 수집 부분), `run_handover_mic_success_path_eval.py`(선별·집계 부분), `run_scaled_qualified_success_path_pipeline.py`(분할 검증).
- **읽지 않은 곳:** `prototype_*`, `stage_*`, `robotwin_*`, 테스트 파일, 거대한 진단 스크립트(`prototype_fixed_instruction_robotwin_client_diag.py` 144 KB 등). 이들은 저장소가 "기본 진입점이 아니다"라고 밝힙니다.
- **공식 학습 경로:** 논문의 설정에 해당하는 것은 `trajectory_update` 실행 모드 + `objective = coherent_tt_consistency` + `coherent_tt_variant = success_path_v1`(`_update_round_success_path_tt`)입니다. 이름은 "일관성(consistency)"이지만 이 변형은 EMA·재잡음 없이 **끝점 pseudo-Huber 회귀**를 합니다.
- **저장소 자체 평가:** README의 릴리스 상태는 **HOLD**("전체 벤치 재현은 인증되지 않음, 보고된 논문 결과와 저장소 점검은 별개", `docs/REPRODUCIBILITY.md` 1–6줄)입니다.

## 2. 대조표 (일치 K1–K16)

| 라벨 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| K1 | LoRA | rank 8, 블록 0–29 (§3.3) | `adapter_rank 8`, `lora_block_indices range(30)`, `lora_alpha 8.0`(스케일 1), 드롭아웃 0, 대상은 어텐션 Q/K/V/O ×2 + FFN 상·하, 300개 선형층(`docs/method/joint_training_method_decision_20260820.md` §4) | 일치 (alpha·대상 층은 코드에만, D8) |
| K2 | 옵티마이저와 학습률 | AdamW, 2e-5 | `torch.optim.AdamW(lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)` (`_fresh_optimizer`) | 일치 (베타·감쇠는 코드에만) |
| K3 | 기울기 클리핑 | 노름 2 | `clip_grad_norm_(max_norm=2.0)` (`_update_round_success_path_tt`) | 일치 |
| K4 | 가중치 | (λ_v, λ_a, λ_FM) = (1, 1, 0.2) | `video_weight 1.0`, `action_weight 1.0`, `action_fm_weight 0.2` | 일치 |
| K5 | pseudo-Huber | ρ_c = √((x−y)²+c²) − c, c = 1e-3 | `_masked_pseudo_huber_loss`: `sqrt(diff² + c²) − c`, `pseudo_huber_c 0.001` | 일치 (v1 정의 δ²(√(1+(e/δ)²)−1)과는 척도 다름) |
| K6 | 에폭, 배치 | 3 에폭, 유효 배치 4 | `inner_epochs 3`, `effective_batch_size 4`. 한 배치는 서로 다른 궤적의 문맥 4개(`_trajectory_distinct_epoch_batches`) | 일치 |
| K7 | 손실 축약 | 궤적 안 평균 후 궤적 동등 평균 (§3.3) | `loss_reduction = mean_trajectories_mean_labels`, 샘플 스케일 = 스텝수/(궤적수 × 궤적 표지수) (`_trajectory_equal_epoch_batch_scales`) | 일치 |
| K8 | stop-gradient | A_S = f_action(h, sg(Ô_S)) (식 2) | `student_video_forward(context, detach_plan_for_action=True)`, 행동 손실이 영상 그래프로 흐르지 않음. 상태 계약(잡음·타임스텝·마스크·토큰 위치·캐시 길이) 일치 검사 | 일치 |
| K9 | Teacher 표지 | Â_T는 Teacher 영상에 조건화 (식 3) | `_on_teacher_video_plan_label`: `action_target_condition = "teacher_on_teacher_z_t"`, "do not inject z_S", Teacher·Student 행동 잡음과 타임스텝 동일 검사 | 일치 |
| K10 | Teacher 스텝 | 25/50 (표 1) | `teacher_video_steps 25`, `teacher_action_steps 50` | 일치. Student 1/1은 설정에 없고 체크포인트 구성에 의존 (확인 못함) |
| K11 | 속도 목표 | v̂ = ε − Â_T, σ = 1에서 Student 속도 (식 4) | 목표 `epsilon_a − target_action`, 예측은 첫 행동 스텝의 `initial_velocity`, `F.mse_loss` 마스크 평균 | 일치 (D10의 동치성 참고) |
| K12 | 분할과 보정 | 24 학습 + 12 보정, 보정 궤적마다 이른/늦은 앵커 2개 | `split_counts train 24 / calibration 12`, `np.linspace(0, count−1, 2)` 앵커, 에폭마다 보정 손실 | 일치 |
| K13 | 수집은 Student만 환경 제어 | §3.1 | `environment_execution "SS"`, `teacher_role "label_only"`, `teacher_controls_environment False` | 일치 |
| K14 | 기본 가중치 동결 | §3.3 | `transformer.requires_grad_(False)` 후 LoRA만 학습 | 일치 |
| K15 | 수집은 1회 | 알고리즘 1 | `rounds 1`, 고정 궤적 묶음을 재사용(`QUALIFIED_SUCCESS_PATH_PIPELINE_V1.md`) | 일치 |
| K16 | 학습·수집 환경 | RoboTwin 2.0 demo_clean | `require_training_task_config` (`demo_clean` 외는 오류) | 일치 |

## 3. 불일치와 빈칸 (D1–D18)

판정 기호: **빈칸**(논문에 없고 코드에 있음), **차이**(논문과 코드가 다름), **미공개**(코드에 없음).

| 번호 | 구분 | 내용 | 위치 |
|---|---|---|---|
| D1 | 빈칸 | **에폭 선택.** 학습 후 에폭 1·2·3 체크포인트를 선별 분할(과제당 4장면 × 잡음 은행 2 = 8쌍)에서 평가해 하나를 고릅니다. 선택 키는 (적응 성공 수 ↓, 단계 향상 합 ↓, 단계 향상 − 퇴보 ↓, 보정 손실 ↑, 에폭 ↑). 논문은 "3 에폭"이라고만 적고 선택을 언급하지 않습니다. 저장소 문서의 place_a2b_left는 선별 E1 1/8 → E2 7/8 → E3 8/8로 에폭 3이 선택됨. 에폭 3이 항상 선택되는지 모름 | `bind_qualified_success_path_task.py` `SELECTION_RULE`; `run_handover_mic_success_path_eval.py` `_candidate_aggregate`; `docs/deployment/REAL_ROBOT_DEPLOYMENT.md` §1 |
| D2 | 빈칸 | **과제 자격 판정.** 바인더는 "preregistered 8-pair strict Teacher/Student gap gate" 결과가 PASS인 과제만 받습니다. 선정 규칙은 (쌍체 순이득 ↓, Teacher 성공 ↓, Student 성공 ↑)입니다. `TASK_SPECS`에는 논문 12과제 외에 8과제(move_stapler_pad, open_microwave, put_bottles_dustbin, place_shoe, scan_object, stack_blocks_three, blocks_ranking_rgb, blocks_ranking_size)가 있고 `open_microwave` 전용 평가 스크립트도 있는데 논문은 이 과제들을 보고하지 않습니다. 12과제 선정 기준이 논문에 없음 | `select_strict_gap_task.py`; `bind_*`: `--qualification-decision`; `opd_task_specs.py` |
| D3 | 미공개 | **세 과제가 바인더에서 거절됩니다.** 바인더는 `task in SUPPORTED_TASKS and task in TASK_SPECS`를 요구하는데 `SUPPORTED_TASKS`에는 논문 12과제 중 place_container_plate, move_pillbottle_pad, place_object_stand가 없습니다(9과제만 있음). 이 세 과제는 공개 코드로 구성할 수 없음 | `stage_h_task_progress.py` 17–37줄; `bind_scaled_qualified_success_path_task.py` 과제 검사 |
| D4 | 미공개 | **평가 규모.** 공개 파이프라인의 held-out은 장면 6 × 잡음 은행 2 = **12쌍**(formal12)입니다. 논문의 "60 held-out trials"에는 저장소 문서가 "frozen extension48"이라 부르는 나머지 48쌍이 필요하며, 공개 코드는 이를 생성하지 않고 "조용히 재생성하지 말 것"이라고 적습니다. 60쌍의 장면 시드와 잡음 은행 구성이 공개되지 않음 | `docs/REPRODUCIBILITY.md` 109, 190–194줄; README "formal12 … not a complete paper paired60" |
| D5 | 미공개 | **절제 학습 경로.** 논문의 행동 전용(m_v = 0)과 영상 전용(m_a = 0)은 같은 JointLoRA에서 손실 스위치만 끄는 것(§3.2)입니다. 그러나 `joint_lora` 설정은 영상·행동 가중치와 행동 흐름 가중치가 모두 양수가 아니면 거절합니다(`"joint stage requires positive video/action weights"`, `_update_round_success_path_tt`의 `"loss weights must be finite and positive"`, `coherent_tt_consistency requires positive action_fm_weight`). 저장소 어디에도 supervision mode 스위치가 없음. 그림 4의 두 단일 모드는 공개 학습기로 만들 수 없음 | `train_iterative_on_policy_flow_opd.py` `_normalize_config`(6660–6665줄), 3366–3370줄; `grep action_only|video_only|supervision_mode` 결과 없음 |
| D6 | 미공개 | **Teacher 이력 학습(표 4).** Teacher가 환경을 제어해 모은 이력으로 학습하는 `offline_teacher` 경로는 서버 학습기에만 있고 공개 학습기에는 없다고 문서가 인정합니다("must be reconciled") | `docs/REPRODUCIBILITY.md` "Important dependency gaps" 4 |
| D7 | 미공개 | **실로봇 전체.** 텔레오퍼레이션 40개 수집, LingBot-VA 적응, Flash-WAM 증류, 36 롤아웃 평가, 배포 코드가 없습니다. 이관 문서가 "검증된 하드웨어 어댑터, 보정, 충돌 감독, 와치독, E-stop이 없음"이라고 명시하고 README의 안전 절도 같은 내용입니다. 문서가 기술한 '가장 좋은 완료 후보'는 시뮬레이션(place_a2b_left)입니다 | `docs/deployment/REAL_ROBOT_DEPLOYMENT.md` §3; README "Real-robot safety" |
| D8 | 빈칸 | **세부 설정.** alpha = 8(스케일 1), 드롭아웃 0, AdamW β = (0.9, 0.999), eps 1e-8, 감쇠 0, LoRA 대상 층 종류, 어댑터 시드 `20260820`, 일관성 시드, Student 초기 가중치(`FlashWAM-RoboTwin`)와 Teacher(`lingbot-va-posttrain-robotwin`) 이름이 논문에 없습니다 | `_training_template`; `joint_lora.py` |
| D9 | 빈칸 | **업스트림 패치.** LingBot-VA 커밋 `58c2ae5`에 WAM-OPD 패치(`0001-wam-opd-portability-and-runtime.patch`, 17.9 KB)를 적용합니다. 서버 트리에는 패치와 별도로 `wan_va/modules/model.py`, `wan_va_server.py` 등에 로컬 수정이 있고 "순정 업스트림과 같다고 보이지 않았다"고 문서가 적습니다. Teacher 성능이 공개 값보다 낮은 이유(`03` V9)와 관련 가능성 | `docs/REPRODUCIBILITY.md` 의존성 공백 2; `repro/dependencies.yaml` |
| D10 | 차이(해석) | **속도 항은 1스텝 Student에서 행동 끝점 항과 같은 양입니다(정적 추론).** 행동 끝점은 `x₀ = x_σ − σ·v`의 한 스텝이고, 속도 평가점은 σ = 1이라 `v_S − (ε − Â_T) = Â_T − A_S`입니다(마스크·첫 프레임 조건 덮어쓰기 제외). 즉 λ_FM 항은 (pseudo-Huber ≈ L1)에 0.2배의 **MSE**를 더하는 것과 같아 "행동 생성에 쓰이는 속도를 따로 맞춘다"는 논문 §3.2의 설명이 독립 신호처럼 읽히는 것과 다릅니다. σ = 1과 스케줄러 이동(shift)이 실제로 1.0인지는 실행 없이 확인하지 못했습니다(문서는 "native scheduler timestep 1000") | 논문 §3.2; `waopd_v0_video_opd.py` 행동 루프(`initial_velocity`, `scheduler.step`) |
| D11 | 빈칸 | **손실 마스크.** 영상 손실은 실행된 행동에 해당하는 프레임에만, 첫 프레임이면 0번 프레임은 제외합니다. 행동 손실은 유효 위치 ∩ 실행 위치입니다. 첫 네이티브 성공 이후 표지를 만들지 않고 성공을 일으킨 행동은 유지합니다. 논문은 "valid elements"만 적음 | `train_video_trajectory_opd.py` `video_execution_mask`, `action_execution_mask`; `docs/method/joint_training_method_decision_20260820.md` §3 |
| D12 | 빈칸 | **죽은 설정.** 설정 템플릿에 `ema_decay 0.995`, `consistency_video_stride 500`, `consistency_action_stride 500`, `consistency_noise_source`, `retention_weight 0`이 있으나 `success_path_v1` 갱신 함수는 EMA·재잡음·보존 항을 쓰지 않습니다. 같은 파일의 다른 변형(`coherent_tt_consistency` 기본형)은 EMA를 씁니다. 객체 이름 "consistency"가 논문의 설명(끝점 회귀)과 달라 독자를 혼동시킬 수 있음 | `_training_template`; 3868–3933줄(기본형 EMA) 대 3330–3855줄(success_path) |
| D13 | 빈칸 | **분할 대 수집 수.** 논문은 36 롤아웃을 적습니다. 바인더의 분할은 학습 24 + 보정 12 + 선별 4 + held-out 6 = 46 시드(학습·보정 36개만 수집)입니다. 시드와 지시문은 "accepted/expert-feasible" 메타데이터(전문가 계획이 성공하는 시드)에서 위치 기준으로 자릅니다. 시드 필터를 논문이 서술하지 않음 | `bind_scaled…` `_indices`, `_load_metadata` |
| D14 | 빈칸 | **롤아웃 길이.** 과제별 최대 제어 스텝(handover_mic 600, place_cans_plasticbox 800, stamp_seal 400 등)과 첫 청크 16, 이후 32 스텝 청크가 코드에 있고 논문에 없습니다. 이 한도가 성공률에 영향을 줌 | `opd_task_specs.py` |
| D15 | 차이 | **정의된 성공 판정.** 평가는 RoboTwin의 "latched `eval_success`"(성공을 한 번이라도 찍으면 유지)입니다. 논문은 성공 정의를 쓰지 않음. 선별에는 "단계 향상" 같은 과제별 진행 지표(`stage_h_task_progress.py`)를 보조 키로 쓰며, 보고 지표는 성공률 | `run_handover_mic_success_path_eval.py` `_candidate_aggregate` |
| D16 | 미공개 | **고정된 경로.** 공식 검증기는 산출물·판정 경로가 `/ssd/data` 아래이기를 요구하고 "임의 경로는 보장하지 않는다"고 README가 적습니다. 모델 가중치, 시뮬레이터 자산, 자격 판정 파일, 선택된 어댑터, 고정된 60쌍 패널은 패키지로 제공되지 않음 | README "Installation"; `docs/REPRODUCIBILITY.md` 의존성 공백 5 |
| D17 | 차이 | **공개 트리와 서버 스냅샷이 다릅니다.** 서버의 학습기에는 공개 트리에 없는 `offline_teacher`와 분산 학습 수정이 있고 "작업 트리가 더럽다(dirty)". 저장소 README도 서버 스냅샷이 "더 큰 작업 스냅샷"이라고 인정하고 불일치 시 어느 쪽이 정본인지 가정하지 말라고 함 | README "Repository status and source of truth"; `docs/SERVER_SOURCE_SYNC.md` |
| D18 | 빈칸 | **통계.** 코드는 쌍별 결과(구조·퇴보 수, 단계 지표)를 계산하지만 검정을 하지 않고 논문도 유의성 검정이나 구간을 보고하지 않습니다. 쌍체 결과 기록(`rescue`/`regression`)은 공개되지 않음 | `_candidate_aggregate` |

### 3.1 정리

- **방법의 핵심(손실, 하이퍼파라미터, 배치, stop-gradient, Teacher 표지)은 논문과 코드가 맞습니다**(K1–K16). 코드 주석과 설계 문서가 논문 §3의 내용과 일관됩니다.
- **결과를 재생하는 데 필요한 것은 대부분 공개되지 않았습니다.** 평가 60쌍(D4), 세 과제(D3), 절제 모드(D5), Teacher 이력(D6), 실로봇(D7), 선택된 체크포인트와 자격 판정 파일(D1, D2, D16).
- **저장소가 스스로 HOLD라고 적습니다.** README와 `REPRODUCIBILITY.md`는 보고 결과와 릴리스 검증을 구분하고, 정본이 서버 스냅샷일 수 있다고 명시합니다(D17).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 수치를 이 맥에서 재현하는 것은 불가능합니다. 코드 정적 읽기와 수치 재계산은 끝냈고, 아이디어의 축소판만 조건부로 가능합니다.**

### 4.1 이 맥에서 가능한 것과 불가능한 것

| 항목 | 이 맥 | 이유 |
|---|---|---|
| 표 재계산, 정수 복원, 검정 | **완료** | `analysis/recompute.py` (numpy, scipy) |
| 코드 정적 대조 | **완료** | 이 문서 |
| 저장소의 CPU 테스트 (`make test`, 54개 가벼운 테스트) | 조건부 | 새 가상환경과 설치(numpy, pytest)와 승인이 필요. 모델 로딩 없음. 보고된 결과와 무관한 계약 테스트라 가치는 제한적 |
| 공식 파이프라인 학습·평가 | **불가** | 수집 GPU에 65 GB 이상 여유 메모리를 요구(`--collection-min-free-mib 65000` 기본값), A800급 GPU 가정, CUDA 12.4/PyTorch 2.6 환경, 저장소의 안정 할당은 GPU 4장(`QUALIFIED_SUCCESS_PATH_PIPELINE_V1.md`) |
| Flash-WAM·LingBot-VA 가중치 로드 | **불가** | CUDA 의존 코드와 오프로딩, 가중치 크기 미확인(받지 않음). RoboTwin 시뮬레이터(SAPIEN)와 자산도 필요 |
| 60쌍 평가 재생 | **불가** | 공개되지 않은 extension48과 자격 판정 파일 필요 (D4) |
| 절제(그림 4), Teacher 이력(표 4) | **불가** | 공개 코드에 경로가 없음 (D5, D6) |
| 실로봇(표 3) | **불가** | 로봇, 데이터, 배포 코드 없음 (D7) |
| 축소판: 장난감 흐름 매칭 영상-행동 WAM에서 "Student 이력 + Teacher 영상·행동 쌍 목표" 대 "오프라인 이력" 대 "단일 모달 목표" 비교 | **조건부 가능** | 직접 짠 torch(MPS) 또는 numpy. 새 설치 없음. 메커니즘(이력 분포, 쌍 목표, 조건 불일치)을 시험하지만 논문 수치와는 무관 |

### 4.2 노력 추정 (추정값, 측정 없음)

- 이 맥에서 축소판(소형 2D 영상-행동 흐름 모델, 장난감 환경): **1–2주**.
- CUDA 기기(80 GB급 GPU 1–4장)에서 공식 파이프라인으로 **과제 1개의 formal12까지**: 의존성·가중치·시뮬레이터 준비 3–5일 + 수집·학습·평가는 과제당 수 시간~하루로 추정(측정 없음). 단, 자격 판정 파일과 메타데이터를 직접 만들어야 하고(D2, D16), 세 과제는 코드 수정이 필요(D3).
- 논문 12과제 + 60쌍 + 절제 + Teacher 이력 + 실로봇을 모두 재현하는 것은 공개 코드만으로는 **불가능에 가깝습니다**(D4–D7).

## 5. 저자에게 물을 질문

1. 60 held-out 시도는 몇 개의 장면 시드와 몇 개의 잡음 은행입니까? 12쌍(formal12)과 나머지 48쌍은 어떻게 정해졌습니까? (D4)
2. 에폭 선택을 선별 분할로 했습니까? 표 2·그림 4·표 4의 어댑터는 에폭 몇입니까? (D1)
3. 12과제는 어떻게 골랐습니까? 자격 판정(PASS)이 안 된 과제, 시도했으나 보고하지 않은 과제가 있습니까? (D2)
4. place_container_plate, move_pillbottle_pad, place_object_stand를 공개 코드로 돌리려면 무엇이 필요합니까? (D3)
5. 행동 전용과 영상 전용은 어떤 코드로 학습했습니까? 설정을 공개할 수 있습니까? (D5)
6. 속도 항은 1스텝 Student에서 행동 끝점 항과 같은 양이 아닙니까? 둘을 분리하는 절제를 했습니까? (D10)
7. v1에서 v2로 학습 궤적을 늘렸는데 handover_mic와 put_object_cabinet의 성공률이 낮아진 이유는 무엇입니까? (`03` V14)
8. Teacher 참조 성능이 공개 LingBot-VA 값보다 낮은 이유(특히 put_object_cabinet 51.7% 대 85%)는 무엇입니까? (D9)
9. 같은 Teacher 호출 예산으로 SFT, 행동 전용 증류, 반복형 OPD와 비교한 결과가 있습니까?
10. 실로봇 평가는 장면·물체 배치가 짝입니까? 몇 번의 리셋 변이로 36회를 만들었습니까?
