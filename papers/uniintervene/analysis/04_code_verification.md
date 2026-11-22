# 04. 공식 코드 대조

대상: `github.com/Denghaoyuan123/UniIntervene`, 기본 브랜치 `main`, 커밋 **`40f461f97573383c9cc8fdba43df56f9eb5bbea1`**(2026-09-19). 얕은 복제를 `repo/`에 두고 **읽기만** 했습니다. 실행, import, 설치, 모델과 데이터 다운로드는 하지 않았습니다(복제된 내용은 신뢰할 수 없는 자료로 취급). 논문과 코드는 모두 분석일 2026-11-23 기준입니다.

## 1. 저장소 개요

| 항목 | 내용 |
|---|---|
| 라이선스 | MIT (`LICENSE`, "Copyright (c) 2026 UniIntervene authors"). GitHub API도 MIT |
| 규모 | 파일 약 40개, 948 KB, Python 약 1.5만 줄 (`hil_vlm_train.py` 4,616줄, `hil_vlm_dataset.py` 1,596줄, `JEPA/train_vf_stitched.py` 1,619줄) |
| 범위 | **오프라인 학습 파이프라인만**: NTTG 라벨링 → 프록시 가치 함수(VF) → 가치 주석 → 개입 라벨 채굴 → VQ1(미래·쌍둥이 Q·위험 머리) → 복구 기억 → VQ2(목표 조건부 FAST 복구 정책). 진입점 `run.py --stage {nttg,vf,annotate,mining,vq1,memory,vq2,all}` |
| 없는 것 | 로봇 배포, HIL-SERL 통합, 알고리즘 1의 온라인 루프, 기준선(π0.5 SFT, HIL-SERL, FA-RL), 표 1·2 평가 코드, 데이터, 체크포인트, 기억 은행. README가 "Robot deployment and HIL-SERL integration are outside this package"라고 밝힙니다 |
| 예시 설정 | `configs/fold_towel.json` 한 개(Fold Towel). 다른 네 과제의 설정은 없습니다 |
| 테스트 | `tests/test_launch_config.py`(명령 조립 검사), `tests/test_data_contract.py`, `tools/smoke_fast_overfit.py`와 기록 `reports/fast_overfit_fold_towel.json`(16개 행동 덩어리의 토큰 CE 7.63 → 0.0035, 토큰 정확도 1.0, 복원 MAE 0.0040. "구성 요소 시험, 홀드아웃 평가가 아님") |
| 서드파티 | FAST 토크나이저 `physical-intelligence/fast`(Apache-2.0, 리비전 `ec4d7aa7…` 기록), Qwen3-VL, SigLIP, Gemma, V-JEPA2는 별도 배포(`THIRD_PARTY_NOTICES.md`) |
| 데이터 형식 | `list[dict]` 피클 파일(`observations`, `next_observations`, 7차원 `actions`, 보상/종료, 과제 정보). README: "Only load pickle files from trusted sources." |

## 2. 논문 ↔ 코드 차이 (D1–D18)

판정 기호: 일치 / 다름 / 논문에 없음 / 미확인.

### 범위와 평가

**D1. 코드로는 표 1, 표 2 어느 수치도 만들 수 없습니다 (논문에 없음).** 온라인 루프(알고리즘 1), HIL-SERL 결합, 개입 모듈이 정책 행동을 덮어쓰는 부분, 평가 스크립트가 없습니다. README는 모듈의 학습 파이프라인만 약속합니다. 논문의 "IR"이 어떻게 집계되는지(개입 스텝 정의, 복구 스텝을 어디에 세는지)를 코드에서 확인할 수 없습니다.

### 트리거

**D2. 트리거 점수에 두 손실이 같은 스칼라에 걸립니다 (다름).** 논문 식 6/§3.3: `R̂ = f_risk(ẑ, f_hist(V))`를 `R_t`에 Smooth-L1로 회귀하고 `s = σ(R̂)`로 쓴다. 부록 D.3/E: 별도 "트리거 머리"가 풀링 래턴트와 분리된 가치 스칼라를 읽어 초점 손실로 학습. 코드(`hil_vlm_train.py` L2256–2299, `tvr_history` 설정): `risk_pred = risk_head(z, 값 이력 인코딩)` 한 스칼라에 (a) `smooth_l1(risk_pred, R_t)`(λ_tvr = 1) 와 (b) `binary_focal_loss_with_logits(risk_pred, 채굴 라벨)`(α = 0.75, γ = 2, λ_int = 1)을 **동시에** 걸고, 평가 확률도 `σ(risk_pred)`입니다. 회귀 목표 `R_t`는 평평한 창에서 0.03 안팎이고(`03` V15), 초점 손실은 로짓이 클수록 손실이 줄어드는 방향이라 두 손실이 같은 출력에서 당기는 방향이 다를 수 있습니다(추정, 학습 곡선은 없음). 부록 D.3의 "분리된 가치 스칼라를 읽는 개입 머리(`InterventionHead`)"는 `head_type: binary`일 때만 만들어지고(L1927), 기본 설정 `full_tvr_history.yaml`(`head_type: tvr_history`)에서는 만들어지지 않습니다.

**D3. 트리거 입력의 가치는 쌍둥이 Q(q̂)가 아니라 사전 계산된 프록시 가치입니다 (다름).** 논문은 "미래 조건부 행동가치"가 안정된 진행 신호라고 서술(초록, §3.2)하지만, `tvr_history`에서 값 이력 `V_{t−K:t}`는 데이터의 `vf_value_pred`(주석 단계에서 프록시 VF가 계산)이고(`_value_history_window`, L1040–1056), 쌍둥이 Q 머리는 학습되지만(`L_q`) 위험 머리와 복구의 입력이 아닙니다. `Q_min`을 개입 머리에 붙이는 경로는 `binary` 변형에만 있습니다. 따라서 배포에서 트리거를 내려면 프록시 VF(SigLIP + Gemma-3-270M)를 별도로 매 스텝 돌려야 합니다(알고리즘 1 7행은 "append current proxy value"). 절제 표 2의 "w/o Value Pred."(Q-loss 없음)에 대응하는 변형은 공식 `VARIANT_SPEC`(`full`, `no_future`, `no_value_cond`, `no_future_value_cond`, `no_intervention`, `no_future_loss`)에 없습니다. 가장 가까운 `no_value_cond`는 Q 머리를 학습합니다(Q-loss "–"와 불일치). 변형 구성 미확인.

**D4. 프록시 가치 함수 손실이 식 9–13과 다릅니다 (다름).** `JEPA/train_vf_stitched.py` L925–990, `run.py` 설정(λ_td 1, λ_label 0.3, λ_prog 1, λ_cql 0.05, λ_mc 0, 워밍업 5에폭 진행 항만).
- TD(식 10)와 라벨(식 11): Smooth-L1 β = 1이 아니라 **가중 MSE**(`twin_weighted_mse`), 성공/실패 표본 가중.
- 라벨 목표: `mc_returns + α_prog · progress`(`α_prog` = 1, 성공 에피소드 한정). 논문은 `v_t = t/(T−1)`(식 11)입니다. 코드의 MC 수익은 희소 터미널 보상의 할인 수익이라 성공 에피소드에서 `γ^{T−1−t}`이고, 합하면 마지막 스텝에서 목표 2.
- 단조성 항(식 12, `max(0, V_t − V_{t+1})` 힌지): **코드에 없습니다**(저장소 전체에서 `monoton`은 VQ2의 "L_tail" 옵션 설명에만 있음). 대신 `loss_prog`(MSE로 `α_prog · progress`에 회귀, 성공 에피소드 한정)가 "진행" 항이고, 같은 표본에 라벨 항(목표 `G_t + p_t`)과 진행 항(목표 `p_t`)이 함께 걸립니다.
- "CQL"(식 13, 이웃 상태 N(o_t)의 log-sum-exp − 관측 값): **없습니다**. `loss_cons`는 `relu(pred − mc_returns)`(MC 수익을 넘는 예측에 대한 한쪽 벌칙)이고 N(o_t) 표본이 없습니다. 가중치 0.05는 일치.
- 추가된 항: 보정 `loss_cal`(사용 안 함), VQA 보조 손실(λ_vqa 0). 에폭 150, 배치 256(`configs/fold_towel.json`)은 논문 표 3에 없음.

**D5. 가치 범위와 이력 처리 (논문에 없음).** VF 출력에 sigmoid나 clip이 없습니다(`scripts/annotate_buffer_vf_value_pred.py`, `train_vf_stitched.py`에 `clamp`는 가중 평균 분모에만). 식 5의 `(1 − V_t)`는 V ∈ [0, 1]을 가정하지만 라벨 목표(D4)가 1을 넘을 수 있습니다. 채굴 규칙은 에피소드별 최소-최대 정규화한 값을, TVR 목표와 이력은 **정규화하지 않은** `vf_value_pred`를 씁니다. 이력 창은 에피소드 처음에서 모자라면 첫 값으로 채웁니다(`idx = max(ep_start, fi − k)`). 건강한 궤적의 처음 2–3단계에서 `R_t > 0`이 됩니다(`03` §4 토이 D).

**D6. 오프라인 개입 라벨은 가치에서 규칙으로 채굴됩니다 (논문에 정성 서술만).** `scripts/mining_multitask_full_3rules.py`: 에피소드별 최소-최대 정규화(에피소드 전체의 최솟값과 최댓값을 쓰므로 인과적이 아님) 후 창 8에서 decline(기울기 ≤ −0.01, 하락 ≥ 0.05, 하강 비율 ≥ 0.7), plateau(`|Δ| ≤ 0.03`인 비율 ≥ 0.6, 진폭 ≤ 0.12), abrupt-drop(하락 ≥ 0.12 후 3스텝 내 절반 미회복)의 합집합. 논문 부록 E에는 임계값이 없습니다. 학습의 `y_int`는 사람 개입 플래그가 아니라 이 라벨입니다(`intervention_label_source: auto`가 `vf_mining_intervention_label`을 우선, `hil_vlm_dataset.py` L572–612). 따라서 표 2의 "Int. F1"은 **가치 규칙 라벨에 대한 일치도**이고 사람 개입 일치도가 아닙니다. plateau 규칙은 7/(T−1) ≤ 0.12, 곧 T ≥ 60에서 선형 상승만으로도 모든 창을 양성으로 표시합니다(`03` §5 토이). 로그에 고정 임계 F1과 임계 스윕 `best_f1`(0.05–0.95 격자)이 모두 기록됩니다(`hil_vlm_train.py` L2562–2633). 표 2가 어느 쪽인지 논문에 없습니다.

### 기억과 복구

**D7. 기억 구성 규칙이 부록 F.1과 다릅니다 (다름).** `scripts/build_vf_delta_bank_fast.py`, `memory_constraints.py`.
- 적격 조건: `end_vf − start_vf ≥ δ` 그리고 `end_vf ≥ target_vf_min`(기본 0.75, 논문 표 5에 없음). 논문의 "시작 값이 이미 어느 정도 높을 것"에 해당하는 하한은 **없습니다**.
- 값은 정규화하지 않은 `vf_value_pred`.
- 진행 구간(bin) 배정: 기본은 에피소드 안 **시간 위치**(`s/(n − span − 1) × bins`). 그림 7(b)의 "progress bin (by start value)"과 다릅니다. 값 기준 재배정은 `--bin_by_keyframe_vf`(그리퍼 키프레임 값) 옵션뿐입니다.
- 구간 안 선택은 무작위가 아니라 `(그리퍼 키프레임 등급, δ, 기울기, 이른 시작, 짧은 span)` 순으로 정렬한 결정적 상위 선택이고, 에피소드당 상한(Fold 8, 기본 2)이 있습니다(`_score`).

**D8. Fold Towel 기억 설정이 논문 표 5와 다르고 저장소 자체 테스트가 "paper numbers"라고 부릅니다 (다름).** 논문 표 5: Fold Towel span 16, δ 0.50, 20구간 × 12. 공식 `configs/fold_towel.json`: **span 32, δ 0.40**, 20구간 × 12(항목 240), `target_vf_min` 0.75. `tests/test_launch_config.py::test_paper_numbers`는 `(span, bins, items_per_bin, target_items) == (32, 20, 12, 240)`를 단언하며 이름이 "paper numbers"입니다. 논문 본문과 스크립트 기본 인자(`--fold_fixed_span` 16)는 16이므로 저장소 내부에서도 값이 갈립니다. README는 "per-task values used in the paper are listed in its appendix"라며 예시 설정이 논문과 같다고 주장하지 않습니다. 비Fold 네 과제의 스크립트 기본값은 `--delta_thr 0.5`로 논문 0.40과 다릅니다.

**D9. 복구 정책의 학습 목표는 "검색된 다른 에피소드의 궤적"입니다 (다름).** 논문 §3.4, 부록 F.3: "각 `o_fail`에서 목표 상태로 이어지는 롤아웃 구간"의 행동 복제. 코드(`hil_vlm_dataset.py` `__getitem__` L1431–1520, `_retrieve_round2_green`, `run.py` L174–183): 채굴 라벨이 양성인 표본마다 기억에서 상위 1개를 코사인으로 검색하되 **같은 에피소드는 제외**(`exclude_episode_id`)하고, 감독 목표를 **검색된 기억 항목의 8스텝 복구 행동 궤적**(`recovery_chunk_indices`로 구간의 처음 8스텝)으로 둡니다. `round2_target_space: action`, `bridge_steps: 0`, CE만 사용(`round2_ce_only: true`). 목표 문자열은 검색된 궤적의 **마지막 행동**을 7개 bin으로 적은 것(`format_memory_goal_state_text`)이고 목표 이미지로 기억 항목의 목표 프레임이 추가됩니다. 기억 항목의 궤적은 구간의 처음 8스텝뿐이라(`build_vf_delta_bank_fast.py` L701, `recovery_chunk_indices`), span 32인 Fold Towel 예시 설정에서는 감독 궤적이 구간 끝의 목표 상태까지 가지 않습니다. 따라서 정책은 "검색된 궤적 복사"에 가까운 과제를 학습하며, 목표 문자열이 출력 궤적의 끝 행동을 담고 있어 정보가 새는지는 코드만으로는 확정할 수 없습니다. 이 점은 논문의 "목표에 도달하는 방법을 배운다" 서술과 다릅니다.

**D10. 검색 규칙: 목표 가치 제약이 추가됩니다 (논문에 없음).** 논문 식 7은 순수 `argmax cos`입니다. 코드는 후보를 과제 필터(`task_text`)로 좁히고 학습 시 같은 에피소드를 빼며, 목표의 `vf_value_pred`가 현재 값보다 높은 후보만 남깁니다(`_retrieve_round2_green_from_memory_bank_strict`, L1197–1199). 검색 키는 VQ1 모델이 만든 failure embedding(Qwen 은닉 상태 기반)입니다. 논문 표 6의 상위 1 코사인 0.957–0.993은 이 키 공간 값이고, 같은 과제의 다른 에피소드 항목끼리 0.99라는 것은 키가 거의 한 방향에 모여 있다(이방성)는 가능성을 배제하지 못합니다. 코드만으로는 키 분포를 알 수 없어 **미확인**입니다.

### 학습 설정

**D11. 일치하는 하이퍼파라미터 (일치).** VQ1: LoRA r 16, α 32, dropout 0.05, 대상 `q,k,v,o_proj`; 학습률 백본 2e-5, 머리 1e-4; 30에폭; 손실 가중치 모두 1; K = 8, 값 이력 임베딩 32, ε = 0.005, γ_r = 0.9, τ = 0.5; 초점 α 0.75, γ 2; 미래 머리 2층 MLP(은닉 4배), 정규화 MSE, V-JEPA2 ViT-L(`facebook/vjepa2-vitl-fpc64-256`) 풀링 1024차원; 복구 H = 8, 7차원, FAST + 토큰별 CE, 공유 잔차 LoRA r 16, α 32. 가치 함수 γ 0.99, 가중치 (1, 0.3, 1, 0.05), 워밍업 5, 성공/실패 균형 0.5, 조기 종료 인내 6(표 3과 일치). 쌍둥이 머리 = 최솟값. 표 4의 배치 "12–100"은 설정에서 12.

**D12. 입력 구성의 구체화 (논문에 없음).** (i) VQ1의 VLM 입력은 두 카메라 이미지 + 지시문 + 현재 제안 행동을 **[0, 1000] 정수 bin으로 이산화한 문자열** + `<VQ_QUERY>`입니다(`_build_vq1_chat_text`, `_action_to_text`). 논문 식 3의 `f_vlm(o, ℓ, a)`와 같은 구조지만 행동을 텍스트 정수로 넣는다는 점과 bin 수는 논문에 없습니다(고유감각 상태 텍스트는 VQ2의 `state` 목표 공간에서만 쓰이고 기본 `action` 모드에서는 쓰이지 않습니다). (ii) V-JEPA2 목표는 두 카메라 프레임 각각을 한 장을 `tubelet_size`만큼 복제한 정지 영상으로 인코딩해 토큰 축으로 이은 뒤 평균해 1024차원 풀링 벡터를 만듭니다(`_OnlineVJEPA2Encoder`). "256 토큰 집계"(부록 D.2)와는 일치하나 동영상 인코더에 정지 영상을 넣는 점은 논문에 없습니다. (iii) Q 손실은 종료 전이를 제외합니다(`non_term`).

**D13. 데이터 분할 (논문에 FA-RL 분할만 있음).** 에피소드 단위, 과제와 라벨 유무별 층화, 80/10/10, 시드 42(`episode_split.py`). 기억은 학습 분할만으로 만들고 검증 질의도 학습 기억에서 검색합니다. 논문 본문에는 UniIntervene 자체의 분할 서술이 없고(FA-RL만 80/10/10, 부록 H) 평가 지표(F1, Q-loss)가 어느 분할에서 계산되는지 알 수 없습니다.

**D14. 오프라인 전용 학습과 온라인 갱신 (논문 모호).** 알고리즘 1의 마지막 줄 "update π_θ from B with off-policy RL"과 부록 G.2("프록시 가치 학습 → 동결 → 개입 모듈 학습, 기억은 같은 점수 롤아웃에서 구성")는 모듈이 한 번 학습된 뒤 고정되는 것으로 읽힙니다. 공식 코드도 고정 버퍼에 대한 오프라인 학습만 담고 있습니다. 복구 전이를 재생 버퍼에 넣어 모듈을 다시 학습하는 코드는 없어, 학습 롤아웃의 출처(사전 HIL-SERL 실행인지, 사람 시연인지)는 코드로도 확인되지 않습니다. NTTG 스크립트 설명이 "NTTG-label a HIL-SERL buffer"인 점이 사전 HIL-SERL 버퍼를 가리키는 정황입니다.

**D15. 실패 에피소드 처리 (다름).** NTTG: 성공 `raw_value = t − (T−1)`, `raw_norm = t/(T−1)`, 실패는 `raw_value = −500`, `raw_norm = 0`(`nttg_label_buffer.py`). 라벨 단계는 논문의 "큰 음수 수익" 서술과 맞고 −500이라는 값은 논문에 없습니다. 그러나 VF 학습은 `raw_value`를 성공 판정(`infer_success`: 마지막 전이의 `raw_value == 0`)에만 쓰고, 목표는 터미널 보상(성공 1, 실패 0)의 MC 수익입니다(`train_vf_stitched.py` `preprocess_episode`). 즉 −500은 학습 목표가 아닙니다. 논문 부록 C.1의 "large negative raw return"이 가치 학습을 억누른다는 서술과 달리 실패 에피소드의 가치 목표는 0 근처입니다.

### 안전과 제공 범위

**D16. 피클 로드의 보안 (논문 외).** 데이터 계약이 `list[dict]` 피클입니다. 신뢰할 수 없는 피클은 임의 코드 실행이 가능하고 README가 이를 경고합니다. 원격 모델 코드는 기본 비활성, W&B 기본 꺼짐.

**D17. 기억 은행과 체크포인트 비공개 (논문에 없음).** 기억 은행, 쿼리 임베딩 에피소드, VQ1/VQ2 체크포인트, 데이터가 모두 빠져 있어 표 5, 6의 통계(후보 수, 검색 유사도)를 재현할 수 없습니다.

**D18. 스모크 테스트의 의미 (논문 외).** `reports/fast_overfit_fold_towel.json`은 16개 실제 Fold Towel 행동 덩어리에 FAST 디코더를 과적합시킨 결과(토큰 CE 7.63 → 0.0035)이며 정책 평가가 아닙니다.

## 3. 논문과 일치하는 것 정리

핵심 구조(공유 은닉 토큰 `<VQ_QUERY>`, 미래 머리, 쌍둥이 Q, 값 이력 인코더, 위험 머리, 분리된 가치 조건)와 하이퍼파라미터 대부분(D11), FAST 복구, 기억 은행의 코사인 검색, 학습 분할 기반 기억 구성은 논문과 일치합니다.

## 4. 구현 가능성 판정 ("구현할 수 있는지")

이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음.

**한 줄 판정: 논문 수치(표 1, 2)의 재현은 이 맥에서 불가능합니다. 방법의 핵심 아이디어는 시뮬레이션에서 직접 짠 소형 코드로 축소 구현할 수 있습니다(조건부). 공식 코드의 학습은 데이터가 없고 CUDA 전제라 사실상 불가입니다.**

| 항목 | 판정 | 이유 |
|---|---|---|
| 표 1 실로봇 평가 (UR7e, SpaceMouse, 5과제) | **불가** | 하드웨어와 HIL-SERL 통합 코드가 저장소에 없음(D1) |
| 공식 학습 파이프라인 전체(NTTG → VF → VQ1 → 기억 → VQ2) | **사실상 불가** | 입력 데이터(HIL-SERL 버퍼)와 체크포인트 미공개(D17). `run.py`는 `CUDA_VISIBLE_DEVICES`를 설정하고 `hil_vlm_train.py`가 CUDA 분기(`_cuda_bf16_autocast`, 피크 메모리 로그)를 가짐. Qwen3-VL-2B + V-JEPA2 ViT-L + LoRA 학습은 이 맥 MPS에서 속도와 bf16 지원이 미검증 |
| 프록시 VF 학습 (SigLIP + Gemma-3-270M) | **불가(데이터 없음)** | 모델 가중치 다운로드(수 GB)와 로봇 궤적이 필요. 데이터가 있다면 270M급은 MPS 소규모가 가능할 수 있으나 미측정 |
| NTTG 라벨링, 채굴 규칙, TVR 계산, 기억 선택 로직 | **가능(합성 데이터)** | 순수 numpy. 이 폴더의 `toy_tvr_labels.py`가 규칙 일부를 다시 구현해 확인했습니다(`03` §5) |
| FAST 토큰화 왕복 시험 | **가능(조건부)** | CPU. `physical-intelligence/fast` 토크나이저 파일 다운로드와 `transformers` 사용 승인 필요(다운로드 크기 미확인) |
| 축소 재구현: MuJoCo 접촉 과제 + 소형 CNN/MLP 값 모델 + TVR 트리거 + 최근접 기억 복구 + 사람 개입 시뮬레이터 | **가능(조건부)** | mujoco 4.0, torch 2.15 설치 완료. 새 패키지 설치 없음. 논문과 다른 실험(승인 필요). 정책은 HIL-SERL이 아닌 단순 SAC/TD3 |

**노력 추정(측정 전 추정):**
- 합성 값 곡선에서 TVR/채굴 규칙 민감도 분석 확장: 1–2일(토이 완료분 위에 파라미터 스윕).
- MuJoCo 접촉 과제 축소 구현(값 모델, 트리거, 기억 복구, 개입 시뮬): 2–4주.
- 공식 코드 포팅(CUDA → MPS) + 데이터 자체 수집: 데이터 수집 장비에 좌우되며 이 저장소 범위를 벗어납니다. 추정하지 않습니다.
- 기준선(HIL-SERL, FA-RL) 재현: HIL-SERL 코드는 공개(Apache-2.0)지만 JAX 실로봇 스택이고 FA-RL 코드는 찾지 못했습니다.
