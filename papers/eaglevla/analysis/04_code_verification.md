# 04. 논문 ↔ 코드 대조 (정적 읽기)

공식 저장소 두 개를 얕게 클론해 **읽기만** 했습니다. 실행, import, 설치, 체크포인트 다운로드는 하지 않았습니다. 클론은 `repo/`에 있고 `.gitignore`에 들어 있습니다(`repo/EagleVLA`, 75 MB, HEAD `852ef5b`, 2026-10-02 / `repo/EagleVLA-Edge`, 273 MB, HEAD `619e92e`, 2026-10-23). 파일 경로는 `repo/EagleVLA/` 기준입니다. 이 저장소는 OpenPI(JAX) 포크이고, 엔진 저장소는 llama.cpp(ggml) 포크입니다.

## 1. 요약

- **학습과 평가 코드는 논문의 LIBERO 실험에 해당합니다.** 보정 모듈 구조, 토큰 감소기, 행동 인코더, 두 단계 이상의 학습, κ 기반 스케줄, 네 LIBERO 군집 평가가 있습니다.
- **그러나 목적함수, 신뢰도 정의, 스케줄 규칙, 학습 단계 설정은 논문과 다릅니다.** 가장 큰 차이는 D1–D4입니다. 논문 Alg. 1의 "VLM 재호출"은 코드에서 "두 단계 full π0 폴백"으로 바뀌어 있고, 신뢰도는 절대 임계값이 아니라 초기값 대비 하락량으로 판단합니다.
- **실로봇 코드(X2-W, XR-1)는 없습니다**(D10). 엔진 저장소의 지연 표는 논문 Table 4 최종 행과 같지만, 그 행 라벨과 구성이 논문 설명과 다릅니다(D12–D14).
- **이 맥에서 재현 가능한 범위:** 구조 재구현(학습 제외)은 가능하고, 학습과 평가와 지연 측정은 불가입니다(§4).

## 2. 대조표 (D1–D15)

| 번호 | 항목 | 논문 | 공식 코드·저장소 | 판정 | 근거 |
|---|---|---|---|---|---|
| D1 | 2단계 손실 | Eq. 1: L_predict = ‖ĥ−h‖² + λ‖ĉ + L_predict‖² | 기본 레시피의 3단계 L_cond는 이분산 가우시안 음의 로그우도 `heteroscedastic_gaussian_nll`(0.5·(오차²/σ² + log σ²))이고 μ와 로그 분산을 함께 학습. 로그 분산 보정 `logvar_calibration_loss`는 옵션 `wm_logvar_only_finetune`에서만 쓰이며 공식 스크립트는 켜지 않음 | **불일치** | `src/openpi/models/pi0_world_model.py` L57–71, `world_model_training.py` L318–330(옵션 경로), `world_model_training.py` L387, `world_model_training_four_stage.py` L1036–1050(`train_step_stage2` 호출) |
| D2 | 신뢰도 정의 | 신뢰도 헤드 ĉ, 학습 목표 −L_predict | κ = −mean(log σ²) (`global_confidence_from_log_var`). 기본 레시피에서 로그 분산 헤드는 1·2단계에서 제외되고 3단계에서만 NLL로 학습됨 | **다름** (정의는 같은 방향, 목표가 다름) | `pi0_world_model.py` L91–92, `pi0_async_inference_policy.py` L130, `world_model_training_four_stage.py` L654 |
| D3 | VLM 재호출 규칙 | Alg. 1: ĉ > θ이면 VLM 생략, ĉ ≤ θ이면 VLM으로 KV와 은닉 상태 갱신 | 청크마다 VLM 접두 1회 계산(`h_t`), 라운드 r ≥ 1은 보정 모듈과 AE만 실행. κ_r < κ₀ − δ이면 **두 단계 폴백**(롤아웃 후 full π0 재실행과 접합), κ_r > κ₀ + δ이면 조기 종료. 롤아웃 중에는 VLM을 재호출하지 않음 | **불일치** | `pi0_async_inference_policy.py` L231–299(단일 라운드), L318–540(다중 라운드), L469–495 |
| D4 | 매개변수 이름과 기본값 | 임계값 θ < 0 (절대값), Fig. 8은 θ = −0.2..−0.4 | 상대 하락 임계값 `kappa_delta` = 0.4 (기본), 트리거 K = 9 → 1 스윕, overlap = H − K, 적응형 모드의 WM 간격 DELTA = ⌊(overlap+2)/3⌋ (= ⌈overlap/3⌉) | **다름** (표의 ∆ ↔ 코드의 K 대응이 논문에 없음) | `scripts/eval_wm_libero_spatial.sh` L38–44, L80–83, `scripts/libero_wm_eval_spatial_k9to1_adaptive_kappa_low_replan_gpu2_kd0p4.sh` |
| D5 | 행동 전문가의 고유감각 입력 | 논문 방법 설명에 없음 (보정 항목만 AE에 들어감) | 기본 `ae_proprio_source = "vlash_last_action"`(VLASH식으로 고유감각을 최근 행동으로 전진), 클라이언트 예제는 `"prefix_t"`를 씀 | **누락 (논문에 없는 요소)** | `pi0_async_inference_policy.py` L95–104, L266–290, `examples/libero/main_asy.py` L47 |
| D6 | 학습 단계와 스텝 | Table 6: 2단계, 1단계 30,000 / 2단계 25,000, 배치 32, lr 3e-4, AdamW, 워밍업 500, 코사인 | 스크립트 기본: 3단계, 1단계 30,000 / 2단계 15,000 / 3단계 55,000, 배치 16(`BATCH_SIZE`), grad accum 1. 학습률 기본은 코드 기본값 3e-4 | **불일치** (단계 수와 스텝, 배치) | `scripts/train_wm_libero_spatial_four_stage.sh` L35–41, README "Recipe", `world_model_training.py` L930 |
| D7 | 청크 길이 H | Fig. 3: H = 20 (동기 분석). 본문 주 실험의 H는 없음 | 학습 `--handover-horizon-min/max 10`, `--max-delta-t 10`, 평가 `AH=10`, `K=9` | **불일치** (H 값의 적용 범위가 다름) | `train_wm_libero_spatial_four_stage.sh` L69–71, `eval_wm_libero_spatial.sh` L41 |
| D8 | 추가 손실과 학습 데이터 범위 | 목적함수에 의미 일관성 항 없음. 학습 과제 범위 미기재 | 세미그룹 일관성 손실 L_sg(λ_sg = 0.1)는 코드에 있지만 공식 스크립트는 4단계 파이프라인에서 `--stage4-steps 0`으로 실행하므로 기본 레시피에서는 꺼져 있음. 학습 데이터는 `--libero-task-index-min 30 --max 40`으로 제한 | **누락 (논문에 없는 요소; 기본 레시피에서는 비활성)** | `world_model_training.py` L805–870, `world_model_training_four_stage.py` L672–681, `train_wm_libero_spatial_four_stage.sh` L63–68, L78 |
| D9 | 토큰 감소기와 행동 인코더 | Q-Former로 T' = 4 토큰(Fig. 10), 행동은 MLP → Transformer 블록 → 마지막 토큰 | `learned_cross_attn` 토큰 감소기, `num_condition_tokens = 4`, `action_encoder_kind = transformer_block` (스크립트 설정, 코드 기본값은 GRU) | **일치** (스크립트 설정 기준; 기본값과 다름) | `pi0_world_model.py` L24–46, L147–240, 스크립트 L72–73 |
| D10 | 실로봇 코드 | X2-W 로봇, XR-1 모델, Orin 배포, 15 Hz | 저장소에 LIBERO 학습·평가만 있음(README "This release open-sources LIBERO training and evaluation code") | **재현 불가** (코드와 데이터 없음) | README 첫 소개 단락 |
| D11 | LIBERO 평가 설정 | 네 군집, 시행 수와 시드 미기재 | 군집 변수로 네 군집 선택 가능, 기본 50 trials/task | **일치** (시행 수 기본값이 논문 표 해석의 근거가 됨) | README "Other LIBERO suites", `eval_wm_libero_spatial.sh` L37–38 |
| D12 | 엔진 표의 행 라벨과 구성 | Table 4 최종 행 "+Intermediate Buffer & Unroll" (v7 값 66.6/137.0/67.6/271.1, 29.0/54.9/48.9/132.8) | 엔진 README에 같은 수치가 있고 행 라벨은 "+ time cache, and GEMM tuning (PR #6)". 이 행은 "earlier rows are historical measurements, not a new same-checkpoint controlled ablation"이라고 적힘. 이 최종 행은 FP16 GEMM 자동 튜닝 스위치, 시간 변조 캐시, FP16 캐스트, "new FP16 model/projector"를 포함 | **라벨 불일치, 귀속 불가** | `repo/EagleVLA-Edge/README.md` 성능 절(PR #6 표, 약 644–705행), 커밋 `f26a2f4`, `619e92e`, `6cec0d9`, `b727abf` |
| D13 | Thor 정확도 게이트 | "negligible success rate loss" (§5.2, 결론) | Thor 최종 구성은 "highest measured throughput, not a same-device accuracy pass", 최소 cosine 0.999075 < 게이트 0.9994. Orin 최종 구성은 게이트 통과(273.36 ms 엔드투엔드 P50) | **단서 누락** | `repo/EagleVLA-Edge/README.md` (Thor 절 주석 †), 같은 파일의 Thor 정확도 주의 절 |
| D14 | 나이브 기준 값 | Table 1 RTX 4090 π0.5 총 76.4 ms(ViT 13.2, LLM 35.9, AE 25.2). Table 4 Orin/Thor 나이브 값은 1420.8 / 457.9 ms | 엔진 README의 4090 "Naive PI0.5"는 162.9 ms(커밋 `619e92e`, "correct RTX 4090 baseline"). Orin 나이브 행 1420.8 ms는 목록 합(1320.1 ms)과 100.7 ms 차이가 그대로 있음 | **기준값 불일치** (4090: 76.4 대 162.9 ms) / Orin 목록 차이 공통 | 논문 Table 1, `repo/EagleVLA-Edge/README.md` 성능 절 |
| D15 | 라이선스와 출처 | 논문 본문 저작권 표기 없음 | 공식 EagleVLA: Apache-2.0(`LICENSE`, OpenPI 기반, `LICENSE_GEMMA.txt` 포함). EagleVLA-Edge: MIT(ggml 저작권 2023–2026). README는 OpenPI, LIBERO 저작권 유지 명시 | **확인됨** | `LICENSE`, `repo/EagleVLA-Edge/LICENSE` |

### 2.1 추가로 확인한 항목

- **학습 손실 구성(D1–D2, D8):** 공식 스크립트는 4단계 파이프라인을 쓰며 1단계는 행동 손실과 토큰 감소기, 2단계는 로그 분산 헤드를 뺀 L_cond, 3단계는 λ_cond·L_cond(이분산 NLL, 토큰 감소기 제외)와 λ_act·L_act(μ를 분리)입니다. 4단계(L_sg)는 0 스텝입니다(`world_model_training_four_stage.py` L224–300, `world_model_training.py` L845–870, 스크립트 L68–82). 논문 Eq. 1의 두 항과 대응하지 않고, 로그 분산 헤드는 NLL로만 학습됩니다.
- **추론 경로의 VLM 호출 횟수:** `Pi0AsyncInferencePolicy.infer`는 청크 요청마다 VLM 접두를 한 번 계산합니다(L142–163). 그래서 "VLM 호출 비율" 측정은 클라이언트 쪽 로그(`openpi/wm_confidence_kappa`, `wm_stitch_n`)에서 나옵니다. 논문 Fig. 9의 신뢰도 궤적은 이 로그에서 나온 것으로 보이지만, 확인은 불가합니다.
- **예측 대상:** 보정 모듈의 mu는 VLM 최종 층 토큰 표현(b × m × d)이고 행동 전문가의 `future_condition_tokens`로 들어갑니다(`pi0.py` L146–160). 논문 Fig. 5와 맞습니다.
- **예제 클라이언트:** `examples/libero/main_asy.py`의 `ae_proprio_source`는 `"prefix_t"`이고, 서버 기본값은 `"vlash_last_action"`입니다. 어느 설정으로 Table 3를 만들었는지 논문에 없습니다(D5).
- **평가 스크립트 기본값:** 시행 수 50, 군집 libero_spatial, H 10, K 9, δ 0.4(적응형 모드에서만). 논문 Table 3의 ∆ 대응은 확인되지 않습니다(D4).
- **Edge 엔진 구성:** Orin 최종 구성의 환경 변수(`GGML_CUDA_LT_AUTOTUNE=1`, `GGML_CUDA_F16_F32_OUT=1`, `PI05_FP16_SHARED_CAST=1`, `PI05_CACHE_TIME_MODULATION=1`)는 저장소가 "fastest tested"라고 적은 조합입니다. 논문 방법 설명에는 이런 설정이 없습니다(D12).
- **엔진의 CUDA 아키텍처:** 빌드 옵션 87(Orin)과 110(Thor). CUDA 전용이므로 맥에서는 빌드하지 않습니다.

## 3. 저자에게 물을 질문 (10개)

1. Table 3의 값은 몇 번의 시행을 어떻게 집계한 값인가요? 시드 수, 군집당 에피소드 수, 평균 방식을 알려 주세요. 500회 단일 집계로는 148개 중 85개 값이 맞지 않습니다(`03` §2).
2. Table 3의 "+Sched" 열과 "Ours" 열 중 본문 14.8 %p와 3.9 %p는 어느 열로 계산했나요?
3. Table 3의 θ 값은 얼마였나요? 공식 코드의 δ = 0.4와 논문 Alg. 1의 절대 임계값 θ는 어떻게 대응하나요?
4. 공식 저장소의 두 단계 폴백(full π0 재실행)은 논문 Alg. 1에 없습니다. 논문 결과는 어느 쪽 규칙으로 얻었나요?
5. 최종 지연 행의 값은 v6에서 v7로 바뀌었습니다. 무엇을 바꿔 측정했나요(모델 체크포인트, FP16 설정, 측정 장치 전력 모드, 튜닝 스위치)?
6. Thor 최종 구성(132.8 ms)이 정확도 게이트(cosine > 0.9994)를 통과하지 못한 것을 논문에 어떻게 반영할 계획인가요?
7. 반응 시간의 정의와 계산 방법은 무엇인가요? +Schedule 행의 674.9 ms와 244.7 ms는 어떻게 얻었나요?
8. Orin 50 W의 나이브 행(1420.8 ms)에서 목록 합과 100.7 ms 차이는 무엇인가요?
9. 실로봇 실험에 쓴 XR-1 모델은 어떤 가중치와 학습 데이터인가요? 15 Hz 실행과 Table 4의 제어 주기는 어떤 관계인가요?
10. 학습 설정은 논문 Table 6(두 단계, 배치 32)과 공식 스크립트(세 단계, 배치 16) 중 어느 쪽이 최종 결과에 쓰였나요? L_sg 항과 학습 과제 인덱스 제한(30–40)도 논문에 적어 주실 수 있나요?

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 구조는 구현할 수 있습니다(노력 중간). 학습, 평가, 지연 측정은 이 맥에서 할 수 없습니다.**

| 작업 | 가능 여부 | 이유 | 노력 (추정, 측정 없음) |
|---|---|---|---|
| 보정 모듈과 스케줄러를 PyTorch 소형 모델로 재구현 | 가능 | 보정 모듈은 40M 수준(§4.1)이고 Alg. 1은 짧다. 공식 코드는 JAX이므로 옮겨야 함 | 1–2주 (학습 제외) |
| 2단계 학습 (30k·25k 스텝, 배치 32, A100 기준) | 불가 | 학습 설정이 CUDA 48 GB급(README), JAX 0.5.3, Python 3.11 전제. 맥의 MPS로는 시간 내 불가 | - |
| 공식 코드로 LIBERO 1 suite 평가 (50 trials/task) | 불가 | 평가 경로가 JAX 서버와 LIBERO 시뮬레이터(EGL 또는 Xvfb)를 씀. 체크포인트 다운로드가 필요하고 맥에서 3B 규모 추론은 500회 평가에 비현실적(추정) | CUDA 리눅스 머신이 있을 때 3–5일 (추정) |
| 엔진 저장소 빌드와 Orin 지연 재측정 | 불가 | CUDA, Jetson 전용 빌드(아키텍처 87, 110) | - |
| 실로봇 재현 | 불가 | 로봇, 데이터, 코드 없음(D10) | - |
| 손실 비교 토이 (Eq. 1 대 이분산 NLL, 합성 동역학) | 가능 | torch와 작은 MLP로 CPU/MPS에서 수 분 | 1–2일 (추정) |

- 맥에서 가능한 것은 재계산, 구조 재구현, 합성 토이입니다. 논문 수치 재현은 CUDA 환경이 있어야 합니다.
- 구현할 때 먼저 확인해야 할 점: 손실 목적함수(D1), 스케줄 규칙(D3), 고유감각 입력(D5). 셋 다 논문에 명확히 적혀 있지 않아 구현 결과가 논문과 달라질 수 있습니다.

## 5. 라이선스와 공개 범위

- **논문:** arXiv 초록 페이지의 "Rights to this article" 링크는 `http://creativecommons.org/licenses/by/4.0/`입니다(CC BY 4.0, 2026-11-22 확인). PDF 텍스트에 ©, copyright, All rights reserved, 회사 권리 고지가 없습니다(소속 기관 이름만 있음). CoRL 목록의 license 필드도 "CC BY 4.0"입니다. 따라서 논문 그림을 출처 표기와 함께 쓸 수 있습니다. 이 폴더의 공개 산출물에는 논문 그림을 넣지 않고 우리가 그린 그래프만 넣기로 했습니다.
- **공식 저장소:** EagleVLA는 Apache-2.0이고 OpenPI 기반입니다. EagleVLA-Edge는 MIT이고 ggml 저작권 표기를 유지합니다.
- **체크포인트와 데이터:** 라이선스 미확인. 다운로드하지 않았습니다.
- **공개본 주의:** 공개본에 저장소 클론 내용은 넣지 않습니다. 저장소 링크와 라이선스 정보만 적습니다.
