# 04. 공식 코드 대조와 구현 가능성

대상: `github.com/NVlabs/cosmos-policy`, 커밋 `18a2acc` (2026-01-22). `repo/`에 얕은 복제로 정적으로만 읽었습니다. 실행, 설치, import, 가중치 다운로드는 하지 않았습니다. 코드는 신뢰할 수 없는 자료로 취급했습니다.

## 1. 저장소와 라이선스

- 라이선스: **Apache-2.0** (`repo/LICENSE`). 소스 파일 헤더도 SPDX Apache-2.0입니다(예: `cosmos_policy/_src/imaginaire/utils/high_sigma_strategy.py`).
- `ATTRIBUTIONS.md`는 제3자 구성요소(GitPython BSD-3, imageio BSD-2)를 적습니다.
- 가중치와 데이터셋(HuggingFace `nvidia/…`)의 라이선스는 확인하지 않았습니다. 기반 모델 Cosmos-Predict2의 라이선스도 이번 분석에서 확인하지 않았습니다.
- 학습 데이터 중 ALOHA rollout 648개는 공개되지 않았습니다(§3, D7).

## 2. 논문 수치와 코드 설정의 일치

| 항목 | 논문 | 코드 (파일) | 판정 |
|---|---|---|---|
| 학습 노이즈 혼합 | log-normal 0.7 + U[1, 85] 0.3 (A.2.1) | `HybridEDMSDE`, `hybrid_sigma_distribution=True`, `uniform_lower=1.0`, `uniform_upper=85.0`, `p_mean=1.386`, `p_std=1.2` (`cosmos_policy/modules/hybrid_edm_sde.py`, `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py` L130–136) | 일치 |
| 추론 σ | σ_max=80, σ_min=4 (A.2.1) | `…__inference_only`에서 `sigma_max=80, sigma_min=4` (L192–193, L275–276, L360–361, L452–453) | 일치 |
| 행동 청크 / 실행 | LIBERO 16/16, RoboCasa 32/16, ALOHA 50/50 (A.2) | LIBERO `chunk_size=16`, RoboCasa `chunk_size=32`, `num_open_loop_steps=16`, ALOHA `num_open_loop_steps 50` | 일치 |
| 추론 디노이징 스텝 | LIBERO/RoboCasa 5, ALOHA 10, 계획 10/5/5 (A.3.1) | `--num_denoising_steps_action 5` (LIBERO 예시), ALOHA 정책 이름 `10stepsA` | 일치 |
| 계획 앙상블 | 세계 모델 3, 가치 5 (§4.3, A.3.1) | `num_future_state_predictions_in_ensemble=3`, `num_value_predictions_in_ensemble=5` (`run_libero_eval.py` L226–229) | 일치 |
| 계획 N | ALOHA N=8 (A.4.2) | `--num_queries_best_of_n 8` (`run_robocasa_eval.py` 예시; ALOHA 문서는 같은 플래그 사용) | 일치 |
| 가치 집계 | majority mean (§4.3) | `value_ensemble_aggregation_scheme="majority_mean"` 구현 (`cosmos_utils.py` L794–812) | 일치 (단, 기본값은 `average`; D6) |
| 배치 분할 | demo 50%; rollout 절반씩 WM·가치 (§4.2) | `demonstration_sampling_prob=0.5`, `p_world_model=0.5` (`robocasa_dataset.py` L816–821) | 일치 |
| ALOHA 계획 학습 비율 | 90% rollout, 10% 정책 (§4.2) | `demonstration_sampling_prob=0.1` (ALOHA planning config, L388–390) | 일치 |
| 전역 배치 | 1920 / 800 / 200 (A.2) | LIBERO 30 x 8 x 8 accum, RoboCasa 25 x 8 x 4, ALOHA 25 x 8 (`LIBERO.md` L99, `ROBOCASA.md` L106) | 일치 |
| 학습 스텝 | 40K / 45K / 50K (A.2) | 설정은 `max_iter=1000000`(실행 상한). 스텝은 실행 명령과 문서가 정함. LIBERO.md는 "30K 뒤 학습률 급감"과 "원래 실험은 40K 스텝" 기록 | 부분 일치 (D4) |
| LIBERO 데이터 | 성공 demo 500/스위트, 실패 포함 rollout (§5.1) | `success_only` (demo), `all_episodes` (rollout) | 일치 |
| 인간 demo RoboCasa | 과제당 50 (§5.1) | README "1200 demos" = 24 x 50 | 일치 |

## 3. 차이 (D1–Dn)

**D1. 학습 노이즈 하한과 상한.** 코드 학습 설정은 `sigma_max=200`, `sigma_min=0.01`입니다(L133–134). 논문은 학습 시 하한과 상한을 적지 않았습니다(A.2.1은 추론 σ_min=4만 적음). 학습 시 σ 범위는 논문에 없는 설정입니다. 영향 크기는 측정하지 않았습니다.

**D2. 학습률과 스케줄.** 코드는 `lr=1e-4`, 첫 주기 30K 스텝, 이후 5배 감소(`f_min=[0.3, 0.06]`, `cycle_lengths=[30000, …]`) 설정입니다(L92–106). 논문은 학습률과 스케줄을 적지 않았습니다. 재현에 필요한 값이 본문에 없습니다.

**D3. 할인율 γ.** 코드는 LIBERO와 RoboCasa `gamma=0.99`, ALOHA `gamma=0.998`을 씁니다. 논문은 γ를 쓰지만 수치를 적지 않았습니다.

**D4. 학습 스텝 수.** 논문 40K/45K/50K는 `LIBERO.md`, `ROBOCASA.md`와 일치합니다. 다만 설정 파일에는 고정 값이 없고 실행 명령이 정합니다. "30K 뒤 감쇠"와 "40K 스텝" 사이의 관계는 문서에 있고 논문에는 없습니다.

**D5. 학습 데이터 필터링.** 논문은 실패 demo를 정책 학습에서 뺀다고 합니다(§5.1). 코드의 LIBERO 설정은 `success_only`를 쓰고, ALOHA는 `treat_demos_as_success_rollouts=True`로 demo를 성공 rollout으로 봅니다(L299). 두 설정 모두 논문과 맞지만 ALOHA에서 demo 처리는 논문 본문에 없습니다.

**D6. 가치 집계 임계값과 기본값.** 논문은 "fixed threshold"라고만 씁니다. 코드는 0.05를 씁니다(`cosmos_utils.py` L798). 집계의 기본값은 `average`이고, `majority_mean`은 명시해야 씁니다(`run_libero_eval.py` L230, `run_robocasa_eval.py` L197). 논문 결과를 재현하려면 이 두 값을 맞춰야 합니다.

**D7. 학습 데이터 공개 범위.** 논문 §7은 "training data"를 공개한다고 씁니다. 코드 설정에는 "This rollouts dataset is not released"라는 주석이 있습니다(L373–375). ALOHA 계획 학습용 648 rollout은 공개되지 않았습니다. LIBERO와 RoboCasa 데이터는 HF에 있습니다(`LIBERO.md`).

**D8. 계획 모델 체크포인트의 장치.** `get_planning_model(cfg, device="cuda")`(`cosmos_utils.py` L298)가 기본값입니다. 장치를 명시하지 않으면 CUDA를 가정합니다.

**D9. CUDA 하드코딩.** `cosmos_utils.py`에 `.cuda()` 호출이 28곳 있습니다(예: L892, L1038–1078). 학습 샘플러 `hybrid_edm_sde.py`는 `device="cuda"`를 직접 씁니다(L66–85). 논문은 이 의존성을 언급하지 않습니다. `DEVICE`는 CUDA 없으면 CPU로 대체하지만(L48), 위 호출은 대체하지 않습니다.

**D10. 가속 라이브러리.** `flash_attn`, `flash_attn_3`, `transformer_engine`을 여러 모듈이 import합니다(예: `attention/flash2/functions.py` L24, `utils/fused_adam.py` L17, `utils/graph.py` L24). 어떤 경로에서 이 import가 필수인지, 없을 때 대체 경로가 있는지는 검사하지 않았습니다(§5의 미확인 항목).

**D11. 분산 학습.** 학습 코드는 65개 파일에서 분산 학습 API(torch.distributed, FSDP 등)를 씁니다. 논문의 8/32/64 H100 설정은 이 경로를 전제합니다.

**D12. 값의 정규화와 빈 값.** 논문 A.1은 가치를 [0, 1]로 되돌린다고 씁니다. 코드는 demo의 가치 목표를 -100으로 채우고(`robocasa_dataset.py` L1040), 가치 손실 대상에서 빼는 것으로 보입니다. 정확한 마스크 처리는 검사하지 않았습니다.

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 학습은 불가능합니다. 추론은 포팅 후 조건부로 가능합니다. 실로봇 재현은 불가능합니다.**

| 작업 | 판정 | 근거 | 노력 추정 (측정 없음) |
|---|---|---|---|
| 학습 (LIBERO, RoboCasa, ALOHA 모두) | **불가** | 8–64 H100 x 48시간을 전제합니다(A.2). 이 맥에서 2B 모델 미세조정은 시간과 메모리 모두 맞지 않습니다 | 해당 없음 |
| 추론 (정책 모드, 체크포인트 로드) | 조건부 | README는 추론 VRAM을 6.0–8.9 GB로 적습니다(체크포인트 파일 크기는 적지 않음). 24 GiB에는 들어갈 수 있으나 `.cuda()`(D9)와 가속 라이브러리(D10)를 고쳐야 합니다. MPS에서 bf16과 주의 연산 지원은 검증하지 않았습니다 | 포팅 1–2주 (추정) |
| LIBERO 시뮬 평가 (500 에피소드 x 4) | 조건부 | 추론 포팅 후 MuJoCo/LIBERO 설치가 필요합니다. 설치와 가중치 다운로드는 승인 후 진행합니다 | 포팅 포함 2–3주 (추정) |
| 계획 (best-of-N) 재현 | 불가에 가까움 | 한 행동에 약 5 s, N=8, 8 GPU 병렬을 가정합니다(A.4.2). MPS에서는 시간이 크게 늘고 병렬화 코드(`multiprocessing` spawn, CUDA 전제)를 다시 짜야 합니다 | 미정 |
| 실로봇 ALOHA 재현 | **불가** | 로봇과 ALOHA 설정, 648 rollout(미공개, D7)이 필요합니다 | 해당 없음 |
| 논문 수치의 CPU 재계산과 토이 | 가능, 완료 | `analysis/recompute.py`, `analysis/toy_majority_mean.py` | 완료 |

## 5. 미확인 항목

- 가중치와 데이터의 라이선스(HF 모델 카드).
- 가속 라이브러리 없이 추론이 도는지(D10). 실제 실행으로 확인하지 않았습니다.
- 공식 코드의 테스트나 CI가 있는지. 확인하지 않았습니다.
- 학습 스텝 40K에서 L1 손실 0.012 (LIBERO.md, A.2.2)가 재현되는지. 학습을 하지 않았으므로 확인할 수 없습니다.
