# WAMotor 코드 대조 (논문 ↔ 공식 저장소)

공식 저장소 `Overdue-Lin/WAMotor`, 커밋 `2de16b8d6fdd227a7a409c3ee1ba2300b10b92e6`(2026-11-20, "Update README"), 얕은 복제본 `repo/`를 **정적으로만** 읽었습니다. 설치, 실행, import는 하지 않았습니다. 코드 조각은 저장소의 파일명과 줄 번호를 씁니다. 원본 복제본은 `repo/`에 있고 `.gitignore` 대상입니다.

## 0. 요약

- **핵심 알고리즘은 논문과 맞습니다.** 임계값 식(6), 앵커 재사용, 접두사 커밋, 보정 스텝(Alg. 1 줄 11–15), 마지막 스텝(줄 17)이 모두 대응합니다. 대표 구현은 `src/wamotor/models/wan22/speculative/action.py`(152줄)입니다.
- **지연 비교가 공정하지 않을 가능성이 큽니다.** 투기 경로의 예측 함수만 CUDA에서 `torch.compile(mode="reduce-overhead")`(CUDA 그래프)로 감쌉니다. 기준 경로(Vanilla)는 이를 쓰지 않습니다. 논문에는 이 사실이 없습니다(D5). 속도비의 일부는 알고리즘이 아니라 컴파일 효과일 수 있습니다.
- 기준선(VLA-Cache, TeaCache, BAC, ParaStep, SpeCa, FlowCast) 구현도 없습니다(D11).

## 1. 대조표

| # | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 기본 ε | 1×10⁻² (§5.1) | 세 모델 `fastwam.py:822, 1014`, `fasterwam.py:171`, `jointwam.py:145`, `speculative/*.py` 기본값 1e-2 | 일치 |
| D1b | 지연 측정의 ε | 본문은 ε = 10⁻²라고만 씀. Fig. 3 지연 실험의 ε는 명시 없음 | `experiments/latency/benchmark_latency.py:82` 기본 `--spec-epsilon 5e-2`, `run_latency.sh:31` 같음 | **불일치(확인 필요)**. 지연 실험이 ε=5e-2로 돌았다면 논문 설정과 다릅니다 |
| D2 | 임계값 식 | 식 (6): ‖u−v̂‖²·(h_k/h_mid)² ≤ ε, h_mid는 간격의 중앙값 | `action.py:65–66`: `thresholds = ε·(median|Δt| / |Δt_k|)²`, 검사 `mse < threshold` (`:130`) | 일치. 논문은 ≤, 코드는 <. 실질 차이 없음 |
| D2b | 오차 집계 | "토큰별 최대 MSE" (식 (2) 아래) | `mse = ...mean(dim=-1)` 후 `.all(dim=-1).all(dim=-1)` (`:124–130`) | 일치(모든 토큰 통과 = 최대 MSE ≤ ε) |
| D3 | 초기 앵커(RAVS, K=1) | 첫 Euler 한 번 뒤 u_1을 앵커로 (§4.3, Alg. 1 줄 1–2) | `action.py:96–104`: t_0에서 Euler 1회, 이어서 m=1에서 u_1 평가 | 일치(K=1 하드코딩) |
| D3b | K 민감도 | K ∈ {1,2,3,4} (Table 5) | 샘플러에 K 인자 없음. 시작 스텝이 1로 고정 | **재현 불가**: 표 5 실험 코드 없음 |
| D4 | 재사용 식 | â_{m,j} = a^s_m + u_m Σ Δt (식 (1)) | `action.py:~111–118`: `cumsum(Δt[m:end]) · velocity` | 일치 |
| D5 | 컴파일 | 논문 어디에도 `torch.compile` 언급 없음 | `action.py:12–28`: 투기 예측기만 `torch.compile(mode="reduce-overhead")`(CUDA). Vanilla 경로(`fastwam.py:992`)는 직접 호출 | **불공정 가능성**. 속도비에 컴파일·CUDA 그래프 효과가 섞일 수 있음. 논문 측정 설정을 따로 확인해야 함 |
| D6 | 지연 측정 횟수 | 30회 측정, 워밍업 5회 (Fig. 3 설명) | `benchmark_latency.py:58–59`: 워밍업 5, 기본 측정 10회 | 명령행 인자로 바꿀 수 있음. 기본값은 논문과 다름(경미) |
| D7 | 수용률·NFE 보고 | 논문 본문에 수용 스텝 수나 NFE 없음 | `benchmark_latency.py:230–280`: `_SpeculationTracker`가 별도 무측정 재실행에서 `accepted` 값을 셈 | 코드는 측정하지만 논문에는 없음. **수용률과 NFE 보고를 요청할 만함** |
| D8 | 접두사 수용과 보정 | Alg. 1 줄 8–15 | `action.py:133–151`: 전부 수용 시 `m=end`, 부분 수용 시 `draft[accepted−1]`에서 `fallback_velocity`로 Euler 1회, `m=step+1` | 일치. `accepted==0`일 때 검증 결과 재사용은 논문의 추가 호출을 피하는 등가 최적화 |
| D9 | 마지막 스텝 | Alg. 1 줄 17 | `action.py:106–109` | 일치 |
| D10 | LIBERO·LIBERO-Plus 평가 | 표 1·3의 성공률 | `scripts/eval_libero.sh:81`, `eval_libero_plus.sh`가 `experiments/libero/run_libero_manager.py` 호출. **파일 없음**. `third_party/`에 RoboTwin만 있음 | **재현 불가(저장소 누락)** |
| D11 | 기준선 구현 | VLA-Cache, TeaCache, BAC, ParaStep, SpeCa, FlowCast (§5.1) | 저장소 전체 검색에 해당 구현 없음 | **재현 불가** |
| D12 | 체크포인트 | 세 공개 WAM 사용 (§5.1) | README: "download the checkpoints following the setup instructions"라고만 씀. 링크 없음. 백본은 `Wan-AI/Wan2.2-TI2V-5B`(`fastwam.yaml`), 행동 DiT `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | 공개 경로가 명시되지 않음. Joint-WAM은 저자 구현(`jointwam.py`) |
| D13 | LIBERO-Plus 평균 | 표 3의 Avg는 7개 범주의 **단순 평균**(검산 잔차 0.00) | `scripts/summarize_libero_plus_categories.py:~80–100`: "Average"는 **과제 수 가중** 평균 | 두 값이 다릅니다(범주별 과제 수가 같지 않다면). 논문 값은 단순 평균과 맞음 |
| D14 | 인용 | Joint-WAM [49] | 저장소 README BibTeX 제목이 "Faster-WAM: Efficient Inference-Time Future Conditioning..."이고 `journal = {arXiv}` | BibTeX 제목 오류(복사 실수로 보임) |
| D15 | 라이선스 | — | 저장소 루트 라이선스 파일 없음(GitHub API `license: null`). `third_party/RoboTwin/LICENSE`만 있음 | 재사용 조건 불명 |
| D16 | 제외 설계 | 비디오 상태는 검증하지 않고 커밋(Joint) (§4.5) | `joint.py:173`: 검증은 `action_mse`만. `:215`에서 비디오 속도는 수용 시 그대로 씀 | 일치 |

## 2. 세부 메모

### D5 (컴파일 비대칭)

`action.py:12–28`:
- CUDA 기기이면 `torch.compile(model._predict_action_noise_with_cache, mode="reduce-overhead")`.
- 그 밖이면 원래 함수.

Vanilla 경로(`fastwam.py:992`)는 `_predict_action_noise_with_cache`를 직접 부릅니다. 따라서 CUDA 지연을 재면 투기 경로만 CUDA 그래프의 이득을 받습니다. 이 이득이 얼마나 되는지는 이 코드만으로 알 수 없습니다. 확인 방법은 "Vanilla에도 같은 컴파일을 적용한 지연"을 재는 것입니다. 이것은 GPU가 필요하므로 이 맥에서는 할 수 없습니다.

### D7 (수용률 추적)

`benchmark_latency.py`의 `_SpeculationTracker`는 `sys.settrace`로 `if accepted == count:` 줄에서 `accepted` 값을 모읍니다. 시간 측정과 분리된 무측정 재실행입니다. 출력 키는 `average_spec_diffusion_steps`, `spec_rounds`입니다. 논문은 이 값을 보고하지 않습니다. 보고하면 "지연이 줄어든 것이 수용 스텝 때문인지, 호출 수 때문인지"를 나눌 수 있습니다.

### D10 (누락된 평가 엔트리포인트)

`scripts/eval_libero.sh` 81행: `"${PYTHON_BIN}" experiments/libero/run_libero_manager.py "${MANAGER_ARGS[@]}"`. 저장소의 `experiments/`에는 `latency/`와 `robotwin/`만 있습니다. `third_party/`에는 `RoboTwin/`만 있습니다. `scripts/setup/install_libero.sh`가 가리키는 `third_party/LIBERO`는 없습니다. 따라서 LIBERO와 LIBERO-Plus 성공률은 이 스냅샷으로 다시 만들 수 없습니다. README의 파일 구조 그림에도 `experiments/libero/`가 있어서 누락이 분명합니다.

### D13 (집계 방식)

논문 Table 3의 Avg는 7개 범주 평균입니다. 예: Joint-WAM 기준 행 36.44, 62.69, 91.39, 90.74, 55.15, 49.03, 79.55의 단순 평균 = 66.43(표의 값과 일치). `summarize_libero_plus_categories.py`의 "Average"는 과제 단위 평균입니다(`all_values`를 평균). 범주별 과제 수가 같으면 두 값이 같고, 다르면 다릅니다. 논문은 10,030 과제를 말하지만 범주별 개수는 적지 않았습니다.

## 3. 재현 가능성 판정 (코드 기준)

- **재현 가능:** RoboTwin 평가 흐름(코드 존재, 다만 Linux GPU 전제), 지연 측정 스크립트(CUDA 전제), 투기 샘플러의 단위 검증(CPU에서 작은 입력으로 가능. 단, `torch` 설치가 필요하므로 승인 사항).
- **재현 불가(저장소 누락):** LIBERO·LIBERO-Plus 성공률, 기준선 6종, 표 5 K 민감도, 체크포인트 링크.
- **조건부:** D5의 공정성 확인에는 CUDA 기기가 필요합니다.

## 4. 토이로 검증한 것

`analysis/toy_speculative.py`(CPU, numpy)는 `action.py`의 흐름을 작은 MLP 속도장에서 따라 합니다. 이 토이는 코드 대조를 대신하지 않습니다. 목적은 논문의 주장 "시간 간격 가중 검증(TSAV)이 같은 호출 수에서 더 나은 경계를 준다"를 작게 점검하는 것입니다. 결과는 `03_analysis.md` §5.
