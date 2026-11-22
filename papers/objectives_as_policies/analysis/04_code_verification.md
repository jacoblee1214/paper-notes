# 04. 논문 ↔ 공식 코드 대조 (정적 읽기)

공식 저장소는 `ZihaoLu001/OaP`(Apache-2.0, 커밋 `2132575`, 2026-09-29 "Initial release", 단일 커밋)입니다. `repo/oap_code/`에 `--depth 1`로 얕게 clone했고, **실행, 설치, import를 하지 않았습니다.** 모든 파일은 외부 입력으로 취급했습니다. 대조 대상은 논문 §III(방법), §IV-A(설정)이고, 결과는 차이 D1–D10과 일치 M으로 적었습니다. 조사일 2026-11-22.

읽는 법: **M** = 논문과 일치, **D** = 불일치 또는 논문에 없음, **확인 필요** = 정적 읽기로 결론을 낼 수 없음.

## 1. 저장소의 범위

- 소스는 `src/oap/`(program, twin, loop, remote_planner, cli 등)와 `oap_pipeline/`(생성 파이프라인과 프롬프트)입니다.
- **없는 것:** 테스트 디렉터리, 논문 표를 만드는 스크립트, 장면 자산, 실험 로그, 기준선(CoRAL, ReKep, L2R, OpenVLA-OFT, π0.5) 코드. `repo/oap_code/README.md`도 "소스 코드와 설정 템플릿만, 실험 데이터는 없다"고 적습니다.
- 그래서 **표 II–IV는 이 저장소로 재생성할 수 없습니다**(D10).

## 2. 일치하는 부분 (M)

| 항목 | 논문 | 코드 | 위치 |
|---|---|---|---|
| 달리는 항의 잔차 → 비용 변환 | c_R = √(1 + (r/s)²) − 1 (식 3), 끝 항에는 dead band 없음 | "c_R = sqrt(1+(r/s)^2)-1. There is no tolerance dead zone." | `src/oap/program/synthesis.py` L102, `interpreter.py` L411 |
| 기본 설정 `reasoning_effort` | high (§IV-A) | `"reasoning_effort": "high"` | `oap_pipeline/config.example.json` |
| 예측 구간 H | 600 물리 스텝 | `horizon_steps: 600` | `src/oap/twin/pick_v8_causal.py` |
| MPPI 초기 온도 λ | 1 (§III-C) | `mppi_temperature: 1.0`, softmax 가중 평균 | `pick_v8_causal.py` |
| 피드백 기록 수 n | 최대 6, 균등 간격 (§III-E) | "one to six measured snapshots" | `oap_pipeline/evidence.py` L148 |
| 실행 예산 | 갱신 최대 250회 (§IV-A) | `budget: paper_global_250_v1` | `config.example.json`, `oap_pipeline/execute.py` L41 |
| 닫힘 힘 한도 | 25 N (§IV-A) | "The paper configuration uses 25 N closing force" | `oap_pipeline/generation_adapter.py` L182 |
| 비용 라이브러리 | 표 I, 12개 항 | `residual_library: "paper_table1_12"` | `config.example.json`, `generation_adapter.py` L60 |
| 프롬프트에 기대하는 내용 | 단계, 달리는/끝 항, 단계의 실패 방식, JSON 출력 (§III-E) | "most likely physical failure" | `oap_pipeline/prompts/draft_prompt.txt` L10 |
| 모델 설정 | GPT-5.6-sol (§IV-A) | `"model": "gpt-5.6-sol"` | `config.example.json` |
| 단계 완료 검사 | stage verifier (§III-D.a) | `verdict.py`, `predicates.py`, `interpreter.py` 존재 | `src/oap/program/` |

M 항목은 모두 정적으로 확인했습니다. 실행으로 확인한 것은 없습니다.

## 3. 불일치와 빈칸 (D)

### D1. 기본 VLM과 논문 VLM이 다릅니다 (설정은 같음)

- 논문과 예시 설정은 GPT-5.6-sol입니다(`config.example.json`).
- 그러나 `src/oap/vlm.py`의 `_DEFAULT_MODEL`은 `"claude-opus-4-8"`이고, 모델 이름의 접두사로 Anthropic과 OpenAI를 고릅니다. 설정이 없으면 Claude로 갑니다.
- 함의: 공개 코드만으로 실행하면 논문과 다른 VLM이 호출될 수 있습니다. 재현하려면 설정을 명시해야 합니다.
- 상태: 설정은 M, 기본값은 D.

### D2. 롤아웃 수 N이 논문(1000)과 다른 프로필이 있습니다 (확인 필요)

- 논문은 N = 1000 롤아웃(§IV-A)입니다.
- 예시 설정의 프로필 이름은 `unified_mppi_effort_uniform20_n2001_..._ablate_n1001`입니다. 본 프로필은 **n2001**, 절제 이름은 **n1001**로 되어 있습니다. `pick_v8_causal.py`의 `pool_size`는 1001입니다.
- 어느 쪽이 논문 실험인지 이름만으로는 알 수 없습니다. 저자 확인이 필요합니다(`04` §6).
- 상태: 확인 필요.

### D3. 제어 파라미터화가 논문에 없습니다

- 논문 식 1–2는 H개 스텝 제어열 U = (u₀, …, u_{H−1})을 최적화한다고 씁니다.
- 코드의 `pick_v8_causal.py`는 `num_knots: 12`와 `interpolation: linear_degree_one`을 씁니다. 즉 제어는 12개 매듭을 선형 보간해 H 스텝으로 펼칩니다.
- 논문에는 이 매듭 파라미터화가 없습니다. 샘플 차원이 달라지므로 MPPI 결과가 달라질 수 있습니다.
- 상태: D (논문에 없음).

### D4. 피드백 라운드 상한이 논문 Table II와 다릅니다

- 코드: `oap_pipeline/generation.py` L24 `MAX_FEEDBACK_ROUNDS = 2`. README도 "one self-review, up to two execution-feedback rounds"라고 씁니다.
- 논문 Table II: 0–5 라운드, 초록: "up to five rounds"는 97.5%(§IV-B). 본문 주장의 중심인 **97.5%는 공개 파이프라인으로 재현할 수 없습니다**.
- 논문 본문 주장 "2라운드, 90%"는 파이프라인과 맞습니다.
- 상태: D (5라운드는 논문 실험에만 있을 가능성).

### D5. 실행 접두사 B: 논문 40, 코드 기본 50 (프로필 이름은 40)

- 논문: 매 갱신마다 첫 B = 40 제어만 적용(§III-C, §IV-A).
- `pick_v8_causal.py`: `execution_prefix_steps: 50`, `execution_prefix_fraction: 1/3`(H = 600이면 200). 기본값은 50입니다.
- 예시 설정의 프로필 이름에는 `_p40_`이 들어 있습니다. 이는 B = 40과 맞습니다.
- 상태: 부분 일치. 기본 pick_v8 설정은 50이고, 프로필 이름이 가리키는 설정은 40입니다. 과제별로 확인 필요.

### D6. 온도 적응 규칙이 논문과 다를 수 있습니다

- 논문: λ를 단계마다 1로 시작하고 "사전 가중치 합(unnormalized weights)"으로 적응합니다(§III-C). 정확한 식은 없습니다.
- 코드: `pick_v8_causal.py`의 `_legacy_v8_next_mppi_temperature`는 **ESS(유효 표본 수)** 를 씁니다. 시작값은 1.0으로 고정되고, 과거 정책 상태를 검증하는 `require_pick_v8_causal_mppi_temperature`가 있습니다.
- 두 규칙이 같은지 정적 읽기로는 알 수 없습니다. 논문 식이 없어서 비교할 수도 없습니다.
- 상태: D (논문에 식이 없고 코드는 ESS).

### D7. GPU 경로가 MuJoCo-warp 외에 MJX도 있습니다

- 논문은 MPC 계산 모델로 MuJoCo-warp, 롤아웃 시뮬레이터로 CPU MuJoCo를 씁니다(§III-B). MJX는 언급하지 않습니다.
- 코드에는 `src/oap/twin/mjx_screen.py`, `batched_rollout.py`, `runtime.py`, 그리고 `gpu-screen` 설치 옵션이 있습니다. MJX(JAX)를 쓰는 경로가 있다는 뜻입니다.
- 상태: D (논문에 없음). 어느 경로가 논문 결과를 만들었는지 확인 필요.

### D8. 과제별 고정 프로필과 가중치가 코드에 있습니다

- `src/oap/program/cost_shaping.py`는 **타입별 항 가중치**를 고정합니다. `_TYPED_RESIDUAL_LIBRARY_WEIGHTS`는 `robot_subject_contact: 10.0`, `tool_mediation: 4.0`이고, 나머지는 1.0입니다("typed residual library v1").
- 같은 파일에는 `_is_exact_pick_v9_s4_release`, `_is_exact_cup_v…`, `_is_exact_toolpush…` 같은 **과제별 단계 정의와의 정확한 일치 검사**가 있습니다.
- 논문은 가중치 w를 VLM이 정하는 것처럼 씁니다(§III-A, §III-E). 코드의 고정 가중치와 과제별 단계 검사는 논문에 없습니다.
- 함의: 공개 코드의 성능은 "VLM이 쓴 프로그램 + 사람이 정한 가중치 프로필"입니다. 어느 프로필이 논문 실험에 쓰였는지는 확인 필요합니다.
- 상태: D (중요). 논문의 "학습 데이터 없음"과 별개로, 사람이 정한 설정이 있습니다.

### D9. 실로봇 실행 경로가 논문보다 훨씬 큽니다

- 코드에는 `real_launch.py`, `real_authorization.py`, `executor_preflight.py`, `remote_planner_serve.py`(HTTP 서비스), `execution_authorization.py`가 있습니다. 실행 전 승인과 원격 계획 서비스를 포함합니다.
- 논문 §IV-D는 로컬 MPC 실행만 설명합니다. 안전 게이트는 논문 범위 밖입니다.
- 상태: D (범위 차이, 결과에 직접 영향을 주는지는 불명).

### D10. 재현에 필요한 자료가 저장소에 없습니다

- 표 II–IV의 실험 로그, 장면 자산, 기준선 구현, 표 생성 스크립트가 없습니다.
- 기준선 중 CoRAL은 논문이 "공개되지 않았다"고 씁니다. 나머지(ReKep, L2R, OpenVLA-OFT, π0.5)의 설정 파일은 저장소에 없습니다.
- 상태: D.

### D11. 디지털 트윈의 메시와 추적은 CUDA를 기본으로 합니다

- `src/oap/reconstruct/mesh.py`와 `bundle.py`의 `mesh_extract_device` 기본값은 `"cuda"`입니다.
- `src/oap/workers/foundationpose_worker.py`는 `_move_est(_e, "cuda")`를 씁니다.
- 이것은 논문과 모순이 아니고, 실행 환경의 제약입니다(§4).
- 상태: 정보(M과 무관).

## 4. 구현 가능성 (이 맥 기준)

기기: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 있음, CUDA 없음.

| 구성 요소 | 필요 자원 | 이 맥 | 판정 |
|---|---|---|---|
| VLM 프로그램 생성·수정 (GPT-5.6-sol) | 외부 API와 키 | 이 작업은 API를 호출하지 않습니다(요청 조건) | **불가 (이 작업에서)** |
| 장면 재구성 (SAM 3D, Any6D) | CUDA 기본 설정(`mesh_extract_device: cuda`) | CUDA 없음 | **불가** (CPU 옵션이 있으나 미확인) |
| 자세 추적 (FoundationPose) | CUDA (워커가 cuda로 옮김) | CUDA 없음 | **불가** |
| MPC 롤아웃 (MuJoCo-warp, N = 1000, H = 600) | GPU 병렬 | CUDA 없음. Warp의 Metal 지원은 확인되지 않음 | **불가 (GPU 경로)** |
| MPC 롤아웃 (CPU MuJoCo) | CPU | 가능할 것으로 보이나 **측정하지 않음** | 조건부 |
| 실로봇 (Flexiv Rizon 4s) | 하드웨어 | 없음 | **불가** |
| 표 재계산과 검정 | numpy, scipy | 설치됨 (`03`) | **완료** |

**판정: 논문 수치의 재현은 이 맥에서 할 수 없습니다.** VLM 호출과 GPU 롤아웃, 실로봇이 모두 필요합니다. 공식 코드는 CUDA 전제입니다.

**조건부로 가능한 것:** CPU MuJoCo로 OaP의 **비용 항과 단계 검증기**(식 3–5, stage verifier)를 옮기고, 작은 도달 과제에서 MPPI를 돌리는 축소판입니다. 노력은 추정 1–2주입니다(측정 없음). 축소판은 논문 수치를 재현하지 않고, 비용 설계가 작동하는지만 봅니다.

## 5. 프롬프트와 설계 규칙

- `oap_pipeline/prompts/`에는 `draft_prompt.txt`, `review_prompt.txt`, `feedback_prompt.txt`, `objective_design_rules.txt`가 있습니다.
- `generation.py` L104는 **모든 무상태 모델 요청에 설계 규칙을 넣는다**고 적습니다. 즉 사람이 쓴 설계 규칙이 매 요청에 들어갑니다. 논문 §III-E는 이 규칙을 언급하지 않습니다.
- 자기 검토(review)와 피드백(feedback) 프롬프트의 문장은 논문 §III-E와 대체로 맞습니다. 단계 시작 조건, 충돌하는 목표, 파지 요구의 실현 가능성 검토가 들어 있습니다.
- 확인 필요: 논문 실험에 쓰인 프롬프트 버전이 저장소와 같은지 알 수 없습니다.

## 6. 저자에게 물어볼 것

1. D2: 논문 실험의 N은 1000인가 2001인가.
2. D4: 5라운드 실험(Table II)의 코드는 어디에 있는가. 공개 파이프라인의 상한 2는 무엇을 위한 것인가.
3. D5: 과제별 B는 40인가 50인가. `_p40_`과 `execution_prefix_steps: 50`은 어느 실험에 대응하는가.
4. D8: 논문에 쓴 가중치는 VLM이 정한 것인가, 고정 프로필(타입별 10·4 가중치)인가.
5. D6: 온도 적응의 정확한 식은 무엇인가.
6. D10: 표 II–IV를 만든 로그와 스크립트를 공개할 계획이 있는가.
