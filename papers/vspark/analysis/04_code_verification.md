# 04. 논문 ↔ 공식 코드 대조 (정적 읽기)

- 대상: `suho-yoo/VSpark`, 커밋 `f151152` (2026-11-21, "initial commit"), Apache-2.0. `repo/`에 얕은 clone.
- 방법: 파일을 **읽기만** 했습니다. 실행, import, 설치, 가중치·데이터 다운로드는 하지 않았습니다. 클론 안의 내용은 신뢰하지 않는 자료로 취급했습니다.
- 읽은 파일: `README.md`, `vspark/tree.py`(전체), `vspark/decode.py`(전체), `vspark/loss.py`(전체), `vspark/rollouts.py`(전체), `vspark/targets.py`(클래스 구조와 `decode`, `generate`, `scores`, StarVLA 설명), `vspark/draft.py`(설정 클래스와 층 수), `training/train.py`(전체), `training/calibrate_noise.py`(전체), `training/collect_rollouts.py`(머리말), `configs/*.json`(키와 값), `inference/eval_libero.py`(AR 경로와 시드).
- 읽지 않은 파일: `training/prepare_demo_states.py`, `inference/speedup.py`, `vspark/libero.py`(일부만), `docs/index.html`(프로젝트 페이지는 `01`과 `02`에서 따로 읽음), `requirements.txt`(내용만 확인). 이 파일들이 논문과 어긋날 가능성은 남아 있습니다.

## 1. 대조표

| 논문 항목 | 코드 위치 | 판정 |
|---|---|---|
| 식 (2) Markov 헤드, softmax(U_i + W₁[x_{i−1}]W₂) | `draft.py`의 `markov`, `tree.py`의 `build` | 일치 |
| 식 (3)·(5) 손실 CE·특징 회귀 가중합, λ = 0.1 | `loss.py` `loss_batch`, `draft.py` `ce_weight = 0.1`, `feature_weight = 0.9` | 일치 (TV 항은 제거됨) |
| 특징 회귀 Smooth L1, 타깃의 pre-head 은닉 상태 | `loss.py`: `smooth_l1_loss(out, h[...])`, `h`는 최종 은닉 상태 | 일치 |
| 식 (4) 잡음 공분산 = 차이의 2차 모멘트 (비중심) | `calibrate_noise.py`: `errors.T @ errors / len(errors)` | 일치 (고유값 하한 1e-10 추가, D6) |
| 식 (4) 잡음 적용: 병진·회전 6차원, 그리퍼 제외, [−1, 1] 클리핑 | `rollouts.py` `perturb`: `out[:6] += …`, `np.clip(out, −1, 1)` | 일치 |
| 잡음은 실행 행동에만, 라벨은 잡음 전 타깃 출력 | `rollouts.py`: 타깃 토큰은 잡음 전에 기록 | 일치 |
| 식 (6) 점수 올림: 같은 부모의 최고 드래프트 점수 | `tree.py` `promote`: `maximum(logprob, max_row.clamp_max(0))` | 일치 |
| 연속 추출: 거부된 형제마다, 자식 일치 따라가기, 없으면 마지막 예측, EOS·길이 7에서 멈춤 | `tree.py` `continuations(max_length=7)` | 일치 |
| 트리: 부모당 후보 8, 노드 60, 깊이 7 | `decode.py` 기본값 `depth=7, nodes=60, top_k=8` | 일치 |
| 조기 종료: 남은 최고 프런티어가 선택된 최하위보다 낮으면 멈춤 | `tree.py` `stop_extending` | 일치 |
| 검증은 탐욕적, 트리 주의 마스크, 정확 검증 | `decode.py` `generate`, `tree.py` `tensors()` | 일치 (실행하지 않아 수치 동일성은 미확인) |
| 드래프트 1층, 10 epoch, AdamW (0.9, 0.95), lr 1e-4, 배치 16, 클리핑 0.5 | `train.py` 기본값, `draft.py` `layers=1` | 일치 |
| 워밍업 5% 후 선형 감쇠 | `train.py` `WarmupDecay`: **로그 워밍업** 후 선형 감쇠 | 불일치 (D3) |
| 시드 7 (평가) | `libero.py` `SEED = 7` | 일치 |
| 탭 층 (표 A.5): π0 2,5,9,13,16 / StarVLA 2,10,18,26,34 / AC 2,7,12,17,22 | `configs/*.json` `tap_indices` | 일치 |
| 드래프트 규격 (표 A.4): 폭 2048, 층 1, 헤드·KV·FFN, RoPE θ, 출력 어휘 1030/2050/2056 | `configs/*.json`, `support_token_ids` 길이 | 일치 (G0.5 열은 없음, D1) |
| 학습 샘플 27,548 (표 A.2) | `configs/*.json` `target_chunks` 합 (세 타깃 모두) | 일치 (`03` V12) |
| 잡음 σ (표 A.6) | `configs/*.json` `noise_covariance` 대각 제곱근 × (50 mm, 28.65°) | 18/18 일치 (`03` V13) |
| 보정용 롤아웃 16개 × suite = 64 (표 A.6) | `calibrate_noise.py` 기본 `--chunks-per-suite 16` | 일치 |
| 앵커 샘플링 (DeepSpec 출처) | `loss.py` `sample_anchor_positions`, `num_anchors = 512` | 논문 본문에 없음 (D5) |
| 손실 가중치 w_i (앞 위치에 더 큰 가중) | `loss.py`: `decay = exp(−offset / 4.0)` | 논문에 형태 없음 (D5) |
| StarVLA 생성의 반복 벌점 | `targets.py` `StarVLA.scores`: 원본 `repetition_penalty`를 토큰 이력에 적용 | 논문에 없음 (D2) |
| 생성 오류 에피소드의 처리 | `eval_libero.py`와 `01` §5.6 | 미확인 (D13) |
| G0.5 타깃 (표 1, 3, A.3–A.6의 G0.5 항목) | `targets.py`에 클래스 없음, `configs/`에 파일 없음, README 모델 옵션에 없음 | **불일치, 재현 불가** (D1) |

## 2. 차이 목록 (D1–D13)

### D1. G0.5 구현과 설정이 공개되지 않았습니다 (높음)
- 논문: G0.5를 네 번째 타깃으로 쓰고, 표 1과 3에 G0.5 행을, 표 A.3·A.4·A.6에 G0.5 항목을 둡니다.
- 코드: `targets.py`에 G0.5 클래스가 없습니다. `configs/`에 `g05.json`이 없습니다. README의 `--model` 선택지는 `pi0_fast`, `actioncodec`, `starvla`뿐입니다. `docs/`를 제외한 코드·설정·README에서 "G0.5"나 "g05"를 찾지 못했습니다. README의 체크포인트 표에도 G0.5 드래프트가 없습니다.
- 결과: 표 1, 3의 G0.5 행(속도 1.28×, 1.29×)과 본문의 "기준선 세 개가 1× 미만" 주장은 공개 자료로 확인할 수 없습니다. 논문 §5의 "선형 어텐션이 트리 검증을 제한한다"는 설명도 확인할 수 없습니다.

### D2. StarVLA 반복 벌점 (중간, 논문 누락)
- 코드: `StarVLA`의 독스트링은 "Released generation applies a repetition penalty over the prompt and generated tokens"라고 적고, `scores()`에서 토큰 이력마다 벌점을 적용합니다. 이는 자기회귀 기준(`generate`)과 트리 검증이 같은 벌점을 쓴다는 뜻이라 무손실 성질에는 맞습니다.
- 논문: "greedy decoding"만 씁니다. 벌점에 대한 언급이 없습니다. 독자가 `argmax(logits)`로 재구현하면 결과가 달라질 수 있습니다.

### D3. 학습률 워밍업 형태 (낮음)
- 논문: "warmup over the first 5% of training followed by linear decay".
- 코드: `WarmupDecay`는 `log(s+1)/log(warmup)` 형태의 로그 워밍업을 씁니다. 독스트링도 "Logarithmic warmup"이라고 적습니다.

### D4. 연속 노드 삽입 점수 (중간, 논문 누락)
- 논문: 연속을 트리에 "병합하고, 없는 노드는 삽입하고, 예산까지 가지치기"라고만 씁니다(§3.4).
- 코드: `tree.py` `insert()`는 새 노드의 점수를 `parent_score + log(0.5)`로 둡니다. 이 값이 프루닝 순서를 결정합니다. 논문에 없는 상수입니다.
- 영향: 검증 정확성에는 영향이 없습니다(검증은 트리 구조와 타깃 예측으로만 정함). 어떤 후보가 노드 예산 안에 남는지에 영향이 있어 τ에 영향이 있을 수 있습니다. 측정하지 않았습니다.

### D5. 손실 가중치와 앵커 수 (낮음~중간, 논문 누락)
- 논문 식 (5): 가중치 w_i는 "앞 위치를 더 강하게"라고만 씁니다.
- 코드: 블록 손실은 512개 앵커를 균등 샘플링하고, 블록 안 위치 k에 `exp(−k/4)`를 곱합니다(γ = 4, `loss_decay_gamma`). 이 형태와 앵커 수는 논문 본문과 부록에 없습니다.
- 영향: 학습 신호의 세기와 위치별 비중이 달라집니다. 재현하려면 이 값을 써야 합니다.

### D6. 잡음 공분산 보정의 추가 필터 (낮음)
- 코드: `calibrate_noise.py`는 |오차| > 10인 블록을 버리고, 고유값을 1e-10 아래로 올립니다. 논문 §3.2와 부록 A.6에는 없습니다. 기본 설정의 σ는 이 필터를 거친 값입니다(추정).

### D7. 학습 데이터의 저장 용량 (논문에 없음, 정보)
- 코드 README: 청크마다 타깃 은닉 상태와 탭 특징을 저장하며 pi0-FAST는 청크당 약 20 MB이므로 "수백 GB의 저장 공간"이 필요하다고 씁니다.
- 논문: 학습 샘플 수(27,548)만 있고 저장 용량과 디스크 요구는 없습니다. 이 값이 맥 재현의 장벽입니다(`03` §7).

### D8. 롤아웃의 적격성 필터 (낮음, 논문 누락)
- 코드: `rollouts.py` `eligible()`는 종료된 청크이고, 모든 토큰이 드래프트 어휘 안에 있고, (pi0-FAST에서) "Action:"으로 시작하는 청크만 학습에 씁니다.
- 논문: "Generation-error episodes and token-limit chunks are excluded from all results"(§4.1)만 씁니다. 어휘 필터와 "Action:" 접두사 조건은 적혀 있지 않습니다.

### D9. 오류 처리의 불일치 (논문 내부, 중간)
- 논문 §4.1: "Generation-error episodes and token-limit chunks are excluded from all results."
- 논문 부록 A.1, B.8: "Generation errors are counted as failures in success rates but excluded from efficiency metrics."
- 두 문장은 함께 성립하지 않습니다. 코드의 `rollouts`는 디코딩 실패가 나면 그 에피소드의 남은 스텝을 버립니다(`break`). 평가 쪽의 처리는 `eval_libero.py`에서 확인하지 않았습니다(D13).

### D10. CUDA 전제 (정보, 재현성)
- `targets.py`의 모든 타깃 클래스 기본 `device='cuda'`, `draft.py`의 `load(path, device='cuda')`, `train.py`의 `.cuda()`와 `torch.autocast('cuda', …)`.
- README: `torch==2.7.1+cu128`로 설치. 논문의 RTX 4090과 일치합니다. 맥(MPS)에서는 장치 인자와 `autocast`를 고쳐야 하고, 고친 뒤에도 수치 동일성은 보장되지 않습니다.

### D11. LIBERO-Plus와 LIBERO-Pro 설정 (정보)
- 논문 §4.1: "We follow the official protocol of each benchmark." 코드는 `--benchmark plus|pro`를 지원합니다.
- README: LIBERO-Pro의 `env` 변형 초기 상태는 `official_generate_init_states.py`(시드 28, 50 상태)로 만들어야 합니다. 이 절차는 논문에 없습니다.

### D12. ActionCodec 환경 (정보)
- README: ActionCodec은 transformers 4.57.1이 필요하므로 별도 환경을 만들라고 합니다. 같은 저장소의 `requirements.txt`는 transformers 5.5.4를 고정합니다. 논문은 환경에 대해 아무 말도 하지 않습니다.

### D13. 평가 쪽 오류 처리 (미확인)
- `eval_libero.py`와 `libero.py`의 생성 오류 처리는 읽었지만, 성공률 분모에서 어떻게 빼는지는 확인하지 못했습니다. 논문 A.1의 "실패로 센다"와 맞는지 판단할 수 없습니다.

## 3. 맞는 것 (요약)

- 구조 수준의 수식은 모두 맞습니다: 식 (2), (3)에서 TV 항 제거(식 (5)), 식 (4)의 2차 모멘트, 식 (6)의 점수 올림.
- 트리와 검증: 노드 60, 깊이 7, 부모당 8, 조기 종료, 연속 추출의 길이 7 제한이 논문 설명과 일치합니다.
- 학습 설정(업데이트 수 17,210, 샘플 27,548, 잡음 σ 18/18)의 수치가 서로 맞습니다.
- 정확 검증은 정확히 구현된 것으로 보입니다. 다만 수치 동일성은 실행하지 않아서 확인하지 못했습니다.

## 4. 구현 가능성 (맥과 공식 재현)

**판정:** 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 **논문 표의 재현은 불가능합니다.** 아이디어 수준의 구현(트리, 연속 재사용, 정확 검증)은 CPU에서 새로 짜서 검증할 수 있습니다. 공식 코드를 쓰는 재현은 CUDA 기기에서만 현실적입니다.

| 요구 | 공식 요구 | 이 맥 | 판정 |
|---|---|---|---|
| 장치 | CUDA (D10) | MPS만 있음. 코드 수정 필요 | 불가 (수정 후 수치 동일성 미확인) |
| 타깃 모델 | π0-FAST 3B, StarVLA 3B, ActionCodec 2.2B | 메모리는 가능. 속도 미측정 | 조건부 |
| 학습 데이터 | 청크당 약 20 MB, 27,548 청크, π0 약 551 GB | 여유 디스크 약 398 GiB | **불가** (`03` §7) |
| 학습 | 드래프트 10 epoch, 17,210 업데이트 | CUDA 시간이 필요 | 불가 (맥) |
| 시뮬레이터 | LIBERO, `MUJOCO_GL=egl` (README) | EGL은 Linux용. macOS 렌더링 경로 미확인 | 조건부 |
| 환경 | torch 2.7.1+cu128, transformers 5.5.4, ActionCodec은 별도 환경 (D12) | 설치 가능 여부는 승인 뒤 | 조건부 |
| 체크포인트 | HF `suhoyoo/VSpark` (세 개), 타깃은 HF에서 자동 다운로드 | 수 GB, 승인 필요. G0.5 없음 (D1) | 조건부 |

- **노력 추정 (미측정):** CUDA 기기에서 pi0-FAST 한 타깃의 롤아웃 수집, 학습, 평가까지 수 주. 이 추정은 README의 저장 용량과 학습 업데이트 수에서 나온 것이며 실측이 아닙니다.
- **맥에서 가능한 부분 (조건부):** 드래프트 체크포인트 하나를 받아 LIBERO 몇 에피소드에서 추론만 돌려 보기. 이는 MPS, EGL 대체, bf16 지원 여부에 따라 달라집니다. 속도 비교는 의미가 없을 가능성이 큽니다(MPS와 RTX 4090의 차이).
- **공식 코드를 실행하지 않은 이유:** 이 분석의 규칙입니다. 실행 승인을 받은 뒤에만 고려합니다.

## 5. 저자에게 물을 것

1. G0.5 구현과 드래프트 체크포인트를 공개할 계획이 있는지(D1).
2. 연속 노드의 점수 `log(0.5)`(D4)과 손실 가중치 γ = 4, 앵커 512(D5)가 논문 설정과 같은지.
3. 표 4의 SR이 어느 실행인지(시드, 정밀도), 그리고 BF16에서 AR과 VSpark의 짝지은 결과 일치율 분모(`01` §5.1).
4. 기준선(EAGLE-3, P-EAGLE, DFlash, DFlash2, DSpark)의 트리 확장 코드를 공개할 수 있는지.
