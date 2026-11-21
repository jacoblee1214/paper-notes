# 04 논문 ↔ 공식 코드 대조 (π0.5, openpi)

- 대상: `github.com/Physical-Intelligence/openpi`, commit `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-25), Apache-2.0. 얕은 clone을 `repo/`에 두고 **정적으로만** 읽었습니다. 실행, import, 설치, 가중치 다운로드는 하지 않았습니다.
- 읽은 파일: `README.md`, `src/openpi/models/pi0.py`, `pi0_config.py`, `gemma.py`, `model.py`(전처리·증강), `tokenizer.py`, `transforms.py`, `shared/normalize.py`, `training/config.py`(pi05 항목).
- 대조 기준: `01_paper_analysis.md`의 §2 (Eq. 1, Sec. IV, Appendix E).

## 1. 결과 요약

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 학습 손실 | Eq. (1): 텍스트·FAST CE + α·flow MSE (post-training α = 10) | `pi0.py`에 flow MSE만. FAST 토큰 경로 없음 | **불일치 (D1)** |
| 고수준 subtask 텍스트 생성 | 있음 (Sec. IV-A) | 추론 쪽 자기회귀 디코딩 코드가 pi0.py에 없음 (`sample_actions`는 flow만) | **불일치 또는 미공개 (D2)** |
| 상태 입력 | 이산 텍스트 토큰 | `discrete_state_input=True` → 256 bin 텍스트 | 일치 (D3) |
| 타임스텝 주입 | adaRMSNorm, MLP | `adarms_cond`, `time_mlp_in/out` | 일치 (D4) |
| 타임스텝 분포 | Beta((s−τ)/s; 1.5, 1), s = 0.999 | `Beta(1.5,1)·0.999 + 0.001`, 시간축은 1−τ | 일치 (D5) |
| 어텐션 마스크 | VLM 임베딩은 액션 전문가를 보지 않음 | suffix 시작에서 `ar_mask=True`로 prefix 비주의 | 일치 (D6) |
| Knowledge insulation (stop-gradient) | 논문 본문에 없음 (어텐션 마스크만) | `stop_gradient` 없음 (모델 경로) | **공개 코드에 미구현 (D7)** |
| 정규화 | 1%/99% 분위수 → [−1, 1] | `q01, q99`, `(x−q01)/(q99−q01)·2−1` | 일치 (D8) |
| 이미지 증강 | crop 95%, ±5° 회전, color jitter 0.3/0.4/0.5 | 같은 값 (`model.py`) | 일치 (D9) |
| 액션 차원 | 모든 로봇에 고정 차원 | `action_dim = 32` | 일치 (D10) |
| 액션 청크 | H = 50 (H = 49 인덱스) | `action_horizon = 50` (기본) | 일치 (D11) |
| VLM 크기 | 2B, width 2048, depth 18, MLP 16384 | `gemma_2b`: 같음 | 일치, 단 **헤드 수 문서 오류 (D12)** |
| 액션 전문가 | 300M, width 1024, MLP 4096 | `gemma_300m`: 같음 | 일치 (D13) |
| 사전학습 데이터·FAST 사전학습 | 280k 스텝, 혼합 데이터 | 코드·데이터 미공개, 체크포인트 `pi05_base`만 | 재현 불가 (D14) |

## 2. 주요 차이 (D1–D2, D7)

### D1. Eq. (1)의 혼합 손실이 이 저장소의 π0.5 학습 경로에 없음

- 논문 Sec. IV-B: “텍스트 토큰과 FAST 토큰의 교차엔트로피 + α × flow matching MSE”. post-training에서 α = 10 (Sec. IV-D).
- README 1행 근처: “we currently only support the flow matching head for both π0.5 training and inference”.
- `pi0.py`의 손실 계산(`compute_loss`)은 flow 예측 `v_t`와 목표의 MSE 하나만 씁니다. FAST 토큰을 만드는 `pi0_fast.py`는 별도 모델 클래스이며, π0.5 설정(`Pi0Config(pi05=True)`)에서는 쓰지 않습니다.
- 해석: 이 저장소의 π0.5는 논문의 **post-training 단계에 해당하는 flow-only 모델**입니다. 사전학습 단계(FAST 토큰 CE)를 포함한 전체 레시피는 공개되지 않았습니다.
- 확인 방법: `grep -n "fast\|FAST" repo/src/openpi/models/pi0.py` → 결과 없음 (2026-11-21 확인).

### D2. 고수준 텍스트 추론 경로

- 논문: 추론 시 “텍스트를 자기회귀로 생성 → 그 subtask 조건으로 10 denoising 스텝” (Appendix E).
- `pi0.py`의 `sample_actions`는 노이즈에서 시작해 flow 스텝만 돕니다. 텍스트 생성 루프는 pi0.py에 없습니다.
- openpi의 예제와 정책 래퍼는 tokenized prompt를 입력받을 뿐, π0.5 고수준 subtask 생성을 제공하는 것으로 보이지 않습니다. 이 부분은 “없다”고 단정하지 못하고 **미확인**으로 둡니다 (`repo/src/openpi/policies/` 전체를 훑지 않음).
- 논문의 “한 모델이 두 수준을 모두 한다”는 핵심 주장을 이 코드로 확인할 수 없습니다.

## 3. 일치하는 구현 세부 (D3–D6, D8–D11, D13)

### D3. 상태를 이산 텍스트 토큰으로 입력

- `pi0_config.py`: “the state input is part of the discrete language tokens rather than a continuous input”. `discrete_state_input = pi05`.
- `tokenizer.py`: 상태를 `np.linspace(-1, 1, 257)`로 256 bin에 나눠 공백으로 이은 문자열을 프롬프트에 붙임. 논문 Sec. IV-A와 같은 설계입니다. bin 수(256)는 논문에 없고, 코드의 선택입니다.

### D4. 타임스텝은 adaRMSNorm으로 주입

- `pi0.py` `embed_suffix`: `time_mlp_in → swish → time_mlp_out → swish`로 `adarms_cond`를 만들어 액션 전문가의 RMSNorm에 넣음 (`gemma.py`의 `RMSNorm(..., adarms_cond)`). Appendix E의 “separate MLP for projecting τ … adaptive RMSNorm”과 같습니다.
- π0 경로(`else` 분기)는 액션과 타임스텝을 concat하는 방식으로, 논문이 π0와 비교해 바꾼 점이 코드에 그대로 있습니다.

### D5. 타임스텝 분포와 시간축 규약

- 논문 Appendix E: τ의 샘플은 p(τ) = Beta((s−τ)/s; 1.5, 1), s = 0.999, 액션 조합은 a^{τ,ω} = τ·a + (1−τ)·ω.
- 코드 `pi0.py` `compute_loss`: `time = Beta(1.5, 1)·0.999 + 0.001`, `x_t = time·noise + (1−time)·actions`.
- 대수 확인: u ∼ Beta(1.5, 1)이면 논문 τ = 0.999(1−u)이고 1−τ = 0.001 + 0.999u. 코드의 `time`은 **1−τ**와 같습니다. 즉 코드는 논문과 같은 분포를 쓰되 시간축을 뒤집었습니다. 결과적으로 같습니다.
- 주의: 추론 루프는 `time`이 1에서 0으로 줄어듭니다 (`dt < 0`). 이것도 같은 축과 일관됩니다.

### D6. 어텐션 마스크

- `pi0.py` `embed_suffix` (pi05): 액션 토큰의 `ar_mask`는 첫 토큰 `True`, 나머지 `False` (`[True] + [False]*(H−1)`). prefix는 `False`(양방향)이며, `make_attn_mask`에서 prefix는 suffix를 보지 못합니다.
- 논문 Appendix E: “VLM 임베딩은 액션 전문가를 보지 않음, 정보는 VLM → 액션 전문가로 단방향”. 일치합니다.
- 단, 논문은 FAST 토큰과 연속 액션 사이 마스크도 말하는데, 이 코드의 π0.5 경로에는 FAST 토큰이 없으므로 그 부분은 해당하지 않습니다 (D1).

### D8. 분위수 정규화

- `shared/normalize.py`: `_compute_quantiles([0.01, 0.99])` → q01, q99.
- `transforms.py` `_normalize_quantile`: `(x − q01)/(q99 − q01 + 1e-6)·2 − 1`.
- `training/config.py`: `use_quantile_norm = model_type != PI0`이므로 π0.5는 분위수 정규화를 씀. 논문 Sec. IV-C와 맞습니다.

### D9. 이미지 증강

- `model.py` (train 모드): `RandomCrop(0.95·w, 0.95·h)`, `Resize`, `Rotate((−5, 5))`, `ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)`. 논문 Appendix E와 같은 값과 순서입니다.
- 해상도 224×224, 긴 쪽 기준 `resize_with_pad` (논문에 해상도 기재 없음).

### D10–D11. 차원과 청크

- `action_dim = 32` (`pi0_config.py`). `pi05_droid_finetune`에 “pi05 is trained with 32-dim actions”라는 주석이 있습니다. 논문의 “고정 차원, 0 패딩” (Sec. IV-C)과 맞습니다.
- `action_horizon = 50` 기본. 단, 설정별로 `pi05_droid`는 15, `pi05_libero`는 10으로 다릅니다. 논문 설정이 아니라 미세조정 설정이므로 논문과 비교 대상이 아닙니다.

### D13. 액션 전문가 크기

- `gemma.py` `gemma_300m`: width 1024, depth 18, MLP 4096, heads 8, kv 1, head_dim 256. 논문 Appendix E와 같습니다.

## 4. 문서 불일치와 미공개 (D7, D12, D14)

### D12. 논문 Appendix E의 헤드 수

- 논문 원문 텍스트: “{width=2048, depth=18, mlp dim=16,384, num heads=18, num kv heads=1, head dim=256}”.
- 코드 `gemma_2b`: `num_heads = 8`. Gemma 2B의 공개 설정도 8입니다.
- 판정: 논문의 “18”은 오타로 보입니다. 다른 값은 모두 같습니다. 이 항목은 논문 문서의 오류로 기록합니다.

### D7. Knowledge insulation (stop-gradient)은 이 모델 경로에 없음

- openpi README는 π0.5가 “knowledge insulation으로 학습됐다”고 말합니다.
- 논문 본문에는 “knowledge insulation”이라는 말이 없고, 액션 전문가 쪽 어텐션 마스크(D6)만 있습니다.
- 코드: `src/openpi`의 `stop_gradient`는 `models/utils/fsq_tokenizer.py`(3곳, 별도 토크나이저)에만 있고, `pi0.py`와 `gemma.py`에는 없습니다.
- 해석: 어텐션 마스크는 **순방향 정보 흐름**만 막습니다. 액션 손실의 기울기는 prefix의 키·값을 거쳐 VLM 가중치까지 들어갑니다. 따라서 이 코드에서는 “VLM을 액션 손실로부터 격리”하는 효과가 없습니다. README 문장은 공개 체크포인트를 학습한 방식(별도 논문)을 설명하는 것으로 읽히므로, 모순이라기보다 학습 방식이 공개되지 않은 것입니다. 이 저장소의 코드로 미세조정하면 knowledge insulation은 적용되지 않습니다.
- 판정: README의 설명과 논문 Appendix E는 이 코드와 맞지 않는 부분이 있습니다. 확인하려면 학습 코드(공개되지 않음)가 필요합니다.

### D14. 재현 불가 항목

- 사전학습 혼합 데이터(MM 400시간, ME, CE, OXE, HL 라벨, WD)와 그 주석은 공개되지 않았습니다.
- FAST 토크나이저를 사전학습에 쓴 학습 스크립트, 280k·80k 스텝 학습 설정도 이 저장소에 없습니다. `training/config.py`의 `pi05_*` 설정은 모두 미세조정 설정입니다 (`pi05_aloha`, `pi05_droid`, `pi05_libero`, `pi05_full_droid_finetune`, `pi05_droid_finetune`).
- 학습률 등: 논문에는 값이 없습니다. openpi의 `pi05_libero`는 배치 256, peak LR 5e-5, warmup 10k, EMA 0.999, 30k 스텝입니다. 논문 설정과 직접 비교할 수 없습니다.

## 5. 체크포인트 관련

- 공개 base 체크포인트 `gs://openpi-assets/checkpoints/pi05_base` (README). 이는 논문에서 말한 “사전학습 + post-training”의 결과물인지, 아니면 flow-only 재학습인지 **이 저장소에서는 확인할 수 없습니다.** 가중치 자체는 다운로드하지 않았습니다.
- 따라서 “논문 성능이 이 체크포인트에서 재현된다”는 주장은 이 분석으로 검증되지 않았습니다.

## 6. 정리

- π0.5 구조(상태 이산화, adaRMS, 타임스텝 분포, 어텐션 마스크, 분위수 정규화, 증강, 차원)는 논문과 코드가 일치합니다 (D3–D11, D13).
- 손실(D1)과 고수준 텍스트 추론(D2)은 공개 코드에 없습니다. 논문의 hybrid 학습 설계는 이 저장소에서 재현할 수 없습니다.
- README가 말하는 knowledge insulation은 공개 미세조정 코드에 구현되어 있지 않습니다 (D7). 체크포인트 학습 방식에 대한 설명으로 보입니다.
- 논문 문서의 헤드 수 “18”은 오타로 보입니다 (D12).
