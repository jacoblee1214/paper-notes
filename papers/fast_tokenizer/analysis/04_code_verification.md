# 04. 공식 코드 대조: FAST (arXiv:2501.09747)

**방법.** 공식 코드는 두 곳에 있습니다. 토크나이저 본체는 Hugging Face 저장소 `physical-intelligence/fast`의 `processing_action_tokenizer.py`와 설정 파일입니다. 사용 예(VLA 입력 구성, 디코딩 실패 처리)는 `Physical-Intelligence/openpi`(커밋 `215abfb`, 2026-08-24)의 `src/openpi/models/tokenizer.py`, `pi0_fast.py`, `training/config.py`, `transforms.py`, `shared/normalize.py`입니다. 얕은 복제와 파일 4개 내려받기로 가져왔고, **정적으로 읽기만 했습니다.** 설치, 실행, 가중치 다운로드, API 호출은 하지 않았습니다. 가져온 코드는 신뢰하지 않는 자료로 취급했습니다.

파일 위치: `repo/` (openpi 얕은 복제, gitignore), `repo/hf_fast/` (HF 파일 4개, gitignore).

## 1. 대조 결과 (D1–D15)

판정 기준: **일치** (논문과 같음), **차이** (논문에 없거나 다른 선택), **위험** (정적 읽기로 실패 가능성이 보임, 실행으로 확인 필요).

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 분위수 정규화 | §V-B: 차원별 1%·99% 분위수를 [-1, 1]로 | HF `__call__`은 정규화하지 않음(README가 "미리 정규화"를 요구). openpi는 FAST 모델에서 `use_quantile_norm = True`로 `(x−q01)/(q99−q01+1e-6)·2−1`을 적용. 클리핑 없음 | 일치 (책임 분리). 범위 밖 값의 처리는 어디에도 없음 (D3) |
| D2 | 어휘 크기 | §V-B: 단일 데이터셋 어휘 1024. FAST+ 어휘는 논문에 없음 | HF `processor_config.json`의 `vocab_size`는 **2048**, 코드 기본값은 `fit()`에서 1024 | 차이 (논문은 FAST+ 어휘를 밝히지 않음). `tokenizer.json`(약 0.7 MB)은 받지 않아 실제 어휘 크기는 확인하지 못함 |
| D3 | 최솟값 클리핑 | 논문 알고리즘에 클리핑 없음 | `np.maximum(elem − min_token, 0)`. FAST+는 `min_token = −354`. 정적 계산: 직교 DCT에서 \|C_k\| ≤ √H·max\|a\|이므로 H = 50, γ = 10에서 하한 −354에 닿으려면 max\|a\| ≈ 5 이상이어야 함 | 차이·위험. 정상 범위 (|a| ≤ 1)에서는 발생하지 않음. 범위 밖 입력은 조용히 잘림 |
| D4 | 디코딩 실패 처리 | 논문은 "모든 연산이 가역" (§V-B) | `decode()`의 `except Exception`에서 **0 행동을 반환**하고 메시지만 출력 | 차이·위험. 잘못된 토큰이 오류 없이 0 행동으로 바뀜 |
| D5 | openpi의 "Action:" 누락 | 해당 없음 | `extract_actions()`가 텍스트에 "Action: "이 없으면 `zeros((action_horizon, action_dim))` 반환 | 차이·위험. 실패가 정책 출력 0으로 보임 |
| D6 | 디코딩 왕복 | 토큰 ID를 그대로 사용 | openpi는 토큰 ID → 텍스트(SentencePiece 디코드) → 다시 인코드 → ID 순서로 복원 | 위험 (정적으로 판단). 재인코딩이 원래 ID와 같다는 보장이 없음. 실행 검증이 필요함 |
| D7 | 토큰 덮어쓰기 위치 | §VI-A: VLM 어휘에서 **사용 빈도가 낮은** 토큰을 덮어씀 | openpi는 PaliGemma 어휘 끝에서 특수 토큰 128개를 건너뛴 **고정 위치**(`V − 1 − 128 − t`)를 씀 | 차이 (구현 방식). 목적은 같음 |
| D8 | 잘림 (truncation) | 언급 없음 | `FASTTokenizer.tokenize`는 `max_len`을 넘으면 `tokens[:max_len]`로 자르고 경고만 함. pi0-FAST 기본 `max_token_len = 250`, LIBERO 설정 180 | 차이·위험. 끝의 `|`와 뒤쪽 행동 토큰이 잘리면 디코딩이 실패(D5로 이어짐) |
| D9 | 추론 온도 | §C: 양팔 과제에서 β = 0.7 | `sample_actions`의 기본 `temperature = 0.0` (greedy) | 차이 (실험별 설정). 공개 기본값은 greedy |
| D10 | DROID 학습 설정 | 부록 D: 15 스텝 청크, 8 또는 15 스텝 실행, 240k 반복, 배치 256, EMA 0.999, 상수 학습률 5e-5 | `pi0_fast_droid`: `action_horizon = 10`, 배치 기본 32, 30k 스텝, EMA 기본 0.99, 코사인 최대 학습률 2.5e-5 | 차이. openpi README가 이를 "근사 구현"이라고 밝힘 |
| D11 | LIBERO 학습 설정 | 부록 E: 40k 반복 (약 40 에폭), 묶은 샘플 270k | `pi0_fast_libero`: `action_horizon = 10`, `max_token_len = 180`, 30k 스텝. 배치 크기는 기본값 | 차이 (스텝 수, 청크 길이). 논문의 배치 크기는 본문에 없음 |
| D12 | DCT 정규화 | 직교 여부 미기재. 스케일 γ = 10 | `dct(..., norm="ortho")`, 스케일 10 | 일치 (γ). 직교 정규화는 논문이 밝히지 않은 선택 |
| D13 | 평탄화 순서 | §V-B: 주파수 낮은 것부터, 차원을 섞어 펼침 (열 우선) | DCT를 시간축(axis 1)으로 적용한 뒤 (H, D) 배열을 행 우선 `flatten()` → 주파수마다 모든 차원이 이어짐 | **일치** (정적 판독). 실행 검증은 아님 |
| D14 | BPE 학습 세부 | BPE 어휘 크기만 밝힘 | `ByteLevelBPETokenizer`, 초기 알파벳은 최소~최대 범위의 문자, `min_frequency=2`, `max_token_length=10000`, 특수 토큰 없음 | 차이 (논문에 없는 값). 재학습 시 필요 |
| D15 | 상태 저장 | 언급 없음 | `__call__`이 `time_horizon`, `action_dim`을 저장하고 `decode()`가 이를 재사용. 인자를 주지 않으면 마지막 호출 값을 씀 | 차이·위험. 호출 순서에 따라 디코딩 형태가 달라질 수 있음 |
| D16 | 범용 토크나이저 학습 | §V-C: 약 1M 청크, 32차원 패딩, 여러 행동 공간 | 공개된 것은 `fit()` 함수뿐. 학습 데이터와 스크립트는 없음. 패딩 로직도 없음 | 차이 (재현 불가 요소). FAST+ 가중치는 공개되어 있음 |

**추가 대조 (일치).**
- 고유감각 상태 256 구간 이산화: openpi `np.digitize(state, linspace(−1, 1, 257)[:-1]) − 1`은 §VI-A·부록 C와 같습니다.
- 학습 옵티마이저 기본 설정은 `pi0_fast_*` 설정에서 확인했으나, 배치와 EMA는 논문과 다릅니다(D10).

## 2. 실행하지 않은 검증 (위험 항목 이어서)

- **D3, D6:** 실제 토큰 열과 극단 입력을 넣어 확인해야 합니다. 설치(transformers, scipy, tokenizers, sentencepiece)와 PaliGemma 토크나이저 내려받기가 필요해 승인 뒤로 미뤘습니다.
- **D4, D5:** 잘못된 토큰 열을 넣어 0 반환이 실제로 일어나는지 확인해야 합니다.
- **D8:** 긴 행동 청크(H = 50, D = 14)에서 토큰 길이가 250을 넘는지 측정해야 합니다.

## 3. 라이선스

- `physical-intelligence/fast` 모델 저장소: 모델 카드 태그 `license: apache-2.0`. 저장소 파일 목록에 LICENSE 파일은 없습니다.
- `Physical-Intelligence/openpi`: Apache-2.0 (GitHub API, 루트 LICENSE 파일 확인).
- 코드는 저장소에 넣지 않았습니다. 이 폴더에 있는 것은 얕은 복제본(`repo/`, gitignore)입니다.

## 4. 구현 가능성 판정 (코드)

**이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 가능한 것과 아닌 것.**

- **토크나이저 검증 (가능).** FAST+ 토크나이저는 SciPy DCT와 `tokenizers` BPE뿐입니다. CPU에서 충분히 돕니다. 설치(transformers, scipy 등)와 몇 MB 파일 내려받기에 승인이 필요합니다. 추정 노력은 반나절입니다(추정).
- **π0-FAST 추론 (조건부).** openpi는 JAX 기반입니다. 맥 CPU나 Metal에서 돌릴 수는 있지만 속도는 측정 전입니다. 체크포인트(`gs://openpi-assets/checkpoints/pi0_fast_base`)는 수 GB로 추정되며(크기 미확인) 1 GB 이상이므로 승인이 필요합니다. 추정 노력은 1–2일입니다(추정).
- **π0-FAST 학습과 LIBERO·DROID 평가 (불가).** 논문 부록 D는 8×H100에서 약 4일을 씁니다. CUDA 기기와 시뮬 환경이 필요하며 이 맥에서는 실행하지 않습니다.

## 5. 프로젝트 페이지

`pi.website/research/fast`는 보안 확인 페이지(429)에 막혀 읽지 못했습니다. 추가 코드, 추가 체크포인트, 학회 표기가 있는지 확인하지 못했습니다.
