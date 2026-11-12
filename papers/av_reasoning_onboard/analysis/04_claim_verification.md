# 04. 주장 검증과 구현 가능성

이 문서는 블로그 주장 대조(§1), 공개 코드·설정의 정적 대조(§2), 확인 못 한 항목(§3), 구현 가능성 판정(§4)으로 이루어집니다. 가중치와 체크포인트는 받지 않았고, 공개 코드는 실행하지 않았습니다. `gh api`로 읽은 텍스트와 config.json만 사용했습니다.

판정 표기: **일치**, **부분 일치**(조건이 빠짐), **불일치**, **확인 불가**(공개 자료 없음 또는 실행 필요).

## 1. 주장 대조 (C1–C21)

| # | 블로그 주장 | 확인한 결과 | 판정 | 근거 |
|---|---|---|---|---|
| C1 | "2B-parameter" 모델 | 전체 2.649B (백본 2.45 + expert 0.2) | 부분 일치 (공칭 명칭) | `03` §1 |
| C2 | Thor에서 픽셀→궤적 160–190 ms | 레시피 단계 합 165+12r ms. 160은 r=0보다 낮음. 측정 조건 미공개 | 부분 일치, 조건 불명 | `03` §3 |
| C3 | Figure 2: 폐루프 중대 사고 간 거리 약 60배 | 15.15/0.25 = 60.6. 모델 카드 15.15 ± 0.04 km | 일치 (분모 0.25 출처 불명) | 모델 카드, `03` §7 |
| C4 | "Outperforms across major AV tasks" (Figure 2) | LingoQA: Edge 75.2 < Qwen 79.4 | **불일치** (한 과제에서 낮음) | 모델 카드 75.2 |
| C5 | Figure 2 Edge 값 (Ego3D, VLADBench, SURDS, WaymoQA, nuReasoning, KITScenes) | 모델 카드 표와 0.1 이내 일치 | 일치 | 모델 카드 Table |
| C6 | 폐루프 주행 성능 +48.9% (AlpaGym RL) | 공개 수치 없음. 공개 정책은 Alpamayo 1.5 10B뿐 | 확인 불가 | alpagym 트리 |
| C7 | Figure 5: AlpaJudge 보상 +10.2% (자체) / +11.4% (사람) | 그림 판독값. 데이터 공개 없음 | 확인 불가 (그림 판독만) | `03` §7 |
| C8 | Figure 5: 기준선 보상 +13.4% (자체) / +1.8% (사람) | 자체·사람 평가가 크게 어긋남 | 확인 불가 (그림 판독만), 해석에 주의 | `03` §7 |
| C9 | AlpaJudge "human-calibrated" | 보정 데이터, 일치도, 방법이 저장소에 없음 | 확인 불가 | alpajudge README |
| C10 | RF 점수 식: action = p_match(1+p_safe)/2, overall = 0.6·action + 0.4·component | `reference_free/model.py`와 일치 | 일치 | rf_model.py |
| C11 | 노트북 RF 결과 overall 0.842 | 기록 부분점수로 재계산 0.8412. 절사·반올림 양쪽과 양립 | 일치 | `03` §6 |
| C12 | 노트북 RB alignment 0.999 | 체크포인트 실행 필요 | 확인 불가 | — |
| C13 | 샘플링 설정: seed 43, top_p 0.98, temperature 0.6, 256 토큰, 10 Euler 단계 | `inference.py` 기본값과 동일 | 일치 | inference.py L278–285 |
| C14 | "<NOMINAL>" 특수 토큰으로 추론 생략 | **학습 쪽은 확인:** alpamayo-recipes `scripts/prepare_pai_annotations.py`가 `<NOMINAL>` 라벨을 세고, "`<NOMINAL>` 라벨이 없으면 미세조정 중 행동이 바뀔 수 있다"는 경고를 둠. **추론 쪽은 미확인:** 공개 추론 코드에는 해당 문자열이 없고, 레시피 어휘는 ID 목록(2211) | 부분 일치 (추론 빈도 확인 불가) | `02` §4 |
| C15 | 6 카메라, 22 프레임, 카메라별 픽셀 예산 | config와 README 일치 | 일치 | config.json, README |
| C16 | 오프라인 증류: Alpamayo 2 Super 자동 라벨 | 증류 코드·데이터 없음. Super SFT 레시피는 있음 | 확인 불가 | alpamayo-recipes |
| C17 | FP8 양자화 + TensorRT·Edge-LLM 레시피 | `quantize.py`, `action_graph.py`, 레시피 README와 일치 (FP8 KV, expert FP8 GEMM, ViT 앞 2층 FP16) | 일치 (구현 존재) | `04` §2, D5 |
| C18 | DFlash2 추측 디코딩이 품질을 떨어뜨리지 않음 | 레시피: 10개 검증 클립에서 greedy 체인과 완전히 같음. 초안 모델은 릴리스에 없음 | 부분 일치 (자체 보고, 초안 모델 미공개) | 레시피 README |
| C19 | 행동 expert 5 Euler 단계 (Thor) | 레시피 README 확인. 블로그 코드 예제는 10단계 | 일치 (단, 블로그 안에서 조건이 섞임) | `03` §3 |
| C20 | Embedl: AGX Thor에서 10 Hz | 10 Hz = 100 ms. 레시피 단계 합 하한 165–177 ms | **불일치** (추가 병렬화 없이는 불가) | `03` §3.2 |
| C21 | 파트너 주장 (PORT 2배 계획 속도, Niulinx minADE 13.6%, 횡방향 46.1%) | 파트너 자체 보고 | 확인 불가 | 블로그 인용 |

## 2. 공개 코드·설정 정적 대조 (D1–D10)

| # | 대조 항목 | 발견 | 함의 |
|---|---|---|---|
| D1 | 치수: config.json ↔ config.py ↔ 블로그 | 텍스트 28층/2048, 비전 27층/1152, expert 28층/512로 일치 | 구조 주장은 신뢰 가능 |
| D2 | 모델 리비전 | PyTorch README는 `f8951d09…`, Thor 레시피 테스트는 `aaaa32c…`. 블로그 코드 예제는 리비전 없이 `load_model("nvidia/Alpamayo2-Edge")` | **두 리비전이 같은 가중치인지 확인 필요.** 재현 시 리비전을 명시해야 함 |
| D3 | 지연 수치의 기재 | 블로그 160–190 ms, 레시피 README "약 200 ms", 결과 절 "180–250 ms (10 클립)", 커밋 메시지 "약 180–250 ms" | 기재가 서로 다름. 블로그의 하한이 가장 낮음 |
| D4 | 검증 클립 설명 | 레시피 README는 "10 internal validation clips", 커밋 메시지는 "ten public golden-bundle clips" | 어느 쪽 클립인지 불명. 재현 자료 확인 필요 |
| D5 | 추측 디코딩 구현 | `thor/runtime/speculative.py`: 초안 block−1개, 검증 1회, 맞은 연속 길이 + 교정 1개 | 구현은 명확함. 블로그 수치와 조건 연결은 불명 |
| D6 | 초안 모델 공개 여부 | 레시피 README: "DFlash2 block-16 drafter ... not part of the model release" | **블로그 수치의 핵심 요소가 공개되지 않음.** 공개 저장소의 `speculative` 검색은 `alpamayo2-edge`에서 0건 |
| D7 | PyTorch 가속 경로 | `--accel tuned`: torch.compile, CUDA 그래프, FlashAttention 2(선택) | 가속은 CUDA 전용 |
| D8 | 샘플링 스텝 | PyTorch 기본 10단계, Thor 5단계. 블로그 예제 10단계 | 같은 모델이라도 결과가 다를 수 있음 (궤적 차이를 재야 함) |
| D9 | 품질 비교 기준 | 레시피는 FP16 엔진과 minADE₆ 비교(1.547 대 1.570 m, 10 클립). BF16 PyTorch와의 직접 비교는 없음 | 양자화 품질은 PyTorch 기준으로 보지 않았음 |
| D10 | RL 경로 | alpagym 공개 정책은 `alpamayo_r1`(1.5, 10B). Edge용 RL 정책 없음. 블로그 Workflow 3의 그레이더 인터페이스(`BaseReasoningGrader`, `GRADER_REGISTRY`)는 `alpamayo1_x_rl`에 있고 예시는 LingoJudge. AlpaJudge 어댑터는 없음 | 블로그의 RL 보상 흐름은 공개 코드로 **재구성만 가능**(어댑터 직접 작성) |
| D11 | AlpaJudge API 이름 | `load_reference_based`, `score_reference_based`, `load_reference_free`, `score_reference_free` 모두 존재 | 블로그 예제 코드는 정확함 |
| D12 | 노트북 기록값 | 노트북 셀 출력은 지워졌고(`outputs: []`), 수치는 마크다운 셀에 적힘 | 실행으로 재현한 값이 아님 |
| D13 | 모델 카드 vs 블로그 | Edge 벤치마크 값, 2.45/0.2 B, 리비전, 16 GB 권장 | 일치 |
| D14 | 라이선스 | 코드 Apache-2.0, 가중치 OpenMDW-1.1 (모델 카드, AlpaJudge README) | 사용 조건은 가중치 쪽을 따라야 함 |
| D15 | 개방 루프 minADE 값 | 모델 카드 minADE₆@5 s = 0.52 m (1,392 샘플). 레시피 minADE₆ = 1.55 m (10 클립, 기간 미표기) | 두 수치는 데이터·기간이 달라 **비교할 수 없음**. 레시피 값을 정확도 주장으로 읽으면 안 됨 |

## 3. 확인하지 못한 항목 (실행 또는 추가 다운로드 필요)

1. AlpaJudge RB 0.999, RF 노트북 실행 결과 (체크포인트 실행)
2. NOMINAL 토큰의 추론 시 출현 빈도 (tokenizer.json 17.8 MB로 어휘 존재는 확인 가능, 빈도는 실행 필요)
3. 두 모델 리비전(`f8951d09`, `aaaa32c`)의 가중치 동일성 (HF 파일 해시 비교로 가능)
4. Thor 지연 160–190 ms의 측정 조건 (블로그 작성자 문의 필요)
5. 드래프터 DFlash2의 공개 여부와 수락률 (블로그·레시피 어디에도 수락률 수치 없음)
6. 양자화 품질 대 BF16 (PyTorch 기준 실행 필요)
7. RL +48.9%의 기준선과 시나리오 수
8. AlpaJudge 사람 보정 데이터와 일치도
9. 273 GB/s 대역폭 (M4 Pro, Thor) — 사전 지식, 공식 명세로 확인 필요
10. Qwen-Drive-1.0 (4B) 수치의 출처

## 4. 구현 가능성 판정

### 4.1 한 줄 판정
**이 맥(Apple M4 Pro, 통합 메모리 24 GiB, Metal/MPS, CUDA 없음)에서는 블로그의 온보드 시스템(Thor 실시간 160–190 ms)을 구현할 수 없습니다.** 모델의 기능적 PyTorch 추론은 메모리상 들어가고, 느린 형태로 **포팅은 가능(조건부)**합니다. Thor 엔진, FP8 보정, 추측 디코딩 초안 모델은 하드웨어와 공개되지 않은 부품 때문에 불가능합니다.

### 4.2 항목별 판정

| 항목 | 필요 자원 | 이 맥 판정 | 근거·추정 | 노력 (추정) |
|---|---|---|---|---|
| 공식 PyTorch 추론 (Edge, 22프레임 → CoT + 궤적) | CUDA GPU 16 GB 이상 권장, BF16, transformers 5.17, SDPA 또는 FA2 | **조건부 가능**: 가중치 BF16 4.9 GiB는 들어감. `device="cuda"`, `torch.Generator(device="cuda")`, `autocast("cuda")`를 MPS 분기로 바꾸고 BF16 수치 차이를 확인해야 함 | 프리필 약 12.7 TFLOP (LM 1.96B × 2 × 3236). 맥 GPU 실효 수 TFLOPS 가정 → **수 초 (추정 2–4 s)**. 지연 목표 0.2 s 대비 10배 이상 | 포팅 1–2주 (가정), 수치 검증에 GPU 필요 |
| 블로그 지연 목표 재현 (160–190 ms) | Thor 또는 H100급 | **불가** | 프리필만 맥에서 수 초 | — |
| 디코드 (CoT 25 토큰) | 메모리 대역폭 | **가능하지만 느림** | BF16 상한 약 73 tok/s (273 GB/s 가정) → 25토큰 약 0.35 s | — |
| AlpaJudge RB (텍스트 전용) | CUDA 명시, 가중치 약 4–5 GB | **조건부 가능** | 텍스트 모델 2B급. CPU 또는 MPS 실행은 미확인 | 포팅 수일 (가정) |
| AlpaJudge RF (6프레임 + 텍스트) | GPU 권장, 가중치 약 5 GB | **조건부 가능** | 비디오 프레임 디코딩과 PAI 데이터 접근(게이트) 필요 | 포팅 1주 (가정) |
| Thor TensorRT 엔진 빌드·실행 | DRIVE AGX Thor, TensorRT 11.3 SDK, DRIVE OS | **불가** | 보드와 자동차용 SDK 필요 | — |
| FP8 보정·ONNX 내보내기 | x86 호스트 + NVIDIA GPU 16 GB 이상, ModelOpt 0.45, TensorRT Edge-LLM | **불가** | CUDA·ModelOpt 전용 | — |
| 추측 디코딩 초안 모델 학습 | 공개되지 않음. 유사 레시피는 80 GB GPU 8장 | **불가** (현재 조건) | `alpamayo1_5_speculative_decoding`은 8× 80 GB 기준 | — |
| 폐루프 RL (AlpaGym) | 멀티 GPU, AlpaSim | **불가** | 공개 정책이 Edge가 아님 | — |
| 데이터 선별 (판정기 점수 기반) | PAI 데이터 접근, 판정기 추론 | **조건부** | 판정기 추론이 가능해야 함 | 판정기 배치 실행 시간에 비례 |
| 산술·toy 재현 (본 폴더 스크립트) | Python, numpy | **가능, 완료** | 이 저장소의 `analysis/*.py` | 완료 |

### 4.3 "구현할 수 있는가"에 대한 답
- **재구현(개념·산술·수식)**: 가능. 지연 모델, 추측 디코딩 수학, RF 점수식, 대역폭 상한은 이미 맥에서 재계산했습니다.
- **기능 포팅(맥에서 같은 모델이 같은 출력을 내는지)**: 조건부. 메모리는 충분하고 CUDA 호출만 고치면 되지만, 수치 검증에는 GPU 기준값이 필요합니다. 승인과 다운로드가 필요합니다.
- **성능 구현(160–190 ms)**: 불가. 이 맥에서는 같은 모델이라도 수 초가 걸릴 것으로 추정됩니다. Thor 엔진과 드래프터가 없으면 블로그 수치에 가까워질 수 없습니다.
- **라이선스·접근**: 가중치는 OpenMDW-1.1이고 PAI 데이터는 게이트가 있습니다. 실행 전에 사용 조건을 확인해야 합니다.

### 4.4 맥에서 할 수 있는 다음 단계 (승인 필요)
1. 가중치 없이 할 수 있는 것: tokenizer.json 검사(NOMINAL 확인, 17.8 MB)와 두 리비전의 해시 비교(D2).
2. 설치 승인 후: torch와 transformers 5.17을 맥에 설치하고, 모델 로드 대신 **config 기반 메모리 측정**(빈 모델 생성, meta device)으로 포팅 가능성 확인.
3. 가중치 다운로드 승인 후: RB 체크포인트(약 4 GB)를 맥 MPS에서 실행해 노트북의 RB 0.999와 비교.

## 5. 검증 방식 요약
- 수치 재계산: config, 레시피 단계 시간, 노트북 기록값으로 계산 (`analysis/*.py`)
- 공개 코드 대조: 블로그 코드 예제와 `inference.py`, 점수 식과 `rf_model.py`, 레시피 `speculative.py`·`quantize.py`
- 실행: 하지 않음. 가중치 다운로드 없음
