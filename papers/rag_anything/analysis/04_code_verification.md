# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `HKUDS/RAG-Anything` HEAD `2cd2b33`(2026-10-28, v1.4.4)와 논문 시점 커밋 `806ac9b`(2025-10-13; `config.py`, `modalprocessors.py`, `query.py`, `prompt.py`만 받아 비교). 기반 라이브러리 `HKUDS/LightRAG` HEAD `aaebfe1`(2026-10-14).
- **범위:** 코드 읽기와 CPU 재계산만 했습니다. 외부 API 호출, 패키지 설치, 모델 다운로드는 하지 않았습니다. 공개 저장소에는 논문 실험의 질의별 결과, 색인, 로그가 없어서 표를 결과 파일로 재계산할 수는 없습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| 문서 분해, 모달 유형 (Eq. 1) | `parser.py` (MinerU 기본, Docling·PaddleOCR 선택), `config.py: parser="mineru"` | ✅ 논문은 MinerU만 씀 |
| 모달별 처리기 (그림·표·수식) | `modalprocessors.py: ImageModalProcessor, TableModalProcessor, EquationModalProcessor, GenericModalProcessor` | ✅ 오디오·비디오 처리기는 이후 추가(기본 꺼짐) |
| 상세 설명 d^chunk + 개체 요약 e^entity | 각 처리기의 `generate_description_only`, JSON `detailed_description` + `entity_info{entity_name, entity_type, summary}` | ✅ 요약은 최대 100단어 |
| 문맥 창 C_j, δ | `ContextExtractor`, `config.py: context_window=1, context_mode="page", max_context_tokens=2000, context_filter_content_types=["text"]` | ⚠️ D6 |
| 내부 개체 추출 R(d^chunk) (Eq. 2) | `_process_chunk_for_extraction` → LightRAG `extract_entities` | ✅ LightRAG의 일반 추출 프롬프트 그대로 |
| `belongs_to` 간선 (Eq. 3–4) | `modalprocessors.py` 857–887행: keywords `"belongs_to,part_of,contained_in"`, weight 10.0 | ✅ weight는 논문에 없음 |
| 두 그래프의 개체 정렬·병합 | 같은 `knowledge_graph_inst`에 `upsert_node`, LightRAG `merge_nodes_and_edges` | ⚠️ D1 |
| 임베딩 표 T (Eq. 5) | LightRAG의 `entities_vdb`, `relationships_vdb`, `chunks_vdb` | ✅ 그림 설명도 같은 chunks_vdb에 들어감 |
| 모달 인식 질의 인코딩, 융합 점수 | 해당 코드 없음 | ❌ D4 |
| 구조 탐색 + 의미 매칭 | `query.py: aquery(mode="mix")` → LightRAG `kg_query` mix 모드 | ⚠️ D3 |
| 시각 자료 복원 + VLM 답 생성 (Eq. 6) | `aquery_vlm_enhanced`: 검색 프롬프트의 `Image Path:` 줄을 정규식으로 찾아 base64 이미지로 VLM에 전달 | ✅ 단, 재현 스크립트는 끔(D5) |
| 실험 설정 (Sec. 3.1) | `reproduce/env.example`, `reproduce/query.py` | ✅ D7 |
| 판정 프롬프트 (A.4, Fig. 10) | `reproduce/llm_answer_evaluator.py` | ⚠️ D8 |
| 분석 프롬프트 (A.3, Fig. 7–9) | `prompt.py` | ✅ 논문 시점과 일치, HEAD는 다름(D9) |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **"두 그래프를 만든 뒤 개체 정렬로 병합"은 같은 그래프 저장소에 넣는 것입니다.** 교차 모달 쪽 노드와 텍스트 쪽 노드가 같은 LightRAG 그래프에 upsert되고, 이름 문자열이 정확히 같은 노드만 합쳐집니다. 별도의 정렬 단계(동의어, 표기 변형, 임베딩 유사도)는 없습니다. 앵커 이름에는 `" (image)"` 같은 유형 접미사가 붙으므로(HEAD 1139–1141행, 논문 시점도 같음) 앵커 자체는 절대 병합되지 않습니다. Fig. 1의 "Text VDB"와 "Multi-modal VDB"도 실제로는 하나의 벡터 DB입니다. | 두 그래프는 내부 개체 이름이 우연히 같을 때만 이어집니다(`03` §5). "개체 정렬"의 기여를 따로 잰 절제는 없습니다. |
| D2 | **사례 연구가 설명하는 표·그림 내부 구조 그래프가 코드에 없습니다.** 행 머리글, 열 머리글, 셀, 단위 노드와 row-of, column-of, header-applies-to, unit-of 간선(Sec. 3.4), 패널·축·범례 노드(Fig. 3 설명), bar-of, label-applies-to 간선(A.2)을 코드 전체에서 검색해도 찾을 수 없습니다. 표는 MinerU가 준 HTML 본문과 LLM 분석문을 이어 붙인 **청크 하나**(`table_chunk` 프롬프트)로 저장되고, 그 분석문에서 일반 개체 추출을 돌릴 뿐입니다. | 사례 연구의 "왜 맞혔는가" 설명은 코드가 하는 일이 아닙니다. LLM 추출이 우연히 그런 개체를 낼 수는 있지만, 스키마로 보장되지 않습니다. 셀 위치를 맞히는 일은 실제로는 reader가 HTML 표를 읽으며 합니다. |
| D3 | **"구조 탐색"은 LightRAG의 키워드 벡터 검색 + 1-hop입니다.** 논문은 "질의어와 개체의 정확한 매칭 후 지정 hop까지 이웃 확장"이라고 씁니다. 코드에서는 LLM이 질의에서 저수준·고수준 키워드를 뽑고, 개체·관계 벡터 DB를 코사인 유사도로 검색합니다(LightRAG `_get_node_data`, `top_k` 기본 40, 재현 설정 50, 임계값 0.2). 그다음 검색된 개체의 1-hop 간선만 펼칩니다. hop 수를 정하는 설정은 없습니다. | "정확한 매칭"이 아니라 임베딩 매칭입니다. 논문의 두 경로(구조/의미)는 코드에서 둘 다 벡터 검색에서 출발합니다. |
| D4 | **모달 인식 질의 인코딩과 다중 신호 융합 점수가 없습니다.** 질의에서 "figure", "table" 같은 단어를 찾아 모달 선호를 정하는 코드, 그리고 구조 중요도·의미 유사도·모달 선호를 합치는 점수가 `query.py`(논문 시점, HEAD 모두)에 없습니다. 후보는 LightRAG mix 모드가 개체·관계·벡터 청크를 토큰 예산 안에서 합치고, 선택적으로 reranker가 다시 정렬합니다. | 논문이 Sec. 2.3에서 서술한 핵심 검색 장치 두 개가 공개 코드로 재현되지 않습니다. 부록 A.5가 말한 "텍스트 중심 검색 편향"은 모달 선호 신호가 없는 코드 구조와 잘 맞습니다. |
| D5 | **재현 스크립트는 VLM 답 생성을 끕니다.** `reproduce/query.py`가 `rag.aquery(..., mode="mix", response_type="One Sentence", vlm_enhanced=False)`로 질의합니다. 그러면 검색 문맥(그림의 텍스트 설명 포함)만 LLM에 들어가고, Eq. 6의 원본 이미지 복원은 일어나지 않습니다. 재현 스크립트는 논문 6개월 뒤인 2026-04-13(`a9fb942`)에 추가되었습니다. | 논문 수치가 VLM 답 생성으로 나온 것인지 알 수 없습니다. 공개 스크립트로 재현하면 Eq. 6이 빠진 설정이 됩니다. |
| D6 | 문맥 창 δ는 **쪽 단위**가 기본입니다(`context_mode="page"`, `context_window=1` → 앞뒤 1쪽). 텍스트 항목만 문맥으로 쓰고(`context_filter_content_types=["text"]`) 2,000 토큰에서 자릅니다. 논문은 δ를 단위 인덱스 거리 |k − j| ≤ δ로 정의하고 값을 밝히지 않습니다. 재현 설정은 `CONTEXT_WINDOW`를 주석 처리해 두어 기본값 1쪽을 씁니다. | 정의가 다릅니다. 재현할 때 값과 단위를 명시해야 합니다. |
| D7 | 실험 설정은 재현 파일과 맞습니다. `reproduce/env.example`은 GPT-4o-mini(OpenRouter), text-embedding-3-large(3,072), bge-reranker-v2-m3, `MAX_ENTITY_TOKENS=12000` + `MAX_RELATION_TOKENS=8000` = 20,000(논문 "개체+관계 20,000")이고, `MAX_TOTAL_TOKENS=32000`에서 20,000을 빼면 청크 12,000(논문 "청크 12,000")입니다. `TOP_K=50`, `CHUNK_TOP_K=10`, mix 모드, 한 문장 답도 같습니다. | ✅ 다만 LightRAG 버전은 논문 시점에 고정되지 않았습니다(`lightrag-hku` 버전 미지정). |
| D8 | **판정 스크립트에 두 가지 문제가 있습니다.** (a) 기본 판정 모드가 `comprehensive`로, 논문 Fig. 10의 정확도 전용 프롬프트가 아니라 여러 지표(관련성, 완결성, 충실성 등)를 함께 요구합니다. (b) JSON 파싱 실패 시 fallback이 긍정 키워드를 부분 문자열로 먼저 검사해서 "incorrect", "mismatch", "not accurate"가 정답(1)으로 채점됩니다(`03` §6; toy 9개 중 4개 오채점). 판정 모델은 GPT-4o-mini, temperature 0.1입니다. | (b)는 실패 빈도만큼 정확도를 부풀립니다. 빈도는 공개되지 않았습니다. 고치는 법은 간단합니다(부정 키워드를 먼저, 단어 경계로 검사하거나 구조화 출력 강제). |
| D9 | 논문 부록의 분석 프롬프트(Fig. 7–9)는 논문 시점 `806ac9b`의 `prompt.py`와 같습니다. HEAD는 2026-05-22(섹션 경로, 주변 텍스트 메타데이터)와 2026-08-13(다이어그램 노드·간선을 모두 열거하라는 지시, 파일명 대신 의미 있는 개체명)에 바뀌었습니다. | HEAD로 돌리면 논문과 다른 설정입니다. 논문 재현에는 `806ac9b`나 v1.2.8(2025-09-22)을 써야 합니다. |
| D10 | 앵커의 내부 개체 추출은 LightRAG의 gleaning(기본 1회 추가 추출)을 그대로 씁니다. 비텍스트 단위 하나당 VLM 설명 호출 1회 + 추출 호출 2회가 듭니다. | 논문에 비용 보고가 없습니다(`03` §7). |

## 3. 수치 재계산 (`analysis/table_check.py`, `analysis/code_logic_toy.py`)

| # | 확인 항목 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | Overall = 영역 정확도의 문항 가중 평균 (Table 2–5) | 11개 행 중 9개가 0.1%p 안에서 일치. 나머지는 **MMGraphRAG(Table 2) 61.0 vs 56.7**, MMLongBench GPT-4o-mini 33.5 vs 34.05 | ⚠️ MMGraphRAG 행 불일치 |
| C1b | 단일 칸 오기 가설 | Law 40.0 → 64.7 또는 Fin 52.8 → 69.2면 61.0. Fig. 2 가중 평균 60.9가 Overall 61.0을 지지 | 가설 |
| C2 | 유형 3개(Txt, Mm, Una)로 Overall 설명 | 7개 행 공통 비율 맞춤의 잔차 최대 2.7%p | 표에 없는 유형이 있을 가능성 |
| C3 | 차이의 표본 오차 | DocBench vs MMGraphRAG +2.4 (p ≈ 0.25), vs LightRAG +5.0 (p ≈ 0.016). MMLongBench vs MMGraphRAG +5.1 (p ≈ 0.016), vs LightRAG +3.9 (p ≈ 0.065). Chunk-only −3.4 (p ≈ 0.10) | 비대응 근사 |
| C4 | 평균 쪽수 (Table 1 vs 5·6) | DocBench 66 vs 61.0 (반올림 허용 60.5–61.5 밖), MMLongBench 47.5 vs 48.3 (47.8–48.8 밖) | ⚠️ 불일치 |
| C5 | Fig. 2 (픽셀 판독) | 본문 길이별 수치와 0.1%p 안에서 일치. 문항 수 합 DocBench 약 1,079, MMLongBench 약 971(표 1,082). MMLongBench 200쪽 이상 n = 5(3 vs 2), 1–10쪽 n = 16(3 vs 2) | 극단 구간은 근거 아님 |
| C6 | GPT-4o-mini 기준선의 50쪽 제한 | DocBench 금융(평균 192쪽) 약 26%, 정부 72%만 봄 | 기준선이 구조적으로 불리 |
| C7 | 이득의 출처 | LightRAG 대비 Txt 0.0, Mm +16.6, Una −0.8. MMGraphRAG 대비 Una −14.5 | 이득은 Mm, 손해는 Una |
| C8 | 비용 (텍스트 처리만, 가격은 가정) | DocBench 약 $22, MMLongBench 약 $10 / 방법 1회 | 논문에 없음 |
| C9 | 그림 정합성 | Fig. 12의 질문이 Fig. 3 질문의 복사본 | 편집 오류 |
| E | 판정 fallback | toy 9개 중 4개 오채점 | D8 |
| G | 앵커-텍스트 그래프 연결 | P = 1 − (1 − p)^m | `03` §5 |

## 4. 구현 가능성 판단

### 4.1 공식 코드를 그대로 쓰는 경우 (도구로 사용)

| 항목 | 요구 사항 | 비고 |
|---|---|---|
| 설치 | `pip install raganything` (+ MinerU 또는 Docling). Python 패키지라 macOS에서도 설치될 것으로 보임 | 이 맥에서는 설치하지 않았습니다. MinerU의 Apple Silicon 가속(MPS) 지원 여부는 확인하지 않았습니다 |
| 모델 | LLM, VLM, 임베딩 함수 세 개를 사용자가 넘김. 논문 설정은 GPT-4o-mini + text-embedding-3-large + bge-reranker-v2-m3(API) | 저장소에 Ollama, LM Studio, vLLM 연동 예제가 있어 로컬 모델로도 돌릴 수 있습니다 |
| GPU | 필수 아님. API 모델이면 CPU만으로 됨. 로컬 모델이면 이 맥의 Apple GPU(Metal, 통합 메모리 약 24 GiB)로 7–14B급 LLM·VLM이 현실적 | CUDA 전용 요소는 vLLM 예제뿐입니다 |
| 비용 | 벤치마크 하나에 방법 하나를 돌리면 텍스트 처리만 수십 달러(C8). 이미지 처리 비용 별도 | 내 문서 몇십 개면 몇 달러 수준으로 추정 |
| 데이터 | DocBench, MMLongBench-Doc는 공개 데이터셋 | 각 라이선스 확인 필요 |
| 프라이버시 | API를 쓰면 문서 전체(텍스트와 그림)가 외부 서버로 갑니다 | 개인 vault나 비공개 PDF에는 로컬 모델이 맞습니다 |

**판단:** 도구로 쓰는 것은 **가능하고 장벽이 낮습니다.** MIT 라이선스이고, 설치가 pip 하나이며, 이 맥에서 API 모델로도 로컬 모델로도 돌릴 수 있습니다. 다만 이번에는 설치와 API 호출을 하지 않았으므로(재현 보류) 실제로 도는지는 확인하지 않았습니다.

### 4.2 논문 수치를 재현하는 경우

| 걸림돌 | 내용 |
|---|---|
| 버전 | 논문 시점 코드 `806ac9b`(또는 v1.2.8)와 그 당시 LightRAG 버전을 맞춰야 합니다. LightRAG 버전은 기록이 없습니다 |
| 설정 차이 | 공개 재현 스크립트는 VLM 답 생성을 끕니다(D5). 논문이 켰는지 알 수 없습니다 |
| 없는 장치 | 모달 선호와 융합 점수(D4), 표·그림 구조 그래프(D2)는 코드에 없어서, 논문 설명대로의 시스템은 재현할 수 없습니다 |
| 판정 | 판정 버그(D8)를 고치고 정확도 전용 프롬프트를 쓸지 정해야 합니다. 판정 모델이 같으면 결과가 판정자에 따라 달라질 수 있습니다 |
| 기준선 | MMGraphRAG 결과(특히 Table 2의 불일치 행, C1)를 확인하려면 기준선도 다시 돌려야 합니다 |
| 비용·시간 | 방법 4개 × 벤치마크 2개면 텍스트 처리만 약 $100–150, VLM 처리와 재시도를 더하면 그 이상입니다. 시간은 API 속도 제한에 달려 있습니다 |

**판단:** 코드와 설정 파일이 있어서 "대략 비슷한 숫자"를 다시 내는 것은 비용만 내면 가능합니다. 하지만 논문 Sec. 2.3이 설명한 시스템 그대로 재현할 수는 없습니다. 공개 코드는 논문보다 단순한, LightRAG mix 모드 + 멀티모달 앵커 시스템입니다.

### 4.3 아이디어만 다시 구현하는 경우

| 구성 요소 | 난이도 | 메모 |
|---|---|---|
| 비텍스트 단위 → VLM 설명 + 앵커 노드 + belongs_to 간선 | 낮음 | 처리기 하나가 프롬프트 + JSON 파싱 + upsert 몇 번입니다. LightRAG 위에 수백 줄 |
| 검색 이미지의 원본 복원 + VLM 답 | 낮음 | 청크에 이미지 경로를 남기고 정규식으로 되찾는 방식이 단순하고 실용적입니다 |
| 논문이 서술했지만 코드에 없는 모달 선호·융합 점수, 표 셀 그래프 | 중간 | 직접 설계해야 합니다. 효과가 검증되지 않은 부분이라 연구 질문이 됩니다 |
| 평가 | 중간 | 판정자 편향과 파싱 실패 처리를 제대로 하려면 구조화 출력과 사람 검수 표본이 필요합니다 |

**요약:** RAG-Anything은 "멀티모달 문서를 LightRAG 그래프에 넣는 실용 도구"로서는 바로 쓸 수 있습니다. 하지만 논문의 검색 설계(모달 인식, 융합 점수, 구조 그래프)는 코드로 뒷받침되지 않습니다. 성능 주장은 표 정합성 문제(C1, C4)와 작은 차이(C3)를 감안해서 읽어야 합니다.
