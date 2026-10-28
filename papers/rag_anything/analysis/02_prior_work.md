# 02. 선행연구와 계보, 관련 GitHub

논문의 참고문헌에는 번호가 없어서 저자·연도로 적고, arXiv ID는 참고문헌 목록에 적힌 값을 옮겼습니다. 코드와 GitHub 정보는 2026-10-28에 `gh api`로 직접 확인했습니다. 그 밖의 논문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[텍스트 RAG]
  Graph RAG 계열
    GraphRAG (Edge et al., 2024) arXiv:2404.16130   커뮤니티 요약 기반 질의 중심 요약
      └─ LightRAG (Guo et al., 2024) arXiv:2410.05779   ← 같은 그룹(HKUDS). 개체·관계 그래프 + 이중 수준(low/high) 키워드 검색
    HippoRAG (NeurIPS 2024), GNN-RAG arXiv:2405.20139, RAPTOR (ICLR 2024), ArchRAG arXiv:2502.09891
        │
[멀티모달 RAG]
  MM-VID arXiv:2310.19773 (영상 → 텍스트), VideoRAG arXiv:2502.01549 (같은 그룹)
  VisRAG arXiv:2410.10594 (문서를 이미지 그대로 검색, 파싱 없음)
  MMGraphRAG arXiv:2507.20804 (장면 그래프 + 텍스트 그래프, 표·수식은 텍스트 취급)
        │
  ★ RAG-Anything (2025-10): MinerU로 파싱 → 비텍스트 단위를 VLM 설명 + 앵커 노드로 LightRAG 그래프에 삽입
```

**차별점 요약:** 구성 요소는 대부분 같은 그룹의 LightRAG에서 옵니다(개체 추출, 그래프 저장, 키워드 이중 검색, `mix` 모드). 새로운 부분은 세 가지입니다. 비텍스트 단위를 모달별 프롬프트로 설명하는 처리기, 설명문에서 뽑은 개체를 앵커로 묶는 `belongs_to` 간선, 검색된 이미지 경로를 원본 이미지로 되돌려 VLM에 넣는 답 생성입니다. 비교 대상 중 가장 가까운 것은 MMGraphRAG인데, 저자들은 MMGraphRAG가 표와 수식을 개체로 다루지 않는다는 점을 차이로 듭니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 문서 파싱 | MinerU (Wang et al., 2024) arXiv:2409.18839 | 텍스트·이미지·표·수식 추출. 코드 기본 파서이고 Docling, PaddleOCR도 선택 가능 |
| 텍스트 그래프·검색 | LightRAG (Guo et al., 2024) | 개체·관계 추출 프롬프트, 그래프·벡터 저장소, `local/global/hybrid/mix/naive` 질의 모드 |
| 그래프 RAG 원형 | GraphRAG (Edge et al., 2024) | 텍스트 그래프 구성 방식의 원형으로 인용 |
| 비교 기준선 | MMGraphRAG (Wan & Yu, 2025) | 같은 백본으로 비교 |
| 임베딩·reranker | text-embedding-3-large, bge-reranker-v2-m3 | 실험 설정 |
| 벤치마크 | DocBench (Zou et al., 2024) arXiv:2407.10701, MMLongBench-Doc (Ma et al., NeurIPS 2024) | 평가 |
| 판정 | GPT-4o-mini (부록 A.4) | 0/1 정확도 |

## 3. 이 저장소의 다른 논문과의 관계

이 저장소의 논문들(로봇 학습, 제어)과 방법상 직접 연결은 없습니다. 연결점은 연구 도구 쪽입니다.

| 대상 | 관계 |
|---|---|
| 이 저장소 자체 (`research_roadmap`) | 논문 PDF, 그림이 있는 쪽 이미지, 표 재계산 문서가 쌓이는 코퍼스입니다. "그림 몇 번이 무엇을 보여 주나", "Table 2의 어느 칸" 같은 질의를 다루려면 이 논문 같은 멀티모달 문서 RAG가 필요합니다. 다만 이 저장소의 분석 습관(표 가중 평균 검산, 문항 수 확인)을 이 논문에 적용하면 MMGraphRAG Overall 불일치 같은 문제가 바로 드러납니다(`04` C1). |
| ZeroDex (`zerodex/`) | 둘 다 VLM에 그림을 보여 주고 구조화된 출력(JSON)을 받아 쓰는 파이프라인입니다. VLM 출력 파싱이 실패할 때의 처리(이 논문 코드의 fallback 버그, `04` D8)는 두 경우 모두 결과 신뢰도에 직접 영향을 줍니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **ColPali / ColQwen** (문서 쪽 이미지를 late-interaction으로 임베딩) | 파싱 없이 쪽 이미지를 바로 검색하는 강력한 대안입니다. VisRAG와 같은 방향이고, MinerU 파싱 오류(부록 A.5의 병합 셀 실패)를 우회합니다. 비교 기준선에 없습니다. | 내 사전 지식. 서지 확인 필요 |
| **M3DocRAG** (다쪽 문서 멀티모달 RAG) | 같은 과제(MMLongBench-Doc 포함)를 다룬 직접 경쟁 방법으로 보입니다. | 내 사전 지식. 서지·수치 확인 필요 |
| **RAG** (Lewis et al., 2020) | 검색 증강 생성의 원형입니다. 논문은 서베이만 인용합니다. | 내 사전 지식 |
| **LLM-as-a-judge 편향 연구** | 생성 모델과 판정 모델이 같을 때의 자기 선호 편향은 알려진 문제입니다(`01` §4.2 5). | 내 해석 |

**한계:** 4절의 서지 정보는 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 원문을 직접 확인해야 합니다.

## 5. 관련 GitHub 저장소 (2026-10-28 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인한 커밋 |
|---|---|---|---|---|---|---|
| [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) | 이 논문의 구현. `raganything/`(처리기, 질의), `reproduce/`(색인·질의·판정 스크립트, 2026-04-13 추가) | **공식** (논문 초록에 링크) | MIT | 24,244 (fork 2,837) | 2026-10-28 | HEAD `2cd2b33`(v1.4.4), 논문 시점 `806ac9b`(2025-10-13). 가장 가까운 릴리스 v1.2.8(2025-09-22) |
| [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | 기반 라이브러리. 그래프 추출, 저장소, 질의 모드 | 같은 그룹의 공식 저장소 | MIT | 40,388 | 2026-10-27 | `aaebfe1` (shallow clone) |
| [opendatalab/MinerU](https://github.com/opendatalab/MinerU) | 기본 문서 파서 | 외부 의존성 | Apache-2.0 + 추가 조건(상업적 사용 임계값) | 82,671 | 2026-10-28 | `b7f3dd0` (clone 안 함) |
| [docling-project/docling](https://github.com/docling-project/docling) | 대체 파서 | 외부 의존성 | MIT | 69,738 | 2026-10-28 | `9051777` (clone 안 함) |
| [wanxueyao/MMGraphRAG](https://github.com/wanxueyao/MMGraphRAG) | 비교 기준선 MMGraphRAG | 논문 저자 저장소로 보임 | Apache-2.0 | 94 | 2026-03-10 | `ec0065d` (clone 안 함) |
| [OpenBMB/VisRAG](https://github.com/OpenBMB/VisRAG) | 파싱 없는 비전 기반 RAG (관련 연구) | VisRAG 공식 | Apache-2.0 | 986 | 2026-08-09 | `7205787` (clone 안 함) |
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | GraphRAG 원형 | 공식 | MIT | 36,473 | 2026-10-27 | `c600ff8` (clone 안 함) |

- 실제로 코드를 읽은 것은 RAG-Anything(HEAD와 논문 시점 파일 4개)과 LightRAG(HEAD)입니다. 나머지는 메타데이터만 확인했습니다.
- 논문 시점에는 `lightrag-hku` 버전이 고정되어 있지 않았습니다(`requirements.txt`). 지금은 `lightrag-hku<1.5`입니다. 논문 결과를 재현하려면 2025-10 당시의 LightRAG 버전을 따로 맞춰야 합니다.
- MinerU 라이선스는 현재 Apache-2.0에 상업적 사용 조건을 덧붙인 형태입니다. 연구 사용에는 문제가 없어 보이지만, 상업적으로 쓸 때는 조건을 확인해야 합니다.
