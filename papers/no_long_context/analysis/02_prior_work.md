# 02. 선행연구와 관련 자료

논문에는 관련 연구 절이 없습니다. 참고문헌 52개는 벤치마크, 모델 카드, 하네스, 문학 작품, SSM 원논문, Turing(1936)이 대부분입니다. 아래 §1–§2의 "논문이 인용하지 않은" 항목은 내가 알고 있는 선행연구이고, 이 저장소에서 원문을 다시 확인하지는 않았습니다.

## 1. 계보

| 흐름 | 논문이 인용한 것 | 논문이 인용하지 않은 가까운 것 |
|---|---|---|
| 긴 문맥 비용 | 어텐션 계산량이 문맥 깊이에 선형이라는 분석 [25], NVFP4 [1], 양자화·희소화 [5, 8, 14] | KV 캐시 축출·압축(StreamingLLM, H2O, SnapKV 계열). 같은 비용 문제를 추론 쪽에서 푸는 방법입니다 |
| 상수 비용 구조 | SSM 원논문 [18–20]. 각주 1에서 "상수 비용 구조는 여러 번 제안됐지만 완전히 받아들여진 적이 없다"고 씁니다 | 2026년에는 하이브리드가 주류 중 하나입니다. 논문의 로컬 모델 둘(Qwen 3.6 27B, Qwen 3 Coder Next)이 Gated DeltaNet 하이브리드이고, NVIDIA 자신의 Nemotron 3.5도 Mamba 하이브리드입니다(§5) |
| 긴 문맥 평가 | NIAH [24], RULER [22], LongBench v2 [6], LOONG [45], BABILong [28], Counting-Stars [39], AA-LCR [4], Lost in the Middle [29] | — |
| 에이전트 하네스 | ReAct [49], SWE-bench [23], SWE-agent·mini-swe-agent [41, 47], SWE-bench Multilingual [48], WebArena [52], FRAMES [27] | SWE-agent의 history processor(오래된 관찰을 접어 둠), OpenHands의 condenser 같은 하네스 쪽 문맥 관리 |
| 에이전트 메모리·압축 | 없음 | MemGPT(Packer et al., 2023, arXiv:2310.08560) → Letta. 상용 코딩 에이전트의 자동 압축(Claude Code의 `/compact`와 auto-compact 등) |
| 긴 문서 읽기 에이전트 | 없음 | ReadAgent(Lee et al., 2024, arXiv:2402.09727): 문서를 쪽으로 나눠 gist 메모리를 만들고 필요한 쪽만 다시 읽습니다. RQ2의 "선택 읽기 + 압축"과 가장 가깝습니다 |
| 검색 증강 | 없음 | RAG 일반, 이 저장소의 RAG-Anything(§5) |

**정리:** 실험 설계(EoCC, 선택 읽기)는 이미 널리 쓰이는 하네스 기법입니다. 이 논문이 새로 보탠 것은 세 가지입니다. (1) 닫힌 최신 모델 둘로 13개 벤치마크에 걸쳐 같은 방법을 체계적으로 잰 것, (2) 처리량 측정으로 "짧은 문맥의 비용 이점"을 정량화하려 한 것, (3) SAC와 SSSM이라는 극단으로 밀어 본 것입니다.

## 2. 가까운 대안과 빠진 기준선

| 대안 | 무엇을 대신하나 | 논문에서 |
|---|---|---|
| 관찰 마스킹·자르기(LLM 호출 없음) | EoCC의 요약 호출 | 부록 A가 "message pruning"을 탐색했다고만 쓰고 수치가 없습니다 |
| RAG(임베딩 검색) | RQ2의 grep 선택 읽기 | 비교하지 않습니다. 기준선은 "문서를 통째로 붙여 넣기" 하나입니다 |
| 같은 길이 예산으로 자른 붙여 넣기 | RQ2 기준선 | 기준선은 문서를 작업 대화에 통째로 붙여 넣습니다(부록 F). 비교 대상이 "붙여 넣기 대 파일 + grep"이라서, 차이가 짧은 문맥 덕인지 정확한 문자열 검색 덕인지 나눌 수 없습니다(`04` C11) |
| 하이브리드·선형 어텐션 구조 | "짧은 문맥이 싸다"는 전제 | 구조에 따라 비용 비율이 크게 달라진다는 점을 다루지 않습니다(`03` §1) |
| KV 축출·압축 | 문맥 길이 자체를 줄이는 것 | 다루지 않습니다 |

## 3. 구성 요소별 출처

| 구성 요소 | 논문 위치 | 출처 |
|---|---|---|
| FHA 루프 | §2.4, 부록 G | mini-swe-agent v2.4 [41], ReAct [49] |
| EoCC | §2.4.1, Fig. 3 | 하네스 관행. 한도/임계값 두 값과 대화 재시작 |
| 압축 프롬프트 6항목 | 부록 A, G.1 | 저자 설계. 자리표시자 내용은 비공개 |
| mini-agent | 부록 G.2 | mini-swe-agent에서 SWE 전용 지시를 뺀 일반화 |
| 선택 읽기 | §2.4.2, 부록 E | sed·awk·head·grep. 문서를 파일로 둠 |
| SAC | §2.4.1, §5 | 저자 설계 |
| 처리량 → 비용 | §2.5, Fig. 1·4, Eq. 1 | vLLM, H100, $3.85/GPU-h(Nebius, 2026-09) |
| EORC | 부록 B, Eq. 2–4 | 저자 설계. r = 0.20, 선형 보간 |
| SSSM | §5.1 | 개념 제안 |

## 4. GitHub와 공개 자료 (2026-11-09 확인)

공식 코드는 없습니다. 연구실 페이지 두 곳은 arXiv 링크만 겁니다.

| 저장소 | 역할 | 라이선스 | ★ | 마지막 push | 커밋 |
|---|---|---|---|---|---|
| (공식 코드) | — | — | — | — | 없음. nv-dler.github.io, research.nvidia.com/labs/lpr/lfi/ 모두 arXiv 링크만 |
| [che-shr-cat/short-context-agent](https://github.com/che-shr-cat/short-context-agent) | 비공식 재현. Claude skill + EoCC/SAC 참조 하네스. "방법 재현이지 결과 재현이 아님"이라고 README에 명시 | MIT | 3 | 2026-11-05 | `bf534c4` (shallow clone, 정적 대조만) |
| [Macro-Harder/context-compaction](https://github.com/Macro-Harder/context-compaction) | 비공식 구현 | Apache-2.0 | 0 | 2026-11-09 | clone 안 함 |
| [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | 논문이 쓴 하네스(v2.4). 태그 v2.4.4–v2.4.6 존재 | MIT | 9,430 | 2026-11-02 | HEAD `04d809c` (clone 안 함) |
| [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | SWE-bench Verified·Multilingual 평가 | MIT | 6,162 | 2026-09-18 | — |
| [harbor-framework/terminal-bench-1](https://github.com/harbor-framework/terminal-bench-1) | Terminal-Bench (laude-institute/terminal-bench에서 이동) | Apache-2.0 | 2,615 | 2026-07-11 | — |
| [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | τ²-bench | MIT | 2,419 | 2026-11-05 | HEAD `033bf08` |
| [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) | BFCL | Apache-2.0 | 13,065 | 2026-10-17 | — |
| [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | RULER | Apache-2.0 | 1,631 | 2026-11-03 | HEAD `db43f92` |
| [gkamradt/needle-in-a-haystack](https://github.com/gkamradt/needle-in-a-haystack) | NIAH | 표기 없음(NOASSERTION) | 2,406 | 2026-06-08 | — |
| [THUDM/LongBench](https://github.com/THUDM/LongBench) | LongBench v2 | MIT | 1,254 | 2025-01-15 | — |
| [MozerWang/Loong](https://github.com/MozerWang/Loong) | LOONG | Apache-2.0 | 154 | 2025-12-22 | — |
| [booydar/babilong](https://github.com/booydar/babilong) | BABILong | Apache-2.0 | 259 | 2026-06-01 | — |
| [nick7nlp/Counting-Stars](https://github.com/nick7nlp/Counting-Stars) | Counting-Stars | MIT | 83 | 2025-11-24 | — |
| [letta-ai/letta](https://github.com/letta-ai/letta) | MemGPT 계열 에이전트 메모리(인용 안 됨) | Apache-2.0 | 25,388 | 2026-10-18 | — |

AA-LCR은 Hugging Face 데이터셋(ArtificialAnalysis/AA-LCR)입니다. 같은 연구실의 다른 작업으로 AgentBaton이 연구실 페이지에 올라 있습니다.

## 5. 이 저장소의 다른 논문과의 관계

연결이 강한 순서로 적었습니다. 억지로 잇지 않으려고 약한 연결은 약하다고 적었습니다.

| 폴더 | 관계 | 강도 |
|---|---|---|
| `nemotron35_super_vl/` | 이 논문의 비용 모델을 검증하는 데 직접 썼습니다. Super VL은 Mamba 40층 + attention 8층이라, 요청당 SSM 상태(FP16 80 MiB + conv 2.34 MiB)가 FP8 KV 약 2만 토큰 분량입니다. 그래서 같은 측정 규약에서 256K 대 16K 비용 비율의 메모리 한계가 3.4–5.1배로, 순수 어텐션(약 15배)보다 훨씬 작습니다(`03` §1). 같은 NVIDIA에서 나온 두 결과가 "짧은 문맥이 얼마나 싼가"에 다른 답을 줍니다 | 강함 |
| `nemotron35_bolt/` | 로컬 장치 쪽 근거입니다. Bolt 분석은 "Mamba 층 덕분에 256K에서도 KV(F16)가 2.15 GB이고, 메모리 한계는 문맥이 아니라 가중치가 정한다"고 결론 냈습니다. Fig. 2의 "뒤쪽 토큰이 약 2배 느리다"는 활성 가중치가 큰 하이브리드 MoE에는 맞지 않습니다(Super VL 대역폭 모델에서 256K에 1.10배). 반면 활성 3B인 하이브리드 Qwen 3 Coder Next는 Fig. 2에서 약 1.9배라, 구조 이름보다 활성 가중치와 KV 읽기의 비율이 중요합니다(`03` §1.4) | 강함 |
| `rag_anything/` | RQ2의 대안입니다. 이 논문은 grep·sed로 필요한 부분만 읽고, RAG-Anything은 지식 그래프와 임베딩 색인으로 찾습니다. 이 논문은 RAG 기준선과 비교하지 않았습니다. 논문 코퍼스 검색 도구로서는 색인 없이 grep만으로 어디까지 되는지가 실용적 질문입니다 | 중간 |
| `agentic_wam/` | 이력을 압축하는 방식의 대비입니다. AW-0 논문은 이전 코드 조각을 학습형 K개 슬롯으로 압축한다고 쓰지만, 공개 설정은 그 메모리를 켜지 않고 이전 조각을 텍스트(`# Completed:`)로 넣거나 마지막 조각만 넣습니다(`agentic_wam/analysis/04_code_verification.md` D2). 실제 공개본이 쓰는 방식은 이 논문의 자연어 상태(SSSM)에 더 가깝습니다. 영역(로봇 정책 대 LLM 에이전트)은 다릅니다 | 중간 |
| `weight_space_programming/` | 블로그의 실험 8에서 MATH-500 답의 93%가 32,768토큰 상한에 걸렸고(반복), 블로그도 "문맥 길이와 하네스 비효율"을 한계로 꼽습니다. 장기 에이전트 실행에서 문맥이 병목이라는 사례입니다. 압축으로 풀리는 문제인지는 확인할 수 없습니다 | 약함 |
| `chronospolicy/` | 로봇 정책은 원래 짧은 관측 이력(카메라당 영상 2장 + 자세 2개)으로 동작합니다. 이 논문의 주장이 로봇 정책 쪽에서는 새롭지 않다는 정도의 연결입니다 | 약함 |
| `openwam/` | OpenWAM은 현재 관측 프레임과 지시문만 조건으로 씁니다. 긴 이력 문제가 평가 설정에 나오지 않아 의미 있는 연결이 없습니다 | 없음 |
