# 02. 선행연구, 코드, 학회, 라이선스

## 1. 선행연구 계보

| 축 | 대표 연구 (논문 참고문헌 번호) | 이 논문과의 관계 |
|---|---|---|
| 흐름/확산 정책 | Diffusion Policy [1], π0 [2], π0.5 [3], 흐름 매칭 [22] | 대상 방법 |
| 세계 행동 모델(WAM) | Fast-WAM [4], World action models are zero-shot policies [5] | 실험 대상 (Fast-WAM 6B) |
| 스텝 수 감소 | 일관성 정책 [6], One-step diffusion policy [7], Flash-WAM [8], Few-step WAM 블로그 [9] | "스텝을 줄이면 된다"는 비교 기준. [9]이 2스텝 95.8%를 보고 |
| 확산 캐시(궤적 내) | DeepCache [10], Block caching [11], TeaCache 계열 [12], MagCache [13], TaylorSeers [14] | 같은 계보의 일반 캐시 |
| 청크 간 캐시 | Falcon [17], Streaming DP [18], **C³ache [21]** | C³ache가 직접 기준선 |
| 제어 주기 간 캐시 | VLA-Cache [20], **X-Cache [19]** | X-Cache는 잔차를 청크 사이에 재사용하며 노이즈 재샘플 문제를 언급 |
| 비동기 실행 | Real-time execution of action chunking [27] | 지연 실험의 이론 배경 |

**핵심 차이.** 선행 캐시 논문들은 캐시가 이득이라는 전제에서 출발합니다. 이 논문은 "2스텝 무캐시"라는 더 단순한 기준선을 먼저 세웁니다. 캐시는 2스텝 호출 비용을 낮출 수 없고(모든 호출이 이미지 인코딩과 비디오 전문가를 다시 돌립니다), 그래서 이 논문의 주장은 "캐시 비율"과 "샘플러" 쪽으로 옮겨 갑니다.

## 2. 가장 가까운 대안

- **C³ache [21] (arXiv:2606.08962)**: 저자들이 재현 대상으로 삼은 방법. 이 논문의 캐시는 C³ache의 "전체 스택 잔차"를 그대로 옮긴 것입니다(Sec. 2, A.4 등). C³ache 공식 코드의 존재는 확인하지 않았습니다.
- **X-Cache [19] (arXiv:2604.20289)**: 자기회귀 세계 모델의 청크 간 블록 캐시. 노이즈 재샘플 문제를 같은 식으로 짚었다고 논문이 인용합니다.
- **Yang [9] 블로그**: 2스텝 무증류 기준선의 출처 (95.8%). 이 논문의 Table 1 수치와 직접 맞춰 봅니다(A.7).

## 4. 학회와 공식 목록

- arXiv 댓글: "Honorable Mention at the CoRL 2026 Workshop on Efficient Foundation Models for Real-Time Embodied AI (EEAI)." 저장소 README도 "CoRL 2026 EEAI Workshop"이라고 씁니다.
- `conferences/corl2026/papers.json` (755항목): 제목으로 찾으면 해당 논문이 없습니다. 이 목록은 본회의 채택 목록입니다.
- 웹 검색(2026-11-22): EEAI 워크숍 채택 목록이나 수상 목록을 찾지 못했습니다.
- 판정: 1쪽 머리말과 arXiv 댓글이 모두 워크숍을 밝히므로 **"CoRL 2026 워크숍"으로 기록합니다.** 본 트랙 채택이 아니라는 점을 venue_note에 적습니다(이 저장소의 다른 워크숍 논문과 같은 표기, 메인 세션 교정 2026-11-22).

## 5. 공식 코드, 프로젝트 페이지, 관련 저장소

확인 날짜 2026-11-22. GitHub API 조회(메타데이터만).

| 구분 | 이름과 링크 | 라이선스 | 별 | 마지막 push | 커밋 | 조치 |
|---|---|---|---|---|---|---|
| 공식 코드 | [gabhinav001/cache-or-denoise-less](https://github.com/gabhinav001/cache-or-denoise-less) | **없음** (API `license: null`) | 0 | 2026-11-08 23:58 UTC (생성 2026-09-09) | 커밋 1개 `71867934ea` "initial commit" (2026-11-08) | 희소 체크아웃(blobless)으로 `repo/`에 복제. 코드·README·repro 스크립트·로그·요약 JSON만 받음. `inputs/`(73 MB 번들, 가중치 JSON)와 `results/`의 raw 파일(수십 MB)은 받지 않음. 정적 읽기만 함 |
| 프로젝트 페이지 | 없음 | - | - | - | - | arXiv 댓글과 README에서 찾지 못함 |
| C³ache 코드 | 미확인 | - | - | - | - | 검색하지 않음 (범위 밖) |

**라이선스 함의.** 저장소에 라이선스가 없으므로 코드는 기본적으로 저작권이 유보됩니다. 우리는 코드를 재배포하지 않고, 정적 읽기로 대조한 결과만 보고합니다. 실행하거나 설치하지 않았습니다.

## 6. 권리 판단 (논문 그림과 공개 범위)

- arXiv abs 페이지의 라이선스 링크: `http://creativecommons.org/licenses/by/4.0/` ("view license"). 확인 날짜 2026-11-22. **CC BY 4.0**입니다.
- PDF 전체 텍스트에서 `©`, `copyright`, `All rights reserved`, 회사 고지를 찾았으나 없었습니다. 1쪽 머리에 "CoRL 2026 Workshop ... (EEAI)"만 있고, 이는 저작권 고지가 아닙니다. `pdfimages -list`에는 래스터 그림이 없어 그림은 벡터입니다.
- **판정: CC BY 4.0이고 상충하는 고지가 없으므로 논문 그림을 공개물에 넣는 것은 허용됩니다.** 다만 이번 게시본은 **우리가 그린 그래프만** 씁니다(`analysis/plot_own.py`의 두 장). 출처 표기 부담을 피하고, 그림 속 수치가 우리 재계산과 같은 표에서 나오도록 하기 위해서입니다.
- 공개 범위 규칙: 개인용 연구 적용 문서는 공개물에 넣지 않습니다. 게시본은 `publish/`와 `report/report_public.pdf`만 만듭니다.

## 7. 재현 가능성 (요약)

- 저장소 README: 전체 GPU 재현은 **48 GB 이상 VRAM과 CUDA 12.8, MuJoCo 3.3.2**가 필요하다고 합니다. 빠른 스모크 테스트도 A6000에서 약 1.7시간입니다.
- 이 맥(M4 Pro, 24 GiB 통합 메모리, CUDA 없음): 6B 모델 롤아웃은 권장 사양에 못 미치고, LIBERO 시뮬레이션과 MPS 호환성은 확인되지 않았습니다. 따라서 **GPU 재현은 불가**로 판정합니다.
- CPU로 가능한 것은 이미 한 재계산(03), 식 (1)과 비용 모델 확인(03), 정적 코드 대조(04)입니다.
