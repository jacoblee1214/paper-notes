# 02. 선행연구와 자료 탐색: TacBridge (arXiv:2611.20939)

## 1. 학회 표기 확인

| 확인 경로 | 결과 |
|---|---|
| arXiv abs (v1) | 댓글(Comments), 저널 참조(Journal reference) 없음. Subjects: cs.RO |
| 논문 본문 (8쪽) | 학회 표기 없음. 첫 쪽 각주는 소속과 Nvidia Academic Grant만 씀 |
| 웹 검색 (제목 정확 일치) | 결과 없음. 가장 가까운 항목은 TLV-CoRe(arXiv:2511.11512), CTSRL(ICML 2026 PMLR v306), UniTac-NV(arXiv:2506.19699), CTTP(arXiv:2410.11834)로, 모두 제목이 다릅니다 |
| CoRL 2026 공식 목록 `conferences/corl2026/papers.json` | 제목·arXiv 번호로 검색해 일치 항목 없음 |

**판정:** 학회 표기 없음 → `venue: "학회 표기 없음 (arXiv)"`. 제출 5일 뒤 분석이라 채택 여부를 알 수 없습니다.

## 2. 코드와 프로젝트 페이지

| 항목 | 확인 결과 | 상태 |
|---|---|---|
| 공식 코드 저장소 | arXiv Code 탭(CatalyzeX/alphaXiv 토글)에 연결 없음. GitHub 저장소 검색(`TacBridge`, `tacbridge tactile`) 0건. 웹 검색에서도 저자 이름과 함께 나온 저장소 없음 | 없음 |
| 프로젝트 페이지 | 논문에 URL 없음. 본문 HTML(`arxiv.org/html/2611.20939v1`)에도 프로젝트 링크 없음 | 없음 |
| 데이터, 체크포인트 | 논문에 공개 계획 문구 없음 | 없음 |
| 관련 저장소 | TacBridge 이름의 저장소 없음. 같은 저자(Guanqun Cao)의 촉각 연구 저장소는 이 분석 범위에서 확인하지 않았습니다 | — |

코드와 프로젝트 페이지가 없으므로 `repo/`는 만들지 않았고 (clone 없음), 검증은 `04_claim_verification.md` (C1–C14)로 했습니다.

## 3. 선행연구 계보 (논문 참고문헌 기준)

| 계열 | 논문 인용 | 이 논문과의 관계 |
|---|---|---|
| 촉각 표현 학습, 센서 불변 | Gupta 외 (ICLR 2025, Sensor-invariant tactile representation) [7], AnyTouch [8], Zhao 외 (CoRL 2025, Transferable tactile transformers) [11] | 가장 가까운 대안. AnyTouch는 Table I–III의 주 비교 대상 |
| 멀티모달 촉각 정렬 | Yang 외 (CVPR 2024, Binding touch to everything) [10], Sparsh [9] | 비교 기준선은 Sparsh의 MAE(S)와 V-JEPA(S) |
| 마스크 촉각 재구성 | Cao 외 (IROS 2023) [12], VideoMAE [16], MAE [26] | 단일 센서 MAE 계열. 구조의 출처 |
| 교차 센서 번역 | Touch2Touch와 T2D2 [17] (Rodriguez 외, CoRL 2025), Grella 외 (2024) [18] | 번역 과제 자체의 선행. 논문 §II-B(115–117행)는 Touch2Touch와 T2D2를 둘 다 [17]로 인용하는데, 두 방법이 한 문헌인지 원문이 명확하지 않습니다. 저자들은 T2D2를 "교차 센서 쌍 없이 깊이를 매개로 번역"으로 소개합니다 |
| 힘 평형 기반 잠재 공간 | UniForce [19] | 쌍 재구성과 힘 평형 제약. 저자들은 힘 제약을 쓰지 않는다고 명시 |
| 촉각 언어 통합 | UniTac [20], OCTOPI [34] | 다른 축(언어)이라 직접 비교하지 않음 |
| 시뮬 촉각 | Taxim [21], TACTO [22], TacSL [23], Tactile Gym [24] | ToucHD-Sim의 배경. 저자는 "시뮬 쌍으로 교차 센서 대응을 학습"한다고 씀 |
| 흐름 매칭 | Lipman 외 (ICLR 2023) [27] | 흐름 손실 식 (10)–(11)의 출처 |
| 데이터셋 | Touch and Go [28], VisGel [29], ObjectFolder Real [30], TVL [31], YCB-Slide [32], SSVTP [33], TouchSlide [9], TacQuad [8], ToucHD [13] | 사전학습 코퍼스. 센서 종류와 쌍 여부가 표마다 다름 |

**저자가 주장하는 차이:** 기존 멀티센서 방법은 "센서가 달라도 같은 특징 공간에 넣는다"는 수준이고, 이 논문은 **같은 접촉 사건의 쌍을 손실로 직접 연결한다**는 점입니다 (Introduction 2쪽, §III-B).

## 5. 검색 방법과 한계

- 웹 검색 두 번(제목, 저자와 "code"), GitHub 저장소 검색(`gh search repos`와 API), CoRL 2026 목록 확인, arXiv abs 페이지와 HTML의 링크 확인.
- 학회 공식 목록(RSS, PMLR, ICRA/IROS)은 제목 검색으로 확인하지 못했습니다. 2026-11-16 제출이라 아직 색인되지 않았을 가능성이 큽니다.
- 같은 저자의 다른 저장소나 프로젝트 페이지는 찾지 못했습니다. 저자가 공개 계획을 알리면 이 파일을 갱신합니다.
