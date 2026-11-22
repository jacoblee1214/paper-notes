# 02. 선행연구, 구성 요소의 출처, GitHub 자료

확인일 2026-11-22. 논문 v1 본문과 참고문헌(Section 1, 2.2, [1]–[20])을 기준으로 계보를 정리했습니다.

## 1. 계보: "sink"라는 이름이 가리키는 것

| 축 | 논문 | 무엇을 주장하나 | 이 논문과의 관계 |
|---|---|---|---|
| 언어모델 sink 정의 | Xiao et al., StreamingLLM, ICLR 2024 [19] | 초기 토큰이 attention을 많이 받는 현상 | 이름의 출처 |
| sink 특성화 | Gu et al., ICLR 2025 [8] | key 쪽 bias 하나, value ℓ2가 작음, 출력 기여가 작음. 첫 토큰이 30% 이상을 받는 head가 90% 이상 | 논문이 검증하는 세 번째 속성(작은 value와 출력 기여)의 기준 |
| 위치 기반 설명 | Barbero et al., COLM 2025 [1] | 첫 토큰에 attention을 쏟으면 head가 거의 쉬어 과혼합을 막음 (approximate no-op) | prefix `<bos>`가 이 역할에 가깝다는 해석의 근거 |
| 위치 인코딩 | Su et al., RoFormer [15]; Chen et al., HoPE [5] | RoPE의 회전이 상대 위치를 인코딩. 학습된 성분이 전역 U자 패턴을 만들 수 있음 | 끝점 질량을 RoPE로 설명하는 근거 |
| 고정 토큰 역할 | Darcet et al., ICLR 2024 [7]; Sun et al., COLM 2024 [16] | 레지스터 토큰, massive activation | 논문의 `<bos>` residual-stream 크기 검사 (1,049–3,784배) |
| 확산 트랜스포머 sink | Wu and Summa, ICML 2026 [18] | 이미지 확산 트랜스포머에서 sink score와 value 편집이 생성물을 바꿈 | 양방향 attention에서 sink 개념을 옮기는 선례 |
| 행동 토큰 horizon 앵커 | Wang et al., "VLA knows its limits", ECCV 2026 [17] | 첫 행동 토큰과 마지막 행동 토큰이 높은 attention을 받아 "radial action sink"로 불림. 이를 실행 horizon 설정에 씀 | 논문이 **검증하지 않는 규칙**의 원출처. 이 논문의 `arXiv:2602.21445` |
| 행동 청크 생성 | Chi et al. [6]; Zhao et al. [20] | 확산 정책과 행동 청크 | 배경 |
| 중간 개입 선례 | Häon et al. [9]; Kim et al. OpenVLA [10]; Brohan et al. RT-2 [4] | 활성값 방향을 조작하는 해석 연구 | 개입 방식의 배경 |
| 정책 구조 | Black et al., π0 [3] (RSS 2025); π0.5 [2] (CoRL 2025) | flow 기반 VLA, action expert 18층, prefix에 `<bos>` | 측정 대상 아키텍처 |
| 비교 모델 | Shukor et al., SmolVLA [14] | 인과 행동 attention, BOS 없음 | Appendix G 프로브 대상 |
| 시뮬레이션 | Nasiriany et al., RoboCasa [11] (RSS 2024); 체크포인트 RoboCasa365 [13] | 과제, 공개 π0.5 체크포인트 (`pi05_pretrain_human300`, step 75,000) | 측정 환경 |

## 2. 가장 가까운 대안 (논문이 명시적으로 비교한 것)

- **Wang et al. [17] (AutoHorizon 계열)**: 끝점 attention을 horizon 앵커로 사용. 논문은 horizon 규칙을 평가하지 않고, 끝점의 역할 설명만 다룸 (Section 1, 2.2).
- **Gu et al. [8]의 sink 프로파일**: 기준선. 논문은 끝점이 이 프로파일의 value 조건을 만족하지 않는다고 보고 (표 3).
- **Barbero et al. [1]의 no-op 해석**: prefix `<bos>`에 적용. 논문은 `<bos>` value를 0으로 하면 성공이 오른다는 결과로 이 해석과 긴장을 보임 (Section 5.4, 7).

## 4. GitHub와 코드 자료

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 논문 공식 코드 | 없음 | - | - | - | arXiv abs와 HTML 본문에 코드·데이터 링크 없음. `gh search repos` 3개 질의(제목, "radial action sink", "sink tokens action expert")와 GitHub 코드 검색 0건. 웹 검색 0건. `repo/` 클론 안 함 |
| 논문 프로젝트 페이지 | 없음 | - | - | - | abs와 HTML에 프로젝트 링크 없음 |
| 선행 AutoHorizon 코드 (Wang et al.) | https://github.com/hatchetProject/AutoHorizon | Apache-2.0 | 34 | 2026-03-17 | ECCV 2026 표기. 이 논문의 공식 코드가 아님. 클론 안 함 |
| 선행 AutoHorizon 프로젝트 페이지 | https://hatchetproject.github.io/autohorizon/ | - | - | - | 페이지 자체의 정적 확인만 |
| 측정 체크포인트 (RoboCasa365) | https://huggingface.co/robocasa/robocasa365_checkpoints (`pi05_pretrain_human300/multitask_learning/75000`) | 미확인 | - | 미확인 | 내려받지 않음. 크기 미확인 |

- 공식 코드가 없으므로 `04_code_verification.md` 대신 `04_claim_verification.md`(주장 C1–Cn)를 씁니다.
- 논문이 올라온 지 3일(2026-11-19 제출)이라 코드 공개를 기다릴 수 있습니다. 저자 저장소가 나오면 그때 정적 대조를 다시 할 수 있습니다.

## 5. 학회 표기 확인

- arXiv abs의 comments에는 "17 pages"만 있고 학회 표기나 journal-ref가 없음.
- 논문 본문에 학회 표기나 "submitted to" 문구 없음.
- `conferences/corl2026/papers.json`(755편)에서 제목 "Beyond the Sink Interpretation"과 arXiv 번호 2611.26385 모두 없음.
- `conferences/iros2026/`에도 없음.
- 결론: **venue는 "학회 표기 없음 (arXiv)"**.
