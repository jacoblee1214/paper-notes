# 02. 선행연구와 관련 자료

## 1. 계보

RT-2의 핵심은 새 구조가 아니라 **기존 VLM의 출력 어휘에 행동을 넣는 것**입니다(01 §2). 그래서 계보는 세 줄로 나뉩니다.

| 줄기 | 논문(본문 인용) | RT-2와의 관계 |
|---|---|---|
| 로봇 트랜스포머 | RT-1 (Brohan 외, arXiv:2212.06817) | 같은 데이터·같은 행동 이산화(256 구간). RT-2는 백본을 VLM으로 바꾼 것 (§3.2, 부록 C) |
| VLM 백본 | PaLI-X (Chen 외, 2023a), PaLI (2023b), PaLM-E (Driess 외, arXiv:2303.03378) | 백본. 두 모델의 혼합과 크기가 실험 변수 (부록 D) |
| 로봇에 VLM 쓰기 | SayCan (Ahn 외, arXiv:2204.01691), 계획기(Huang 2022), CLIPort(Shridhar 2021), MOO(Stone 2023) | 계획기나 의미 채널로 쓰는 방식. RT-2는 VLM을 정책 자체로 씀 (§2) |
| 표현 학습 | VC-1(Majumdar 2023a), R3M(Nair 2022b) | 기준선 (§4). 행동 텍스트 출력 없음 |
| 정책 기준선 | BC-Z(Jang 2021), LAVA(Lynch 2022) | Language-Table 표 1 기준선 |

**같은 시기의 가까운 대안(본문 밖, 일반 지식):** 저자들이 언급하지 않은 이후 연구는 이 분석의 범위가 아닙니다. 다만 후속 연구를 세야 하면 다음을 봅니다. 이 저장소에 분석이 있는 것은 §3에 적었습니다.

## 2. 공식 코드와 GitHub 자료

확인 날짜 2026-11-22. GitHub 검색은 `gh search repos`, 저장소 메타는 `gh api`로 했습니다. 저장소는 클론하지 않았습니다(공식 코드가 없어서 정적 대조 대상이 아님).

| 구분 | 주소 | 라이선스 | 별 | 마지막 push | 메모 |
|---|---|---|---|---|---|
| RT-2 공식 코드 | 없음 | - | - | - | 논문·프로젝트 페이지 어디에도 RT-2 코드 링크 없음. 검색으로도 없음 |
| 프로젝트 페이지 | https://robotics-transformer2.github.io/ | - | - | - | 데모 영상, 논문 PDF 링크 (`assets/rt2.pdf`) |
| 논문 Table 1 환경 | https://github.com/google-research/language-table | Apache-2.0 | 368 | 2026-10-07 | Language-Table 데이터와 시뮬 (Lynch 2022). RT-2 코드가 아니라 **환경과 데이터**. 프로젝트 페이지가 링크 |
| 비공식 구현 | https://github.com/kyegomez/RT-2 | MIT | 585 | 2024-07-26 | 저자 무관 "democratization" 구현. 공식 가중치나 본 논문 결과 재현이 아님. 클론하지 않음, 내용 검증 안 함 |
| 비공식 데모 | https://github.com/vengalavignesh/Vision-Language-Action-Integration-for-Robotic-Control- | 없음 | 0 | 2025-11-22 | PyBullet 8차원 토큰화 데모. 개인 데모로 보임(README 미확인). 클론하지 않음 |

**판정:** 공식 코드 없음. 그래서 `04_claim_verification.md`(주장 C1–C)로 대조했습니다. 공식 가중치도 없어서 RT-2 자체를 재현할 수 없습니다.

## 4. 구성 요소별 출처

| RT-2의 구성 요소 | 본문 출처 | 비고 |
|---|---|---|
| 행동 256 구간 이산화 | RT-1 (§3.2) | 같은 이산화 |
| 행동 토큰을 VLM 어휘에 넣기 | Symbol tuning, Wei 외 2023 (§3.2) | PaLM-E는 가장 적게 쓰는 토큰 256개 덮어쓰기 |
| "multimodal sentence" | PaLM-E, Driess 외 2023 (§1) | VLM을 로봇 입력에 쓰는 틀 |
| 공동 미세조정 | 본문 자체 주장 (§3.2, §4.3) | 웹 데이터와 로봇 데이터 혼합. 선행 근거는 본문에 없음 |
| 사고 연쇄 | Wei 외 2022 (§4.4) | 계획 토큰 추가 |
| A/B 평가 설계 | Fisher 1936 (§4.2) | 같은 조건에서 순서대로 평가. 이 설계가 분산을 줄인다는 주장은 근거를 싣지 않음 (04 C12) |
| 기준선 RT-1, VC-1, R3M, MOO | 각 논문 (§4, 부록 C) | 기준선은 같은 로봇 데이터 사용 |

## 5. 독립 확인 필요 항목

- Language-Table 환경 링크(`google-research/language-table`)는 RT-2 코드가 아니라 환경입니다. 프로젝트 페이지가 이 링크를 두어서 "RT-2 코드"로 오해하기 쉽습니다.
- arXiv v1 외에 공식 본문 수정판은 없습니다. CoRL 2023 판본은 수치가 같습니다(01 §0).
- PaLI-X의 arXiv 번호는 본문 참고문헌에 없어서 여기 적지 않았습니다.
