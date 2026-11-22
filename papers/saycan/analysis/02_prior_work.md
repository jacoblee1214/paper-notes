# 02. 선행연구와 관련 자료

## 1. 계보

SayCan의 새 요소는 두 가지입니다(01 §1–2). 하나는 LLM 점수(log-likelihood)를 **고정된 선택지에 대한 점수**로 쓰는 것이고, 다른 하나는 그 점수에 **가치함수를 곱하는 것**입니다. 나머지는 기존 부품입니다.

| 구성 요소 | 출처 (논문 본문 인용) | SayCan에서의 역할 |
|---|---|---|
| LLM | PaLM 540B (Chowdhery 외, 2022), FLAN 137B (Wei 외, 2021), GPT-3 (Brown 외, 2020) | 지시를 기술 순서로 바꾸는 "Say" 부분 |
| 가치함수를 affordance로 쓰는 생각 | Value function spaces (Shah 외, ICLR 2022, arXiv:2111.03189) | 가치함수 공간에서 기술 가용성 읽기 |
| affordance 개념 | Gibson (1977) | 개념 출처 |
| 언어 조건 BC | BC-Z (Jang 외, CoRL 2021) | 피킹 정책 구조 |
| 언어 조건 RL | MT-Opt (Kalashnikov 외, arXiv 2021) | 가치함수 학습 |
| 문장 임베딩 | Universal Sentence Encoder (Cer 외, 2018) | 기술 설명을 입력으로 |
| sim-to-real | RetinaGAN (Ho 외, ICRA 2021) | 시뮬레이션 이미지 보정 |
| 사고 연쇄 | Chain of thought (Wei 외, 2022) | 부정 처리 개선(§5.2) |
| 오픈소스 노트북의 정책 | CLIPort (Shridhar 외, CoRL 2022), ViLD (Gu 외, 2021) | 노트북의 객체 탐지와 피킹·배치 |
| 같은 시기의 LLM 계획 | Huang 외, "Language models as zero-shot planners" (arXiv:2201.07207, 2022) | 동시 연구. 점수화 없이 생성으로 계획 (본문이 "Generative" 기준선과 대응된다고 적음, §5.1) |
| 폐루프 후속 | Huang 외, "Inner Monologue" (arXiv:2207.05608, 2022) | 실패 감지와 피드백으로 재계획 (본문 §5.2) |

**같은 계열 (본문 밖, 일반 지식):** 작업 계획과 운동 계획의 결합(TAMP), 계층 강화학습의 옵션, 기술 가치로 계획하는 방법들이 있습니다. 이 분석은 문헌 조사를 하지 않았고, 위 목록만 논문 본문과 공식 자료로 확인했습니다.

## 2. 공식 코드와 GitHub 자료

확인 날짜 2026-11-22. GitHub 메타데이터는 공개 REST API로 확인했습니다. 저장소는 `saycan/repo/`에 **얕은 희소 복제(sparse clone, `saycan/` 폴더만)**로 받았고, 실행하거나 설치하지 않았습니다.

| 구분 | 주소 | 라이선스 | 별 | 마지막 push | 메모 |
|---|---|---|---|---|---|
| **공식 코드 (노트북)** | https://github.com/google-research/google-research/tree/master/saycan | Apache-2.0 (노트북 SPDX 헤더, 상위 monorepo) | monorepo 39,253 | monorepo 2026-11-21 · `saycan/` 최근 커밋 2026-04-28 (a0080d3) | 테이블탑 UR5e 시뮬, GPT-3, ViLD, CLIPort 변형. 논문 PaLM 실험 아님 |
| 프로젝트 페이지 | https://say-can.github.io/ | — | — | — | 영상, 논문 링크, 노트북 링크. 헤드라인 PaLM 실험 자료는 없음 |
| 프로젝트 저장소 | https://github.com/say-can/say-can.github.io | 없음 | 17 | 2022-12-14 | 웹사이트 정적 파일과 `data/` 폴더(저장소 크기 약 415 MB, 데이터 미다운로드) |
| 비공식 구현 | https://github.com/kyegomez/SayCan | MIT | 26 | 2026-10-26 | 저자 무관 재구현 (설명: "Implementation of … by Google"). 검증하지 않음 |
| 후속 공식 코드 | https://github.com/RishiHazra/saycanpay | 없음 | 56 | 2025-10-22 | SayCanPay (AAAI 2024). 같은 계열의 후속 연구이지 이 논문의 코드가 아님 |

- **판정:** 공식 코드가 있습니다. 다만 **논문의 PaLM 실험 코드가 아니라 공개 노트북**입니다. 그래서 `04_code_verification.md`(D1–D17)로 대조했습니다.
- 노트북 커밋 이력: `8c56e5d`(2022-08-22, "SayCan minor updates…"), `6dedab7`(2025-02-06, JAX 경고 수정), `a0080d3`(2026-04-28, gsutil→gcloud). 최신은 `a0080d3`입니다.
- 노트북 실행에는 OpenAI API 키(유료)와 Google Drive·GCS에서 받는 체크포인트·데이터가 필요합니다. 이 분석에서는 받지 않았습니다.

## 4. 분석 범위 밖

- 최신 LLM 기반 계획기(예: 도구 사용 에이전트)와의 직접 비교는 하지 않았습니다.
- 이후 연구의 인용 수나 후속 채택은 확인하지 않았습니다.
