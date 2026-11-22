# LIBERO 선행연구, 공식 코드, 관련 분석

## 0. 학회 확인과 판본

- **학회.** NeurIPS 2023, Datasets and Benchmarks 트랙. 근거는 NeurIPS 공식 포스터 페이지(neurips.cc/virtual/2023/poster/73415, 발표 목록)입니다. 논문 v2 첫 페이지의 "37th NeurIPS 2023 Track on Datasets and Benchmarks"와 맞습니다. 논문 제목은 학회 기록과 arXiv에서 "Benchmarking Knowledge Transfer for Lifelong Robot Learning"입니다. 한 곳(Peter Stone 공개 BibTeX)은 "in Lifelong Robot Learning"으로 씁니다. 인용할 때 제목을 확인해야 합니다.
- **판본.** v1(2023-06-05, "LIBERO: A Benchmark for Lifelong Robot Learning", 심사 중 표기)과 v2(2023-10-14, 학회 판형)가 있습니다. 분석 기준은 v2입니다(`01` §0).
- **arXiv 라이선스.** abs 페이지 "view license" 링크는 `creativecommons.org/licenses/by/4.0/`(CC BY 4.0)입니다. PDF 본문에서 ©, All rights reserved, 출판사 고지를 찾지 못했습니다.

## 1. 선행연구 계보

### 1.1 평생학습 벤치마크

| 계열 | 대표 연구 | LIBERO와의 차이 (논문 §6) |
|---|---|---|
| 이미지·텍스트 분류 | MNIST, CIFAR, ImageNet, Core50, GLUE, SuperGLUE | 절차적 지식을 다루지 않음 |
| 게임 강화학습 | Atari, XLand, ViZDoom | 인간 활동을 대표하지 않음 |
| 로봇 연속 강화학습 | ContinualWorld (Wołczyk 외 2021, [69]) | MetaWorld 50 과제 기반. 같은 계보의 강화학습 벤치마크 |
| 연속 강화학습 벤치마크 | CORA (Powers 외 2021, [51]) | Atari, Procgen, MiniHack, ALFRED 기반 |
| 실로봇 객체 학습 | F-SIOL-310 (Ayub 외, [3]), OpenLORIS (She 외, [61]) | 실제 장면 촬영, 조작 과제 없음 |

### 1.2 로봇 학습 벤치마크

| 계열 | 대표 연구 | 관계 |
|---|---|---|
| 메타학습·다과제 | MetaWorld (Yu 외 2020, [73]), RLBench (James 외, [27]) | 과제 집합의 다양성 |
| 일반화 조작 | ManiSkill (Mu 외, [47]), ManiSkill2 (Gu 외, [24]) | 객체 일반화 |
| 가정 활동 | BEHAVIOR (Srivastava 외, [63]), BEHAVIOR-1K (Li 외, [35]) | LIBERO의 과제 발상(Ego4D 템플릿)과 맞닿음. 논문은 직접 비교하지 않음 |
| 합성 과제 | CompoSuite (Mendez 외, [44]) | 구성 과제 |

**기반 시뮬레이터.** robosuite (Zhu 외 2020, [76]) 위에 세웠습니다. 생성 파이프라인은 PDDL로 장면과 목표를 씁니다(§4.1, 부록 C.2).

### 1.3 정책과 평생학습 알고리즘의 출처

- 정책 구조: ResNet-RNN(ResNet + LSTM, FiLM 언어 결합, [42] Mandlekar 외 2021, robomimic), ResNet-T(VIOLA [75] 계열), ViT-T(ViLT [31] 계열). 출력은 GMM(Bishop 1994, [8]).
- 알고리즘: ER (Chaudhry 외 2019, [13]), EWC (Kirkpatrick 외 2017, [33]), PackNet (Mallya & Lazebnik 2018, [41]), 순차 미세조정과 다과제 학습.
- 과제 임베딩: BERT [19], CLIP [52], GPT-2 [53].

## 2. 공식 코드와 관련 저장소

| 저장소 | 관계 | 라이선스 | 상태 (조회 2026-11-22) | 대조 기준 | 비고 |
|---|---|---|---|---|---|
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 공식 코드, 과제, 시연 링크 | 코드 MIT, 데이터 README는 CC BY 4.0, HF 카드는 apache-2.0 | 스타 2,563, 마지막 push 2025-03-15, 기본 브랜치 master | `8f1084e` (2025-03-15) | `repo/`에 얕은 복제 (650 MB). 정적 읽기만 (`04`) |
| [libero-project.github.io](https://libero-project.github.io) | 공식 프로젝트 페이지 (논문과 README에 명시) | — | 내용 미확인 | — | 링크만 확인 |
| 데이터 `yifengzhu-hf/LIBERO-datasets` (HF) | 시연 hdf5 (미러) | apache-2.0 (카드) | 약 100 GB, 과제 130개 | 2025-05-18 갱신 | 받지 않음 (`04` §1) |
| 평생학습 실험 저장소 | 저자 측 실험 코드 (별도 저장소 없음) | — | — | — | 논문은 코드 저장소와 같은 곳을 가리킴 |

- **체크포인트.** 저장소에 학습된 가중치 링크는 README에 없습니다. 이 분석은 체크포인트를 찾지 않았습니다.
- **인용 정보.** 저장소 README의 BibTeX는 "arXiv:2306.03310"과 "2023"을 씁니다. 학회 정보는 없습니다.

## 4. 참고 메모

- 논문 [20] (Díaz-Rodríguez 외 2018)은 FWT를 제로샷 정의로 씁니다. 이 논문의 FWT와 다릅니다(`01` §3.3, `03` §4.1).
- 논문 [42] (Mandlekar 외 2021, "What Matters in Learning from Offline Human Demonstrations")는 Robomimic의 체크포인트 선택 관례를 제공합니다. 논문이 "관례"라고 부르는 것이 이 논문입니다(`04` D6).
- v1의 참고문헌 번호는 v2와 다릅니다(예: [63] 대 [65] 등). 인용할 때 판본을 확인합니다.
