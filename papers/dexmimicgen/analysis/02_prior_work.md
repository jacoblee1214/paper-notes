# DexMimicGen 선행연구, 대안, 자료 (02)

분석일 2026-11-22. 논문 인용 번호는 `paper/paper.txt`의 참고문헌 번호([n])입니다.

## 1. 계보

| 연도 | 연구 | 역할 | 출처 |
|---|---|---|---|
| 2018–2019 | RoboTurk, 원격 조작 기반 대규모 수집 | 시연 수집 방식의 출발점. DexMimicGen은 원천 시연 수를 줄이는 쪽 | [4], [18] |
| 2020–2022 | Mandlekar et al., GTI(RSS 2020), robomimic 계열 | 오프라인 시연 학습과 BC-RNN, BC-RNN-GMM의 출처 | [1], [43] |
| 2023 | Dalal et al., Imitating task and motion planning (CoRL 2023) | 작업 계획과 시각 운동 정책의 결합 | [15] |
| 2024 | Wang et al., RoboGen (ICML 2024, 41회; 참고문헌에는 "2023"으로 적힘) | 생성형 시뮬레이션으로 데이터 생성 | [16] |
| 2023 | **MimicGen** (CoRL 2023, PMLR v229) | 직접 기반. 객체 중심 하위과제와 SE(3) 변환으로 단일 팔 데이터 생성 | [17] |
| 2023 | Hoque et al., Interventional data generation (CoRL 2023 워크숍) | MimicGen 계열의 다른 갈래, 개입 데이터 | [37] |
| 2023 | Diffusion Policy (RSS 2023) | 정책 구조 비교에 쓰는 DP | [54] |
| 2024 | RoboCasa (RSS 2024) | 같은 저자 계열의 시뮬레이션 과제 프레임워크. 관련 인용 | [38] |
| 2024 | SkillMimicGen (arXiv:2410.18907) | 충돌 회피용 모션 플래닝을 붙이는 후속 방향으로 논문이 명시 | [79] |
| 2025 | **DexMimicGen** (ICRA 2025) | 본 논문 | 본 분석 |

**계보의 요지.** MimicGen은 "과제를 객체 중심 하위과제로 나누고, 원천 세그먼트를 객체 자세에 맞게 옮기면 새 장면의 데이터를 얻는다"는 틀을 만들었습니다. DexMimicGen은 같은 틀에서 **팔 두 개와 다지 손**이 생기면서 바뀌는 세 가지 문제(독립, 조정, 순서)를 다룹니다. 구현상 가장 큰 차이는 하위과제 분할을 팔마다 따로 한다는 점입니다.

## 2. 가까운 대안

| 대안 | 방식 | 본 논문과의 비교 | 근거 |
|---|---|---|---|
| **DemoNoise** (논문의 기준선) | 원천 시연을 행동 잡음과 함께 재생 | 표 III에서 같은 원천 시연으로 DexMimicGen이 세 과제 모두 앞섬. 초기 분포 D1, D2 데이터는 만들 수 없음 | §VI-C, 표 III |
| **SkillMimicGen** [79] | 모션 플래닝으로 충돌을 피함 | 본 논문이 향후 과제로 들었지만 실험 없음 | 부록 IX |
| **DexFlyWheel** (arXiv:2509.23829) | 다지 손 조작 데이터를 자기 개선으로 생성하는 프레임워크 | 검색 결과로만 확인. 본 분석에서 내용을 읽지 않음 | [WebSearch 결과] |
| **HumanoidGen** (NeurIPS 2025) | 양손 다지 손 데이터 생성을 LLM 추론으로 확장 | Crossref 메타데이터로 게재만 확인. 본문은 읽지 않음 | [Crossref, DOI 10.52202/085713-5220] |
| **PriDex** (arXiv:2611.21908) | 사람 영상에서 양손 다지 손 데이터 | 다른 입력(영상)에서 같은 목표. 실험 방식이 다름 | 외부 논문 |

DexFlyWheel과 HumanoidGen은 이 분석에서 자세히 읽지 않았습니다. 비교 주장을 쓰려면 원문을 따로 읽어야 합니다.

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 본 논문에서의 위치 |
|---|---|---|
| 원천 시연 분할, SE(3) 변환, 보간, 성공 필터 | MimicGen [17] | §III |
| BC-RNN, BC-RNN-GMM | robomimic [1] | §VI-A, 표 I |
| Diffusion Policy | [54] | §VI-A, 표 I |
| 시뮬레이션 프레임워크 | RoboSuite [70], MuJoCo [71] | §V |
| Panda OSC 제어 | Khatib [72] | §V |
| GR-1 IK 제어 | mink [73], Pink [74] | §V |
| 손 리타게팅 | OmniH2O [13] | §V |
| Vision Pro 텔레오퍼레이션 | VisionProTeleop [75] | §V |
| BiGym 과제 | [76] | §VI-B |
| 실로봇 객체 위치 맞춤 | Ditto [77], GroundingDINO [78] | §VI-D |
| 병렬 하위과제 확장, 조정 동기화, 순서 제약 | 본 논문 | §IV |

## 5. GitHub, 프로젝트, 데이터셋 표

| 자료 | 관계 | 라이선스 | 기준 커밋·날짜 | 스타·포크 | 상태 |
|---|---|---|---|---|---|
| [ARISE-Initiative/robomimic `dexmimicgen` 브랜치](https://github.com/ARISE-Initiative/robomimic) | BC-RNN 학습 (README가 안내) | robomimic 저장소 라이선스(API NOASSERTION) | 브랜치 존재 확인 | - | 복제하지 않음 |
| [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) | 시뮬 기반 (README는 "최신 버전" 설치를 안내) | MIT (README 기준) | 최근 push 2026-08-09 | 2721 | 복제하지 않음. 버전 고정 안 됨 (`04` D7) |
| [dexmimicgen.github.io](https://dexmimicgen.github.io) | 프로젝트 페이지 | 사이트 CC BY-SA 4.0 (사이트 명시) | 2026-11-22 확인 | - | 데이터셋 링크 없음, 생성 영상 안내 |
| [HF MimicGen/dexmimicgen_datasets](https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets) | 데이터셋: 과제별 HDF5 9개, GIF 1개, 총 약 59.9 GB (메타데이터 기준) | **cc-by-nc-sa-4.0** (카드 메타데이터와 README). GitHub README는 "CC-BY 4.0"이라고 씀 (`04` D8) | 파일 목록 2026-11-22 확인 | - | 내려받지 않음 |

**라이선스 판단 근거.** 코드는 NVIDIA 라이선스의 3.3절이 비상업 연구·평가로 제한합니다. 데이터셋은 카드 메타데이터가 NC-SA를 가리키므로 보수적으로 NC-SA로 봅니다. 논문은 non-exclusive라 그림 재배포가 막힙니다.
