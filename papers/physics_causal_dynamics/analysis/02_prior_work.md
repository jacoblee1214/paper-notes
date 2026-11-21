# 02. 선행연구, 공식 코드, 이 저장소의 관련 분석

분석일 2026-11-21. 자료는 arXiv 논문 본문과 참고문헌, 프로젝트 페이지(jungbbal.github.io/PICR), GitHub 메타데이터(`gh repo view`, 2026-11-21 조회)입니다.

## 1. 선행연구 계보

PICR은 세 흐름을 잇습니다. 저자들은 자기 리뷰(On et al., 2026, IJCAS)에서 같은 주제를 다뤘다고 적었습니다(Intro, 참고문헌).

| 흐름 | 대표 연구 | PICR과의 관계 |
|---|---|---|
| 문맥 기반 적응 (context-based) | Rakelly 2019 (PEARL), Zintgraf 2020 (VariBAD), Lee 2020 (CaDM), Kumar 2021 (RMA), Li 2025 (RWM), O'Connell 2022 (Neural-Fly) | 최근 이력에서 잠재 문맥을 추정한다는 점이 같습니다. 대부분은 문맥을 일반 신경망에 넣고 기계 구조를 주지 않습니다 |
| 물리 구조 신경망 | Greydanus 2019 (HNN), Cranmer 2020 (LNN), Raissi 2019 (PINN), Lutter 2019 (DeLaN) | DeLaN의 Lagrangian 파라미터화를 그대로 씁니다 |
| 문맥 + Lagrangian | Schulze 2025 (CaDeLaC, IROS) | **가장 가까운 대안.** 문맥을 LSTM으로 만들고 입력 연결로 DeLaN에 넣습니다. 논문은 CaDeLaC가 URDF 같은 공칭 모델에 의존한다고 주장합니다. 실제 비교에서는 공칭 잔차를 쓰지 않고 원시 토크로 전체 시스템을 학습하도록 고쳤습니다(부록 B.4) |
| 조건화 방식 | Perez 2018 (FiLM) | FiLM을 Lagrangian 망에 적용합니다. 표 5에서 입력 연결, 시프트만, 저랭크 하이퍼네트워크와 비교합니다(하이퍼네트워크 변형은 본문에 출처 인용이 없습니다) |
| 인과 표현 | Pearl 2009, Peters 2017, Schölkopf 2021, Wang 2022 (causal dynamics), Huang 2022 (AdaRL), Kirchmeyer 2022 | 논문의 "인과" 표현은 주로 개입과 반사실 질문을 뜻합니다. 인과 발견이나 인과 식별은 다루지 않습니다 |

**구성 요소별 출처.**
- 식 (1)(H, c, g, V) → Greenwood 2003, Featherstone 2008, Canudas de Wit 1996.
- 인코더 구조와 KL → 표준 VAE형 문맥 추정(Rakelly 2019 계열).
- 대조 학습 식 (3) → 지도형 대조 손실(SupCon)의 가중 변형으로 보입니다(저장소 docstring이 SupCon 형태라고 적음, `04` D9). 논문 본문은 이 식의 출처를 인용하지 않았습니다. 가중치 w_ij는 질량과 CoM의 가우시안입니다.
- 게이트 ψ(u) = αT tanh(u/T) → 논문 본문에 출처 인용이 없는 변형입니다. 저장소 주석은 T를 "roll-off"를 밖으로 미는 온도로 설명합니다(`04` D9).
- 데이터: Isaac Lab 2.3.2 (Mittal 2023, ORBIT), Franka 자료(Haddadin 2022), 평가 지표 ADD(Hinterstoisser 2012).

**가장 가까운 두 논문과의 차이.** CaDeLaC는 문맥을 입력 연결로 넣고 Lagrangian 망을 씁니다. 표 5와 부록 C.3의 "입력 연결"이 이 방식에 해당합니다. RWM은 문맥을 GRU 상태로 두고 상태 변화를 직접 예측합니다. 표 1에서 PICR의 학습 밖 ADD(6.76 cm)는 RWM 11.15, CaDM 11.40, CaDeLaC 21.41 cm보다 낮습니다.

## 2. 공식 코드와 저장소

| 이름 | 관계 | 라이선스 | 마지막 push | 별 | 기준 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [jungbbal/PICR](https://github.com/jungbbal/PICR) | 공식 구현, 학습과 평가, 체크포인트(seed 0), 논문 결과 스크립트 | **없음** (GitHub API `licenseInfo: null`, LICENSE 파일 없음) | 2026-11-13 (생성 2026-11-09) | 0 | `d22568a8` (2026-11-13) | 얕은 복제: `repo/PICR/`, 정적 읽기만 |
| [jungbbal.github.io/PICR](https://jungbbal.github.io/PICR/) | 프로젝트 페이지, 데모 영상 | (해당 없음) | (확인 안 함) | — | — | 페이지 텍스트와 링크를 읽음. 학회 표기 없음, "Under review", 논문 링크 "coming soon" |
| [milutter/deep_lagrangian_networks](https://github.com/milutter/deep_lagrangian_networks) | DeLaN 원 구현 (논문 기준선) | MIT | 2024-11-23 | 157 | 확인 안 함 | 저장소에 없음, 정적 읽기도 안 함 |
| [leggedrobotics/robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) | RWM 공식 구현 (기준선) | Apache-2.0 | 2026-10-29 | 737 | 확인 안 함 | 저장소에 없음 |
| [younggyoseo/CaDM](https://github.com/younggyoseo/CaDM) | CaDM 구현 (기준선; 논문은 TF1을 PyTorch로 다시 씀) | 확인 안 함 | 2020-05-20 | 64 | 확인 안 함 | 저장소에 없음 |
| [aerorobotics/neural-fly](https://github.com/aerorobotics/neural-fly) | Neural-Fly 구현 (기준선) | 없음 (`null`) | 2022-05-21 | 229 | 확인 안 함 | 저장소에 없음 |
| CaDeLaC 공식 코드 | 기준선 | 검색 2회에서 찾지 못함 | — | — | — | 논문은 "release의 크기"를 따랐다고 적었습니다. 공식 저장소 URL 미확인 |

**데이터와 가중치.**
- 학습과 평가 데이터셋은 공개되지 않았습니다. `data/README.md`와 `data/contracts/*.json`에 데이터셋 ID와 에피소드 목록만 있습니다(`04` §3).
- 체크포인트 `checkpoints/picr.pt`(26 MB, seed 0, epoch 40, EMA 가중치)가 저장소 안에 있습니다. 얕은 복제에 함께 받아졌습니다. 이 분석은 가중치를 로드하지 않았고, 로컬 복제에서 파일을 삭제했습니다. SHA-256은 `c9a710751de0…`입니다.
- 시뮬레이션 영상(`docs/*.mp4`)과 논문 그림은 저장소에 있습니다. 논문 그림은 공개본에 쓰지 않습니다(라이선스).

**프로젝트 페이지 수치.** 페이지의 ADD 표는 PICR 4.58 / 6.92, RWM 7.09 / 11.63, CaDM 6.79 / 10.90, CaDeLaC 9.07 / 24.75, DeLaN 6.82 / 25.65, Neural-Fly 7.66 / 24.78입니다. 논문 표 6의 시드 0 값과 같습니다. 논문 표 1(3 시드 평균)과 다릅니다(`03` K2).

