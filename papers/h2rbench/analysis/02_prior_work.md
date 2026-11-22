# 02. 선행연구, 가까운 대안, 코드 저장소

## 1. 계보

H2RBench는 두 갈래의 연구 위에 서 있습니다. 하나는 **사람 영상에서 로봇 정책을 배우는 H2R 방법**, 다른 하나는 **real-to-sim으로 로봇 정책을 평가하는 벤치마크**입니다.

### 1.1 Real-to-sim 평가 (벤치마크 쪽)

| 연구 | 하는 일 | H2RBench와의 관계 |
|---|---|---|
| SIMPLER [19] | 실제 로봇 설정의 시뮬 대응물을 만들고 시뮬–실세계 성능의 상관(Pearson r)과 MMRV를 보고 | 상관 지표와 MMRV(식 3)를 그대로 가져옴. 로봇 전용 정책만 평가 |
| PolaRiS [20] | 신경 재구성(가우시안 스플랫)으로 실제 장면을 Isaac Lab 환경으로 변환 | **환경 구축 파이프라인과 코드 베이스.** 공식 코드는 PolaRiS 저장소의 포크이고 LICENSE 저작권 표기도 PolaRiS 저자(2025)입니다(`04` §1) |
| RobotArena ∞ [21] | 생성형 real-to-sim 환경에서 범용 정책 자동 평가 | Table 1 비교 대상 |
| RoboTube [7] | 사람 영상 + 시뮬 환경, 보상 학습(온라인 RL) 중심 | 가장 가까운 선행. 논문은 "로봇 데모 예산을 표준화하지 않고 사람 데모 양을 바꾸지 않는다"는 점으로 구별(§2) |

Table 1의 신규성 주장은 두 가지입니다. H2R 평가와 정량적 Sim-vs-Real 검증을 **동시에** 갖춘 것은 H2RBench뿐이라는 것(Table 1), 그리고 H2R 설정의 "첫 체계적 Sim-vs.-Real 검증"(§2)입니다. 이 분석이 확인한 범위에서 이 비교는 4개 벤치마크 대비 표라서 한정된 주장입니다(`03` C14).

### 1.2 H2R 방법 (평가 대상)

| 방법 | 핵심 아이디어 |
|---|---|
| Phantom [13] | 사람 팔 인페인팅 + 로봇 팔 렌더, 로봇 데이터 없이 학습 (원 논문은 사람 영상 268–950개로 Franka와 Kinova 평가) |
| Point Policy [12] | 사람과 로봇을 공유 3D 키포인트로 표현 |
| AMPLIFY [14] | 행동 없는 영상에서 키포인트 모션 토큰을 배우고 로봇 데모로 역동역학을 학습 |
| GHOST [15] | 계층형: 사람 + 로봇으로 하위 목표 예측, 로봇만으로 저수준 제어 |
| XSkill [9] | 교차 신체 스킬 발견 (부록 B.5에서 실패) |

그 밖에 논문이 사전학습, 보상 학습, 어포던스로 분류한 인용([4, 29], [30, 31], [5, 32])이 있습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 |
|---|---|
| 환경 재구성, 평가 인터페이스 | PolaRiS (Isaac Lab, 가우시안 스플랫 렌더) |
| 장면 메시와 스플랫 | Marble (PolaRiS 파이프라인 대체) |
| 물체 자산 | TRELLIS (다시점 이미지에서 3D) |
| 로봇 데모 생성 | cuRobo 모션 플래너 (시뮬), GELLO 원격 조작 (실세계) |
| 로봇 | Franka Panda + Robotiq 2F-85 (코드 `robot=franka_robotiq_2f_85`) |
| 행동 표현 | 3D 이동 + 6D 회전 [33] + 이진 그리퍼 |
| 지표 | 진행도와 성공률, 로그 오즈 이득(저자 정의), Pearson, Spearman, MMRV(SIMPLER) |

## 4. GitHub와 프로젝트 자원

조회일 2026-11-23. 별 수와 최근 푸시는 `gh api`로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 별 | 최근 푸시 | 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [xiaochy/H2RBench](https://github.com/xiaochy/H2RBench) | **공식 코드** (벤치마크 환경, 데이터 생성, 평가 서버와 클라이언트) | MIT (저작권 표기는 PolaRiS 저자, `04` §1) | 9 | 2026-11-03 | `155026875c6ddf3bedfedb8ef1fefb6e8c9a2744` (단일 커밋) | 얕은 복제, 정적 읽기만 (`repo/`) |
| [프로젝트 페이지](https://davidz2851.github.io/h2rbench.github.io/) | arXiv가 가리키는 주소. 저장소 README는 `h2rbench.github.io`로 안내 | 해당 없음 | | | | 읽음 (Table 3 수치가 논문과 같음) |
| [arhanjain/polaris](https://github.com/arhanjain/polaris) | 기반 코드 (PolaRiS) | MIT | 252 | 2026-07-13 | `129abc45` | 읽지 않음 (포크의 차이만 `04`에서 봄) |
| [r-pad/ghost](https://github.com/r-pad/ghost) 브랜치 `haotian/h2rbenchmark` | GHOST 학습 코드 | MIT | 14 | 2026-09-14 | `b30284d9` | 확인만 (복제하지 않음) |
| [r-pad/lerobot](https://github.com/r-pad/lerobot) 브랜치 `polaris` | GHOST 저수준 정책용 LeRobot 포크 | Apache-2.0 | 2 | 2026-10-02 | `c344bb35` | 확인만 |
| [DavidZ2851/Point-Policy](https://github.com/DavidZ2851/Point-Policy) 브랜치 `feat/polaris` | Point Policy 포크(H2RBench용) | MIT | 0 | 2026-09-14 | `ce506da0` | 확인만 |
| [siddhanthaldar/Point-Policy](https://github.com/siddhanthaldar/Point-Policy) | Point Policy 원 구현 | MIT | 92 | 2025-07-21 | | 확인만 |
| [MarionLepert/phantom](https://github.com/MarionLepert/phantom) | Phantom 원 구현 | MIT | 120 | 2025-08-29 | | 확인만. H2RBench 저장소에는 Phantom 파이프라인이 없음 |
| [pairlab/AMPLIFY](https://github.com/pairlab/AMPLIFY) | AMPLIFY 원 구현 | MIT | 56 | 2026-04-13 | | 확인만. H2RBench 서버가 `../../../AMPLIFY` 경로를 기대 |
| [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv) | MMRV와 상관 지표의 원형 | MIT | 1,189 | 2026-09-30 | | 확인만 |

사람 데이터는 Hugging Face 데이터셋 네 개(`Kovavavvavava/pick_place_red_mug_human` 180 에피소드 9.1 GB, `pick_toys_human` 315 에피소드 27.7 GB, `insert_donut_human` 300 에피소드 14.6 GB, `stack_bowls_human` 300 에피소드 11.5 GB, 합계 약 62.9 GB)와 Google Drive에 있다고 README가 밝힙니다. 크기는 Hugging Face API의 `usedStorage`와 파일 목록으로 확인했고 데이터는 받지 않았습니다. 이 네 데이터셋에는 라이선스 필드가 없습니다.
