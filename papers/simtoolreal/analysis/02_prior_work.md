# SimToolReal 선행연구와 관련 자료

## 1. 연구 계보

SimToolReal은 세 흐름의 교차점에 있습니다.

1. **기능적 리타게팅(functional retargeting):** 사람 손 동작을 그대로 옮기지 않고 물체의 운동 결과를 맞춥니다. 모캡이나 RGB-D 사람 영상에서 물체 궤적을 얻은 뒤, 시뮬레이션 RL로 로봇 접촉 전략을 찾습니다(논문 II절, 참고문헌 [49, 17, 39]). 최근에는 RGB-D 영상[46, 21]과 AI 생성 영상[50]까지 확장되었습니다. 각 과제마다 RL을 다시 돌려야 한다는 것이 한계입니다. SimToolReal은 이 목표 궤적을 **학습 없이** 같은 정책에 넣습니다.
   - 직접 선행: Lum 외(참고문헌 [46]), 궤적 추종과 특화 정책 비교의 기준입니다.
2. **다지 손 파지와 재배치(dexterous grasping, in-hand reorientation):** DexFunc[1]은 제한된 물체 인식으로 기능적 파지를 학습합니다. RMA[62]와 함께 SimToolReal이 "거친 물체 표현"을 고르는 근거입니다. 재배치 연구[14, 44, 27]와 물체 회전[78, 40]은 하위 기술로 다룹니다.
3. **단일 정책의 대규모 절차적 학습:** 보행 로봇에서 절차적 몸체로 정책을 묶는 연구[43, 2]를 물체 쪽으로 가져왔습니다. 실행 기반은 DexPBT[59]의 Kuka-Allegro reorientation 환경과 SAPG[71]입니다. 비대칭 critic은 [60]입니다.

원격 조작 쪽 대안: DexterityGen[86]은 물체별 정책을 학습한 뒤 손 동작 입력만 받는 모델로 증류합니다. 논문은 이것을 "물체 맹목(object-blind)"이라고 부릅니다. 외골격과 텔레오퍼레이션[74, 82, 24]은 접촉 촉감 피드백이 없어 한계가 있습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 논문 참고 | 역할 | 이 저장소의 관련 분석 |
|---|---|---|---|
| SAPG | [71] | 정책 최적화 (PPO 대체) | 없음 |
| 비대칭 actor-critic | [60] | critic에 특권 상태 | 없음 |
| 키포인트 기반 거리 | [4, 59] | 목표 거리 d | 없음 |
| SAM 3D | [16] | 미터 단위 메쉬 | 없음 |
| SAM 2 | [65] | 첫 프레임 분할, 손잡이 영역 | 없음 |
| FoundationPose | [80] | 6D 포즈 추적 | 없음 |
| HaMeR | [58] | 키네마틱 리타게팅의 손 추정 | 없음 |
| Viser | [85] | 시각화 | 없음 |
| PyRoki | [33] | 충돌 회피 궤적 최적화 (고정 파지 기준) | 없음 |
| IsaacGym | [48] | 물리 시뮬레이터 (논문 결과) | 없음 |

## 3. 이 저장소의 다른 논문과의 관계


## 4. GitHub, 프로젝트, 관련 저장소 (확인 2026-11-17)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **공식 코드** | https://github.com/tylerlum/simtoolreal | MIT (Copyright 2026 Tyler Lum) | 435 / 42 | 2026-09-21 | `repo/` 얕은 clone, HEAD `313d5ae` (2026-08-13), 정적 읽기만. 커밋 1개 (얕은 clone이라 이력 없음) |
| 프로젝트 페이지 | https://simtoolreal.github.io/ | - | - | - | RSS 2026 표기 |
| 사전학습 정책 | https://download.cs.stanford.edu/juno/simtoolreal/ (스크립트 `download_pretrained_policy.py`의 기본 URL) | 미확인 | - | - | 받지 않음 |
| DexToolBench 데이터 | `download_dextoolbench_data.py` (URL 미확인) | 미확인 | - | - | 받지 않음 |
| FoundationPose 포크 | https://github.com/kushal2000/FoundationPose | 원저작물 라이선스 확인 필요 | - | - | 외부 저장소 |
| IsaacGymEnvs (원형) | https://github.com/isaac-sim/IsaacGymEnvs | 기타 | - | - | `repo/`에 포크된 코드 |
| rl_games (SAPG 포함) | `repo/rl_games/` (동봉) | 저장소 내 LICENSE | - | - | |

- 공식 저장소에는 `assets/licenses/`(YCB, KUKA iiwa)가 있어 자산 라이선스를 따로 적어 둡니다.
- Isaac Sim 경로(`isaacsimenvs/`, Python 3.11, pip 설치)가 **권장**이고, Isaac Gym 경로(`isaacgymenvs/`, Python 3.8)가 **논문 결과를 낸 것**입니다(README).
