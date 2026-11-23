# 02. 선행연구와 관련 저장소

참고문헌 번호는 HumanoidNDM v1의 참고문헌 목록 번호입니다. 인용한 논문의 내용은 이 분석에서 다시 읽지 않았고, HumanoidNDM이 그 논문을 어떻게 쓰는지만 적었습니다. 저장소 정보는 2026-11-24에 `gh api`로 확인했습니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 직접 출처 | HumanoidNDM에서의 사용 |
|---|---|---|
| 동작 추종 정책 | DeepMimic [57], GMT [8], BeyondMimic [9], Track Any Motions Under Any Disturbances [7], OmniH2O [6], ExBody2 [5] | 기본 정책은 MJX에서 PPO [1]로 학습. 어느 방법을 따랐는지는 명시하지 않음 |
| 시뮬-현실 격차를 줄이는 시뮬 쪽 방법 | 시스템 식별 [19, 20], 전이 보상(ASAP [12]의 델타 행동 모델, 운동 보상 [21]), 궤적 정렬(DexNDM [22]) | 표 2의 기준선 ASAP, 5.1절. "OOD 위험이 높고 학습이 느려 온라인 갱신이 어렵다"는 주장은 인용 없이 서술(2.1절) |
| 실제 데이터로 정책 갱신(모델 없는 RL) | A Walk in the Park [23], Legged robots that keep on learning [24], Grow Your Limits [25], Safe RL [26], Robot Trains Robot [27], Bloesch 외 [28] | "제한된 실제 데이터와 복잡한 인간형 동역학에서 불안정"이라는 서술(2.1절). 직접 비교 실험은 없음 |
| 모델 기반 RL | RWM [31], RWM-U [32], 이 외 [29, 30, 33, 34] | RWM [32]를 기준선으로 구현(5.1절, 부록 B). "세 단계 모두 다르다"고 서술 |
| 모델 기울기 정책 최적화 | DreamerV1–V3 [15–17], SHAC [18], 효율적 모델 기반 모터 스킬 학습 [35], SuperTrack [13], ControlVAE [14] | 이 계열을 채택(4.3절). 이중 공간(구성 + 유클리드) 목적은 물리 기반 애니메이션 연구 [12–14]를 따랐다고 서술(4.2절) |
| 신경 동역학 모델 | Neural Robot Dynamics [55], Contact-aware Neural Dynamics [56], DexNDM [22](이름과 "NDM"이 같음) | 이력 기반 트랜스포머 + 앙상블 머리로 구성(부록 A.2, A.5) |
| 탐색 | OU·행동 잡음 [39, 45, 46], 색 잡음 [40], 부드러운 탐색 [10], 모델 기반 탐색 증강 [36], 활성 탐색 SysID [20], 안전 탐색 [38], Go-Explore의 셀 이산화 [54] | 이웃 점유율 `C`는 [54]의 셀 방식을 연속 제어로 확장했다고 서술(4.1절). 저주파 잡음 + 밴딧은 새로 제안 |
| 밴딧 | UCB(문헌 번호 없음) | 식 9–10 |
| 분포 이동과 모델 악용 | MBPO [50], 목표 불일치 통합 관점 [51], MOPO [53], Ready Policy One [52] | 데이터가 정책 부분공간을 덮어야 한다는 동기(4.1절) |
| 시뮬레이터 | MuJoCo Playground(MJX) [49] | 시뮬 평가와 정책 학습 |

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [shenbiachao/HumanoidNDM](https://github.com/shenbiachao/HumanoidNDM) | 논문 프로젝트 페이지의 "Code" 링크 대상. **빈 저장소**(커밋 0개) | 없음(API license: null) | 0 | 2026-09-27 (생성과 같은 시각) | 얕은 clone이 "empty repository"로 끝남. 비교할 코드가 없음 |
| [shenbiachao/HumanoidNDM_Website](https://github.com/shenbiachao/HumanoidNDM_Website) | 프로젝트 페이지 소스(HTML, 영상 에셋 약 132 MB) | 없음 | 0 | 2026-11-04 | 읽지 않음, 받지 않음 |
| [pku-epic.github.io/HumanoidNDM](https://pku-epic.github.io/HumanoidNDM/) | 프로젝트 페이지 | - | - | - | 요약만 읽음. 논문과 같은 초록, 영상 8개, BibTeX |
| [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP) | 기준선 ASAP의 공식 구현(IsaacGym 계열). HumanoidNDM은 MJX로 이식했다고 서술(부록 B) | MIT | 2,147 | 2026-01-06 | 이식본은 공개되지 않음 |
| [leggedrobotics/robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) | 기준선 RWM(RWM, RWM-U)의 공식 구현 | Apache-2.0 | 740 | 2026-10-29 | 이식본은 공개되지 않음 |
| [leggedrobotics/robotic_world_model_lite](https://github.com/leggedrobotics/robotic_world_model_lite) | RWM 경량 구현 | Apache-2.0 | 254 | 2026-04-08 | - |
| [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | 논문이 쓴 MJX 기반 시뮬 환경 [49] | Apache-2.0 | 2,337 | 2026-11-23 | 인간형 추종 환경이 이 위에 있는지는 논문에 없음 |

- DexNDM(arXiv:2510.08556, 같은 연구 계열)과 BeyondMimic, GMT, Track Any Motions 같은 추종 정책의 저장소는 이 분석에서 검색하지 않았습니다(검색 한도). GitHub 검색 `DexNDM`은 공식으로 보이는 저장소를 반환하지 않았습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **"ASAP는 델타 행동 모델이 동역학 격차를 줄인다"** [12]: 표 2에서 ASAP가 scaled KP 같은 구동기 격차에 강한 결과(balance Scaled KP의 Elink pos 2.61 등)는 이 전제와 일치합니다.
- **"모델 기울기 정책 최적화가 모델 기반 RL보다 데이터 효율적"** [31, 32, 35]: 이 논문은 RWM과 직접 비교로 보입니다(표 2). 단 RWM 기준선은 이식과 격자 탐색을 거쳤습니다(부록 B).
- **참고문헌 사소한 점.** [25]와 [44]가 같은 논문(Grow Your Limits)을 두 번 싣습니다. "uncorrelated action noise"에 Sutton–Barto 교재와 REINFORCE [45, 46]를, "temporally correlated noise"에 DDPG [39]를 인용합니다. 서술과 인용이 1:1로 대응하지는 않는다는 점만 적습니다.
