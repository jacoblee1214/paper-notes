# SuperMimic: 선행연구와 자료 계보

## 1. 계보

1. **모션 모방 기반 휴머노이드 제어.** DeepMimic [2]이 모방 보상과 PPO로 캐릭터 기술을 배우는 기본 틀을 세웠습니다. PHC [11]는 큰 모션 집합과 복구 동작으로 확장했습니다.
2. **대규모 추적 학습.** SONIC [1]은 모션 데이터, 정책 용량, 계산량을 함께 키운 일반 추적기입니다(Science Robotics, 2026). SuperMimic의 직접 기준선이며, 128 GPU × 7일(약 21,000 GPU-h)이 보고되어 있습니다. BeyondMimic [3]은 같은 추적 보상 식을 제공하고, 목표 조건부 확산으로 과제를 바꿉니다. SuperMimic은 이 보상을 거의 그대로 쓰되 자기충돌 항을 빼고, 손끝 편차 종료 조건도 뺍니다(`01` §2).
3. **학습된 미분 가능 동역학을 쓰는 제어.** SuperTrack [12], ControlVAE [6], MoConVQ [13], SuperDyno [14]가 학습된 미분 가능 대리 모델로 모션 제어 정책을 최적화합니다. SuperDyno는 수만 개 클립을 다룹니다. SuperMimic은 이 흐름을 단일 GPU의 휴머노이드 추적기 학습으로 옮깁니다.
4. **짧은 지평의 미분 기반 actor-critic.** SHAC [19]는 미분 가능 시뮬레이터에서 짧은 지평 actor-critic을 씁니다. SuperMimic은 시뮬레이터 대신 학습된 모델을 쓰고, 목표 네트워크 없이 같은 형태의 비평가를 씁니다(`01` §2).
5. **모델 기반 강화학습.** Dreamer와 DreamerV3 [15, 16], TD-MPC [17], DayDreamer [18]는 잠재 동역학과 상상 학습을 씁니다. DMO [21]는 시뮬레이터로 궤적을 굴리고 학습 모델을 통해 미분합니다. SuperMimic은 롤아웃과 역전파를 모두 학습 모델로 하고 모션 집합 전체에서 공유합니다(`01` §2).
6. **학습된 시뮬레이터에서 정책 학습.** RWM [22], RWM-U [23]는 학습된 신경 시뮬레이터에서 PPO로 정책을 학습해 실로봇에 옮깁니다. SuperMimic은 미분으로 정책을 학습하고, 모델을 학습 중에 같은 정책의 자료로 맞춘다는 점이 다릅니다(§3.3, Related Work).
7. **재사용 가능한 모션 표현.** NPMP [5], PhysicsVAE [7], PULSE [8], MaskedMimic [9], LATENT [10]이 잠재 표현과 하위 과제 인터페이스를 다룹니다. SuperMimic은 이 가운데 속도 명령 인터페이스를 RL로 학습하는 실험(§4.5)을 추가합니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 확인 상태 |
|---|---|---|
| 추적 보상, 종료, 도메인 무작위화 | BeyondMimic [3] (부록 A에서 변경점 명시) | 공개 코드에서 값 확인(`04` D18–D20) |
| 상상 롤아웃과 Dyna 방식 시작 상태 | Dyna [24], MBPO [25] | 논문 본문 |
| 일반화 이득 추정 (GAE) | Schulman 외 [26] | 코드 확인(`04` D13) |
| 짧은 지평 actor-critic과 생존 가중치 | SHAC [19] | 논문에 차이 명시 |
| 재설정 분포 (반환 기반) | 논문 부록 D.1 | 코드 확인(`04` D15) |
| 시뮬레이터 | MuJoCo [27], MuJoCo Warp (GPU) | 논문 §4 |
| 모션 데이터 | BONES-SEED [28] (공개 부분 G1 리타게팅), LAFAN1 [29] | 다운로드하지 않음 |

## 4. GitHub와 코드 자료 (확인 2026-11-15)

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 논문 프로젝트 페이지 | https://supermimic.github.io | - | - | - | 논문, 영상, FAQ. 코드 링크 없음 |
| 공식 코드 저장소 | https://github.com/HybridRobotics/supermimic | Apache-2.0 | 1,501 | 2026-11-11 (커밋 abc124c) | 정적 읽기만 함, `repo/`에 클론. 공식성 근거는 `04` §0 |
| 의존 저장소 (mjlab 포크) | https://github.com/HybridRobotics/mjlab_research | Apache-2.0 | 45 | 2026-11-11 (커밋 dd51a9f) | 정적 읽기만 함, `repo/mjlab_research/`에 클론 |
| 모션 데이터 (BONES-SEED) | https://huggingface.co/datasets/bones-studio/seed | 미확인 | - | - | 내려받지 않음 |
| SONIC 코드 후보 | https://github.com/NVlabs/GR00T-WholeBodyControl | NOASSERTION | 3,915 | 2026-11-03 | "GEAR-SONIC" 설명이 있으나 논문 SONIC과의 동일성은 확인하지 않음. 클론하지 않음 |
| 제3자 재현 (비공식) | CuboiLeo/SuperMimic-Reproduced (MIT, 463별), Motphys/Uni-SuperMimic (Apache-2.0, 302별), NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real (MIT, 97별) | 각각 표기 | 위 | 2026-11-10, 11-12, 10-27 | 읽지 않음. CuboiLeo 저장소 설명이 공식 저장소를 가리킴 |
| 목록 저장소 | miao-org/awesome-supermimic (CC0-1.0, 601별) | CC0 | 601 | 2026-11-13 | 목록만, 읽지 않음 |

별 개수는 GitHub API 검색 결과(2026-11-15)이며 시간에 따라 바뀝니다. 제3자 저장소의 결과는 검증하지 않았습니다.

## 5. 비교 대상으로 쓰는 값

- SONIC 공개 체크포인트: 정규화 반환 0.90(Fig 3a 점선), 실패 179(Table A.8). 학습 비용 21,504 GPU-h는 SONIC 보고값입니다.
- BeyondMimic: 추적 보상 식과 종료 임계의 출처(부록 A).
- DMO: 같은 "시뮬레이터 롤아웃 + 모델 역전파" 계열의 대안(v2에서 추가, `01` §7).
