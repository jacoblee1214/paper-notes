# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 출처 (논문의 인용 번호) | WM-Craftnet에서의 쓰임 |
|---|---|---|
| 손바닥 위 물체 돌리기 RL 기본 틀, 시뮬-실제 전이 | Rapid Motor Adaptation 계열 [4] (HORA), DeXtreme [3] | 과제 정의, 도메인 무작위화, 아이덴티티 없는 적응이라는 문제 설정 |
| 시각-촉각 융합 정책 | Robot Synesthesia [6] (in-hand rotation), RotateIt [2], NeuralFeels [15], ViTacFormer [17] | 기준선 "In-Hand Rotation"이 [6]입니다. 이름 "Synesthesia"도 여기서 옵니다 |
| 이진·분산 촉각만으로 회전 | Touch Dexterity [1] | 기준선 "Touch Dexterity" |
| 중력 불변 다축 회전과 촉각 | AnyRotate [5] | 관련 연구로만 인용 |
| 월드 모델을 상태 표현으로만 사용(상상 없음) | World Model-based Perception (WMP) [7], DreamMimic [25] | **직접 계보.** WMP는 stop-gradient한 순환 상태를 이동 정책에 주는 구조이고, WM-Craftnet은 이것을 시각-촉각 손 조작으로 옮기고 "잡음 → 깨끗한 깊이" 복원과 물체 분포 간 재사용을 더했다고 §2 마지막 문단이 설명합니다 |
| RSSM, 이산 잠재, KL 균형 | World Models [19], Dreamer [20], DreamerV3 [21] | `wsm/`은 dreamerv3-torch(NM512, MIT)에서 가져와 수정한 것으로 저장소 NOTICE에 적혀 있습니다 |
| 월드 모델로 상상 롤아웃·정책 최적화 | DayDreamer [22], MoDem-V2 [23], Robotic World Model [24] | 대비 대상: WM-Craftnet은 이 방향을 쓰지 않습니다 |
| 보조 목표로 접촉·형상 구조화 | Hasson et al. [18], GAAC [13], DextrAH-G [14] | 자세, 형상(BPS 변위), 접촉 헤드의 착상(§3.3 "Inspired by") |
| 실제 데이터 기반 액추에이터 보정 | DexNDM [11], Lessons from spinning pens [10] | 대조: 이 논문은 관절 수준 시스템 식별(1 Hz 사인파로 Kp, Kd 조정)만 씁니다 |

### 새로움과 가까운 대안

- 이 논문의 기여는 "월드 모델을 상태 추정기로 쓰는 WMP 계열 + 깊이 복원 감독 + 사전학습 재사용"의 조합입니다. 각각의 요소는 이미 있고(WMP의 stopgrad 문맥, Dreamer의 RSSM, 보조 헤드), 이 논문이 보이려는 점은 손 조작에서 이 조합이 기준선보다 낫다는 것입니다.
- 가장 가까운 대안은 (1) 순환 정책 자체(LSTM/GRU, 표 9에서 비교됨), (2) 특권 정보 증류(RotateIt/HORA류), (3) 비순환 인코더 문맥(표 2의 encoder–decoder z_t)입니다. 논문은 (1)과 (3)을 절제로 비교했고, (2)는 비교하지 않았습니다.
- 비교의 공정성: 모든 기준선은 22자유도 손에서 같은 보상과 무작위화로 직접 재구현했다고 밝힙니다(§4.1). 기준선 구현은 공개되지 않아 확인할 수 없습니다.

## 2. 코드와 저장소 (GitHub 표)

조회일 2026-11-26 (`gh api`, 메타데이터만).

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 상태 |
|---|---|---|---|---|
| [sharpa-robotics/WM-Craftnet](https://github.com/sharpa-robotics/WM-Craftnet) | **공식 코드** (Isaac Gym 학습, z/x/y축, Sharpa Wave 배포 예시) | LICENSE 파일은 Apache-2.0 (Copyright 2026 Sharpa Group). GitHub API는 NOASSERTION으로 표시. 번들된 SDK, 자산은 별도 조건(THIRD_PARTY_NOTICES) | 55 / 2026-10-29 | 정적 읽기만. 커밋 `a1e825acaf50ade298e30b0b13c6671627af3a1e` (2026-10-29), 커밋 11개, 릴리스 v1.0.0 (2026-10-10). 저장소 크기 약 137 MB (GitHub 보고) |
| [SharpaIT/WM-Craftnet (Hugging Face)](https://huggingface.co/SharpaIT/WM-Craftnet) | 공식 체크포인트 `wm_craftnet_set_z.pth` | 모델 카드에 라이선스 표기 없음 | 다운로드 0, 좋아요 0 | 받지 않음. API 보고 크기 400,555,772 바이트(약 382 MiB). 파일은 z축 하나뿐 |
| [wmcraftnet/wmcraftnet.github.io](https://github.com/wmcraftnet/wmcraftnet.github.io) | 프로젝트 페이지 소스 | 없음 | 0 / 2026-11-03 | 페이지는 https://wmcraftnet.github.io 입니다. 페이지 푸터의 CC BY-SA 4.0 표기는 페이지 템플릿 표기입니다 |
| [sharpa-robotics/sharpa-wave-sdk](https://github.com/sharpa-robotics/sharpa-wave-sdk) | 로봇 손 SDK (배포 예시가 의존) | 없음 | 6 / 2026-11-06 | 장비 전용. 받지 않음 |
| [bytedance/WMP](https://github.com/bytedance/WMP) | 직접 계보 (stop-gradient 월드 모델 문맥) | Apache-2.0 | 426 / 2024-12-18 | 코드 README가 인용 |
| [YingYuan0414/in-hand-rotation](https://github.com/YingYuan0414/in-hand-rotation) | Robot Synesthesia (기준선 [6]) 과제 구조 | MIT | 116 / 2024-07-31 | 코드 README가 인용 |
| [HaozhiQi/hora](https://github.com/HaozhiQi/hora) | HORA, 프로젝트 페이지가 링크 | NOASSERTION | 262 / 2026-07-08 | 프로젝트 페이지가 링크 |
| [danijar/dreamerv3](https://github.com/danijar/dreamerv3), [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) | RSSM 구현의 출처 (`wsm/` 파일 헤더 MIT, NM512) | MIT | 4021 / 2026-10-04, 888 / 2026-03-08(보관) | |
| [isaac-sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs), [Denys88/rl_games](https://github.com/Denys88/rl_games) | 시뮬레이션·PPO 기반 | BSD-3-Clause(NOASSERTION 표기), MIT | 2949 / 2024-10-26(보관), 1409 / 2026-11-25 | 저장소가 `isaacgymenvs/`, `rl_games/`를 포함해 수정 |

- 프로젝트 페이지: https://wmcraftnet.github.io (페이지 소스에 "We will release the codes soon!" 문구가 주석으로 남아 있으나 코드 링크는 활성입니다). 포스터 PDF와 Sharpa 블로그 글 링크가 있습니다.
- Isaac Gym Preview 4는 NVIDIA 라이선스 하에 별도 내려받기입니다. 저장소에는 없습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "촉각이 시각과 상보적"이라는 전제는 Robot Synesthesia [6], RotateIt [2]에서 왔고, 이 논문의 절제(표 1)는 그 전제를 테스트 시점 촉각 마스킹으로만 시험합니다.
- "깊이 복원이 sim-to-real에 필요"하다는 전제의 직접 증거는 표 3의 "IHR + WSM-denoised depth" 한 줄이고, 시뮬레이션 쪽 근거는 표 2의 "깨끗한 대 잡음 목표" 한 줄입니다.
- "세계 모델 상태를 정책에 stop-gradient로 준다"는 WMP [7]의 결과(이동 과제)에 의존합니다. 손 조작에서 stop-gradient가 필요한지는 이 논문에 절제가 없습니다.
