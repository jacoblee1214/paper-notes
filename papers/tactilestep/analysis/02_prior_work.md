# 02. 선행연구와 구성 요소별 출처

TactileStep은 새 알고리즘 하나가 아니라 **이미 있는 것들의 조합**입니다. 휴머노이드 지각형 파쿠르 정책(Hiking in the Wild) 위에, 발바닥 촉각 특징 관측과 단계 조건 보상을 얹고, 비평가를 둘로 나눕니다. 이 문서는 각 부품의 출처와 가까운 대안을 정리합니다. 참고문헌 번호는 논문의 것이고, arXiv 번호는 논문 참고문헌에 적힌 값입니다.

## 1. 계보

| 흐름 | 대표 연구 (논문 번호) | TactileStep과의 관계 |
|---|---|---|
| 지각형 휴머노이드 파쿠르 | Humanoid Parkour Learning [7], BeamDojo [8], **Hiking in the Wild [9] (arXiv:2601.07718)**, RPL [10], Perceptive Humanoid Parkour [11], TTT-Parkour [15] | 깊이와 높이 지도로 지형 기하를 보고 발 디딤을 정합니다. 이 논문의 **외부 기준선과 깊이 인코더·학습 틀은 [9]**입니다(4.1절, 3.1절 "following [9]") |
| 발 디딤·지지 목표 | BeamDojo [8], [9], [10]의 모서리 검출·발 부피 점 벌점·겹침 보상 | 기하 기반 발 디딤 대리 보상. 이 논문은 "발 디딤이 가능해도 착지 후 지지가 불안할 수 있다"며 촉각으로 접촉 후 상태를 봅니다(2절) |
| 발바닥 촉각과 균형 | Atlas 부분 발판 [20], 발바닥 촉각 보행 [21], 분산 촉각 전신 접촉 [22], 부분 발판 검출 [23], ZMP [24], CoP 민감도 [25] | 발바닥 접촉 정보로 지지 영역을 추정하고 균형을 잡는 고전·모델 기반 연구. 이 논문은 이를 **학습 정책의 관측과 보상**으로 옮깁니다 |
| 소프트 랜딩·정숙 보행 | QuietWalk [13] (휴머노이드, 평지 보행, 센서 데이터로 GRF 예측기 학습), 사족 소음 [14, 32, 33], 고전 착지 제어 [34–37] | 휴머노이드 소프트 랜딩은 [13]이 가장 가깝습니다. 차이: [13]은 배포 시 발바닥 피드백을 쓰지 않고, 이 논문은 정책 관측에 넣고 파쿠르 지형까지 다룹니다(2절) |
| 촉각 sim-to-real | 이진 접촉 [26], 타셀 표현 [27], 전단·법선 [28–29], TACTO [30], Taxim [31] | 손 조작 중심. 이 논문은 정교한 촉각 모델 대신 강체 합력을 타셀에 나누는 가벼운 모델과 특징 세 개를 택했다고 설명합니다(2절) |
| 이중 비평가 | RobotKeyframing [39], Composite Motion Learning [40], Reward-adaptive RL [41] | dense/sparse 보상 분리에서 영감을 받았다고 적습니다(3.1절) |
| 게이트 보행 단계 | WoCoCo [44], Hold My Beer [45], gait-conditioned [46] | 고정 주기 시계 기반 SWING/STANCE 두 단계를, 이 논문은 촉각·운동 신호로 추정하는 4단계로 확장합니다(3.3절) |

## 2. 구성 요소별 출처

| 구성 요소 | 어디서 왔나 | 이 논문이 바꾼 것 |
|---|---|---|
| PPO + AMP 보조 보상(식 17) | PPO [38], AMP [43] | 계수 0.25, 판별기 [1024, 512] 등 설정만(표 8) |
| PD 이득 | BeyondMimic [42] | 그대로 가져옴(3.1절) |
| 깊이 인코더·지형 커리큘럼·MoE 액터-크리틱 | Hiking in the Wild [9]로 추정 | 논문이 "following [9]"라고 쓴 것은 깊이 이력이고, MoE와 커리큘럼의 출처는 명시되지 않음(표 8, B.2절) |
| Isaac Lab 병렬 학습 | Isaac Lab [47] | RTX 4090, 2048 환경 |
| 이중 비평가·어드밴티지 혼합 | [39–41] | 혼합 가중치 [1.1, 0.9], 두 가치 머리(표 8) |
| 촉각 시뮬(식 3–7) | 논문 고유(출처 명시 없음) | raycast 기반 힘 분배 + kNN 확산 + 특징 세 개 |
| 4단계 접촉 단계와 보상 | 논문 고유 | 알고리즘 1, 표 5–6 |
| 하드웨어 인솔 | 제조사 미기재 무선 인솔(25 Hz) | MLP 보정(제조사 제공 데이터) |

## 4. GitHub와 자료 표

조사일 2026-11-23. 별 수와 마지막 push는 GitHub API 값입니다.

| 저장소·자료 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| (공식 코드 없음) | TactileStep | – | – | – | 프로젝트 페이지에 "Code Soon". 논문에도 코드 링크 없음 |
| [tactilestep/tactilestep.github.io](https://github.com/tactilestep/tactilestep.github.io) | 공식 프로젝트 페이지 소스(GitHub 조직 tactilestep, 2026-09-24 생성) | 없음 | 0 | 2026-10-22 | 코드가 아니라 웹페이지. 표 세 개(표 1–3)가 그대로 재게시되어 있음 |
| https://tactilestep.github.io/ | 프로젝트 페이지 | – | – | – | 영상, 개요 그림, 표 1–3, 초록 |
| [project-instinct/InstinctLab](https://github.com/project-instinct/InstinctLab) | Hiking in the Wild가 쓰는 파쿠르 코드 틀로 보임(`Instinct-Parkour-Target-Amp-G1-v0`, AMP G1 파쿠르 과제). **TactileStep이 이 코드 위인지는 논문에 없음** | NOASSERTION | 883 | 2026-11-01 | 기준선 재현의 후보 틀(추정). 커밋 de6b92f |
| [project-instinct/instinct_rl](https://github.com/project-instinct/instinct_rl) | 위의 학습 알고리즘 저장소(AMP-PPO) | NOASSERTION | 190 | 2026-11-01 | 이중 비평가 구현 여부는 읽지 않음 |
| [project-instinct/hiking-in-the-wild](https://github.com/project-instinct/hiking-in-the-wild) | [9]의 프로젝트 페이지 저장소(`index.html`, `static`, `resources`; README 없음) | 없음 | 9 | 2026-09-23 | 코드 아님 |
| [jie0110/hiking-in-the-wild-sim2sim](https://github.com/jie0110/hiking-in-the-wild-sim2sim) | [9] 정책의 sim2sim 배포 예시(비공식으로 보임) | 없음 | 62 | 2026-06-05 | 정책 파일 교체 방식의 배포 튜토리얼 |
| [amazon-far/php_parkour](https://github.com/amazon-far/php_parkour) | Perceptive Humanoid Parkour [11] 공식 | Apache-2.0 | 303 | 2026-10-13 | 같은 계열의 공개 파쿠르 코드 |
| (미확인) | QuietWalk [13]의 공식 코드 | – | – | – | 검색에서 `sony/QuietWalk`(2025-02 push, 별 1)가 나왔으나 [13](2026)과 같은 것인지 확인하지 못함 |

- 공식 코드가 없어 복제하지 않았습니다. 기준선 코드를 받지도 않았습니다.
- 지원 자료: 프로젝트 페이지의 영상 6개(`platform-ascent` 등)는 내려받지 않았습니다.
