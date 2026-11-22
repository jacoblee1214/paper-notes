# 02. 선행연구와 관련 자료

## 1. 선행연구 계보 (구성 요소별 출처)

Fast-JEPA의 구성 요소는 아래 계보에서 가져옵니다. 번호는 논문 참고문헌 번호입니다 [원문 참고문헌].

| 구성 요소 | 출처 | 역할 |
|---|---|---|
| 잠재 세계 모델 기반 계획 | Visual Foresight [10], PlaNet [15], Dreamer [16, 17], TD-MPC [18, 19] | 예측 결과로 행동을 고르는 틀 |
| JEPA 표현 학습 | I-JEPA [2], V-JEPA 2 [3], V-JEPA [5] | 예측형 표현 학습 |
| 세계 모델 기준 (LeWM) | LeWorldModel [24] (arXiv:2603.19312) | 기반 모델과 과제 설정 (PushT, Reacher, Cube, TwoRoom) |
| 빠른 세계 모델 | Fast-LeWM [13] (arXiv:2606.26217), RC-aux [22], FIRM-WM [36] | 지연 감소, 다중 예측 |
| 목표 조건 행동 생성 | GC-IDM [26] (arXiv:2605.08732) | 역동역학 기반 행동 상각 (Table 1·2 비교 대상) |
| 의도-행동 탐색 | INTACT [32] (arXiv:2607.26056) | 가장 가까운 직접 비교 대상 (Guarded A 변형 포함) |
| 샘플 기반 최적화 | CEM [31], iCEM [30], POPLIN [35] | 제안 주변 탐색의 기반 |
| 목표 조건 학습 | HER [1], Goal-conditioned ISL [14], RIG [25] | 목표 조건 정책 |
| 종점 목표 표현 | DINOv2 [27], DINO [6], LeJEPA/SiGReg [4] | 시각 목표 거리, 정규화 |
| 행동 생성 정책 | Diffusion Policy [7], Diffuser [21], Zhao et al. (ACT) [39] | 행동 청크 표현, 확산 계획 |
| 비교 발표 결과 | IMWM [12], NGC [37], LEAP [29], LeFlow [20], SALT [38], PPWM [11] | Table 1 "Published results" (재측정 아님) |

- 논문과 계보의 관계: Fast-JEPA는 **새 세계 모델을 만들지 않습니다.** 기존 순방향 모델 `W_ψ`에 앵커 질의를 쓰고, 행동 생성기와 CEM을 얹는 추론 절차를 제안합니다(`01` §2). 학습 측면의 새 요소는 행동 모델 식 (8)–(9)입니다.
- 가장 가까운 대안은 INTACT [32]입니다. 논문도 INTACT를 가장 강한 기준으로 두고, 재귀 채점을 쓰는 같은 계열 설계를 Table 5에서 따로 비교합니다.

## 2. 코드와 저장소 (GitHub 및 웹 표)

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [DeepCybo-PhysAI/Fast-JEPA](https://github.com/DeepCybo-PhysAI/Fast-JEPA) (논문의 `Sofilyzia/Fast-JEPA`가 이동) | 공식 저장소 | 없음 (API license null; README "public redistribution license has not yet been selected") | 2026-11-21 | 8 | 클론 깊이 1, 커밋 `4aedeea` (2026-11-20). **구현 코드 없음**: README, 그림 4개, 배지, 로고만 있음. 얕은 클론은 `repo/`에만 둠 (정적 읽기) |
| LeWorldModel 공식 코드 | 기반 모델 | arXiv 2603.19312 "License: CC BY 4.0" (HTML) | - | - | 이 분석에서 저장소를 열어 보지 않음 (미확인) |
| INTACT, Fast-LeWM 공식 코드 | 비교 대상 | - | - | - | 미확인 |

- 저장소에는 **학습 코드, 평가 코드, 체크포인트, DynaGoal 케이스 목록이 없습니다.** README에 따르면 "implementation release"는 추후 공개입니다.
- 공식 저장소의 README 표 값은 논문 Table 1과 같습니다(검산 완료, `03` S12).
- `git log`: 한 커밋 "Remove local paper files" (2026-11-20). 원저자 공개 과정의 흔적입니다.

## 3. 이 저장소의 관련 분석

| 폴더 | 논문 | 관계 |
|---|---|---|
| `vjepa2/` | V-JEPA 2 (arXiv:2506.09985) [3] | 같은 JEPA 계열 계획 모델. 공식 코드 정적 대조 방식이 이 분석과 같음. 연결: 행동 조건 예측과 CEM 계획의 원형 |
| `dino_wm/` | DINO-WM (arXiv:2411.04983) | DINOv2 특징 위 세계 모델과 zero-shot 계획. 연결: 종점 시각 목표 거리 |
| `uncertain_action_wm/` | ACTUS (arXiv:2611.23620) | LeWM 추론 위 행동 불확실성 계획. 연결: LeWM 기반 CEM 계획을 같은 체크포인트로 확장 가능 |

- 루트 `README.md`에서 LeWM, INTACT, Fast-LeWM을 직접 다룬 폴더는 찾지 못했습니다. `uncertain_action_wm/` 분석이 LeWM 체크포인트를 언급합니다.

## 4. 학회 표기 확인 (venue)

- arXiv abs: 코멘트, 저널 참조 없음. 저자 소속에 학회명이 없습니다.
- 저장소 README: 학회 표기 없음. 인용 형식은 `@misc`(arXiv)입니다.
- `conferences/corl2026/papers.json` (755편): 제목과 키워드 "jepa", "latency frontier", "dynagoal"을 검색했으나 일치 항목이 없습니다.
- 웹 검색(제목 기준): 일치하는 공식 목록을 찾지 못했습니다. 검색 결과에는 Fast LeWorldModel(arXiv:2606.26217)만 나왔습니다.
- **판정: `학회 표기 없음 (arXiv)`.**
