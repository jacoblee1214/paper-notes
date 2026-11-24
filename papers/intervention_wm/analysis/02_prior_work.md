# 02. 선행 연구와 자료

논문 WHIRL(arXiv:2609.06009 v1)의 위치를 구성 요소별로 나눕니다. "논문 인용"은 논문 참고문헌에 있는 것이고, "논문이 인용하지 않은 가까운 연구"는 이 분석 작성자의 지식에서 꺼낸 것이라 이번 세션에서 원문을 다시 확인하지 않았습니다.

## 1. 계보

### 1.1 잔차 RL과 실로봇 RL (논문 인용)

- **Residual Policy Learning** (Silver 외 2018, [34])와 **Residual RL for robot control** (Johannink 외 2019, [7]): 고정된 기본 정책에 보정량을 학습하는 틀. WHIRL의 `a = a_BC + Δ_θ`가 이 틀이다.
- **ResFiT** (Ankile 외, arXiv:2509.19301, [25]): 동결한 BC 사전 정책 위에 off-policy 잔차를 학습한다. 논문이 "채택한 골격"이라 부르고 기준선 사다리의 두 번째 단이다.
- **SERL**(Luo 외 2024, [10])과 **HIL-SERL**(Luo 외, arXiv:2410.21845, [14]): 실로봇 RL의 actor–learner 시스템과 사람 개입. **SiLRI**(Zhao 외, arXiv:2512.24288, [15]): 차선 개입에서 배우기. 논문은 둘의 "핵심 메커니즘"을 자체 구현해 기준선 "Residual RL (model-free)"로 쓴다(부록 D).
- 실로봇 RL의 큰 정책 사후학습(π0.6\* RECAP [23], Fleet-Scale RL [24])은 배경으로만 인용된다.

### 1.2 개입에서 배우기 (논문 인용, 일부)

- **HG-DAgger** ([16]), **ThriftyDAgger** ([17], 위험과 새로움으로 개입을 요청), **Robot learning on the job** ([18]): 개입을 모방 학습의 마스크, 가중치, 요청 조건으로 쓴다. 논문의 주장은 이들이 개입 라벨을 "한 번 쓰고 버린다"는 것이다(1절, 2절).
- **논문이 인용하지 않은 가까운 연구(미확인):** SafeDAgger(Zhang & Cho 2017)는 정책이 기준 정책에서 벗어날지를 예측하는 안전 분류기로 개입을 촉발한다. Recovery RL(Thananjeyan 외 2021)은 제약 위반 확률을 학습한 안전 비평가로 복구 구역을 정한다. 둘 다 "위험을 예측하는 학습된 머리"를 쓰므로 WHIRL의 새로움 주장(결론의 "first HIL-RL system ... per-state takeover prediction as an actor-side risk signal on a 16-DoF dexterous hand")은 "개입 라벨을 감독으로 하는 예측을 actor 손실에 넣고, 손이 16-DoF"라는 조건의 합으로 읽어야 한다. 이번 세션에서 두 논문을 다시 읽지는 않았다.

### 1.3 잠재 세계모델 (논문 인용)

- **Dreamer / DreamerV3** ([19], [20]), **MBPO** ([21]), **TD-MPC** ([22]): 잠재 모델로 환경 양을 예측하고 상상 데이터로 actor나 비평가를 학습한다.
- **Hi-WM** (Li 외, arXiv:2604.21741, [26]): 사람이 학습된 세계모델 안에서 개입한다. WHIRL은 반대로 교정을 실로봇에 두고 모델은 1스텝 데이터 증폭과 상태별 위험 예측에만 쓴다(부록 C).
- WHIRL의 모델은 RSSM이 아니라 비평가 인코더의 분리된 잠재 위의 1스텝 결정론적 앙상블이다. 다단계 상상을 피하는 이유는 "작은 실로봇 데이터에서 다단계 오차가 누적된다"는 관찰이며 수치 근거는 없다(부록 C).

### 1.4 텔레오퍼레이션과 시너지 (논문 인용)

- **AnyTeleop**(논문은 "DexRetargeting"이라 부름, [27]), **DexPilot** ([28]), **DexCap** ([29]), **Bimanual Dexterity / Bidex Manus Teleop** ([30]): 손끝 IK나 관절 사상으로 손을 따라가게 한다. WHIRL은 비엄지 손가락은 아핀 사상, 엄지는 손끝 IK + 대립 방향 항으로 나눈 "형태 경로" 리타게터를 쓴다.
- **손 시너지** (Santello 1998 [31], Ciocarlie & Allen 2009 [32], eigengrasps [33]): 손 잔차를 3차원 자세 기저로 제한하는 근거. 기저를 어떻게 만드는지는 이 논문에 없다.

## 2. 구성 요소별 출처와 변경

| 구성 요소 | 출처 | WHIRL에서 바뀐 점 |
|---|---|---|
| 동결 BC + 잔차 off-policy | ResFiT [25], 잔차 정책 [7, 34] | 손 잔차를 K = 3 시너지 공간으로 제한, 비평가가 [a, a_BC]를 받음 |
| 개입 전환 저장 | HIL-SERL [14], SiLRI [15] | 개입 전환에 운영자 명령 저장, 페달 라벨 i_t를 세계모델 감독으로 재사용 |
| 불확실성 게이트 imagined 목표 | MBPO류 모델 기반 데이터 증폭 [21] | 앙상블 표준편차를 소프트 게이트(τ = 0.03, β = 0.01)로 사용 |
| 종료 머리, 보상 머리 | Dreamer류 [19, 20] | 보상 머리는 1[r > 0] BCE 후 R_succ = 10, R_step = −0.01로 보정 |
| 개입 확률 머리 p̂_intv | 새로움(저자 주장) | actor 보조 손실(식 6)에만 사용, 비평가 목표에서 제외 |
| 텔레오퍼레이션 리타게터 | [27, 28, 30] | 형태 경로 + 개입 이음매 제거(식 1, 11) |

## 4. GitHub와 자료 (확인 2026-11-24)

| 자료 | 링크 | 라이선스 | 별 / 마지막 푸시 | 비고 |
|---|---|---|---|---|
| **공식 코드** | 없음 | - | - | 프로젝트 페이지에 "Code coming soon". 논문 부록 B는 "보조 코드의 작업 설정 파일"을 언급하나 아직 공개되지 않음 |
| 프로젝트 페이지 | https://whirl-dexterous.github.io/ | - | 0 / 2026-09-20 (저장소 `whirl-dexterous/whirl-dexterous.github.io`) | 정적 페이지. 영상, 방법 그림, 인용. 수치 없음 |
| 제3자 구현 | 찾지 못함 | - | - | GitHub 검색(제목, WHIRL, LEAP + intervention 조합). `whirl-dexterous` 계정은 위 페이지 저장소 하나뿐 |
| HIL-SERL | https://github.com/rail-berkeley/hil-serl | Apache-2.0 | 1601 / 2025-10-27 | WHIRL의 Residual RL 기준선이 따르는 메커니즘. 논문은 공개 코드로 돌리지 않음 |
| SERL | https://github.com/rail-berkeley/serl | Apache-2.0 | 872 / 2025-10-27 | 실로봇 RL 프레임워크 |
| ResFiT 제3자 구현 | https://github.com/lucidrains/resfit-pytorch | MIT | 18 / 2025-09-29 | 저자 공식 저장소는 이번 세션에서 확인하지 못함 |
| dex-retargeting (AnyTeleop) | https://github.com/dexsuite/dex-retargeting | MIT | 1322 / 2025-08-21 | 논문의 "DexRetargeting" 비교 대상 |
| Bidex Manus Teleop | https://github.com/leap-hand/Bidex_Manus_Teleop | MIT | 109 / 2025-05-08 | LEAP + Manus 관절 사상 기준 |
| LEAP Hand API | https://github.com/leap-hand/LEAP_Hand_API | 파일 판별 불가(NOASSERTION) | 179 / 2025-10-20 | 손 드라이버 |
| SiLRI, Hi-WM | 저장소를 찾지 못함 | - | - | 미확인 |
| 데이터와 가중치 | 공개 없음 | - | - | 시연, 체크포인트 모두 없음 |

별 수와 푸시 날짜는 `gh api`로 읽은 값입니다.
