# 02. 선행연구, 코드와 저장소, 관련 분석

## 1. 선행연구 계보

논문이 서 있는 자리는 네 갈래의 교차점입니다(논문 2절). 인용 번호는 논문 참고문헌 번호입니다.

### 1.1 원격조작 인터페이스

- 리더-팔로워 장치: ALOHA [1], Mobile ALOHA [3], GELLO [7]. 정밀하지만 전용 하드웨어가 필요하고 과제 수준 제약은 없습니다.
- 비전·XR 인터페이스: AnyTeleop [14](카메라 추적), OPEN TEACH [9](VR 손 추적), Open-TeleVision [8](스테레오 능동 시각 피드백). GLIDE의 기준선 "OpenTV"가 이 계열이고, 코드도 Open-TeleVision 위에 있습니다.
- 촉각·힘 피드백·외골격: [15–18]. 이들은 명령이 과제를 망칠 위험을 따로 검사하지 않는다고 논문은 구분합니다.
- 가장 가까운 동시기 연구: Sha et al. [19] (arXiv:2603.17016)의 real-to-sim-to-real 공유 자율 부조종사. 학습된 잔차 보조 정책이라는 점이 다르고, GLIDE는 실패에서 추론한 실행 가능한 코드를 데이터 수집과 정책 배치 양쪽에 재사용한다고 대비합니다. 이 저장소 분석 시점에 그 논문 본문은 읽지 않았습니다(인용으로만 확인).

### 1.2 불완전 시연에서 학습

- 사람 개입을 신호로: SIRIUS [20](개입으로 OOD 행동 교정), RLIF [21](개입을 보상으로), ConRFT [22](VLA용 RL).
- 데이터 큐레이션: Data quality [23], DataMIL [24], CUPID [25], VLM 가치 학습기 [11], TOPReward [26]. 시연을 고르거나 가중합니다.
- 알고리즘: DWBC [27](판별자 가중 BC), DAgger [28], 소프트 전문가 안내 RL [29], ADVISOR [30], VRB [31], PATO [32](반복 하위 과제 자동 수행과 불확실성 질의).
- GLIDE의 구분: 이들은 학습 단계에서 데이터를 다루고, GLIDE는 명령이 실행되기 **전에** 제약을 걸어 더 나은 데이터가 모이게 합니다.

### 1.3 로봇을 위한 LLM과 코딩 에이전트

SayCan [33], WildLMa [34], Code as Policies [35], VoxPoser [12], Eureka [10], DrEureka [36], RL-VLM-F [13]. 더 최근의 코딩 에이전트 계열로 CaP-X [37](arXiv:2603.22435), ASPIRE [38](2607.00272), ENPIRE [39](2606.19980, 실세계 정책 자기 개선), VIA [40](2607.11119)을 듭니다. GLIDE는 이들이 전체 컨트롤러를 만들거나 학습 레시피를 개선하는 것과 달리 "기존 조작자나 학습 정책의 실행 시점에 얹는 가벼운 가드"를 쓴다고 구분합니다.

### 1.4 논문이 인용하지 않은 가까운 개념 (우리가 붙인 연결, 논문에 없음)

- **안전 필터와 제어 장벽 함수, 가상 고정구(virtual fixture).** 사람이나 정책의 명령을 제약 집합으로 사영해서 실행하는 구조는 로봇공학에서 오래된 것입니다. GLIDE의 가드는 같은 구조지만 제약을 코딩 에이전트가 쓰고, 증명 가능한 안전성 대신 라운드별 실험적 성공률로 개선합니다. 논문은 이 계보와의 관계를 따로 논하지 않습니다.
- **공유 자율(shared autonomy).** 사람 의도를 추정해 보조하는 틀입니다. GLIDE는 의도 추정이 없고 과제 단계는 그리퍼 닫힘 정도 같은 신호로 규칙적으로 판정합니다(부록 B의 트리거).
- 두 연결 모두 문헌 조사 없이 적은 해석이므로 근거 인용이 필요하면 따로 찾아야 합니다.

## 2. 코드와 저장소 (조사일 2026-11-26)

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [Yuchen-Song/glide_code](https://github.com/Yuchen-Song/glide_code) | 공식 구현 (실로봇 런타임: 조작 3종, 정책 클라이언트 4종, 가드레일 최종 형태). 87개 파일 | Apache-2.0 | 5 / 2026-09-23 | 커밋 `8ebb33a6dca12a2091facccdc45a1a6ffbb50e66` (2026-09-23, 얕은 복제 1커밋) | 정적 읽기만 (`repo/`, 실행·설치 안 함) |
| [yuchensong/glide_data](https://huggingface.co/datasets/yuchensong/glide_data) | 공식 데이터 (LeRobot v2.1, `*_mixed`와 `*_success` 6개) | CC BY 4.0 | 다운로드 1,124 / 2026-09-23 수정 | 저장소 크기 2.75 GB(`usedStorage`), 부분집합 합계 약 4.2 GB | 받지 않음 (API 메타데이터와 README 카드만 읽음) |
| [guardrail-policy.github.io](https://guardrail-policy.github.io/) | 프로젝트 페이지 | - | - | 발표 일정, 코드·데이터·포스터 링크 | 읽음 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 정책(π0.5) 학습·서빙 기반. 공식 저장소에는 `openpi-client`만 복사 | Apache-2.0 | 14,767 / 2026-08-24 | 학습 설정은 GLIDE가 "OpenPI 코드베이스에서 직접 수정"했다고 README가 적음(공개 안 됨) | 참고 |
| [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) | 로봇 드라이버(YAM). 공식 저장소가 `third_party/i2rt`로 수정본을 포함 | MIT | 184 / 2026-11-25 | - | 참고 |
| [OpenTeleVision/TeleVision](https://github.com/OpenTeleVision/TeleVision) | 기준선 OpenTV, Quest 연동 | 라이선스 불명(API NOASSERTION) | 1,321 / 2024-09-27 | - | 참고 |
| [craft-hand/CRAFT-Hand_API](https://github.com/craft-hand/CRAFT-Hand_API) | 15자유도 CRAFT 손 API | 확인 안 함 | 18 / 2026-06-04 | - | 참고 (저장소의 `craft-hand/helpers`가 최소 지원 코드를 포함) |

- 공식 저장소에 **없는 것**: 코딩 에이전트에게 보낸 프롬프트와 호출 코드(프롬프트는 논문 부록 C에만), 라운드별 가드 G^(0), G^(1), 정책 학습 설정과 체크포인트, 평가 로그, OpenTV·Manual 기준선 데이터(공개 데이터는 GLIDE 부분집합 6개뿐). 자세한 대조는 `04`입니다.
- Manual 가드 구현은 저장소에 있습니다(`teleop_gripper_plate_manual.py`, `teleop_craft_manual.py`, 마커는 base 스크립트의 `--lock`).

## 4. 구성 요소별 출처

| GLIDE의 구성 | 출처 |
|---|---|
| 정책 기반 모델과 흐름 정합 목적 (식 5) | π0.5 [4] (openpi) |
| VR 원격조작 파이프라인과 기준선 | Open-TeleVision [8], i2rt 드라이버 |
| 로봇 플랫폼 | I2RT YAM Ultra [41], CRAFT 손 [42], Meta Quest Pro [43] |
| 코딩 에이전트로 제약 코드 작성(식 2, 3) | 코딩 에이전트 계열 [37–40], Code as Policies [35]의 "코드로 쓰기" 아이디어 |
| 가드 형태 (속도·가속 제한, EMA, 자세 고정, 쌍 결합) | 일반적인 필터링과 제약 사영. 논문은 출처를 따로 적지 않음 |
| 정책 출력에 같은 필터를 거는 아이디어(식 6) | 실행 시점 안전 필터 계열(논문은 인용 안 함) |
| 섞은 품질 데이터로 학습 (식 5의 D^mix) | 불완전 시연 계열 [23–27]. 다만 GLIDE는 "가드 아래에서 모은" 불완전 시연이라는 점이 다름 |
