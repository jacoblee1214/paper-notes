# 02. 선행연구와 관련 저장소

UME의 구성 요소가 어디서 왔는지, 가까운 대안과 무엇이 다른지, 공식 코드와 관련 저장소가 어떤 상태인지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| UME의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 조종자–팔로워 텔레오퍼레이션으로 시연 수집 | ALOHA [13], ALOHA 2 [21], Mobile ALOHA [14, 42], GELLO [15], FACTR [43] | 리더 암을 손으로 끌어 말단을 조종하는 계열. 논문은 이 계열이 관절 토크를 기록·되돌리지 않고 말단 수준 제어만 준다고 비판합니다(§1) |
| 정책 학습 | ACT [13] (Zhao 2023), ResNet18 [33], Transformer [34] | UME 정책은 ACT 구조입니다. 공개 코드는 CVAE를 쓰지 않는 변형입니다(`04` D9) |
| 휴대형 시연 장치(손에 든 그리퍼) | UMI [16], DexUMI [17], UMI-FT [18], HoMMI [19] | 말단 자세만 기록하고 IK로 충돌을 처리해야 한다고 논문이 대비시킵니다(§1). UMI-FT(저자 중 Cutkosky 공통)는 힘/토크를 기록하는 UMI 계열입니다 |
| 상체 외골격 텔레오퍼레이션 | HOMIE [49] (등형 외골격, 발 페달 이동), AirExo [50] / AirExo-2 [51] (비등형, 보정 필요), ACE [52] (손목 자세와 손가락) | 가장 가까운 대안. UME는 등형(관절 1:1)에 가깝게 만들면서 **토크 피드백**을 더하고 이동을 IMU로 처리합니다(부록 A) |
| 동축 어깨·손목 외골격 기구 | ANYexo [26], ANYexo 2.0 [27] (Zimmermann), ARMin III [44], Harmony [46] 등 재활 외골격 [44–48], 인간 팔 3-1-3 구조 [24, 25] | 기구 영감. 재활 외골격의 관절 정렬 아이디어를 값싼 PLA+알루미늄+준직접구동 모터로 옮겼습니다 |
| 낮은 감속비 준직접구동 액추에이터 | Damiao [22], Unitree [23] | Dynamixel [20] 대신 백드라이브성·토크 투명성을 얻습니다(§1, 부록 E.1) |
| 매크로–미니 분해 | Khatib [29] | 팔을 세 하위 매니퓰레이터로 분해한 영감(§2.2) |
| 동적 일관 역행렬과 작업 공간 힘 사상 | Khatib 작업 공간 형식 [30] | 토크 사상 F = J̄ᵀ τ (§2.2) |
| 모바일 베이스 | Tidybot++ [32], Holmberg & Khatib [31] | 스워브 모듈을 Hexfellow PCW-25로 교체(부록 E.4) |
| 팔로워 로봇 | OpenArm [22, 28] (Enactic), WowRobo OpenArm 1.0 [65], X-ARM [38], Franka [39] | 이중 팔 모바일 매니퓰레이터는 자체 제작(부록 E.4) |
| 순응 정책 학습 | Adaptive Compliance Policy [60], Compliant Residual DAgger [12], 엔드투엔드 토크 정책 [61, 62], 어드미턴스 [58], 임피던스 [59], teleimpedance [41] | 대비: ACP와 [12]는 고수준 위치 정책 + 하위 순응 제어기, [61, 62]는 토크를 직접 내는 정책. UME는 **위치 목표를 내고 고정 PD가 순응을 정하는 중간형**입니다(`04` D5) |
| 접촉이 불가피한 좁은 공간 과제 | SLIM [35], Gentle retraction [36], Going in blind [37] (Cutkosky 연구실) | GPU 집기·냉장고 과제의 동기(§3, Ch2) |

**가까운 대안과의 차이.**

- **GELLO·ALOHA 계열 대비:** 토크 기록과 되돌림, 팔 전체 관절 구성을 줍니다. 대신 외골격은 사람이 입어야 하고(약 12 kg) 팔로워의 관절 구조와 닮아야 하는 제약이 있습니다.
- **UMI 계열 대비:** 로봇 없이 모으는 이점은 잃지만, 팔 전체 관절 구성과 토크를 얻습니다. 논문의 UMI 기준선은 같은 시연의 말단 자세 + 충돌 제약 없는 IK 구현입니다(`04` D13).
- **AirExo 계열 대비:** 비등형 보정이 필요 없고 토크를 되돌린다는 점이 다릅니다.

## 2. 코드와 저장소 (GitHub 표)

2026-11-23 `gh api`로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 대조 기준·상태 |
|---|---|---|---|---|---|
| [inclusionAI/Universal-Manipulation-Exoskeleton](https://github.com/inclusionAI/Universal-Manipulation-Exoskeleton) | **공식 구현** (하드웨어 CAD, 텔레오퍼레이션, 수집, ACT 학습·평가 코드). 논문 초록·프로젝트 페이지의 Code 링크 | Apache-2.0 | 23 (포크 3) | 2026-07-16 | 커밋 `e087df5` (2026-07-15), 얕은 복제 `repo/`, 정적 읽기만(`04`) |
| [ume-exo.github.io](https://ume-exo.github.io/) | 프로젝트 페이지 (영상, CoRL 2026 로고) | (페이지 템플릿은 CC BY-SA 4.0 표기) | — | — | HTML 확인. 영상은 받지 않음 |
| [enactic/openarm](https://github.com/enactic/openarm) | 팔로워 팔 OpenArm (논문 [28]) | Apache-2.0 | 3,717 | 2026-11-16 | 복제하지 않음. 공식 UME 코드는 OpenArm MJCF를 포함 |
| [jimmyyhwu/tidybot2](https://github.com/jimmyyhwu/tidybot2) | 모바일 베이스 원형 (논문 [32]) | MIT | 624 | 2025-12-18 | 복제하지 않음 |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | ACT 원본 (논문 [13]) | MIT | 2,352 | 2024-07-23 | 복제하지 않음. UME 코드는 `detr/` 하위에 포함한 변형 |
| [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software) | GELLO (논문 [15]) | MIT | 566 | 2026-11-10 | 복제하지 않음 |
| [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) | UMI (논문 [16], 기준선의 입출력 형식 원본) | MIT | 1,647 | 2026-05-26 | 복제하지 않음 |
| [dmBots/motor-debugging-tool](https://github.com/dmBots/motor-debugging-tool) | Damiao 모터 설정 도구 (공식 문서 1단계에서 링크) | 라이선스 없음 | 8 | 2026-11-19 | 복제하지 않음. Windows 도구 바이너리 |

- 시연 데이터와 학습된 체크포인트는 공식 저장소에도, 프로젝트 페이지의 Data 링크(주석 처리되어 템플릿의 원본 URL을 가리킴)에도 없습니다.
- 저장소 이슈 0건, 릴리스 없음, 전체 커밋 3개(논문 공개 2026-06-12 이후 2026-07-01 생성, 07-16 공개 push).
- README의 연락처 이메일과 논문 저자의 개인 연락처는 이 분석 문서에 옮기지 않았습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **동축 외골격의 가동 범위 이점**은 재활 외골격 문헌(ANYexo [26, 27])의 설계 원칙에 기댑니다. UME의 새 점은 이를 모터 16개 $1,900대로 구현한 것과 토크 되돌림입니다.
- **"토크 센서가 없는 시연 데이터로는 순응 정책을 못 배운다"**는 논리는 ACP [60], Compliant Residual DAgger [12]가 이미 강조한 문제의식입니다. UME는 이를 관절 토크 입력 + 위치 목표 출력으로 풀되, 관절 토크는 전류 기반 추정이라는 점에서 힘 센서 기반 연구와 신호의 질이 다릅니다.
- **UMI 대비 우위** 주장은 UMI-FT [18]처럼 힘 입력이 들어간 UMI 계열과의 비교가 없어 "토크가 있는 UMI 계열"에 대한 우위는 확인되지 않았습니다.
