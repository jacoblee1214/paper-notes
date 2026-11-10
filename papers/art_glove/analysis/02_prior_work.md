# ART-Glove 선행연구와 관련 저장소 (02)

## 1. 계보

사람 시연을 로봇 학습에 쓰는 흐름은 세 갈래입니다. 첫째, 로봇을 직접 조종하는 텔레오퍼레이션(TeleVision [1], DexPilot [18], AnyTeleop [19], ALOHA 계열 [20])입니다. 촉각 피드백이 약한 것이 한계로 꼽힙니다. 둘째, 사람이 로봇 손 모양의 장치를 끼는 웨어러블 계열입니다. 외골격(DOGlove [3], DexUMI [5], ExoStart [6], DexOP [7], WHED [8], DexExo [9], MILE [4])과 링크 구동 장치가 여기 속합니다. 셋째, 맨손 영상과 소프트 장갑입니다. 맨손 영상(EgoDex [10], EgoMimic [11], EgoScale [12], EgoVerse [13], EgoLive [30])은 손 기하와 접촉을 영상에서 추론하고, 소프트 장갑(DexCap [14], DexWild [15], OpenTouch [16], Osmo [17])은 손 자세나 촉각을 더하지만 표면이 변형됩니다.

ART-Glove는 세 번째 갈래 가운데 촉각을 가진 장갑이면서, 외골격처럼 강체 면을 쓰는 중간 지점을 노립니다(논문 §2). 촉각 기술은 비전 기반 센서(GelSight [33], 9DTact [34], TacTip [35], DelTact [36])와 압저항 장갑(STAG [37], Flex-glove [38], FlexiTac [39], 패브릭 배열 [40], HIPI [41])으로 나뉩니다. 저자들은 얇고 확장성 좋은 압저항을 택했습니다. 촉각 장갑 가운데 같은 저자의 HIPI [41]가 판독 회로를 제공합니다.

## 2. 가까운 대안 (논문 비교 대상과 공개 상태)

GitHub 조사는 2026-11-10에 `gh search repos`와 `gh api`로 했습니다. 라이선스는 GitHub API의 `license.spdx_id`이고, "none"은 표기 없음입니다.

| 논문 (arXiv) | 센싱과 장치 | 손 동작 보존 (논문 Table 1) | 접촉 정보 | 공개 저장소 (2026-11-10 조회) |
|---|---|---|---|---|
| DexCap [14] (2403.07788) | 손 자세 mocap, 촉각 없음 | 거의 자유 | 없음 | `j96w/DexCap`, MIT, 별 426, 최근 push 2024-10-10 |
| DexUMI [5] (2505.21864) | 손 인터페이스, 로봇손 배치 | 링크형 | 제한적 | `real-stanford/DexUMI`, MIT, 별 273, push 2026-04-18 |
| DOGlove [3] (2502.07730) | 힘 피드백 외골격 | 제한적 | 힘 피드백 (햅틱) | `TEA-Lab/DOGlove`, MIT, 별 147, push 2026-04-18 |
| OpenTouch [16] (2512.16842) | 전체 손 촉각 장갑 | 거의 자유 | 촉각, 희소 | `OpenTouch-MIT/opentouch`, 라이선스 없음, 별 83, push 2026-03-18 |
| Osmo [17] (2512.08920) | 자력계 촉각 장갑 | 거의 자유 | 촉각 (희소) | `jessicayin/osmo_tactile_glove`, 라이선스 없음, 별 88, push 2026-10-26 |
| FlexiTac [39] (2604.28156) | 압저항 촉각 모듈 | 장갑 아님 | 촉각 | `FlexiTac/FlexiTac`, 라이선스 없음, 별 177, push 2026-11-08 (하드웨어·펌웨어) |
| Feel Robot Feels [24] (2603.28542) | 촉각 배열 장갑 | 제한적 | 촉각 + 햅틱 | `InternRobotics/TAG`, 라이선스 NOASSERTION, 별 41, push 2026-04-24 (이름 일치로 판단, 논문 저자 저장소인지는 확인 안 함) |
| DexOP [7], MILE [4], HIPI [41], Fits-like-a-flex-glove [38] | — | — | — | 검색 결과 없음 (2026-11-10) |

공개 상태만 보면 DexCap과 DexUMI는 재사용 가능한 코드(MIT)가 있지만 촉각이 없습니다. 촉각을 가진 장갑(OpenTouch, Osmo)은 라이선스 표기가 없어서 코드를 가져다 쓰기 어렵습니다. ART-Glove 저장소는 기계 설계 파일과 BOM을 라이선스 표기 없이 공개합니다(`04_code_verification.md` §1).

## 3. 이 저장소의 다른 분석과 연결

ART-Glove는 "사람 손 시연에 접촉 정보를 붙인다"는 점에서 아래 분석들과 입력과 출력이 맞닿습니다. 연결의 성격은 세 종류입니다. (a) 같은 촉각 배치를 쓰는 사람·로봇 쌍(DexTouch-WM), (b) 사람 시연을 로봇 손 궤적으로 바꾸는 리타게팅(FlashDexRetarget, GNR, CHORD), (c) 촉각 입력을 쓰는 정책과 sim-to-real(TactSpace, ContactGroundedPolicy, ResTacVLA, CONTACT, GraspSense). 접촉 라벨을 장갑 없이 얻는 대안은 NoContactNoWorries입니다.

| 분석 폴더 (arXiv) | 논문 한 줄 | ART-Glove와의 연결 |
|---|---|---|
| `dextouch_wm/` (2609.20649) | 사람 장갑과 로봇손에 같은 압전 촉각 배치를 달아 사람 촉각으로 로봇 촉각 세계모델을 돕는 연구 | 같은 배치의 사람 장갑이라는 전제를 ART-Glove 같은 장갑이 만족할 수 있음. ART-Glove는 taxel-to-surface 매핑을 제시하지 않아 두 장치의 촉각 좌표를 맞추려면 추가 작업이 필요함 |
| `contact_grounded_policy/` (2603.05687) | 시각-촉각 확산 정책이 접촉 일관성 매핑으로 compliance 목표를 만드는 연구 | "접촉 기반" 시연 표현의 소비자. ART-Glove의 면 단위 촉각을 접촉 일관성 입력으로 쓸 수 있음 |
| `graspsense/` (2604.05697) | 표면별 허용 힘 지도로 파지 후보를 재순위화 | 힘 지도는 법선 힘 정확도가 필요함. ART-Glove는 힘 보정이 없어(§8) 직접 입력으로 쓰기 어려움 |
| `flashdexretarget/` (2610.01849) | 사람 손-물체 시연 모음을 RL 정책 하나로 로봇손 궤적으로 옮김 | ART-Glove가 주는 관절 22개와 면 자세는 FlashDexRetarget의 사람 입력 형식과 맞는지 확인 필요. 접촉 라벨 추가는 보상 설계 문제 |
| `gnr/` (2610.12440) | MPC 리타게팅 궤적을 생성 모델로 보정하는 신경 리타게팅 | 관절 기준 리타게팅. 토이 C의 관절 공간 목적함수와 같은 계열 |
| `chord/` (2607.00033) | 사람 접촉점·법선으로 마찰원뿔 wrench를 만들어 보상에 넣음 | CHORD 입력(접촉점, 법선)을 ART-Glove 면 접촉으로 얻을 수 있는지가 핵심. 법선은 면 기하로 주어지고 접촉점은 taxel 격자로 근사 가능 |
| `tactspace/` (2606.18959) | 시뮬 촉각과 실측 정전용량을 짝지어 sim-to-real | ART-Glove는 압저항이므로 같은 방법을 쓰려면 압저항 시뮬 모델이 필요 |
| `nocontact_noworries/` (2606.24450) | 손목 RGB-D와 관절 명령으로 손가락 접촉 추정 | 장갑 없이 접촉 라벨을 얻는 기준선. ART-Glove 접촉 라벨의 대안 비교 대상 |
| `restacvla/` (2607.03387) | π0.5에 촉각 잔차 토큰을 붙이는 VLA | 촉각 입력 정책의 한 예. ART-Glove 데이터로 학습하는 경우의 입력 형식 참고 |
| `contact_disassembly/` (2603.08560) | 시각, 촉각 영상, 촉각 힘장을 같은 확산 정책에서 비교 (CONTACT) | 촉각 표현 비교 (영상 대 힘장). ART-Glove의 힘 없는 촉각은 TacRGB 쪽 입력에 가까움 |

## 4. ART-Glove 자체의 공개 상태 (GitHub)

| 항목 | 값 |
|---|---|
| 저장소 | `linchangyi1/ART-Glove` (프로젝트 페이지와 arXiv abs에서 링크됨) |
| 생성 / 마지막 push | 2026-05-31 / 2026-11-07 (API `pushed_at`) |
| 별 | 46 |
| 라이선스 | 표기 없음 (README: "Licensing details will be announced later") |
| 확인한 커밋 | `7e210b8a805e68017eac6fbd1285013e09ebb8ef` (2026-11-06 커밋 시각, "fix format") |
| 내용 | 기계 설계 STEP(zip), 3MF, DXF 절단 도면, BOM, 조립 가이드 |
| 출시 예정 | PCB(판독, MCU, FFC 허브, 엔코더), 촉각 FPCB, 펌웨어와 Python 캘리브레이션·시각화·기록·재생 도구, 조립 영상, 통합 판독 PCB, 로봇용 버전 |
| 프로젝트 페이지 | `linchangyi1.github.io/ART-Glove` (영상 "coming") |

자세한 대조는 `04_code_verification.md`입니다.
