# 02. 선행연구와 공식 저장소: YUBI (arXiv:2606.10244)

## 1. 선행연구 계보

논문이 서 있는 흐름은 "로봇 없이 로봇이 실행할 수 있는 시연을 모은다"입니다(§2.1, §2.2).

| 갈래 | 대표 | YUBI와의 관계 |
|---|---|---|
| 리더–팔로워 원격조종 | ALOHA(Zhao et al. 2023), Mobile ALOHA, GELLO(Wu et al. 2024) | 정확한 로봇 공간 시연이지만 수집 장소마다 로봇이 필요해 처리량이 낮다(§1, §2.1). YUBI의 비교 대상이 아니라 동기입니다. |
| 사람 영상, 손 재타깃팅 | Ego4D, AnyTeleop, DexCap, Open-TeleVision, HumanPlus, Being-H0 | 규모는 크지만 사람 손과 로봇 사이 신체 격차가 있다(§2.1). |
| 크로스 임베디먼트 데이터, 합성 | Open X-Embodiment, DROID, MimicGen | 접촉이 많은 양팔 정밀 작업은 부족하다고 주장(§2.1). |
| **손에 쥐는 그리퍼(UMI)** | UMI(Chi et al. 2024), FastUMI와 FastUMI-100K, LEGATO, Data Scaling Laws | **직접 계승.** 로봇 그리퍼와 같은 장치를 사람이 쥐게 해 손목 카메라와 궤적을 얻는다. UMI는 SLAM 추적(§2.2). |
| UMI + 촉각 | ViTaMIn, ViTaMIn-B, exUMI, Touch in the Wild, FreeTacMan | YUBI는 촉각을 넣지 않고 한계로 남긴다(§7). |
| UMI + VR 추적 | ActiveUMI, exUMI | **가장 가까운 대안.** 헤드셋을 머리에 쓰는 대신 책상 프레임에 고정하는 것이 YUBI의 차이(§1, §3.2). |
| 손가락 로봇 손 UMI | DexUMI, DexWild | 다지 손이라 장치가 복잡하다. YUBI는 평행 턱으로 단순화한 대신 손가락 정렬 구동을 택했다. |
| VLA | π0, π0.5, Diffusion Policy | 정책은 π0.5 후학습, 대조로 Diffusion Policy(§6, 부록 C.4). |

## 2. GitHub와 공개 자료

2026-11-22에 `gh api`와 arXiv·Hugging Face 메타데이터로 확인했습니다. 별 수와 push 시각은 그 시점 값입니다.

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 대조 기준 (얕은 복제) |
|---|---|---|---|---|---|
| [Toyota/yubi-hw](https://github.com/Toyota/yubi-hw) | **공식 하드웨어** (STEP, STL, BOM, 조립 가이드, 플랜지) | CERN-OHL-W-2.0 | 197 | 2026-09-21 | `dd8bd13` (2026-09-21). STEP와 STL은 받지 않고 README, BOM, LICENSE만 읽음 (저장소 44 MB) |
| [airoa-org/yubi-sw](https://github.com/airoa-org/yubi-sw) | **공식 수집 소프트웨어** (ROS 2 bringup, Quest 브리지, 엔코더 펌웨어, URDF) | Apache-2.0 | 62 | 2026-09-22 | `1eccb04` (2026-07-31) |
| [airoa-org/yubi-core](https://github.com/airoa-org/yubi-core) | 녹화, 업로드, 기록 게이트 (ROS 2 백엔드) | Apache-2.0 | 5 | 2026-06-01 | `620a069` (2026-06-01) |
| [airoa-org/yubi-app](https://github.com/airoa-org/yubi-app) | 수집 관리 웹 앱 (Go, TypeScript) | Apache-2.0 | 3 | 2026-06-12 | `dc1362d` (2026-06-12) |
| [airoa-org/rebake](https://github.com/airoa-org/rebake) | ROS bag를 Parquet+영상으로, 다시 LeRobot으로 바꾸는 Rust/Python 파이프라인 | Apache-2.0 | 14 | 2026-06-10 | `1ad9453` (2026-06-10) |
| [airoa-org/yubi-real2sim](https://github.com/airoa-org/yubi-real2sim) | 후속 논문 YUBI-Real2Sim(arXiv:2610.17521, CC BY 4.0)의 코드: Franka FR3 + 양손 YUBI, MuJoCo + 3DGS 배경, OpenPI π0.5 평가 | Apache-2.0 (3DGS 렌더러는 비상업) | 18 | 2026-11-09 | `c1d5d41` (2026-11-09). 저장소 17.9 MB (복제 105 MB) |
| [airoa-org/umi-arena-evaluation](https://github.com/airoa-org/umi-arena-evaluation) | UMI Arena(CoRL 2026 워크숍) 평가 러너와 어댑터 | Apache-2.0 | 2 | 2026-10-02 | `f60d4e6` (2026-10-02) |
| 프로젝트 페이지 [yubi.airoa.io](https://yubi.airoa.io/) | 영상, 링크. "Dataset (Coming Soon!)" 표시 | - | - | - | 2026-11-22에 확인 |
| UMI Arena [umi-arena.airoa.io](https://umi-arena.airoa.io/) | 2026-11-12 CoRL 워크숍. 경쟁용 데이터와 평가 안내 | - | - | - | - |
| Hugging Face `airoa-org/yubi-corl2026-umi-arena` 외 5개 | 경쟁용 데이터셋. **접근 제한(수동 승인), 라이선스 "other: airoa-limited-access-dataset-terms"** | 약관 | - | 2026-11-05 ~ 11-22 | 열람하지 않음. 메타데이터만 확인 (`04` §3) |

- 논문 부록 D의 "CERN-OHL-W v2 / Apache-2.0 / CC-BY-NC-SA-4.0"에서 앞의 둘은 저장소 라이선스와 일치합니다. 세 번째는 일치하지 않습니다(`04` D9).
- **정책 학습과 배치 코드(π0.5 후학습 설정, 로봇별 IK, 10 Hz 다운샘플)는 이 저장소들에 없습니다.** yubi-real2sim은 π0.5를 MuJoCo에서 평가하는 도구이고 학습 코드가 아닙니다(`04` D11).
- 원 저장소들은 모두 clone을 정적으로 읽기만 했고 실행, 설치, 가중치와 데이터 다운로드는 하지 않았습니다.

## 4. 구성 요소별 출처

| YUBI 요소 | 출처 | 새로운 정도 |
|---|---|---|
| 핸드헬드 그리퍼 + 손목 카메라 + 궤적으로 로봇 정책을 학습 | UMI | 개념은 그대로 |
| VR 컨트롤러로 그리퍼 6 DoF 추적 | ActiveUMI, exUMI | 헤드셋을 책상 프레임에 고정한 것이 변경점 |
| 손가락 정렬 구동 (엄지 대 검지·중지) | 논문의 주 기여 | 설계 요소 중 가장 새롭다. 정량 비교는 정밀도와 시간 시험뿐 |
| 발판 하위 행동 분할 | 새로움보다 운영 장치. 원 UMI의 시연 구간 지정과 비슷한 목적 | 보통 |
| ROS 2 + rosbag2 → LeRobot 변환, 웹 앱, 책상 22개 병렬 수집 | 엔지니어링 | 규모가 기여 |
| 정책 π0.5 후학습, 상대 말단 포즈 행동, 10 Hz 실행 | π0.5, UMI의 상대 포즈 행동 | 새롭지 않다 |
| 크로스 로봇 배치 | UMI 계열의 임베디먼트 불문 행동 공간 주장 | 로봇 세 대에서 시연(로봇당 과제 1–3개) |
