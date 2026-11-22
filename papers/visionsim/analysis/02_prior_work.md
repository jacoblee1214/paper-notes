# 02. 선행연구와 구성 요소의 출처: VisionSIM (arXiv:2611.08368)

## 1. 계보

VisionSIM은 새 알고리즘이라기보다, 저자 그룹(위스콘신 Gupta 그룹, 포틀랜드 주립 Ingle 그룹, CMU Narasimhan 그룹)이 각자 개발해 온 센서 모델을 Blender 렌더링 위에 한 파이프라인으로 묶은 플랫폼입니다. 계보를 모달리티별로 나누면 다음과 같습니다.

| 줄기 | 선행 | VisionSIM에서의 역할 |
|---|---|---|
| 이벤트 카메라 에뮬레이션 | ESIM (Rebecq 외, CoRL 2018, [7]), v2e (Hu 외, CVPRW 2021, [8]) | 이벤트 모델은 v2e를 따름(S.3). 공식 코드도 v2e 포팅 |
| 단일 광자 영상 | Quanta burst photography (Ma 외, TOG 2020, [27]), 고플럭스 수동 영상 (Ingle 외, CVPR 2019, [28]), 비동기 단일 광자 3D (Gupta 외, ICCV 2019, [6]) | 수동 SPAD의 베르누이 모델, dToF의 파일업 해석 |
| 단일 광자 LiDAR 히스토그램 | 등깊이 히스토그램 (Sadekar 외, ECCV 2024, [85]), 레이스 로직 (Ingle, Maier, TPAMI 2023, [84]) | EWH, EDH 읽기 방식. 구현은 별도 SPCSimLib |
| iToF | 최적 코딩 함수 (Gupta 외, TOG 2018, [41]), 합성 ToF 데이터 FLAT ([61]) | 상관 모델과 코딩 함수 목록 |
| 열 영상 | 빛-열 수송 이론 (Ramanagopal 외, CVPR 2024, [29]), 이중 대역 열 영상 ([95]), 열전도로부터의 형상 ([68]), 마이크로볼로미터 모션 블러 제거 ([97]) | 열전달 식 S5–S7의 틀, 방출 모델, 마이크로볼로미터 1차 모델 |
| 비다양체 메쉬 라플라시안 | Sharp, Crane (SGP 2020, [72]), Wardetzky 외 ([74]) | 열 솔버의 mollified 라플라시안 |
| 렌더링과 합성 데이터 | Blender, Kubric ([23]), BlenderProc ([25]), Spring ([19]), Genesis ([3]) | 세계 시뮬레이션 층의 Blender 선택, 장면 흐름 계산 |
| 프레임 보간 | RIFE (Huang 외, ECCV 2022, [31]), XVFI ([44]) | 고속 프레임 합성 |

논문 자신의 위치 주장은 "센서 폭 + 물리 충실도 + 파라미터 제어를 한 파이프라인에서"이며(Sec. 1, Sec. 2), "기존 범용 시뮬레이터는 센서를 이상화한다"는 대비가 핵심입니다. 센서별 시뮬레이터(ESIM, v2e)는 폭이 좁고, 열 시뮬레이터(MuSES, DIRSIG)는 지형 규모이거나 상용이라는 논리입니다(Sec. 1).

## 2. 가까운 대안

| 대안 | 센서 | 차이 |
|---|---|---|
| Isaac Sim과 Isaac Lab 센서 스택 | RTX 카메라, 라이다(레이캐스트), 접촉 센서, 시각촉각 | 논문 Sec. 1은 센서가 이상적이거나 단순 노이즈라고 주장함(우리가 확인하지 않음). 로봇 학습 처리량이 목적 |
| CARLA, AirSim | 카메라, 라이다, 이벤트(일부) | 도메인 특화, 원시 측정 모델 없음 |
| Genesis, MuJoCo, ManiSkill | 카메라 위주 | 물리와 병렬 처리량 중심 |
| ESIM, v2e | 이벤트만 | 모달리티 하나. VisionSIM이 v2e를 포함 |
| Kubric, BlenderProc | 렌더링과 GT | 센서 에뮬레이션 없음 |
| SPCSimLib | 능동 SPAD (EWH, EDH) | VisionSIM의 dToF가 실제로는 이 라이브러리에 기대고 있음(`04` D1) |

"로봇 학습용 시뮬레이터"와 "센서 물리 시뮬레이터"는 서로 다른 축입니다. VisionSIM은 후자에 속하고, 로봇 폐루프(정책 학습, 접촉 물리)는 다루지 않습니다(Sec. 8). 따라서 Isaac Lab이나 ManiSkill 대비 처리량 비교는 의미가 없고, 논문도 하지 않습니다.

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 위치 |
|---|---|---|
| Blender, Cycles, Bullet | Blender 커뮤니티, Bullet (Coumans) | S.2, S.2.4 |
| RPC 분산 렌더링 | rpyc (공식 코드 의존성), UDP 레지스트리 | S.2.2 |
| HTCondor 대규모 생성 | CHTC ([65]) | S.2.2, S.7 |
| 키프레임 재매핑 | 자체 | S.2.3 |
| 보간 | RIFE ([31]), 대안 FILM ([76]) | S.2.7 |
| 키프레임 배수 탐색 (식 S8) | 자체. ESIM의 광류 임계 트리거 대비 병렬화 쉬움을 주장 | S.2.7 |
| 열 솔버 | Narayanan 외 ECCV 2024 ([68]) 틀, mollified 라플라시안 ([72]), 포아송 디스크 표본 ([73]) | S.2.5.2 |
| 열 방출 모델 | 그레이바디, Sakuma-Hattori ([96]), 이중 대역 열 영상 ([95]) | S.5.4.1 |
| 마이크로볼로미터 | Ramanagopal 외 RSS 2020 ([97]) 계열 1차 모델 | S.5.4.2 |
| 이벤트 | v2e ([8]) | S.3 |
| RGB 모델 | Hasinoff 외 CVPR 2010 ([81]), Grossberg-Nayar ([51]) | S.3 |
| IMU | Crassidis, Leutenegger 외 IJRR 2015 ([53]) | S.3 |
| 저장 형식 | NeRF의 transforms.json 확장, 비트팩 numpy | S.4 |
| 장면 흐름 | Spring ([19]) 방식 대신 후처리로 계산 | S.2.6 |

## 5. GitHub와 코드 표

조회일 2026-11-22 (`gh api`, 메타데이터만).

| 저장소 | 관계 | 라이선스 | ★ | 마지막 push | 확인한 것 |
|---|---|---|---|---|---|
| [WISION-Lab/visionsim](https://github.com/WISION-Lab/visionsim) | 공식 구현 (Python 패키지 `visionsim`, CLI `vsim`) | GPL-3.0-or-later (`LICENSE.md`, 상업용 별도 라이선스 문의 문구 포함). API는 NOASSERTION | 51 (포크 5, 열린 이슈 4) | 2026-11-21 | HEAD `eda2b4b` 얕은 복제(정적 읽기). PyPI 0.2.0은 2026-11-03 업로드 |
| [wision-lab.github.io/visionsim](https://wision-lab.github.io/visionsim/) | 프로젝트 페이지 | (페이지) | - | - | 센서 데모 영상. 데이터셋은 "Coming soon. The prerelease is available in the meantime." |
| [visionsim.readthedocs.io](https://visionsim.readthedocs.io/en/latest/) | 문서 | - | - | - | 200 응답 확인 |
| [wisionlab.com/project/visionsim](https://wisionlab.com/project/visionsim/) | 연구실 프로젝트 페이지 | - | - | - | 200 응답 확인 |
| [kaustubh-sadekar/SPCSimLib](https://github.com/kaustubh-sadekar/SPCSimLib) | 능동 SPAD(dToF) 에뮬레이션 외부 라이브러리. VisionSIM 문서가 의존한다고 밝힘 | `LICENSE.txt`는 저작권 문구만(API NOASSERTION) | 8 | 2025-08-19 | 메타데이터만 |
| [SensorsINI/v2e](https://github.com/SensorsINI/v2e) | 이벤트 모델의 원본 | MIT | 504 | 2026-11-21 | 메타데이터만 |
| [hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE) | 보간 | MIT | 5,714 | 2025-09-10 | 메타데이터만 |
| [uzh-rpg/rpg_esim](https://github.com/uzh-rpg/rpg_esim) | 이벤트 시뮬레이터(비교 대상) | MIT | 745 | 2023-12-25 | 메타데이터만 |
| [google-research/kubric](https://github.com/google-research/kubric) | 데이터셋 생성기(비교 대상) | Apache-2.0 | 2,859 | 2026-05-21 | 메타데이터만 |
| [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/Genesis) | 물리 엔진(이벤트 이식 예제) | Apache-2.0 | 30,285 | 2026-11-22 | 메타데이터만 |

공식 저장소 보조 정보:

- 열린 PR #39 "Add active single photon cameras (LiDARs)" (2026-11-03 생성, 브랜치 `active-spc`, 49개 파일, +10,896줄). 설명에 "단일 픽셀, 단일 프레임 반복 방식만 지원, EWH 지원, EDH 미검증"이라고 적혀 있습니다. 논문의 dToF가 main에 아직 합쳐지지 않았다는 직접 증거입니다(`04` D1).
- 열린 PR #23 "Additions to image sensor simulations" (2026-01-15). RGB ISP(모자이킹, 디노이즈), SPAD 이진 합산 등을 추가하는 제안입니다(`04` D7).
- 릴리스: GitHub 릴리스는 `v0.1.0-alpha`(2025-04-29)뿐이고, 태그 `v0.2.0`, PyPI `0.2.0`(2026-11-03)이 있습니다.
- 데이터셋 장면(100개), 체크섬, 센서 원시 데이터는 저장소에 없습니다. `tests/test_files/scenes/`에는 `cube.blend` 하나뿐입니다(`04` D13).
