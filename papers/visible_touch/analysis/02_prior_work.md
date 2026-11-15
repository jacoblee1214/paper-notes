# Visible Touch 선행연구와 관련 자료 (arXiv:2609.14156)

분석일 2026-11-15. 논문 인용은 `paper/paper.txt`의 참고문헌과 `reference.bib` 키(TeX 소스)로 확인했습니다. GitHub 정보는 2026-11-15에 API로 확인한 값입니다(별, 라이선스, 최근 push).

## 1. 선행연구 계보

논문 §2는 접촉을 정책에 넣는 방법을 세 갈래로 나눕니다. 이 구분을 그대로 씁니다.

| 갈래 | 대표 연구(논문 인용 키) | 입력 위치 | 이 논문과의 차이 |
|---|---|---|---|
| 상태 벡터에 이어 붙임 | Lee 2019(`lee2019making`), FACTR(`liu2025factr`) | 고유감각 벡터 | 논문 Proprio 기준선이 이 갈래를 대표 |
| 별도 촉각 인코더 | ManiFeel(`xu2025manifeel`), CONTACT(`saka2026contact`), Tactile-VLA(`huang2025tactilevla`), ForceVLA(`yu2026forcevla`), FoAR, HapticVLA | 추가 인코더 스트림 | 논문 Tac-View는 같은 정보를 칸만 바꿔 넣은 대안 |
| 기하 정렬 점 구름 | Robot Synesthesia(`yuan2023robotsynesthesia`), 3D-ViTac(`huang20243dvitac`), SaTA | 3D 점 구름 백본 | 논문은 2D 이미지 토큰 구조에 맞춘 대안이라고 주장 |
| 접촉을 과제 명세로 | CAP(`cui2026contact`) | 조건 입력 | 직교 |

영상 위에 보조 정보를 그리는 선행연구도 있습니다. RT-Trajectory(궤적 스케치), TraceVLA(움직임 흔적), AimBot(말단 조준선), PIVOT과 MOKA(VLM 후보 행동과 키포인트)입니다. 논문은 이들이 "조밀하고 부드러운 신호"만 그렸다고 보고, 접촉처럼 희박하고 사건 중심인 신호는 그린 적이 없다고 주장합니다(§2). Audio-VLA(`wei2025audiovla`)는 접촉 음향을 별도 스트림으로 씁니다.

저비용 촉각 센서 계보: GelSight, DIGIT(비전 기반), ReSkin과 AnySkin(자기식 입자 엘라스토머, 자화 장비 필요), uSkin과 여러 자석 주조 설계, eFlesh(3D 프린트 TPU, 24 mm 적층). 논문 센서는 "4 mm 주조 본체, 부품 $5 미만"이라고 위치를 잡습니다(§2 마지막 문단).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 확인 |
|---|---|---|
| 측정 기반 BC-Transformer 학습 설정 | LIBERO(`liu2023libero`), 기본 구현 | App. C.1, LIBERO 기본 설정 사용 |
| miniVLA(Qwen2.5-0.5B, DINOv2+SigLIP) | `belkhale2024minivla` | App. D, 공개 체크포인트 `minivla-libero90-prismatic`과 대조 |
| π0.5, 체크포인트 `pi05_droid` | `black2025pi05` | App. E.1 |
| LoRA | `hu2021loralowrankadaptationlarge` | App. E.2 |
| 원격 조작 수집 | TeleDex(`rayyan2026teledex`, arXiv:2603.17065) | App. F.2 |
| 2D 영상 위 오버레이 기반 | RT-Trajectory, TraceVLA, AimBot | §2 |
| 자기식 센서 회로 | ESP32 구성(`tanaka2025mechanisms`, ICRA 2025 논문) | App. B.1 BOM |
| 핀홀 투영 | 표준 모델, 설명 없음 | App. A |

## 4. GitHub, 코드, 데이터 표 (2026-11-15 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 push | 대조 commit |
|---|---|---|---|---|---|---|
| [qaz1678249/visibletouch](https://github.com/qaz1678249/visibletouch) | CAD-몰드 파이프라인과 센서 YAML 세 개, STL 몰드(프로젝트 페이지가 링크) | 미확인. 논문은 "release"를 말하지만 저장소가 개인 계정이며 저자 이름이 없음 | 없음(LICENSE 파일 없음) | 0 | 2026-06-11 | `b31be56` (2026-06-10, 커밋 1개). `repo/`에 shallow clone, 정적 읽기만 |
| 오버레이 렌더링 코드 | 논문의 핵심(§3.1) | 저장소에 없음 | — | — | — | — |
| 실험 코드(BC, miniVLA, π0.5 학습과 평가) | 논문 Apps. C–F | 저장소에 없음 | — | — | — | — |
| PCB 레이아웃 | 논문 App. B.2("release the PCB board layout") | 저장소에 없음 | — | — | — | — |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | LIBERO 벤치마크(BC 기본 구현 포함) | 원 저자 | MIT | 2,541 | 2025-03-15 | 대조 안 함 |
| [Stanford-ILIAD/openvla-mini](https://github.com/Stanford-ILIAD/openvla-mini) | miniVLA 학습 기반(TeX 링크에서 확인) | 구성 요소 원 저자 | MIT | 383 | 2025-03-19 | 대조 안 함 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 원 구현과 `pi05_droid` 체크포인트 | 구성 요소 원 저자 | Apache-2.0 | 14,648 | 2026-08-24 | 대조 안 함 |
| miniVLA 공개 체크포인트 | `minivla-libero90-prismatic` | 원 저자 | 미확인 | — | — | 저장소 이름 `Stanford-ILIAD/minivla`은 API에서 찾지 못함 |
| TeleDex | 원격 조작 인터페이스(arXiv:2603.17065) | 원 저자 | 미확인 | — | — | 저장소 미확인 |

- 프로젝트 페이지 `https://visibletouch.github.io/`는 `#hardware`, `#appendix-tables`, `#videos` 섹션이 있는 정적 페이지입니다. 영상과 표를 보여 주며, 라이선스나 코드 설명은 확인하지 않았습니다.
- 공식 코드 저장소는 **센서 몰드 생성 도구뿐**입니다. 논문의 방법(투영 오버레이)과 실험은 공개되어 있지 않습니다. 이 점이 구현 판정(`04` §4)의 핵심입니다.
- 저장소 owner가 개인이고 생성일(2026-06-11)이 arXiv 제출(2026-09-12)보다 앞섭니다. 논문 공식 저장소인지는 저자 확인이 필요합니다. 이 분석에서는 "프로젝트 페이지가 링크한 저장소"로만 부릅니다.

## 5. 라이선스

- 논문: arXiv abs 페이지에서 `licenses/by/4.0`(CC BY 4.0) 확인(2026-11-15). PDF 본문에 ©, copyright, All rights reserved 문구는 없습니다. "IEEE"는 참고문헌의 학회명이라 권리 표시가 아닙니다.
- 따라서 논문 그림은 CC BY 4.0 조건에서 출처 표기와 함께 쓸 수 있습니다. 다만 이 분석의 공개 산출물에는 논문 그림을 넣지 않았습니다. 이유는 (1) 그림이 대부분 저자의 장면 사진과 화면 캡처이고, (2) 우리 그래프로 충분하기 때문입니다(`report/figures/own_*.png`).
- 공식 저장소: 라이선스 파일이 없으므로 기본적으로 모든 권리가 유보된 것으로 봅니다. 읽기만 했고 복제본을 공개하지 않습니다.
