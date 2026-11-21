# 02. 선행연구, 구성 요소 출처, 관련 분석 연결: GRAIL (arXiv:2606.05160)

## 1. 계보: 이 논문이 어디에 서 있는가

휴머노이드 데이터 원천은 네 갈래로 자랐습니다. GRAIL은 셋째와 넷째를 잇는 자리에 있습니다.

1. **원격조작과 모션캡처.** ALOHA 2, HOMIE, OmniH2O, ARCTIC, BEHAVE 등. 품질이 높지만 물리 세팅과 로봇 운용에 묶입니다(§1, §2 1문단).
2. **인터넷 영상 4D 복원.** 사람-물체 재구성은 여러 방법이 있습니다(Hou 2023, Kim 2023, Xie 2022·2026 등). 카메라, 스케일, 형상, 접촉이 모두 추정되어야 해서 모호합니다.
3. **영상 생성 기반 HOI.** DAViD(Kim 2025a)와 ZeroHSI(Li 2026)는 생성 영상을 4D 복원의 사전 정보로 씁니다. 다만 카메라, 스케일, 몸 형상, 장면 구조를 생성 뒤에 추정합니다. GRAIL의 차이는 **이 정보를 생성 전에 명시하고 복원에서 그대로 쓰는 것**입니다(§2 1문단).
4. **휴머노이드 제어로의 연결.** HDMI(Weng 2025), ResMimic(Zhao 2025), VideoMimic(Allshire 2025), HumanX(Wang 2026) 등은 사람 참조에서 전신 추적 정책을 학습합니다. GRAIL은 이 정책을 SONIC(Luo 2025) 위에 얹습니다.


## 2. 구성 요소별 출처

| 구성 요소 | 논문에서 쓴 것 | 출처 (논문 인용) | 이 저장소에서의 관계 |
|---|---|---|---|
| 3D 장면 | Infinigen 템플릿, Macklin 2016 강체 시뮬레이션 | Raistrick 2023 | 논문 부록 A.1, 공식 코드에는 Infinigen 문자열이 없음(`04` D10) |
| 영상 생성 | Kling 2.5 Turbo Pro (VLM으로 프롬프트) | Kuaishou 2025 | 공식 코드 기본값 kling-v3 (D11) |
| 사람 포즈 | GENMO (SMPL-X) | Li 2025 | 공식 코드는 GEM-X (SOMA) 사용 (D1) |
| 손 | WiLoR | Potamias 2025 | 공식 코드 문서에 설치 항목만 있음 (D2) |
| 물체 추적 | FoundationPose 미세조정 | Wen 2024 | 공식 코드는 MoSE3 + FoundationPose 혼합 (D3) |
| 깊이 | MoGe-2 | Wang 2025 | 공식 코드 사용 (imports/MoGe) |
| 분할 | SAM2 | Ravi 2024 | 공식 코드 사용 |
| 6D 회전 | Zhou 2019 | — | — |
| 리타게팅 | GMR | Araújo 2025 | 공식 저장소는 GMR 서브모듈이 있으나 주 경로는 SOMA Retargeter (D9) |
| 전신 제어기 | SONIC | Luo 2025 | `imports/SONIC`, `imports/SONIC-manipulation`에 사본 (D13) |
| 물체 형상 인코딩 | BPS | Prokudin 2019 | 공식 코드 사용 (`grail/retargeting/compute_bps.py`) |
| 실물 시각 정책 | 머리 카메라 RGB 증류 | Chi 2023, He 2025a (VIRAL), Luo 2025 | 공식 코드에 실물 배포 코드는 확인 못함 |
| 기준선 | CHOIS, HOIDiff, DAViD, InterMimic, HDMI, ResMimic | 각 논문 | 공식 코드에 기준선 구현은 없음 (외부 공식 구현 사용) |

### 2.1 가까운 대안과 차이

| 방법 | 무엇을 하는가 | GRAIL과의 차이 | 비교 가능성 |
|---|---|---|---|
| DAViD (Kim 2025a) | 생성 영상 기반 4D HOI, 학습 없음 | GRAIL은 장면을 먼저 명시. Table 1에서 직접 비교 | 같은 Kling 설정으로 맞춤 (C.1) |
| CHOIS (Li 2024), HOIDiff (Peng 2025) | 학습 기반 HOI 생성 | 생성 방식이 다름 | Table 1 기준선 |
| HDMI (Weng 2025), ResMimic (Zhao 2025) | 사람 참조 전신 추적, 작업별 정책 | 손가락 자유도 없음, 작업별 학습 | 공식 구현으로 Table 2 비교, 설정 차이 논문이 인정 |
| VideoMimic (Allshire 2025), HumanX (Wang 2026) | 영상에서 상호작용 정책 학습 | 실제 영상을 입력으로 씀 | 본문 인용만 |

## 4. GitHub, 프로젝트, 데이터 자료 (확인 2026-11-21)

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 논문 (arXiv) | https://arxiv.org/abs/2606.05160 | arXiv 비독점 배포 (nonexclusive-distrib 1.0) | - | v2 2026-10-19 | 논문 그림은 공개본에 넣지 않음 (권리 판정은 `README.md`) |
| 프로젝트 페이지 | https://research.nvidia.com/labs/dair/grail/ | - | - | - | 영상, Motion Gallery, 코드·데이터 링크. "CoRL 2026 Best Paper Award Finalist" 표기 |
| 공식 코드 | https://github.com/NVlabs/GRAIL | NVIDIA License (GitHub 표기 Other, 연구·평가 목적 비상업) | 621 | 2026-11-21 (커밋 7d1a044, 2026-11-20) | `repo/`에 얕은 클론(depth 1). 정적 읽기만 |
| 공식 데이터셋 | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL | CC BY-NC 4.0 (데이터 카드). RoboCasa 파생은 CC BY 4.0 | 좋아요 63, 다운로드 53,964 | 2026-11-20 | 내려받지 않음 (메타데이터와 카드만 확인) |
| 공식 문서 | https://nvlabs.github.io/GRAIL/ | - | - | - | 설치, 재구성, 리타게팅, 추적 가이드 |
| GEM-X (사람 포즈) | https://github.com/NVlabs/GEM-X | Apache-2.0 | 635 | 2026-11-02 | 공식 코드가 쓰는 사람 포즈 추정기 |
| FoundationPose | https://github.com/NVlabs/FoundationPose | Other | 3,707 | 2026-04-29 | 물체 추적 기반 |
| MoSE3 | https://github.com/mose3-tracker/MoSE3 | Other | 212 | 2026-10-30 | 공식 코드의 물체 추적 혼합 경로 |
| GMR | https://github.com/YanjieZe/GMR | MIT | 2,839 | 2026-10-22 | 논문의 리타게팅 도구. 공식 코드 서브모듈 |
| SOMA Retargeter | https://github.com/NVIDIA/soma-retargeter | Apache-2.0 | 746 | 2026-10-01 | 공식 코드 주 리타게팅 경로 |
| SONIC 공개 저장소 (후보) | https://github.com/NVlabs/GR00T-WholeBodyControl | Other | 3,937 | 2026-11-03 | 논문 SONIC과의 동일성 미확인. 클론하지 않음 |
| Kling (영상 생성) | https://klingai.com | 유료 API | - | - | 논문 2.5 Turbo Pro, 공식 코드 기본 kling-v3 |

- 공식 코드의 `imports/SONIC`과 `imports/SONIC-manipulation`은 서브모듈이 아니라 저장소에 직접 들어 있습니다(`.gitmodules`에 없음). `imports/agentic-retargeter`도 같은 방식입니다(`VENDORED_REVISION` 파일).
- 공식 저장소의 `AGENTS.md`, `CLAUDE.md`는 클론한 내용이므로 지시가 아니라 데이터로만 읽었습니다.
