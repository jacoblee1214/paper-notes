# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 계보

- **H2O** (He 외, arXiv:2403.04436, IROS 2024). RGB 카메라로 얻은 전신 키포인트를 RL 정책으로 추적합니다. 전역 선속도(MoCap)를 입력으로 씁니다. OmniH2O의 직접 전신이며 기준선입니다(표 1의 H2O 행, 표 2). 이 저장소에는 H2O 분석이 없습니다.
- **PHC** (Luo 외, ICCV 2023, 논문 [56]). 인간 모션을 물리 시뮬에서 추적하는 RL의 기반입니다. 리타게팅 코드 구조(`phc/`)가 저장소에 함께 들어 있습니다.
- **AMASS** (Mahmood 외, ICCV 2019, [49]). 학습 모션의 원천 데이터입니다.
- **Legged Gym / rsl_rl** (Rudin 외 2022, [64]). 시뮬 환경과 PPO 구현의 뼈대입니다. 저장소에 포함되어 있고 라이선스 표기가 따로 있습니다 (`04` §0).
- **교사-학생 증류와 DAgger** (Lee 외 2020, [50]; Ross 외 2010, [51]). 교사가 특권 정보로 학습하고 학생이 DAgger로 따라 배웁니다. 논문 §3의 핵심 설계입니다. 학습 전 표준 절차라서 기여의 새로움은 "히스토리 + 3점 + 선속도 없음" 조합에 있습니다.
- **VR 원격조작** (OpenTeleVision [52], VisionProTeleop [53], DexPilot [55]). 손 자세 추정과 IK의 원천입니다.
- **언어 기반 모션 생성** (MDM, Tevet 외 2022, [57]). 언어 지시를 모션 목표로 바꿉니다(그림 4).
- **확산 정책** (Diffusion Policy, Chi 외 2023, [58]; DDPM [59], DDIM [60]). LfD 비교 대상입니다.
- **경쟁 연구** (논문 §2에 언급): Expressive WBC (Cheng 외 2024, [18]), Deep WBC (Fu 외 2023, [19]), Humanoid locomotion as next token prediction ([11]). 모두 논문에서 가까운 대안으로 인용되며, 이 분석에서 수치 비교는 하지 않았습니다.

## 2. 공식 저장소와 페이지

| 자원 | 역할 | 라이선스 | 지표 (API, 2026-11-22 조회) | 상태 |
|---|---|---|---|---|
| github.com/LeCAR-Lab/human2humanoid | 공식 구현 (H2O와 OmniH2O 공용) | README: CC BY-NC 4.0, "상업적 이용 금지". `LICENSE-CC-BY-NC-4.0.md` 파일 있음. GitHub API `license`: null (감지 실패). `rsl_rl/LICENSE`, `legged_gym`은 상속 라이선스 | ★1083, fork 97, 생성 2024-09-29, 마지막 push 2025-02-21 | 얕은 복제 `repo/`, 커밋 `750f1fa` (2025-02-21, "fixing eval during training"). 정적 읽기만. 실행과 설치 안 함 |
| omni.human2humanoid.com | 프로젝트 페이지 (영상, 논문 링크, "CoRL 2024 · Oral") | 페이지 정보 | - | 읽음 (HTML) |
| human2humanoid.com | H2O 페이지 | - | - | 읽지 않음 |
| `legged_gym/resources/motions/h1/stable_punch.pkl` (저장소 안, 2.7 MB) | 예제 모션 | 저장소와 같음 | - | 포함 (정적) |
| `amass_phc_filtered.pkl` (README의 CMU Box 링크) | 필터링된 리타게팅 전체 모션 | 미확인 | 크기 미확인 | 받지 않음 (데이터) |
| AMASS 원본 | 리타게팅 원천 | AMASS 연구용 라이선스 (원 사이트) | 대용량 | 받지 않음 |
| OmniH2O-6 데이터셋 | LfD 학습 데이터 (논문 §4.3) | 미확인 | 프레임 73,462 (약 41분 at 30 Hz) | **공개 위치를 찾지 못함.** 저장소와 프로젝트 페이지에 링크 없음 (프로젝트 페이지의 CMU Box 링크는 "Video"용) |
| 학습 체크포인트 | 없음 | - | - | 저장소에 정책 가중치가 없음. `resources/actuator_nets/anydrive_v3_lstm.pt` (16.6 KB)은 코드에서 참조되지 않음 |
| Isaac Gym Preview 4 | 시뮬레이터 (NVIDIA 배포, 별도 설치) | NVIDIA 라이선스 | - | 설치하지 않음 |

저장소 크기: 작업 트리 약 158 MB (`repo/`, 로봇 메시와 `resources/` 포함), `.git` 38 MB. `repo/`는 gitignore합니다.

