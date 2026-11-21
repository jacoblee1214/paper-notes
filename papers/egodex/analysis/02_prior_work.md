# 02. 선행 연구, 대안, 자료 계보

## 1. 자리매김

EgoDex는 **데이터셋 논문**입니다. 방법은 기존 모방학습 구성요소(X-IL의 인코더-디코더·디코더 전용, BC·DDPM·FM)를 그대로 씁니다. 새로운 것은 (a) 1인칭 영상과 고품질 3D 손 추적을 함께 모은 규모, (b) 손 궤적 예측과 역동역학이라는 두 벤치마크와 best-of-K 지표입니다.

## 2. 선행 계보 (논문 Table 1과 §2 기준)

| 축 | 대표 연구 (논문 인용) | EgoDex와의 차이 |
|---|---|---|
| 로봇 텔레오퍼레이션 데이터 | RoboTurk, BridgeData V2, RT-X, DROID | 로봇 하드웨어 필요. EgoDex는 사람 손을 공통 embodiment로 둠 |
| 사람 영상 + 3D 손 | EgoMimic (Kareer 2025, 약 4시간) | 같은 방향, 규모 약 200배. 손가락까지 추적(EgoMimic은 손목만) |
| 손–물체 3D 주석 | HOI4D, DexYCB, HOT3D | 수 시간~수천 개 에피소드 수준. 주로 파지 |
| 1인칭 대규모 영상 | Ego4D (HOI 89k), EPIC-KITCHENS | 손 포즈 주석 없음. 조작 위주 아님 |
| 능동 수집 장치 | UMI, DEXCAP, ARCap, Armada, DexHub | 사람이 의도적으로 수집. EgoDex는 "기록" 방식 |
| 사람 영상 → 로봇 | Motion Tracks (Ren 2025), Humanoid Policy ~ Human Policy (Qiu 2025) | 자세 추정망(HaMeR) 후처리 또는 3% 부분집합 사용(v1·v2) |
| 궤적 예측 프레임워크 | X-IL (Jia 2025) | EgoDex 실험이 쓰는 코드베이스(학습 코드는 공개 안 됨, 4절) |


## 3. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 쓰임 |
|---|---|---|
| 6D 회전 표현 | Zhou et al. 2019 (CVPR, 본문은 2018로 표기) | 행동의 회전 6차원 |
| 청크 상대 궤적 | Chi et al. 2024 (UMI, RSS) | 행동 청크 |
| 카메라 프레임 표현 | Kareer et al. 2025 (EgoMimic) | 포즈를 현재 카메라 좌표로 |
| 정책 구조 | X-IL (Jia et al. 2025, arXiv:2502.12330) | 14개 모델의 학습 골격 |
| 이미지 인코더 | ResNet | 224×224 입력 |
| 언어 인코더 | CLIP (Radford 2021), 고정 | 과제 설명 |
| 자연어 설명 생성 | GPT-4 (OpenAI 2024) | 메타데이터 → 설명 |
| 영상 디코딩 | torchcodec | 학습 시 프레임 디코딩 |
| 손 추적 기준 | ARKit (visionOS 2) | 주석 원천 |

## 4. GitHub, 프로젝트 페이지, 관련 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 최근 push | 별 | 비고 |
|---|---|---|---|---|---|
| 공식 코드·데이터 안내 | https://github.com/apple/ml-egodex | Apple 자체 라이선스 (`LICENSE`), GitHub 표기 "Other" | 2026-09-11 | 416 | 생성 2025-08-19. 얕은 클론 `repo/`(커밋 `718d9eb`), **정적 읽기만**. 실행·설치 안 함 |
| 데이터 라이선스 | 위 README "License" 절 | CC-by-NC-ND (README 문장) | - | - | 본문 v3에는 이 문장이 없음(01 §6) |
| 연구 페이지 (프로젝트 페이지 역할) | https://machinelearning.apple.com/research/egodex-learning-dexterous-manipulation | - | - | - | 검색으로 확인. 코드 링크는 GitHub 하나 |
| 데이터 다운로드 | 위 README: part1–5 각 약 300 GB, test 16 GB, extra 200 GB | - | - | - | **받지 않음** (사용자 지시) |
| 커뮤니티: H-RDT | https://github.com/HongzheBi/H_RDT | MPL-2.0 | 2025-12-21 | 161 | 로봇 파운데이션 모델, EgoDex 사전학습과 학습 코드 공개 (README 기재) |
| 커뮤니티: Being-H0 | https://github.com/BeingBeyond/Being-H0 | MIT | 2026-05-04 | 61 | VLA, MANO로 주석 처리 (README 기재) |
| 커뮤니티: EgoDex Viewer | Hugging Face Space (`ego-dex-viewer`, README 기재) | - | - | - | GitHub 저장소가 아니어서 확인 생략 |

- 공식 저장소의 `homepage`는 arXiv 주소입니다. 별도의 프로젝트 사이트 URL은 저장소에 없습니다.
- 공식 저장소에는 **학습 코드, 모델 가중치, 실험 설정 파일이 없습니다**(`04` §1). 학습은 X-IL 코드베이스에 의존합니다.
