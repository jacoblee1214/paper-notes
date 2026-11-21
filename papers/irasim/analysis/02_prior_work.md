# IRASim 선행연구와 관련 자료

분석일 2026-11-21. GitHub 수치는 이 날 API로 확인했습니다.

## 1. 계보

| 흐름 | 대표 연구 (논문에 인용된 것) | IRASim과의 관계 |
|---|---|---|
| 행동 조건 영상 예측 (계획용) | Visual foresight (Finn & Levine 2017) [40], Ebert 2018 [44], FitVid [23], MaskViT [24] | 같은 목적(행동을 넣고 미래 영상을 봄). IRASim은 확산 트랜스포머로 바꾸고 고해상도·긴 궤적을 다룸 |
| 상호작용 실세계 시뮬레이터 | Learning interactive real-world simulators (UniSim, Yang 2024) [2] | 텍스트와 행동으로 영상 모델을 조건화. IRASim은 행동만 쓰고 프레임 단위 정렬을 추가 |
| 자기회귀 행동 조건 모델 | iVideoGPT [25] | RoboNet 비교 대상 (표 2). 두 방법 모두 RoboNet 설정 |
| 영상 확산 기반 백본 | VDM [32], LVDM [33], Latte [28], DiT [35], Sora [6], Open-Sora [7] | VDM과 LVDM은 직접 비교 기준선. Latte는 구현 기반 (README). DiT는 블록 구조, Open-Sora는 정책 실험의 초기 가중치 |
| 로봇 세계 모델과 계획 | FLIP [41], VLMPC [42], GPC (Qi et al. 2025, arXiv:2502.00622) [58], DayDreamer [46], DreamerV3 [45] | GPC는 Push-T 계획 비교 대상 (표 5). FLIP과 VLMPC는 관련 연구로만 인용 |
| 정책 평가를 시뮬레이터 대신 쓰는 흐름 | Li et al. 2024 (arXiv:2405.05941) [3] | 정책 평가 실험(§4.2)의 동기 |
| 평가 지표 | PSNR [51], SSIM [52], FID [53], FVD [54] | 표 1, 3의 지표. Latent L2는 이 논문이 제안한 주 지표 |

## 2. 가까운 대안 (기준선과 비교 관점)

| 방법 | 조건 | 비교 방식 | 이 논문과의 차이 |
|---|---|---|---|
| VDM [32] | 궤적을 임베딩으로 넣음, 픽셀 공간 | 표 1, 그림 4 | 픽셀 공간 U-Net. 가장 약한 기준선 |
| LVDM [33] | 궤적을 임베딩으로 넣음, 잠재 공간 | 표 1, 3, 그림 4 | IRASim과 같은 VAE와 학습 설정 (부록 D). 가장 강한 기준선 |
| Video-Ada (IRASim 내부) | 영상 단위 조건 | 표 1, 3, 그림 4 | 같은 백본에서 프레임 단위 조건만 뺀 것. 핵심 대조 |
| iVideoGPT [25] | 자기회귀 토큰 | 표 2 (숫자 차용) | 대규모 사전학습 (OpenX). IRASim은 RoboNet만으로 학습 |
| MaskViT [24] | 마스크 토큰 | 표 2 (숫자 차용) | 같음 |
| GPC [58] | 확산 영상 예측 + 보상 모델 | 표 5 (숫자 차용) | 자기회귀 다음 프레임 예측. 추가 롤아웃 수를 밝히지 않음 |

## 4. 공식 코드와 저장소

| 항목 | 주소 | 라이선스 | 별·포크 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| 공식 코드 | [bytedance/IRASim](https://github.com/bytedance/IRASim) | Apache-2.0 (LICENSE 파일 확인) | 168 / 18 | 2025-07-08 (HEAD `c72b6dade6`, 2025-07-08 커밋) | 얕은 복제, 정적 읽기 전용 (`repo/`). 설치와 실행 안 함 |
| 프로젝트 페이지 | [gen-irasim.github.io](https://gen-irasim.github.io/) | 확인 안 됨 | — | — | 정적 페이지. 영상 예시와 "Accepted to ICCV 2025" 표기 |
| 프로젝트 페이지 저장소 | Gen-IRASim/Gen-IRASim.github.io | 확인 안 됨 | 0 | 2025-07-30 | 페이지 소스로 보임 (내용 미확인) |
| 학습·평가 데이터와 가중치 | README의 lf-robot-opensource 링크 (RT-1, Bridge, Language-Table) | — | — | — | 받지 않음. 합계 약 0.77 TB (`04` D13) |
| 데이터 미러 | Hugging Face `fangqi/IRASim` (README 2025-05-20 공지) | — | — | — | 확인 안 함 (주소는 README 인용) |
| 커뮤니티 포크 (비공식) | ewykric/irasim_robotwin (별 4, 2025-10-02 push) | 확인 안 됨 | — | — | 이름으로 보아 RoboTwin 적용. 내용 미확인 |

- 같은 이름의 다른 저장소 (jobswei/IRASim, liwan-buaa/irasim, ChuyaoFu/IRASim-starVLA 등)는 별이 0이거나 2026년에 만들어졌고, 공식과 무관한 것으로 보입니다. 이 분석에서 다루지 않았습니다.
- 코드가 기대는 외부 구현: Latte (Vchitect/Latte, Apache-2.0, 별 1,949), pytorch-fid (mseitzer/pytorch-fid, Apache-2.0, 별 3,848), stylegan-v (universome/stylegan-v, 라이선스 없음 (API null), 별 390). FVD는 stylegan-v에 기대는데, 이 저장소에서는 서브모듈이 비어 있습니다 (`04` D9).
- 논문 PDF 자체의 라이선스와 권리는 `README.md`에 적었습니다.
