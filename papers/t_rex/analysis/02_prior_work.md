# 02. 선행연구와 관련 저장소

번호는 논문 참고문헌 번호입니다(`paper/paper.txt`의 References).

## 1. 선행연구 계보 (구성 요소별 출처)

| T-Rex의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 인간 1인칭 영상 사전학습 | EgoScale [1], EgoVLA [2], EgoMimic [3], DexWild [4] | 사전학습 단계는 EgoScale 방식을 그대로 따릅니다(§4.3). 22,889시간 |
| 흐름 매칭 행동 전문가 | π0 [5], GR00T N1 [6] | 행동 생성 방식 |
| 느림-빠름 분리 | Fast-in-Slow [7], Reactive Diffusion Policy [8] | RDP는 병렬 그리퍼용 과제별 모방학습으로 소개됩니다(§1). T-Rex는 이를 기초 모델 크기로 옮겼다고 주장 |
| 촉각을 VLA에 넣기 | Tactile-VLA [21], VTLA [22], OmniVTLA [23], VLA-Touch [25], ForceVLA [26] | 추가 모달리티, 정렬, 힘 인식 MoE |
| 촉각 이력 압축 | ViTacFormer [18], VTAM [19], MimicTouch [20] | 미래 촉각 예측, 촉각 행동 정제 |
| 미래 시각 잠재 예측 | VPP [32], F1 [33], GR-2 [34], FLARE [55] | Lfuture. 코드는 FLARE 계열의 질의 토큰과 코사인 손실을 씁니다(`04` D8) |
| MoT 구조 | BAGEL [27], InternVLA-A1 [35], Motus [36] | 전문가별 독립 투영과 공동 어텐션 |
| 이산 촉각 토큰 | VQ-VAE [68] | 손가락별 코드 1개 |
| 변형 맵 인코더 | ResNet-18 [69]에 자기지도 오토인코더 [70] | 동결 |

**가까운 대안과의 차이.** 가장 가까운 것은 RDP(고속 촉각 반응 경로)와 EgoScale(같은 사전학습)입니다. T-Rex의 새로움은 (a) 계단식 분할로 같은 흐름 궤적의 하위 구간을 촉각 전문가가 맡는 점, (b) 100시간 촉각 중간학습 데이터를 두는 점입니다. 표 2·3과 그림 6이 이 둘을 분리하려는 시도입니다.

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 별 / 마지막 푸시 | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [ZhuoyangLiu2005/T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex) | 공식 구현(main: 사후학습·추론·텔레옵·VQ-VAE·데이터셋 도구) | MIT (LICENSE: UC Regents) | 356 / 2026-10-08 | main `ba4047c` (2026-10-08), 정적 읽기 | 얕은 복제, 실행 안 함 |
| 같은 저장소 `full-pipeline` 브랜치 | 사전학습·중간학습 스크립트 | MIT | — / 2026-06-01 | `b23eafe` | 네 파일만 API로 읽음 |
| [tactile-rex.github.io](https://tactile-rex.github.io/) | 프로젝트 페이지(코드 README는 tactile-reactive-dexterous.github.io) | — | — | — | HTML 읽음 |
| HF 데이터셋 `zekaiwang/trex_dataset` | 공개 데이터 5,464 에피소드, 약 50시간, 1.55 TB | MIT | 다운로드 약 9만 / 2026-10-05 | — | 메타데이터만 |
| HF 모델 `miniFranka/T-Rex_pretrain_mecka22k_epoch1`, `..._midtrain_mecka23k_ucb100_vqvae_epoch6` | 사전학습·중간학습 체크포인트(각 약 10 GB, 8.5 GB) | MIT | — | — | 받지 않음 |
| 기준선 코드: RoboVerseOrg/ViTacFormer, xiaoxiaoxh/reactive_diffusion_policy, Isaac-GR00T, openpi | 논문 부록 E가 각주로 가리킴 | 각각 | — | — | 확인만 |

## 4. 읽을 때의 연결점

EgoScale 분석은 같은 사전학습 가정을 공유합니다. 촉각 이득을 사전학습과 분리하려면 표 3의 2×2가 필요한데, 이 논문은 기준선에 같은 사전학습(EgoScale)을 쓰므로 비교가 비교적 깨끗합니다(`03` V9).
