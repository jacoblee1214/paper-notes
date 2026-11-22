# 02. 선행연구, 대안, 공식 저장소

## 1. 계보

- **암묵 행동 복제 (IBC) [11]와 BC-RNN [35].** 논문 표 II에서 3D 변형과 함께 비교합니다. 둘 다 이 논문의 과제에서 낮은 성공률을 보입니다.
- **3D 기반 정책.** PerAct [60], GNFactor [82], RVT [14], ACT3D [13], 3D Diffuser Actor [28]은 키프레임 기반의 저속(PerAct 2.23 FPS, 3D Diffuser Actor 1.67 FPS) 정책입니다. 논문은 이들이 고차원 제어와 빠른 제어에 맞지 않는다고 봅니다(§II-B).
- **점군 인코더.** PointNet [42], PointNet++ [43], PointNeXt [46](사전학습 포함), Point Transformer [84]. 논문 표 V에서 모두 DP3 인코더에 졌습니다. 점군 샘플링은 FPS [42]를 씁니다.
- **시각 기반 모방학습과 시연 수.** 2D 정책이 대부분이고 3D 정책은 최근에 늘었습니다(§II-B). 저자들은 시연 수를 줄이는 방법으로 알고리즘 자체를 고치는 쪽을 택합니다(§II-C, 손 재설계와 다른 방향).
- **벤치마크.** Adroit [49], Bi-DexHands [8], DexArt [5], DexDeform [31](PlasticineLab [26] 기반), DexMV [47], HORA [44], MetaWorld [77]. 시뮬레이터는 MuJoCo, Sapien, IsaacGym, PlasticineLab입니다.
- **확산 샘플러.** DDIM [62]을 쓰고, DPM-Solver++ [32]를 절제에서 비교합니다(표 VII). 샘플 예측 설정은 고차원 행동에서 DDIM의 클리핑을 피하려는 선택입니다(§III-C).

## 2. 가까운 대안과 후속 (공식 저장소 README 기준)

- **Simple DP3.** 같은 저장소에 있습니다(`policy/simple_dp3.py`). UNet 채널을 줄여 추론을 2배 빠르게 합니다(표 XVI, XVII). 논문 본문 설명은 없고 공식 저장소에만 있습니다(`04` D13).
- **후속 적용 (README 목록).** HumanoidGen (arXiv:2507.00833, DP3가 DP보다 낫다고 보고), Vision in Action (arXiv:2506.15666, DP3를 강한 기준선으로 씀), DemoGen (arXiv:2502.16932, 생성 시연과 함께 씀), EquiBot (arXiv:2407.01479), 사족 보행 로코-조작 (arXiv:2403.20328, DP3를 상위 계획기로 씀). 모두 저희가 읽지 않았고 README 목록을 그대로 옮겼습니다.

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 | 별 / 포크 | 마지막 푸시 | 상태 |
|---|---|---|---|---|---|
| [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) | 공식 구현 (DP3, Simple DP3, 학습·평가 스크립트) | MIT (GitHub API, LICENSE 파일 있음). `third_party/` 아래 다른 라이선스가 따로 있음 | 1,475 / 172 | 2025-10-17 (커밋 `47385d9`, 2025-10-16 PDT, "Update README with community applications and extensions") | 얕은 복제 (136 MB, gitignore). 정적 읽기만 함, 실행·설치 안 함 |
| [3d-diffusion-policy.github.io](https://3d-diffusion-policy.github.io) | 프로젝트 페이지 | 해당 없음 | — | — | 확인함. RSS 2024 표기, "Best Paper Award, RSS"와 "Best Student Paper Award Finalist, RSS" 표기가 있음 (공식 목록에서 확인하지 못함) |

- 공식 저장소에 **학습된 체크포인트는 없습니다**(README 확인). 시뮬 전문가 데이터와 실로봇 데이터는 OneDrive와 Google Drive 링크로 제공됩니다. 크기는 확인하지 않았고 받지도 않았습니다.
- 공식 코드의 **최신 커밋은 2025-10-16이고, 논문 v7은 2024-09-27입니다.** 논문 결과를 낸 코드가 어느 시점의 것인지 확인할 수 없습니다(`04` §1).
- 재현에 필요한 `third_party/`(mujoco-py 2.1.2.14, gym 0.21.0, MetaWorld, VRL3, dexart-release, pytorch3d_simplified 등)는 고정 버전과 CUDA 환경을 요구합니다. 설치 방법은 `INSTALL.md`에 있습니다(`04` §3).
