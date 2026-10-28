# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다. arXiv ID는 논문 참고문헌 목록에 적힌 값을 그대로 옮겼습니다. 이 문서의 각 논문을 별도로 열어 내용을 검증하지는 않았습니다.

## 1. 계보도

```
[고전] MIT copy-demo(1970) [3] → Kang & Ikeuchi 1997 [4] → Efros et al. "Do as I do" 2003 [5]
        │
[RGB-D/3D 스캔 가정] DexMV 2022 [6] ─ 물체 3D 스캔, pick-place
        │
[특수 장비 가정] EgoZero [9], Dexterity from Smart Lenses [8] ─ 스마트글래스 depth, 손 keypoint
        │
[단안 RGB + 파운데이션 모델 + 물리 리타겟팅]  ← 2025–2026 주류
   H2Sim2Robot [16]  LiDAR 스캔 + FoundationPose → RL           (데이터: 직접 촬영)
   VideoManip [18]   MeshyAI + FoundationPose → DRO + DP3       (직접 촬영 + egocentric)
   DexMan [22]       TRELLIS + FPose + SpaTrack → RL            (직접 촬영 + 생성 영상)
   DexImit [7]       SAM 3D + FPose++ → motion planning         (직접 촬영 + 생성 영상)
   ★ Do as I Do      SAM 3D + guided diffusion 추적 → sampling 최적화 (직접 촬영 + 생성 + egocentric + 인터넷)
```

**차별점 요약:** 다른 방법들은 모두 물체 추적에 **FoundationPose 계열**(모델 기반 6-DoF 추적기)을 씁니다. Do as I Do는 추적까지 **생성 모델(SAM 3D)** 로 처리하고, 리타겟팅은 RL이나 휴리스틱 대신 **RL 학습이 필요 없는 sampling 최적화**로 처리합니다.

## 2. 구성 요소별 핵심 선행연구

| 역할 | 논문 | 이 논문에서의 쓰임 |
|---|---|---|
| 이미지 → 3D 물체 (shape + pose 결합 생성) | SAM 3D [11], arXiv:2511.16624 | 메쉬 생성, 그리고 guided 추적의 백본 |
| 가속 | Fast-SAM3D (wlfeng0509/Fast-SAM3D, 논문에는 미인용) | Taylor cache로 sparse-structure 단계 가속 |
| 분할 | SAM 3 [64], arXiv:2511.16719 | 물체(클릭)와 손(텍스트) 비디오 분할 |
| 단안 기하 | MoGe-2 [10], arXiv:2507.02546 | pointmap과 intrinsics. SAM 3D가 MoGe pointmap으로 학습됐기 때문에 같은 것을 사용 |
| 손 | HaWoR [45] (CVPR 2025) | world-space MANO 손 궤적. 스케일 기준으로 사용 |
| 점 추적 | BootsTAP/BootsTAPIR [67], arXiv:2402.00847 | 회전 속도 → 적응형 α_p |
| 중력 | GeoCalib [68] (ECCV 2024) | 궤적의 중력 정렬 |
| 6-DoF 추적 베이스라인 | FoundationPose [17] arXiv:2312.08344, Any6D [47] | 비교 대상(같은 파이프라인에 끼워 넣어 비교) |
| Flow matching 이론 | Lipman et al. [77] arXiv:2210.02747 | Eq. 2 로그우도(instantaneous change of variables) |
| Replacement 가이던스 | RePaint [65] arXiv:2201.09865, Score SDE [66] arXiv:2011.13456 | Eq. 1 blending의 근거 |
| 물리 리타겟팅 | **SPIDER [15] arXiv:2511.09484** | 바로 위 베이스라인. annealed sampling의 틀 |
| RL 리타겟팅 | ManipTrans [59] arXiv:2503.21860, DexMachina [60] arXiv:2505.24853, Dexplore [61] | 성공 기준(E_pos < 0.1, E_rot < 0.5) 관례 |
| 운동학 IK | mink [55] | 사람손 → 로봇손 초기 레퍼런스, 로봇팔 IK |
| 도메인 랜덤화 | OpenAI Rubik's cube [69] arXiv:1910.07113, Rudin et al. [70] arXiv:2109.11978 | 교란(perturbation) 아이디어의 출처 |
| 데이터 | DexYCB [71], HOI4D [72], OakInk2 [73] arXiv:2403.19417, 100DOH [76] | 평가와 필터링 분석 |

## 3. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **DIAL-MPC** (Xue et al., 2024, "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing") | 코드 주석에 "One DIAL-MPC optimization step"이라고 적혀 있습니다. 반복 방향과 horizon 방향으로 노이즈를 annealing하는 방식이 DIAL-MPC 계열입니다. SPIDER를 거쳐 계승된 것으로 보입니다. | 코드 주석으로 확인. 논문 본문에는 미인용 |
| **One-Euro filter** (Casiez et al., CHI 2012) | 적응형 α_p와 구조가 같습니다(`03` §1.3). | 내 해석 |
| **Kalman / α-β 필터** | EMA의 지연 문제를 상수속도 예측으로 없애는 고전적 해법입니다. | 내 해석. 개선 아이디어 (b) |
| **RePaint의 resampling, DPS 등 posterior 샘플링 계열** | replacement 방식 조건부 생성의 편향을 줄이는 방법들입니다. | 내 해석. `03` §2 |
| **EgoDex** [75] arXiv:2505.11709 | 대규모 egocentric 손 데이터입니다. 논문은 필터링 동기 부분에서만 인용합니다. 이 파이프라인의 입력 소스로 바로 쓸 수 있습니다. | 논문 인용 |

**한계:** 3절의 DIAL-MPC 서지 정보는 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 arXiv ID를 직접 확인해야 합니다.
