# 02. 선행연구와 계보

## 1. 계보

```
Genie (2402.15391) ── 영상에서 이산(VQ) 잠재 행동을 라벨 없이 학습, 상호작용 환경
   └─ LAPA (2410.11758) ── 이산 잠재 행동으로 VLA 사전학습 (NSVQ 계열), 로봇 라벨은 미세조정에서만
        ├─ IGOR (2411.00785), AdaWorld (2503.18938) ── 이미지 목표·적응형 세계 모델용 잠재 행동
        ├─ UniVLA (2505.06111) ── 과제 중심 잠재 행동(DINOv2 특징 위 VQ), 구현체 독립
        ├─ Villa-X (2507.23682) ── 잠재 행동 모델링을 VLA에 결합
        ├─ Motus (2512.13030) ── 통합 잠재 행동 세계 모델
        ├─ DreamDojo (2602.06949) ── 사람 영상 44K 시간, 연속 잠재 행동(상대 행동 + 청크 주입), 세계 모델
        ├─ LAOF (2511.16407) ── 광학 흐름 제약으로 잠재 행동을 방해 요소에 강하게
        └─ CLAP (2601.04061), WLA3 (2609.15870), JoyAI-RA 0.5 (2608.05674) ── 사람 영상 잠재 행동의 최근 변형
사람 영상 → 명시적 의사 행동: VITRA (2510.21571, 손 자세 → 행동), EgoScale (2602.16710), EgoDex (2505.11709)
LAPT-Scale (2610.13732): 연속 잠재(VAE, d = 32) + LPIPS·DINOv2·RAFT 정렬 + 23.3K시간 + JoyAI-RA 정책의 3단계 학습
```

LAPT-Scale의 새 부분은 (1) LAM 손실에 지각·의미·흐름 정렬 항을 더한 조합(식 1–4), (2) 잠재 행동을 정책의 1단계 흐름 정합 목표로 쓰고 같은 행동 전문가를 2·3단계에 이어 쓰는 설계, (3) 1K/3K/5K 시간 스케일링과 손 동작·DreamDojo LAM 대조다. 연속 잠재 + 사람 영상 사전학습 자체는 DreamDojo가 먼저 했다. DreamDojo(arXiv:2602.06949)의 LAM은 연속 두 프레임에서 32차원 벡터를 뽑는 VAE이고 KL 가중치가 1e-6이라서, 이 논문의 LAM 뼈대(d = 32, KL 1e-6, 시공간 VAE)와 같고 논문 자신도 "MSE만" 진단을 DreamDojo의 재구성 + KL 레시피로 설명한다(부록 B.1). 즉 새로운 것은 뼈대의 연속성이 아니라 정렬 항 세 개의 추가다. 광학 흐름 제약은 LAOF [41]와 겹친다. 논문은 두 방법의 차이를 "화소 재구성만 쓰는 LAM" 대 "정렬 항 추가"로만 설명한다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 (논문 인용) | 이 논문에서의 역할 |
|---|---|---|
| 잠재 행동의 정의, 이웃 프레임 쌍에서 추출 | Genie [4], LAPA [5], UniVLA [6] | 1단계 감독의 단위(전이 하나) |
| 이산 코드 대 연속 잠재 | VQ-VAE [8]와 위 LAM 계열 | 표 7·17의 VQ-LAM·LAPA·VILLA-X |
| 시공간 VAE, 행동 질의 토큰 | VAE [42], Transformer [43] | LAM 구조(§3.2) |
| 지각 정렬 LPIPS | Zhang 외 [9] | 식 1 |
| 의미 정렬 DINOv2 | Oquab 외 [10] | 식 2(패치 풀링 + CLS) |
| 운동 정렬 RAFT | Teed·Deng [11] | 식 3(L1, 목표 흐름에 stop-gradient) |
| 광학 흐름 제약 아이디어 | LAOF [41] | 관련 연구 절이 명시 |
| 데이터 필터 | EgoVid-5M [34] | 흐림·카메라 운동·손 상호작용 약한 구간 제거 |
| 정책 구조(VLM + Perceiver식 행동 전문가) | JoyAI-RA 0.1 [44] | 같은 저자 그룹의 기반 모델 |
| 조건부 흐름 정합 | Lipman 외 [45] | 식 5 |
| 평가 | RoboTwin 2.0 [46], RoboCasa-GR1 [47, 16], LARY [50] | 표 1, 2, 5, 7 |
| 기준선 | π0 [14], π0.5 [15], Motus [7], LingBot-VLA [18], ABot-M0 [19], GR00T-N1.6, Qwen3PI(StarVLA) [48], Diffusion Policy [49] | 표 1–3, 6 |

**가장 가까운 대안과 차이.**
- **JoyAI-RA 0.1 [44]는 이미 대규모 1인칭 사람 조작 영상을 사전학습에 쓴다.** arXiv 초록(2604.20100, 2026-11-23 확인)이 "web data, large-scale egocentric human manipulation videos, simulation-generated trajectories, real-robot data"와 "explicit action-space unification"을 말한다. 따라서 표 1–3의 JoyAI-RA 대비 이득은 "사람 영상을 쓰는가"가 아니라 "LAM 잠재 감독 대 명시적 행동 공간 통일"의 비교에 가깝다. 두 모델이 같은 2·3단계 데이터를 쓰는지는 논문에 없다.
- **JoyAI-RA 0.5 [20]**(2608.05674)은 초록이 "Implicit action alignment infers latent actions from visual transitions"를 포함한다고 쓴다. 같은 그룹의 더 최근 잠재 행동 정렬 모델이며 LAPT-Scale 논문은 이를 일반 VLA 계열에서 인용만 하고 표 1–3에서는 0.1 [44]를 기준선으로 쓴다. 0.5와의 비교는 없다.
- **WLA3 [40]**(2609.15870, 같은 저자 일부)는 사람 영상과 이질 데이터에서 잠재 행동을 의미·동역학·운동학에 재사용한다. 연속 잠재의 설계 선택(국소 창, 부분 양식 재구성)이 다르다.
- **DreamDojo LAM [37]**: 연속 잠재와 재구성 기반이며 위처럼 뼈대가 같다. 논문의 "DreamDojo LAM" 기준선은 공개 사전학습 LAM 그대로이고, "MSE만" 진단은 같은 레시피를 자체 데이터로 재학습한 것이다(부록 B.1). 두 개는 다른 체크포인트다.

## 4. GitHub와 자료 (확인 2026-11-23)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드** | **없음** | - | - | 논문·arXiv 페이지에 링크 없음. GitHub 저장소 검색(`LAPT-Scale`, `LAPT_Scale`)에서 발견되지 않음 |
| 프로젝트 페이지 | **없음** | - | - | 논문에 링크 없음, `lapt-scale.github.io` 등 추측 주소는 404 |
| 기반 모델 JoyAI-RA 0.1 | arXiv 2604.20100 (코드 없음) | 논문 non-exclusive | - | 저자 조직 `jd-opensource`에서 VLA 코드를 찾지 못함 |
| JoyAI-RA 0.5 | https://joyai-ra-05.github.io (저장소 `JoyAI-RA-05/JoyAI-RA-05.github.io`) | 확인 안 함 | 별 0, 2026-08-07 | 프로젝트 페이지만. 코드 링크 없음 |
| WLA3 | https://wla-3.github.io | 확인 안 함 | - | 프로젝트 페이지만 확인 |
| VITRA | https://github.com/microsoft/VITRA | MIT | 별 519, 2026-06-12 | 명시적 의사 행동 기준(손 자세) |
| DreamDojo | https://github.com/NVIDIA/DreamDojo | Apache-2.0 | 별 1,133, 2026-03-21 | 논문의 "DreamDojo LAM" 기준선, 공식 사전학습 LAM |
| LAPA | https://github.com/LatentActionPretraining/LAPA | MIT | 별 587, 2025-01-22 | 이산 잠재 행동 |
| UniVLA | https://github.com/OpenDriveLab/UniVLA | Apache-2.0 | 별 1,146, 2025-11-19 | 과제 중심 잠재 행동 |
| Motus | https://github.com/thu-ml/Motus | Apache-2.0 | 별 1,321, 2026-10-09 | 표 1 기준선 |
| AdaWorld | https://github.com/Little-Podi/AdaWorld | Apache-2.0 | 별 264, 2025-06-17 | 적응형 잠재 행동 |
| StarVLA | https://github.com/starVLA/starVLA | 미지정(NOASSERTION) | 별 3,971, 2026-11-18 | 표 2의 Qwen3PI 기준선 코드 베이스 |
| 논문 PDF | https://arxiv.org/pdf/2610.13732 (`paper/paper.pdf`) | CC BY 4.0 | - | 7.1 MB |

공식 코드가 없어 clone하지 않았고 위 저장소는 메타데이터(`gh api`)만 확인했습니다. 데이터(Egocentric-10K, Xperience-10M, EgoDex 등)와 가중치는 받지 않았습니다. 사람 영상 데이터 자체는 Hugging Face에 공개된 출처(Egocentric-10K, Xperience-10M)와 비공개 자체 수집 8.5K 시간이 섞여 있어 말뭉치 재구성은 불가능합니다.
