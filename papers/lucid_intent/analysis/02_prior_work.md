# 02. 선행연구와 관련 저장소

LUCID의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| LUCID의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 의도 = 물체 3D 점 흐름 | Any-point trajectory modeling [48], General Flow [49], Motion Tracks [41], Track2Act [30], Flow as cross-domain interface [7], NovaFlow [8], PointWorld [52] | 점 궤적을 사람 영상과 로봇 사이 공통 표현으로 쓰는 계열. LUCID는 여기에 손바닥 자세를 함께 예측하는 토큰을 더했고(§2.3), 흐름만으로는 접촉 시 손 위치가 정해지지 않는다고 봅니다 |
| 손 중심 표현(손 자세, 파지 사전) | DexVIP [53], ZeroMimic [25], DexMV [11], Dexterity from smart lenses [4] | 손 정보만으로 정책을 학습. LUCID는 palm pose만 거칠게 예측하고 손가락은 π에 맡깁니다 |
| 점 추적 트랜스포머 | CoTracker3 [55] (EfficientUpdateFormer) | f_θ의 뼈대. "관측 프레임의 트랙 추정"을 "미래 예측"으로 바꾸고 palm 토큰 하나를 더함 |
| 시각 인코더 | DINOv3 [56] (동결) | 패치 토큰. 깊이는 0 초기화 잔차 어댑터로 합침 |
| 영상 → 3D 감독 추출 | ViPE [61], SAM 3 [62], DELTAv2 (DenseTrack3Dv2) [63], WiLoR [64], MANO [65] | 모두 기존 모델의 조합. 새 모델 학습은 f_θ뿐입니다 |
| 영상 코퍼스 | Panda-70M [57], Action100M [58], Something-Something-V2 [59], EPIC-Kitchens [60], Large Video Planner 메타데이터 [54] | 과제당 20k 클립을 행동 라벨로 걸러 모음 |
| 물체 중심 목표 조건 정책 | SimToolReal [9], Dex4D [13], DexTrack [16], Dexplore [17], DexterityGen [47] | π의 직접 선행. 행동 적분 방식(EMA, 관절 목표 델타)은 SimToolReal을 따릅니다(부록 B.2). Dex4D는 점 궤적을 목표로 하는 과제 무관 정책이라 가장 가까운 대안입니다 |
| eigen-grasp 행동 기저 | Ciocarlie 외 [67], Lum 외 [23] | 재타게팅한 사람 파지의 PCA 상위 5개 |
| 교차 어텐션 정책 구조 | AME [68] (보행에서 온 지형 어텐션) | 지형 격자 대신 물체 점 토큰 사용 |
| 교사-학생과 곡선 | DextrAH-RGB [10], OpenAI Rubik's cube [70] (ADR), PHP [71] (하이브리드 증류), VIRAL [75] (DAgger-BC 대조군) | ρ 변수와 난이도 곡선은 DextrAH-RGB 방식 |
| 개루프 기준선 | Veo 3 [73], Dreamitate [31], NovaFlow [8] | 생성 영상 → 흐름·손 추출 → 실행 계열의 대표로 Veo 3.1을 씁니다 |
| 현실 보정 | CMA-ES [78], 깊이 센서 잡음 [72], Fast-FoundationStereo [79] | 액추에이터 식별, 센서 잡음, 손목 깊이 |

**가까운 대안과의 차이.** (a) **영상 생성 → 실행 계열**(Dreamitate [31], NovaFlow [8], Flow-as-interface [7])은 t=0에 한 번 계획합니다. LUCID는 같은 표현(흐름)을 학습된 f_θ로 1 Hz 다시 질의합니다. (b) **사람 궤적을 정책 입력으로 쓰는 계열**(DexMV [11], Phantom [26] 등)은 녹화된 궤적에 묶입니다. LUCID는 "다음 1초"만 감독합니다. (c) **Dex4D [13]**는 과제 무관 점 궤적 정책이라는 점에서 π와 구조가 비슷하며, 차이는 의도의 출처(Dex4D는 생성 영상, LUCID는 학습된 의도 모델)와 palm pose 참조의 유무입니다(논문 §1, §2.3의 서술에 근거. Dex4D 본문은 직접 읽지 않았습니다).

## 2. 코드와 저장소 (GitHub 표, 확인 2026-11-25)

**공식 코드: 없음.** 프로젝트 페이지의 `main.js`에 외부 링크 표가 있고 `code`는 `'#'`(주석에 "fill these once live")입니다. 렌더된 페이지에서 Code와 Demo 버튼은 비활성입니다. 프로젝트 페이지 저장소의 마지막 커밋은 2026-06-18입니다. GitHub 검색에서 논문 번호, 저자 계정(hgupt3), 프로젝트 계정(lucid-robot)의 공개 저장소는 프로젝트 페이지 저장소 하나뿐입니다. 그래서 이 분석은 코드 대조 대신 주장 대조(`04_claim_verification.md`)로 합니다.

| 저장소 | 관계 | 라이선스 (GitHub API) | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [lucid-robot/lucid-robot.github.io](https://github.com/lucid-robot/lucid-robot.github.io) | 프로젝트 페이지 소스 (정적 사이트) | 없음 | 0 | 2026-06-18 (e504a6e) | 페이지 HTML과 `main.js`만 읽음. 코드 링크는 비어 있음 |
| [tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal) | π의 직접 선행 (SimToolReal [9]) | MIT | 442 | 2026-09-21 (313d5ae, 2026-08-13 커밋) | 복제하지 않음 (확인만) |
| [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) | f_θ 뼈대 (CoTracker3 [55]) | "Other" (자체 라이선스 파일, 조건 미확인) | 5,211 | 2026-03-03 | 확인만. 복제 안 함 |
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | 시각 인코더 [56] | "Other" (조건 미확인) | 11,802 | 2026-11-02 | 확인만 |
| [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | 마스크 [62] | "Other" (조건 미확인) | 12,401 | 2026-11-17 | 확인만 |
| [nv-tlabs/vipe](https://github.com/nv-tlabs/vipe) | 카메라·깊이 추출 [61] | Apache-2.0 | 2,220 | 2026-11-24 | 확인만 |
| [snap-research/DELTA_densetrack3d](https://github.com/snap-research/DELTA_densetrack3d) | DenseTrack3D 계열. 논문의 v2 [63]와 같은 저장소인지는 확인하지 못했습니다 | "Other" (조건 미확인) | 152 | 2025-04-06 | 확인만 |
| [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) | 손 복원 [64] | "Other" (조건 미확인, MANO 모델은 별도 약관) | 749 | 2026-04-07 | 확인만 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 시뮬레이션 [66] (CUDA 전제) | "Other" (조건 미확인) | 8,713 | 2026-11-24 | 확인만 |
| [Denys88/rl_games](https://github.com/Denys88/rl_games) | PPO 라이브러리 [77] | MIT | 1,409 | 2026-11-24 | 확인만 |
| [NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) | 손목 깊이 [79] | "Other" (조건 미확인) | 1,622 | 2026-09-11 | 확인만 |
| [leap-hand/LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API) | 로봇 손 [18] | "Other" (조건 미확인) | 179 | 2025-10-20 | 확인만 |
| [rai-opensource/NovaFlow](https://github.com/rai-opensource/NovaFlow) | 가까운 대안 (생성 영상 → 흐름) [8] | "Other" (NOASSERTION) | 26 | 2026-04-09 | 확인만 |
| [buoyancy99/large-video-planner](https://github.com/buoyancy99/large-video-planner) | 영상 코퍼스 메타데이터 [54] | 없음 | 265 | 2026-01-31 | 확인만 |

별과 push 날짜는 `gh api`로 읽었고, 라이선스 칸의 "Other"는 GitHub이 SPDX로 분류하지 못한 자체 문구라는 뜻이며 조항은 읽지 않았습니다. 이 분석에서 외부 저장소를 복제하거나 실행하거나 설치하지 않았습니다. 가중치와 데이터도 받지 않았습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **점 흐름만으로는 접촉 위치가 정해지지 않는다**(§2.3)는 서술은 DexVIP [53], ZeroMimic [25] 같은 손 중심 연구의 동기와 맞습니다. 이 논문의 palm 토큰이 실제로 흐름 단독 대비 얼마나 기여하는지는 실험에 없습니다(`04` C9).
- **시뮬레이션의 과제 무관 보상 + 점 궤적 목표**는 SimToolReal [9]과 Dex4D [13]에서 이미 실세계에 옮겨졌습니다. LUCID의 기여는 이 정책을 학습된 의도 모델의 출력과 닫힌 루프로 묶은 점과 사람 영상 감독 파이프라인입니다.
- **스케일링 곡선**은 다른 영역(언어, 비전)에서 멱법칙이 흔하지만, 5개 점으로는 지수를 가릴 수 없다는 점을 저자도 부록 C.3에서 인정합니다.
