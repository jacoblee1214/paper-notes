# 02. 선행연구와 관련 자료: Dex4D

## 1. 연구 계보

Dex4D는 네 흐름이 만나는 자리에 있습니다. 참고문헌 번호는 논문 v2 기준입니다.

1. **목표 조건 다지 손 RL(파지와 재배치).** UniDexGrasp [2]·UniDexGrasp++ [18]은 목표 조건 정책과 기하 인식 커리큘럼으로 수천 개 물체를 잡습니다. Dex4D는 이 물체 집합(3,200개 항목)과 형상 특징 파일을 그대로 쓰고, 목표를 "파지 자세"가 아니라 "물체의 임의 자세"로 일반화합니다. DextrAH-G/RGB [19, 20]와 DexterityGen [1]은 일반 목적 다지 컨트롤러라는 목표를 공유합니다. 인핸드 회전 [22–25]은 하위 기술입니다.
2. **물체 궤적 추종 RL.** OmniGrasp [26], DexTrack [27], Dexplore [28], Patel 외 [29]는 참조 물체 궤적을 따라가는 정책입니다. 차이는 참조의 출처(인간 모캡·영상)와 사용 범위입니다. 같은 시기 SimToolReal [30]은 물체 중심 목표 자세 조건 정책으로 도구 조작을 하며, 논문 §2는 인터페이스(희박한 점 궤적, 메시와 파지 상자 불필요)와 범위(3,200개 물체 대 길쭉한 도구 원형)로 구분합니다.
3. **영상 생성과 점 궤적을 로봇 계획으로.** Track2Act [33], Gen2Act [36], Flow as the cross-domain manipulation interface [35], RAM [34]에서 NovaFlow [44], LVP [45], Dream2Flow [46], Patel 외 [42], 3DFlowAction [43]로 이어집니다. 이들은 생성 영상에서 물체 운동을 뽑아 자세 추정 + 모션 플래닝 또는 리타깃팅으로 팔을 움직이는 개루프 계획이 중심입니다. Dex4D의 차별점은 계획 이후 단계를 **폐루프 RL 정책**으로 바꾼 것입니다.
4. **시뮬-실 증류와 월드 모델형 보조 손실.** 특권 교사 → 부분 관측 학생(DAgger [52])에 다음 상태 예측을 더하는 것은 DyWA [47], UWM [54], WorldVLA [55] 계열을 인용합니다. Dex4D 학생은 다음 관절각·속도만 예측합니다(식 2).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 시뮬레이터 | Isaac Gym [4] (Isaac Lab [5]는 인용만) | 코드는 Isaac Gym Preview |
| 환경 코드 기반 | IsaacGymEnvs, DexterousHands, UniDexGrasp (코드 README 감사의 글) | 논문 본문에는 UniDexGrasp만 |
| 물체 집합·형상 특징 | UniDexGrasp [2] (`datasetv4.1`, `meshdatav3_pc_feat`) | 3,200 (메시, 스케일) 항목, 메시 2,145개 (`03` V12) |
| 강화학습 | 대칭 PPO [50], 적응형 KL 스케줄 | 하이퍼파라미터는 표 7 |
| 점군 인코더 | PointNet [48], PointNet++ [49] | 코드는 PointNet 변형(평균-최대 채널 분할 풀링) |
| 키포인트 거리 보상 | ResMimic [51] | 부드러운 보상 지형 근거 |
| 증류 | DAgger [52] | 코드는 교사 개입 없이 학생 롤아웃 |
| 학생 구조 | 트랜스포머 [56], 다음 상태 예측 [47, 53–55] | 토큰 4개, 4층, 차원 128 |
| 행동 공간 | 팔 증분 + 손 절대, 동적 핸드오버 [64], 아이겐그래스프 [62], DexFunc [63] | |
| 하드웨어 | xArm6, LEAP 손 [60], RealSense D435, AprilTag [61] | |
| 영상 생성 | Wan2.6 [9] 온라인 플랫폼 (가중치 비공개로 보임), 중국어 프롬프트 | 코드 예시 스크립트에는 Kling 호출 줄도 있음 |
| 깊이·분할·추적 | Video Depth Anything [65], SAM2 [66], CoTracker3 [57] | 모두 외부 공개 모델 |
| 시각화 | Viser [59] | |
| 비교 대상 | NovaFlow [44], LVP [45], π0.5-DROID [58] | 세 기준선 모두 잡기 단계는 Dex4D 정책(NovaFlow, LVP) 또는 자체(π0.5) |

## 4. GitHub와 자료 (확인 2026-11-26, `gh api`)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (시뮬레이션·학습)** | https://github.com/Dex4D/Dex4D-Simulation | Apache-2.0 | 34 / 2026-02-23 | 커밋 `0dfccd82797e929cda6ab665003bc27a6b8cd31a`. 환경, 교사 PPO, 학생 DAgger, 체크포인트 3개 포함(`repo/simulation/`) |
| **공식 코드 (비전)** | https://github.com/Dex4D/Dex4D-Vision | Apache-2.0 | 20 / 2026-02-23 | 커밋 `13c2f02902dc9b7f87d3db68eda69df417d851da`. 깊이, 오프라인·온라인 CoTracker, AprilTag 보정(`repo/vision/`) |
| **공식 코드 (하드웨어)** | https://github.com/Dex4D/Dex4D-Hardware | 없음 | 8 / 2026-02-23 | 커밋 `b13ec71953e68d4ced0d24e7f83043c136c446a6`. README만 있고 "Coming Soon"(`repo/hardware/`) |
| 시각화 | https://github.com/Dex4D/Dex4D-Viser | 없음 | 1 / 2026-02-12 | `.viser` 기록 파일 약 105 MB. clone하지 않음 |
| 프로젝트 페이지 | https://dex4d.github.io/ | (저장소 `Dex4D/Dex4D.github.io` 별 0, 라이선스 표기 없음) | 2026-10-13 | 영상과 갤러리, Q&A 영역. "3,000 objects" 표기 |
| 논문 | https://arxiv.org/abs/2602.15828 | arXiv non-exclusive 1.0 | v2 2026-10-13 | `README.md` 라이선스 절 |
| UniDexGrasp (물체·형상 특징) | https://github.com/PKU-EPIC/UniDexGrasp | 표기 없음 | 276 / 2024-05-25 | 데이터는 별도 신청·다운로드. 받지 않음 |
| DexterousHands | https://github.com/PKU-MARL/DexterousHands | Apache-2.0 | 1,123 / 2025-02-18 | 환경 코드 기반 |
| Video Depth Anything | https://github.com/DepthAnything/Video-Depth-Anything | Apache-2.0 | 2,366 / 2025-10-07 | 깊이 모델 |
| CoTracker | https://github.com/facebookresearch/co-tracker | 별도 표기(NOASSERTION) | 5,212 / 2026-03-03 | 점 추적기 |
| SAM2 | https://github.com/facebookresearch/sam2 | Apache-2.0 | 20,239 / 2026-05-30 | 첫 프레임 분할 |
| LEAP Hand API | https://github.com/leap-hand/LEAP_Hand_API | 별도 표기 | 179 / 2025-10-20 | 손 하드웨어 소프트웨어 |
| NovaFlow (기준선) | https://github.com/rai-opensource/NovaFlow | 기타 | 26 / 2026-04-09 | 같은 이름의 다른 저장소가 있어 소유자 확인 필요 |
| Large Video Planner (기준선) | https://github.com/buoyancy99/large-video-planner | 없음 | 265 / 2026-01-31 | |
| SimToolReal (비교 대상) | https://github.com/tylerlum/simtoolreal | MIT | 443 / 2026-09-21 | 논문 §2 |
| openpi (π0.5-DROID) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 14,767 / 2026-08-24 | 기준선 |

가중치와 데이터(UniDexGrasp 자산, Video Depth Anything 체크포인트, SAM2, CoTracker 가중치, Wan2.6 영상)는 받지 않았습니다. 시뮬레이션 저장소의 체크포인트 세 개(약 40 MB)는 clone에 포함되어 있지만 불러오지 않았습니다.

## 5. 가까운 대안과 비교

| 접근 | 과제 정보 | 하위 제어 | 폐루프 | 이 논문과 다른 점 |
|---|---|---|---|---|
| Dex4D | 영상 → 점 궤적 | 점 조건 RL 정책(과제 무관) | 예(물체 점 추적) | 기준 |
| NovaFlow-CL | 영상 → 점 궤적 | Kabsch + IK + 모션 플래닝 | 예(자세 재추정) | 하위 제어가 해석적. 이 논문의 구현은 잡기를 Dex4D 정책으로 대체 |
| LVP | 영상 → 손·물체 | 운동학 리타깃팅 | 아니오 | 임베디드 격차를 인정(부록 D) |
| SimToolReal | 목표 자세(물체 중심) | 목표 자세 조건 RL | 예 | 도구 원형 중심, 메시 필요 |
| π0.5-DROID | 언어 | VLA | 예 | 병렬 그리퍼, 다른 로봇 |
| DexTrack 계열 | 인간 참조 궤적 | 추종 RL | 예 | 참조 필요, 과제별 |
