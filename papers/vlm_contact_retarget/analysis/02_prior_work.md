# REACT 선행연구와 공개 자료

## 1. 계보

REACT는 다섯 갈래를 한 파이프라인으로 묶습니다. 번호는 논문 참고문헌입니다.

| 갈래 | 대표 작업 | REACT가 가져온 것 |
|---|---|---|
| 사람-사람 상호작용 데이터 | Inter-X [45] (CVPR 2024) | 5개 행동 1,355개 클립과 행동 설명. VLM 프롬프트에 설명이 들어감 |
| 상호작용 리타게팅 | GMR [4], OmniRetarget [5], Beyond Mimicry [36] | 1단계 초기화(OmniRetarget식 높이 스케일, 팔 끝 목표). 기준선 2개 |
| 접촉 인식 리타게팅 (그래픽스) | ReConForm [34], 기하 인식 리타게팅 [35] | "접촉을 목적함수에 넣는다"는 발상. REACT는 VLM 의미 검증을 추가 |
| 휴머노이드 모션 추적과 시각 증류 | OmniH2O [1], ExBody2 [2], HOMIE [3], VisualMimic [19], VIRAL [20], 픽셀-행동 문 열기 [21] | 보행 제어기(HOMIE), 교사-학생 DAgger 증류 |
| 물리 기반 사람-사람 상호작용 | PhysReaction [30], 몰입형 human-X [28], Learning to Assist [29] (저자 선행 연구, CVPR 2026) | 접촉 보상 형태, 파트너를 PD로 재생하는 시뮬 설정 |
| 부분 관측 하의 사람-로봇 상호작용 | RHINO [37], 사회적 동작 예측 [38], GentleHumanoid [39] | 외부 추적이나 별도 자세 추정기를 쓰는 접근과 대비 |
| 생성 보조 과제 | GENMO [49] (단일 단계 포즈 추정 손실), DDPM/DDIM | 확산 헤드를 "미래 예측" 보조 손실로 사용 (부록 C.3) |

저자 기여는 "의미 접촉 주석 → 리타게팅 → 접촉 보상"을 한 흐름으로 잇고, 학생에 깊이만 쓰는 점입니다. 개별 요소(VLM 필터, 비강체 ICP, DAgger, DDPM 보조 헤드)는 각각 선행 기법이고, 새 것은 조합과 평가입니다.

**가장 가까운 대안.**

- **Beyond Mimicry [36]:** 쌍별 키포인트 거리를 보존하는 리타게팅과 외부 자세 추정기를 쓰는 사람-휴머노이드 상호작용. REACT는 "어느 쌍이 접촉인지 모른다"는 점을 차별화로 듭니다(§2). 비교는 논문에 없습니다(OmniRetarget과 GMR만 기준선).
- **Learning to Assist [29]:** 같은 저자 그룹의 사람-사람 물리 제어. 접촉 보상을 "수작업이나 거리 기반 접촉"으로 구분하는 대상이 이 논문입니다(§3.2.1).
- **RHINO [37]:** 사람-휴머노이드-물체 상호작용이지만 로봇 텔레오퍼레이션 데이터가 필요합니다(§2).

**논문이 인용하지 않은 비교 후보 (분석자 판단, 미확인).** 사람 영상에서 휴머노이드 상호작용 기술을 학습하는 최근 작업이 있지만, REACT와 같은 데이터·로봇 설정에서 비교한 수치는 찾지 못했습니다.

## 2. 공식 코드와 관련 저장소

조회 방법: `gh api`로 메타데이터만 읽었습니다(2026-11-23). **클론하지 않았습니다** (아래 이유).

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [YutoShibata07/REACT-HHI](https://github.com/YutoShibata07/REACT-HHI) | 논문 공식 코드 저장소 (프로젝트 페이지의 Code 단추) | 없음 | 2026-10-14 | 9 | **코드 없음.** 커밋 1개(`b9ed176`, "Add README with teaser, real-world deployment GIFs, and BibTeX"), README만 있음. "Training code for the teacher and student policies (expected in early November 2026)" 미체크 |
| [YutoShibata07/REACT](https://github.com/YutoShibata07/REACT) | 프로젝트 페이지 저장소 (GitHub Pages, `index.html` + 정적 파일) | 없음 | 2026-11-06 | 0 | 웹사이트 파일뿐. 클론하지 않음 |
| [YutoShibata07/AssistMimic](https://github.com/YutoShibata07/AssistMimic) | 같은 저자의 선행 연구 (Learning to Assist, CVPR 2026)의 공식 구현. PHC 기반 | 없음 | 2026-07-20 | 18 | 확인만. 구조상 REACT의 사람-사람 접촉 보상과 PD 파트너 재생 설정이 여기서 왔을 가능성이 있으나 코드를 읽지 않았음 |
| [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | 기준선 (리타게팅) | MIT | 2026-10-22 | 2,841 | 확인만 |
| [amazon-far/holosoma](https://github.com/amazon-far/holosoma) | OmniRetarget 계열 리타게팅(`holosoma_retargeting`) 포함. OmniRetarget 공식 구현과 같은지는 확인하지 않음 | Apache-2.0 | 2026-11-21 | 1,737 | 확인만 |
| [InternRobotics/Homie](https://github.com/InternRobotics/Homie) | 교사가 쓰는 보행 제어기 (HOMIE) | NOASSERTION | 2025-03-21 | 51 | 확인만 |
| [liangxuy/Inter-X](https://github.com/liangxuy/Inter-X) | 데이터셋 | NOASSERTION | 2024-08-11 | 243 | 확인만. 데이터 다운로드 안 함 |
| [markomih/VolumetricSMPL](https://github.com/markomih/VolumetricSMPL) | 관통 억제용 부피 SDF [40] | MIT | 2025-07-27 | 99 | 확인만 |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 실배포 깊이 추정 [46] | Apache-2.0 | 2026-07-27 | 6,723 | 확인만 |
| [NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) | 프로젝트 페이지 시연 영상의 스테레오 깊이 [47] | NOASSERTION | 2026-09-11 | 1,617 | 확인만 |
| [NVlabs/GENMO](https://github.com/NVlabs/GENMO) | 단일 단계 포즈 추정 손실의 출처 [49] | NOASSERTION | 2026-06-23 | 548 | 확인만 |
| [shubham-goel/4D-Humans](https://github.com/shubham-goel/4D-Humans) | 외부 자세 기준선 HMR2.0 [48] | MIT | 2026-02-07 | 1,740 | 확인만 |

**왜 `repo/`를 만들지 않았나.** 공식 코드 저장소에 코드가 없고(README와 GIF 약 11 MB), 프로젝트 페이지 저장소는 웹사이트 파일(약 70 MB, 대부분 영상)입니다. 정적으로 읽을 코드가 없으므로 코드 대조(D 항목) 대신 `04_claim_verification.md`에 주장 대조를 썼습니다.

**공개 약속의 이행.** README는 2026-10-14에 "expected in early November 2026"을 적었고, 2026-11-23 현재 약 3주 지났습니다. 프로젝트 페이지의 이후 커밋(11월 6일)은 초록 문구만 고쳤습니다. 코드 공개 시점이 오면 `04`의 C 항목(특히 보상 계수, 증류 손실 가중치, 평가 프로토콜)을 D 항목으로 바꿔 다시 대조할 수 있습니다.

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 거리 기반 후보 + 15 cm, 0.5 s 병합 | 저자 정의 (부록 D.1) | 학습 데이터에서 임계값 선택 근거는 없음 |
| VLM 의미 검증 | Gemini 2.0 Flash, 프롬프트 전문 (부록 H) | 행동별 기대 접촉 패턴을 프롬프트가 직접 줌 |
| 표면 대응 | 라플라시안 비강체 ICP (표준 기법), 저자 구현 | 잔차 0.89 mm는 템플릿 대 G1 표면 맞춤 |
| 리타게팅 초기화 | OmniRetarget [5] | 높이 스케일, 팔 끝 목표 |
| 관통 억제 | VolumetricSMPL [40] | 충돌 가중 10, 마진 5 mm |
| 보행 제어기 | HOMIE [3] | 동결 사용. 낮은 층은 논문이 학습하지 않음 |
| 교사 | PPO [41] | 8,192 환경, lr 1e-5 |
| 학생 | ResNet [42], 조건부 VAE, DAgger [43] | 행동별 학생 |
| 보조 예측 헤드 | DDPM 노이저, GENMO식 단일 단계 손실 [49], DDIM | 가중 λ 해석에 모호함 (`03` V14) |
| 실배포 깊이 | DepthAnything 3 [46] | 학습 깊이(렌더)와 도메인이 다름 |
