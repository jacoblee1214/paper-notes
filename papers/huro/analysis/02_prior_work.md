# 02. 선행 연구, 대안, 구성 요소, 저장소

분석일 2026-11-23. 논문 번호 `[n]`는 v5의 참고문헌 번호입니다. 논문 본문에 없는 연결(우리의 해석)은 "해석"이라고 표시합니다.

## 1. 계보: 사람 영상을 로봇 학습에 쓰는 네 갈래

| 갈래 | 대표 | 하는 일 | HuRo와의 차이 (논문 §2 서술 + 해석) |
|---|---|---|---|
| **시각 변환** | H2R [20], Masquerade [21] | 사람 손을 렌더링한 로봇으로 바꾸거나(H2R), 3D 손 궤적을 복원해 로봇화 클립으로 2D 말단 예측을 사전학습(Masquerade). 시각 인코더나 보조 목표 중심 | HuRo는 같은 입력에서 **행동까지 로봇 행동으로 바꿔 VLA를 끝까지 사전학습**. 논문은 Masquerade식 변형을 행동 지도로 확장해 부록 B(표 3)에서 비교 |
| **행동 지도 + 사람 관측 유지** | VITRA [18], EgoScale [19] | 손 동작에서 행동을 얻되 관측은 사람 영상 그대로 (VITRA: 사람 손을 말단 장치로 보고 통합 행동 공간에 정렬, EgoScale: 대상 로봇 행동으로 리타게팅해 대규모 egocentric 사전학습) | HuRo는 관측도 로봇화. 부록 C(표 4)에서 H-RDT, VITRA 변형(사람 관측 + 사람 행동)과 비교 |
| **관측·행동 동시 정렬 (과제 맞춤)** | Phantom [11], DexUMI [10], WARPED [12], RwoR [13], MimicDreamer [17] | 사람 시연을 하류 로봇 과제에 맞춰 수집하거나 변환 | 규모와 이질성: HuRo는 5개 출처, 142M 프레임의 일반 영상을 한 형식으로 |
| **영상 생성 기반 변환** | H2R-Grounder [14], Mitty [15], X-Humanoid [16], DreamGen [40] | 사람 영상을 로봇 영상으로 생성하거나, I2V + 역동역학(IDM)으로 가짜 행동을 붙임 | HuRo는 렌더링 + 최적화 기반이라 기하가 명시적. §4.5(그림 6)에서 DreamGen식 I2V+IDM과 프레임 예산 대비 비교 |

**HuRo의 위치(해석):** 방법 자체(손 자세 추정, 카메라 SLAM, IK 리타게팅, 인페인팅, 렌더링)는 모두 기존 구성 요소의 조합입니다 (§3 표). 새로운 것은 (1) 주석 수준이 다른 출처를 **필요한 신호만 추정해** 하나의 형식으로 합치는 구성, (2) **카메라 운동까지 반영한 정렬**(식 2, 7)로 관측 카메라와 행동 좌표를 같은 변환으로 묶는 점, (3) 그것을 VLA 사전학습 규모로 돌려 실로봇으로 평가한 실증입니다.

## 2. 가까운 대안과 비교 지점

- **EgoScale [19] (사람 관측 유지, 행동 지도):** 같은 데이터 출처 계열(egocentric 대규모)에서 "관측 로봇화가 필요한가"를 가르는 가장 직접적인 대조입니다. HuRo의 표 4(Human-HRDT, Human-VITRA)는 이 계열을 "사람 관측 + 사람 손 행동"으로 근사한 것일 뿐 EgoScale 자체와의 직접 비교는 없습니다 (논문 미수행).
- **Masquerade [21]:** 고정 카메라-로봇 외부 파라미터 가정. 부록 B의 Fixed-EEF가 이를 근사하지만 "Rather than reproducing Masquerade's full pipeline" (부록 B)이라 원 방법 재현은 아닙니다.
- **Phantom [11]:** 사람 팔 제거 + 로봇 렌더링의 원형. 과제 맞춤 설정이며, HuRo가 보이는 "사람 팔 제거 후 로봇 합성" 시각 변환의 직계 조상입니다.
- **영상 생성(DreamGen 계열):** 렌더링 방식은 로봇 형태가 정확하지만 가림과 접촉이 없고, 생성 방식은 사실적이지만 행동이 IDM 추정입니다. 그림 6의 I2V+IDM은 ALLEX용으로 적응한 한 구성이라 일반적 결론은 아닙니다 (구성 세부 미공개, `03` C5).
- **대형 로봇 데이터(참고):** π0.5 [4], GR00T N1.6 [2]를 실로봇 기준선으로 두지만, 이들은 대규모 로봇 데이터로 사전학습되어 데이터 종류가 다릅니다.

## 4. 공식 코드와 저장소 (확인 2026-11-23)

| 저장소 | 역할 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [3587jjh/HuRo](https://github.com/3587jjh/HuRo) | 공식 코드 (로봇화 파이프라인 stage 1–10) | Apache-2.0 (자체 코드). 실행에는 비상업 제한이 있는 의존성 포함: `THIRD_PARTY_NOTICES.md`, README 경고 "cannot be run commercially" | 71 (포크 3) | 2026-11-10 | `repo/` 얕은 복제 (`082fb40`, 2026-11-10 "Enable the Hugging Face badge"), **정적 읽기만**, 실행·설치 없음. 열린 PR 3개(#7 Isaac Sim 6.0, #8 소스 코퍼스 분할 도구, #9 EgoDex 문서), 열린 이슈 1개(#1 데이터 공개 요청) |
| [3587jjh.github.io/HuRo](https://3587jjh.github.io/HuRo/) | 프로젝트 페이지 | — | — | — | 읽음. 데이터는 "Coming soon" 표기 |
| [Joocjun/huro](https://huggingface.co/collections/Joocjun/huro) (Hugging Face 컬렉션) | 데이터셋 4개: `HuRo-EPIC-KITCHENS`, `-Egocentric10K`, `-Ego4D`, `-EgoVerse` | 데이터 카드 표기: EPIC cc-by-nc-4.0, Ego4D cc-by-4.0, Ego10K·EgoVerse "other". 4개 모두 **접근 제한(gated)** | — | 최종 수정 2026-11-09 | 목록과 메타데이터만 확인. 내려받지 않음. **EgoDex 부분은 없음** (`04` D2, D3). 저장소 크기(메타데이터 합계, 파일 목록이 10만 개에서 잘렸을 수 있음) 약 10–17 GB씩, 총 약 54 GB 이상 |
| [chungmin99/pyroki](https://github.com/chungmin99/pyroki) | 리타게팅 해결기 (PyRoKi, JAX) | MIT | 1,826 | 2026-04-01 | 주소만 (HuRo가 코드 일부 차용: `create_conn_tree` 등) |
| [wirobotics-rih/allex_model](https://github.com/wirobotics-rih/allex_model) | ALLEX URDF | BSD-3-Clause (메쉬는 평가 전용) | 62 | 2026-08-31 | 주소만 (서브모듈) |
| [ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR) | 손 자세 (HAWOR) | CC BY-NC-ND 4.0 (GitHub: NOASSERTION) | 392 | 2026-04-16 | 주소만 |
| [sczhou/ProPainter](https://github.com/sczhou/ProPainter) | 인페인팅 | S-Lab 1.0, 비상업 (GitHub: NOASSERTION) | 7,092 | 2026-10-27 | 주소만 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 정책 구조 (GR00T N1.6) | Apache-2.0 | 8,403 | 2026-11-17 | 주소만. HuRo 저장소에는 포함되지 않음 |
| boschresearch/DroidCalib, javrtg/AnyCalib, microsoft/MoGe, facebookresearch/sam2, ddshan/hand_object_detector | 파이프라인 구성 요소 | AGPL-3.0, Apache-2.0, (MoGe 라이선스는 `THIRD_PARTY_NOTICES.md`에 MIT), Apache-2.0, MIT | 106, 252, 3,171, 20,210, 303 | 2024-01 ~ 2026-11 | 주소만. 서브모듈 패치 형태로 저장소에 일부 들어 있음 |

## 5. 구성 요소별 출처 (논문 §3.1, 부록 G 기준)

| 단계 | 구성 요소 | 출처 | 비고 |
|---|---|---|---|
| 카메라 내부 파라미터 | DroidCalib, AnyCalib | [24], [25] | AGPL 코드. 코드 쪽은 안정 추정 실패 시 대체 (`04` D12) |
| 손 검출·추적 | 100DoH, BoT-SORT | [26], [27] | ultralytics(AGPL) 경유 |
| 손 자세 | HAWOR (MANO) | [28], [29] | **CC BY-NC-ND**, MANO는 연구용만 |
| 카메라 궤적 | DROID-SLAM + MoGe-2 + GeoCalib | [30], [31], [32] | 단안이라 스케일은 MoGe-2로 보정 |
| 지시문 | Qwen3.5 | [33] | 코드는 `Qwen3.5-9B` 가중치 (`04` D11) |
| 리타게팅 | PyRoKi (jaxls) | [34] | MIT. 두 단계 구성은 저장소의 `retargeter.py` |
| 팔 분할·인페인팅 | SAM2, Detectron2, ProPainter | [35], [36], [37] | ProPainter는 비상업 |
| 렌더링 | Isaac Sim | [38] | RT 코어 필요, 드라이버 R580 이하 (`04` D14) |
| 정책 | GR00T-N1.6-3B | [2] | 학습·평가 코드는 공개되지 않음 (`04` D1) |
