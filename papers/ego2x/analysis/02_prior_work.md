# Ego2X 선행연구와 자료

## 1. 계보

| 계열 | 대표 연구 (논문 참고문헌) | Ego2X와의 관계 |
|---|---|---|
| 사람 영상을 정책 학습에 직접 쓰기 | EgoMimic [20], 잠재 계획 MimicPlay [38], OKAMI [23], 사람 영상 사전학습 VLA [44] | 정책 정렬 쪽. Ego2X는 정책이 아니라 **데이터 변환**에 집중 |
| 영상 합성 (로봇 장면 생성) | Phantom [22] 합성, MIRAGE [7], RoVi-Aug [8], RoboEdit [16] | 합성 쪽. Ego2X는 가림 제거를 **접촉 보호 영역**으로 바꿈 (§3.3.3) |
| 3D 손 추정 → 로봇 관절 리타게팅 | HuRo [19], Ego2Robot [40], DexMV [28], AnyTeleop [29], DexMachina [26], ManipTrans [24] | 가장 직접적인 기준선. 손 궤적 단위 전달 |
| 손–물체 접촉 복원 | ContactOpt [13], GRAB [33], Shan et al. [32], PressureVision [14], C2Dex [30] | 접촉을 **물체 좌표계 앵커**로 바꿈 |
| 파지 분류 | GRASP 분류 [12] | 파지 모드 7종 (표 1)의 근거 |
| 단안 손·물체 추정 | WiLoR [27], ACE-Ego-Hand [25], EgoHand [6], MANO [31], HaWoR [45] (논문 표 3에서 사용), SAM 3 [4], SAM 3D [10], FoundationPose [42], DROID-SLAM [34], DROID-calib [17], MoGe-2 [39], GeoCalib [36] | 입력 파이프라인 구성 요소. Ego2X 고유 기여는 아님 |
| 물리 검증 | RoboTwin 2.0 [9], real-to-sim [35], HSImul3R [3] | 시뮬 수율 평가 환경 |
| 데이터와 벤치마크 | HOT3D [1], EgoDex [18], DexYCB [5], Ego4D [15] | 평가에 쓴 데이터 |
| 로봇 VLA | X-VLA [49], Cosmos류 월드 모델, 교차 형태 정책 [11] | 사후 학습 대상. X-VLA 학습 세부는 원문 참고문헌으로만 확인 |

- **가장 가까운 대안은 HuRo입니다.** 같은 EgoDex 영상을 쓰고, 같은 "사람 영상 → 로봇 데이터" 목표를 가집니다. 차이는 전달 단위(손 궤적 대 접촉)와 깊이 보정 방식입니다.
- **Ego2Robot [40]은 기준선에서 장치 추적 3D 손을 씁니다.** 논문은 이를 "privileged"라고 적고, 단안 추정 조건의 비교가 아님을 밝힙니다(§4.1).
- 참고문헌 [40]의 저자와 저장소는 본문에서 명확하지 않습니다. 아래 GitHub 표의 이름 겹침은 확인하지 않은 후보입니다.

## 2. 공식 코드, 프로젝트 페이지, 관련 저장소 (확인 2026-11-21)

| 자료 | 링크 | 라이선스 | 최근 push | 비고 |
|---|---|---|---|---|
| **Ego2X 공식 코드** | 없음 | - | - | 논문 §5 "We will publicly release"만 있음. `gh search repos "Ego2X"` 0건 |
| **Ego2X 프로젝트 페이지** | 없음 | - | - | 본문에 URL 없음. 웹 검색에서도 찾지 못함 |
| **Ego2X 데이터셋 (CORE 주석)** | 없음 | - | - | 공개 예정만 있음 |
| HuRo (CoRL 2026, 기준선 [19]) | https://github.com/3587jjh/HuRo | Apache-2.0 | 2026-11-10 | 71★. 설명 "CoRL 2026". 기준선 공식 코드. 클론하지 않음 (Ego2X 공식 코드가 아니므로) |
| Ego2Robot 이름 겹침 1 | https://github.com/msunbot/ego2robot | 없음 | 2025-11-23 | 설명 "Egocentric Factory Episodes for Robot Foundation Model Pretraining". 논문 [40]과 같은지 미확인 |
| Ego2Robot 이름 겹침 2 | https://github.com/shaohua-pan/ego2robot | 없음 | 2026-08-23 | 설명 "ego2robot". 논문 [40]과 같은지 미확인 |
| Ego2Robot 이름 겹침 3 | https://github.com/BAp3r/Ego2Robot | 확인 안 함 | 확인 안 함 | 이름만 겹침. 검색 결과에만 존재 |

- 이 표의 저장소는 **Ego2X가 아닙니다.** 기준선과 이름 겹침만 적었습니다.
- 공식 코드가 없으므로 `repo/` 클론과 `04_code_verification.md`는 만들지 않았습니다. 대조 기준은 `04_claim_verification.md`입니다.

## 4. 본문에 없는 선행연구 확인

- 논문은 "first to exploit post-training of embodied models using egocentric videos"라고 적습니다(§1, §5). 이 주장은 본 분석에서 검색으로 반증하지 못했습니다. 다만 해당 기간 전체를 확인하지 않았으므로 **새로움 주장은 미확인**입니다.
