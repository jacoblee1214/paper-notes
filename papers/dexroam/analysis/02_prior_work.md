# 02. 선행연구와 관련 자료: DexRoam

## 1. 연구 계보

DexRoam은 네 흐름이 만나는 자리에 있습니다. 참고문헌 번호는 논문 v1 기준입니다.

1. **전신·모바일 조작 데이터 수집.** Stretch [2], Mobile ALOHA [3], HOMIE [4], HumanoidExo [5], NuExo [6]은 전용 하드웨어 인터페이스로 로봇 시연을 모읍니다. 논문은 이 방식이 "점점 더 특수한 인터페이스를 요구해 규모를 키우기 어렵다"고 봅니다(§I). DexRoam의 로봇 쪽 수집(Quest 3S + 컨트롤러 + MANUS 장갑, §III-A2)은 이 계열의 한 변형이고, 사람 데이터 수집이 이 논문의 차별점입니다.
2. **사람 시연을 로봇 정책 감독으로.** DexCap [10], UMI [11], ARCap [12], Open-TeleVision [13] 같은 휴대형 수집 시스템과 MotionTrans [14], EgoScale [15], DexWild [16], H-RDT [17], EgoVLA [18]이 "사람 데이터가 정책을 개선한다"는 흐름입니다. DexRoam의 §III-A3는 DexCap류의 추적기 기반 수집에서 외부 카메라와 몸에 다는 추적기를 뺀 형태로 읽을 수 있습니다.
3. **모바일·전신 정책으로 사람 데이터를 옮기기.** EgoHumanoid [19], EMMA [20], HoMMI [21], HALOMI [22], BifrostUMI [23], HuMI [24]는 이동·전신 과제에서 사람 시연을 씁니다. 논문 §I, §V-B는 이 계열이 "이산 이동 명령, 단계(phase) 분해, 끝 효과기·키포인트 목표 + 하위 컨트롤러"로 사람 동작을 단순화하는 대가를 지적하고, DexRoam은 **연속·결합 행동을 그대로** 로봇 행동 청크로 옮긴다는 점으로 구분합니다. HERMES [58]는 모바일 양팔 다지 조작을 다루지만 학습한 조작 기술을 별도 항법 모듈에 연결한다고 서술합니다(§V-A).
4. **표준 VLA 백본.** GR00T N1 [31]과 π0.5 [32]를 사람-로봇 혼합 데이터로 미세조정합니다. 사람 데이터 전용 헤드나 입력 분기를 두지 않는다는 점이 설계 제약입니다(§II의 조건 `R(D_human) ⊂ S`).

참고문헌 번호는 논문 참고문헌 목록(번호 19: EgoHumanoid, 20: EMMA, 21: HoMMI, 22: HALOMI, 23: BifrostUMI, 24: HuMI)으로 확인했습니다. 논문은 이들과 정량 비교하지 않고 관련 연구로 인용합니다(실험의 기준선은 Robot-only 하나).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 사람 전신 + 손 추적 | Meta Quest 3, WebXR (Vuer [29]) | 몸 관절 33개, 손당 25개 랜드마크. 코드는 Vuer 기반 스트리밍 |
| 1인칭 스테레오 영상 | ZED Mini (HD1080, 30 fps) | 코드는 ZED SDK 4.x 전제(5.x 비호환 명시) |
| 손 리타게팅 | dex-retargeting [30] (AnyTeleop) | 벡터 리타게팅, 스케일 1.2, 저역 통과 0.5. 코드는 v0.5.0 스냅샷을 포함 |
| 로봇 손·플랫폼 | XHand(12자유도), Astribot | 하드웨어 가이드는 공개 문서(링크만 확인) |
| 로봇 시연 수집 | Meta Quest 3S, MANUS Pro 장갑 | 단일 조작자 제어 |
| 정렬 수식 | 루트 프레임 투영, 상대 변환, SLERP | 새 알고리즘이라기보다 표준 기하 연산의 조합(§III-B) |
| 시간 정렬 | 과제 진행 정규화 + 선형/SLERP 재표집 | 로봇 평균 길이를 목표로 사용 |
| VLA 백본 | GR00T N1.7-3B, π0.5 | 코드는 OpenPI와 Isaac-GR00T를 복사해 수정 |
| 정렬 평가 | MMD(RBF, 편향 V-통계량), SWD(슬라이스 W1, L = 128) | 부록 E.1에 식. 평가 코드는 공개되지 않음 |
| 부드러움 | LDLJ, SPARC | 부록 B2. 표 V |
| 과제 평가 | SR + 5단계 가중 TCS | 표 X |

## 4. GitHub와 자료 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 푸시 | 비고 |
|---|---|---|---|---|
| **공식 코드 (수집 + 정렬)** | https://github.com/zhourui9813/DexRoam | MIT | 9 / 2026-09-29 | 커밋 3개, 최신 `e539387…`(전체 해시는 `04`). 얕은 clone, 정적 읽기만 |
| **공식 코드 (정책 학습)** | https://github.com/zhourui9813/DexRoam-Policy-Training | MIT | 4 / 2026-09-29 | 커밋 2개, 최신 `a82d2c9…`. OpenPI와 Isaac-GR00T 기반 |
| 프로젝트 페이지 | https://dexroam.github.io/ | 저장소(DexRoam/dexroam.github.io)에 라이선스 없음 | - / 2026-09-29 | 논문, 영상, 코드, 데이터, 하드웨어 문서 링크 |
| 하드웨어 가이드 | 프로젝트 페이지의 Google Docs 링크 | - | - | 카메라 마운트 조립 설명. 3D 모델은 코드 저장소 `assets/3d_model/` |
| 데이터셋 | https://huggingface.co/datasets/zhourui9813/DexRoam_Realworld_Data | 카드에 라이선스 미표기 | 다운로드 167 / 2026-09-21 | 로봇 에피소드 50개(Push Chair & Close Laptop). 크기 약 244 GB(HF API 파일 크기 합계, 받지 않음). 사람 데이터와 다른 4과제는 없음 |
| dex-retargeting | https://github.com/dexsuite/dex-retargeting | MIT | 1,322 / 2025-08-21 | 코드에 v0.5.0 스냅샷 포함(requirements 파일은 0.4.6 표기) |
| Vuer | https://github.com/vuer-ai/vuer | MIT | 412 / 2026-11-23 | 사람 쪽 WebXR 스트리밍 |
| OpenPI (π0.5) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 14,767 / 2026-08-24 | 정책 학습 코드의 기반 |
| Isaac-GR00T (N1.7) | https://github.com/NVIDIA/Isaac-GR00T | Apache-2.0 | 8,419 / 2026-11-17 | 정책 학습 코드의 기반 |
| LeRobot | https://github.com/huggingface/lerobot | Apache-2.0 | 29,091 / 2026-11-26 | 데이터 변환 대상 형식(v2.1) |
| MotionTrans (관련 연구) | https://github.com/michaelyuancb/motiontrans | 미확인 | 64 / 2026-03-09 | 논문 [14]. 사람 VR 데이터로 모션 수준 학습 |
| EgoHumanoid (관련 연구) | https://github.com/OpenDriveLab/EgoHumanoid | Apache-2.0 | 264 / 2026-06-06 | 논문 [19] |

모델 가중치(GR00T N1.7, Cosmos-Reason2-2B, π0.5)와 HF 데이터는 받지 않았습니다. 정책 학습 저장소 안에는 학습 실행 설정과 통계 파일(체크포인트 폴더의 `experiment_cfg/`, `processor/`)이 들어 있고 모델 가중치는 없습니다.

## 5. 가까운 대안과 비교

| 접근 | 사람 데이터 형태 | 행동 공간 | 이동 | 이 논문과 다른 점 |
|---|---|---|---|---|
| DexRoam | 1인칭 스테레오 + 전신·손 랜드마크 | 56차원 연속, 결합, 상대 행동 | 연속 베이스 명령 | 기준 |
| EgoMimic | Aria 안경 + 손 추적 | 양손 끝 효과기 | 없음 | 고정 베이스, 상대적으로 단순한 행동 공간 |
| EMMA | 1인칭 영상 + 사람 손 | 이동은 별도 | 이동 포함 | 논문 §I의 "단계 분해"의 예 |
| EgoHumanoid / HoMMI / HuMI / BifrostUMI / HALOMI | 휴대 장비로 전신 시연 | 끝 효과기·키포인트 + 하위 컨트롤러 | 휴머노이드 | 논문은 "하위 컨트롤러에 맡기는 목표 표현"으로 분류 |
| Mobile ALOHA | 로봇 텔레옵만 | 관절 + 베이스 | 있음 | 사람 데이터 없음. 로봇 데이터만 |
| DexRoam의 Robot-only | - | 같음 | 같음 | 이 논문의 기준선(같은 로봇, 로봇 데이터만) |

위 표의 "EMMA" 등 항목은 논문 §I, §V의 서술에 근거한 분류이고, 각 논문을 이 저장소에서 직접 분석한 것은 아닙니다(분석한 것은 §3).
