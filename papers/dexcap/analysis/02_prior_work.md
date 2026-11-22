# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

DexCap은 새 알고리즘 하나보다 **장비 + 리타게팅 + 점군 정책 + 교정**의 조합입니다. 구성 요소마다 출처가 다릅니다.

| 구성 요소 | 논문에서의 역할 | 출처 (논문 참고문헌 번호) | 메모 |
|---|---|---|---|
| EMF 장갑으로 손가락 추적 | 가림에 강한 손가락 관절 | Rokoko 장갑 (§III-A), 원격조작 장갑 [51–53] | 손목 pose는 주지 않음 (§II-B) |
| 손목 SLAM (카메라 두 대와 IMU) | 세계 좌표 손목 6-DoF | Intel T265 (§III-A) | 손목 드리프트를 IMU 대비 줄임 (§3.7) |
| 가슴 RGB-D로 장면 점군 | 관찰 입력 | Intel L515 (§III-A) | 몸통 움직임을 세계 좌표로 제거 |
| 인간 손 접촉 우선 가정 | 손끝 IK 정당화 | HO3D [41], GRAB [44], ARCTIC [45] | 근거는 기존 데이터셋 통계 |
| 손끝 IK 리타게팅 | 로봇 손 관절 계산 | [96–98] (RelaxedIK, CollisionIK 등) | PyBullet IK 구현 (공식 저장소 STEP2) |
| 점군 확산 정책 | 행동 생성기 | Diffusion Policy [14] (Chi 외 2023) | 점군 입력 버전은 [103] (Qin 외) |
| BC-RNN 기준선 | MLP-LSTM 기준선 | [102] (Mandlekar 외) | robomimic 구현 |
| Perceiver 점군 인코더 | 기본값 DP-perc | [105, 106] (Jaegle 외, Set Transformer) | 공식 저장소 코드 경로는 PointNet 고정 (`04` D3) |
| 반복 가중 교정 (IWR) | 교정 데이터 혼합 | [101] (Mandlekar 외) | 원 데이터와 같은 확률로 샘플 |
| 손끝 로봇 손 | 대상 로봇 | LEAP Hand [95] | 인간 손보다 약 50% 큼 |
| OSC 팔 제어 | 저수준 팔 제어 | [108] (Khatib) | Deoxys 기반 (공식 저장소 README) |
| 비전 손 복원 기준 | 비교 대상 | HaMeR [39] | 그림 7 정성 비교 |
| 원격조작 (대조) | 처리량 기준 | DIME [31], Qin 외 [35] | 처리량 3배 주장 (§3.1) |

### 1.1 가까운 대안과 차이

- **원격조작 기반 데이터 (DexPilot [1], DIME [31], VideoDex [3]):** 로봇 하드웨어가 필요합니다. DexCap은 로봇 없이 기록하고, 대신 리타게팅 오차를 떠안습니다.
- **비전 기반 손 추적 (DexMV [33], DexVIP [34], HaMeR [39]):** 카메라 한 대면 가림과 깊이 모호성에 약합니다. DexCap은 장갑 센서를 씁니다.
- **IMU 전신 mocap (§II-B [46–50]):** 드리프트가 쌓입니다. DexCap은 SLAM으로 드리프트를 줄입니다 (§3.7, 실험 조건은 `01` §3.7).
- **휴대형 그리퍼 데이터 ([88–94], 그중 UMI [94]):** 평행 조 그리퍼만 다룹니다. DexCap은 다지 손을 다룹니다.
- **인간 손 데이터 기반 정책 (MimicPlay [10], 손 궤적 [80–82]):** 2D 궤적이거나 상위 계획입니다. DexCap은 저수준 3D 행동을 직접 학습합니다.

## 2. 코드와 저장소 (GitHub 표)

확인 날짜: 2026-11-22. 별 수와 마지막 push는 GitHub API 메타데이터입니다. 공식 저장소는 얕은 복제(`--depth 1`)로 `dexcap/repo/`에 두고 **정적으로만 읽었습니다** (실행, 설치, 가중치 다운로드 없음).

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [j96w/DexCap](https://github.com/j96w/DexCap) | 공식 구현 (RSS 2024 저장소, 설명란에 "[RSS 2024]") | MIT (LICENSE 파일, GitHub 판정 MIT) | 427 | 2024-10-10 (API `pushed_at`) | HEAD `4b0bed0` (2024-08-17 "hardware updates") | 정적 읽기 (`repo/`) |
| [dex-cap.github.io](https://dex-cap.github.io) | 프로젝트 페이지 | 표기 없음 | — | — | 페이지 HTML에서 저장소 링크와 arXiv 링크 확인 | 읽음 |
| [chenwangj/DexCap-Data](https://huggingface.co/datasets/chenwangj/DexCap-Data) (Hugging Face) | 공개 데이터 (처리본 2과제 + 원시 zip 2개) | 메타데이터에 license 없음 | 좋아요 8, 다운로드 794 | 2024-04-15 (lastModified) | 파일 목록과 크기 (메타데이터만) | 받지 않음 (`04` D12) |
| [leap-hand/LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API) | 손 제어 API (저장소의 LEAP 메시와 관련) | NOASSERTION (GitHub 자동 판정 실패, LICENSE 파일 확인 필요) | 179 | 2025-10-20 | 참고 | 읽지 않음 |
| [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | 학습 코드 기반 (공식 저장소 STEP3에 사본) | MIT | 1,608 | 2026-08-09 | 사본과 원본 차이는 보지 않음 | 참고 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 확산 정책 기반 | MIT | 4,737 | 2024-12-24 | 참고 | 참고 |
| [UT-Austin-RPL/deoxys_control](https://github.com/UT-Austin-RPL/deoxys_control) | 팔 제어기 (공식 README가 기반이라고 밝힘) | Apache-2.0 | 312 | 2024-05-23 | 참고 | 읽지 않음 |
| `DexRobot/dexcap-sdk-python`, `DexRobot/dexcap-sdk-cpp` 등 (검색 결과) | 제3자 SDK. 이름만 같고 DexCap 논문 저자와 관계는 확인하지 못함 | 표기 없음 | 0–2 | 2026-09 ~ 2026-11 | 확인 안 함 | 공식 아님 |

- 공식 저장소의 **학습 설정 파일**(`STEP3_train_policy/robomimic/training_config/`)은 두 개뿐입니다 (포장, 닦기). 가위, 차 준비, 공 수집 설정은 없습니다.
- 공식 저장소에는 **로봇 실행 코드 (정책 롤아웃, OSC 제어, 페달 교정)**가 없습니다. `robomimic/envs/env_real_panda.py`는 있지만 LEAP 손 실행 루프가 없습니다 (`04` D10).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **점군 확산 정책 (DP)이 MLP보다 낫다:** DP 원 논문 [14]의 주장. 이 논문은 같은 방향을 Table I에서 재확인합니다 (`03` C6).
- **이미지 입력은 시점 변화에 약하다:** 이 논문의 주장이고, 근거는 0/30 (Table II)입니다. 선행 연구의 방향과 같지만 이 실험에서 원인을 나눠 본 것은 없습니다 (`03` C10).
- **손끝이 가장 많이 접촉한다:** HO3D, GRAB, ARCTIC 통계 [41, 44, 45]. 이 논문의 리타게팅 설계가 여기에 기댑니다. 이 데이터셋들은 평평한 물체 위주라서, 가위처럼 손가락을 손잡이에 넣는 과제에는 가정이 약합니다 (본문 §IV-C가 직접 인정).
- **사람 교정은 데이터 효율을 높인다:** IWR [101] 및 교정 학습 [85–87]. 이 논문은 교정 효과와 "교정 데이터가 고정 카메라에서 얻어졌다"는 설명을 함께 제시하지만 두 요인을 분리하지 않았습니다.
