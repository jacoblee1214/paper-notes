# 02. 선행 연구, 구성 요소의 출처, 관련 분석, 공식 자료

## 1. 연구 계보

ForceMimic은 세 줄기가 만나는 곳에 있습니다.

1. **사람이 드는 데이터 수집:** 원격조작(RoboTurk [4]) → 손에 드는 그리퍼(UMI [5], Song 2020 [20], Young 2021 [21]) → 손 모션 캡처(DexCap [6]). 힘을 기록하는 쪽은 드뭅니다. ForceCapture는 손 장치에 6축 힘 센서를 붙이고, 중력 보정으로 상호작용 힘만 남깁니다(§III-A, §III-B).
2. **모방학습 정책:** BC [23], BeT [24], IBC [25], VINN [26] → 확산 정책 Diffusion Policy [27], 3D 변형 DP3 [28], RISE [29]. HybridIL은 DP [27]를 바탕으로 힘-토크를 출력에 넣습니다.
3. **힘을 쓰는 조작:** 임피던스 제어(Hogan [35], Ott 2010 [36]), 힘-위치 하이브리드 제어, 힘을 쓰는 학습(병 뚜껑 열기 [31], Jenga [33], MOMA-Force [34], ForceSight [37]). 본문은 힘을 **입력**으로 넣는 방식(Force DP)과 **출력**으로 내는 방식을 비교합니다.

껍질 벗기기 연구는 드뭅니다. Dong 2021 [38], MORPHeus [39], 칼을 쓰는 방법 [40], 손을 쓰는 방법 [41]이 있고, 저자들은 도구를 쓰는 기술 학습을 강조합니다(§II-c).

## 2. 구성 요소별 출처

| 구성 요소 | 논문 출처 | 이 저장소 내 대응 | 비고 |
|---|---|---|---|
| 손 장치와 SLAM 추적 | DexCap [6], UMI [5] | `repo/forcecapture` (main, gripper 브랜치) | 장치 설계 자체는 자체 설계 |
| 중력 보정과 힘 변환 | 논문에 수식 없음, 최소제곱 | `forcecapture/tare_pyft.py` (r3kit의 LinearMFTarer 등) | 외부 라이브러리 r3kit에 의존, 저장소에는 없음 |
| 정책 구조 | Diffusion Policy [27] | `hybridil/robomimic/algo/diffusion_policy.py` (DiffusionPolicyUNetDex) | robomimic [DP 코드 기반] |
| 점군 인코더 | 논문은 "MLP" | `hybridil/robomimic/models/base_nets.py` PointNet | 실제는 1D 합성곱 + 최대 풀링 (`04` D6) |
| 힘 프리미티브 | 하이브리드 힘-위치 제어 [35, 36] | `hybridil/devices/flexiv.py` (Flexiv RDK 0.9.1) | 로봇 API에 의존 |
| IK | Pinocchio [42] | `hybridil/devices/ik.py` | pin 3.4.0 |
| 실험 과제 설정 | RH20T [7] (원격조작 설정) | 없음 | 원격조작 기준선은 저장소에 없음 |
| 확산 정책 학습 | robomimic, DP 코드 | `hybridil/train.py`, `configs/train_dp_ftout.json` | 1000 에포크 설정 (`04` D1) |

## 4. GitHub, 프로젝트 페이지, 관련 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 최신 커밋 | 비고 |
|---|---|---|---|---|---|---|
| **공식 학습 코드 (HybridIL)** | https://github.com/ForceMimic/hybridil | MIT | 48 | 2025-03-05 | `1defa54` (커밋 3개) | `repo/hybridil/` 얕은 clone, 정적 읽기만 |
| **공식 수집 코드 (ForceCapture, main)** | https://github.com/ForceMimic/forcecapture | MIT | 37 | 2025-03-05 | `acafb08` (커밋 2개) | `repo/forcecapture/` 얕은 clone, 고정 공구 형태 |
| 수집 코드 (gripper 브랜치) | 같은 저장소, `gripper` 브랜치 | MIT | - | 2025-03-04 | `3eabf58` | `repo/forcecapture_gripper/` 얕은 clone. 그리퍼 형태 |
| 프로젝트 페이지 | https://forcemimic.github.io | (저장소에 라이선스 없음) | 2 | 2025-03-04 | `6ca69c5` | 정적 HTML. 논문과 두 코드 저장소 링크, 영상 |
| 비공식 사본 | https://github.com/0mrrabbit0/Forcemimic | MIT | 0 | 2026-04-13 | `ea6360` | 공식 조직이 아님. 파일이 더 많음(`tare_pyft_1.py`, `tare_pyft_copy.py`). **클론하지 않음**(출처 불명) |
| 데이터셋 | 저장소 README의 Google Drive 링크 | 미확인 | - | - | - | **받지 않음.** 크기 미확인 |
| 학습 체크포인트 | 없음 | - | - | - | - | 공개되지 않음 |
| r3kit, flexiv_rdk 0.9.1 | 외부 의존 | 미확인 | - | - | - | 설치하지 않음. 로봇 SDK는 실로봇 전용 |

- 두 공식 저장소는 2025년 3월 초에 마지막으로 바뀌었습니다. 논문 v3 날짜(2025-03-03)와 맞습니다.
- 공식 저장소에는 **정책 학습과 로봇 실행 코드만** 있습니다. 실험의 평가 코드(성공 판정, 수집 효율 측정)와 세 기준선(Raw DP, Force DP, Force+Hybrid DP)은 없습니다. 기준선 결과는 이 저장소로 재현할 수 없습니다(`04` §3 D21).
- 프로젝트 페이지는 영상과 CAD를 링크하지만, CAD와 설치 가이드의 실제 파일은 이번 분석에서 받지 않았습니다(고정 공구 형태의 CAD는 Onshape 링크입니다, forcecapture README).
