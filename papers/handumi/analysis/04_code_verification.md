# 04. 공식 코드 대조 (논문 ↔ 저장소)

대조 대상은 `repo/handumi-sw/`(커밋 `b9258f07`, 2026-11-08), `repo/handumi-hw/`(커밋 `ad9e4119`, 2026-11-05), `repo/handumi-quest-app/`(커밋 `1ed5935f`, 2026-10-19)입니다. **정적 읽기만 했습니다.** 설치, 실행, import, 테스트 실행, 체크포인트나 데이터 다운로드는 하지 않았습니다. 각 항목은 파일과 줄을 적었고, 확인하지 못한 것은 "미확인"이라고 표시했습니다.

## 1. 대조표 (D1–D10)

| 번호 | 논문 주장 | 저장소의 실제 값 | 판정 |
|---|---|---|---|
| D1 | 동기화 시간차 한계 60 ms (§2, "60 ms skew limit") | `src/handumi/dataset/capture.py:7`: `MAX_SYNC_SKEW_S = 0.06` | **일치** |
| D2 | 불량 행 1% 허용 (추적, 상태, 동기화) (§3.1) | `src/handumi/dataset/quality.py:27–29`: 세 값 모두 `0.01` | **일치** |
| D3 | 센서 워밍업 0.2 s 제외 (§3.1) | `quality.py:39`: `sensor_warmup_s: float = 0.2` | **일치** |
| D4 | 1 s 이상 센서 끊김이면 에피소드 폐기 (§2) | `teleop_record.py`에 `SENSOR_LOSS_TIMEOUT_S` 상수가 있음. 값은 확인 못 함 | 미확인 (값 대조 필요) |
| D5 | 기저 관절 회전 60° 초과 에피소드 제외 (§3.1, 79개) | `dataset/screening.py:165`: `max_base_rotation_deg`의 기본값은 `None`(꺼짐), 주석 "Off by default". 논문 실행 설정은 공개되지 않음 | **설정 차이.** 기본값은 꺼짐. 논문은 60°를 적용했다고 쓰지만 실행 설정 파일과 명령을 제시하지 않음 |
| D6 | IK 위치 오차 p99 > 3 cm이면 거부 (표 3 각주) | `screening.py:153`: `max_position_error_m: float = 0.03`, 878행에서 `position_error_p99_m`과 비교 | **일치** |
| D7 | 거부 기준: 도달 불가 시작 자세 또는 p99 > 3 cm (표 3 각주) | `screening.py`에 `retarget_start_pose_unreachable`를 reject로 기록 | **일치** |
| D8 | 자기 충돌은 거부 기준이 아니라 기록 (표 3 열) | `screening.py:157` `self_collision_margin_m = 0.01`, `self_collision_frames` 기록. 자기 충돌이 reject인지는 확인 못 함 | 미확인 (C6과 연결) |
| D9 | 로봇별 IK 오차 (표 3, 위치 중앙값) | 논문 값 (PiPER 1.17 mm)은 실행 결과라서 저장소로 재현 불가. 저장소 내 메타 값은 없음 | 재현 안 함 |
| D10 | 보정 게이트: 내부 파라미터 평균 재투영 ≤0.8 px (§2) | `scripts/recover_workspace_camera.py:45`: `--max-error-px` 기본 `6.0` (작업 공간 카메라 복구용). 내부 파라미터 게이트 상수는 이번 조사 범위에서 찾지 못함 | **불일치 가능.** 작업 공간 카메라 게이트는 6 px. 논문 0.8 px 게이트의 위치는 확인 못 함 |
| D11 | TCP 피벗 RMS < 5 mm, 최대 잔차 < 10 mm (§2) | `calibration/control_tcp.py:210–242`: 잔차의 RMS와 최댓값을 계산. 임계값 상수는 확인 못 함 | 계산 방식 일치. 임계값 미확인 |
| D12 | 평행 턱 벌림 0–66 mm → PiPER 행정 선형 사상 (§2) | 조사 범위에서 66 mm 상수를 찾지 못함 | 미확인 |
| D13 | 로봇 프로파일의 시뮬 배치 (표 3) | 시뮬 배치 파일 9개 모두 `verified: false`. 금속(Metal)은 0.45 m → 0.35 m로 변경 주석 있음. PiPER는 0.325 m (실제 배치 측정, 2026-10-11) | **불일치 가능.** D14 참고 |
| D14 | Metal 행 표 3은 현재 배치에서 나온 결과 (표 3 캡션) | `configs/calibration/table/sim/metal.yaml` 주석: "The previous 0.45 m offset was checked on episode 0 ... those metrics do not validate this adjusted placement." 커밋 `b9258f07` 메시지도 "Clarify that historical Metal IK metrics apply to the previous placement" | **불일치.** 표 3의 Metal 행은 옛 배치 값으로 보임 |

### 1.1 D5 설명

논문 §3.1은 PiPER 기저 관절이 60°를 넘는 79개를 뺐다고 씁니다. 저장소 기본값은 이 규칙을 **끈 상태**입니다. 주석은 "what the base may do is a property of the mount and the task"라고 적었습니다. 따라서 논문 결과를 재현하려면 실행 설정에서 `max_base_rotation_deg=60`을 켜야 합니다. 논문은 그 설정을 공개하지 않았습니다.

### 1.2 D8 설명

논문 표 3의 R1 Lite 자기 충돌 16.4%는 거부 열에 포함되지 않습니다. 저장소에서도 자기 충돌은 `self_collision_frames`로 기록되는 신호입니다. 거부로 바뀌는지는 여기서 확인하지 못했습니다. 따라서 초록의 "≤4.7% rejected"가 자기 충돌 에피소드를 포함하는지는 **미확인**입니다.

## 2. 설정과 논문의 학습·검증 차이

- `configs/calibration/table/sim/` 아래 9개 파일은 모두 `verified: false`입니다. 시뮬 배치는 확인된 값이 아니라고 저장소가 스스로 적어 둡니다.
- 논문 §2는 PiPER와 OpenArm의 재생이 MuJoCo [37]와 IK 설정을 공유한다고 씁니다. 이 대조는 하지 않았습니다.

## 3. 라이선스와 권리 (정적 확인)

| 저장소 | 라이선스 (GitHub API) | 확인 내용 |
|---|---|---|
| `handumi-sw` | Apache-2.0 | `LICENSE` 파일과 API 값 일치 |
| `handumi-hw` | Apache-2.0 | README 배지와 `LICENSE` 파일 |
| `handumi-quest-app` | Apache-2.0 (예외 있음) | `NOTICE`: "independent, best-effort behavioral reconstruction of YubiQuestApp" (AIRoA의 APK 기반). APK에서 나온 `evidence/`는 Apache 부여 범위에서 **제외** |

- 원래 Quest 앱(YubiQuestApp)의 라이선스는 NOTICE에 "not stated"라고 적혀 있습니다. 재구성은 APK에서 추출한 프로토콜 이름과 상수를 "상호운용 목적"으로 썼다고 적습니다. 이 저장소를 가져다 쓸 때는 `evidence/` 파일을 복사하지 않는 것이 안전합니다.
- 논문은 "Apache-2.0"이라고만 씁니다. Quest 앱이 예외 조항을 가진 재구성이라는 점은 논문에 없습니다.

## 4. 구현 가능성 판정

### 4.1 판정

**이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 전체 재현은 불가능합니다. 소프트웨어 파이프라인 일부(시뮬 재타깃팅과 기록 검사)는 조건부로 가능합니다.**

### 4.2 자원별 구분

| 구분 | 항목 | 이 맥에서 | 근거 |
|---|---|---|---|
| 가능 | 표 재계산과 통계 (`recompute.py`) | 완료 | `03_analysis.md` |
| 조건부 | `handumi-sw` 설치 후 PiPER 시뮬 재타깃팅 1개 에피소드 | 설치 필요. JAX와 PyRoki 필요 (CPU 가능 여부 미확인). 설치 승인 필요 | README "Quick Start", `pyproject.toml` |
| 조건부 | 표 3 Metal 행을 현재 배치로 다시 계산 | 1번과 같은 설치 필요. 원본 데이터(`handumi-towel-raw-562`) 다운로드 필요 | D14 |
| 조건부 | 기록 검사(`handumi dataset qa`)를 원본 데이터에 적용 | 원본 데이터가 1 GB 이상일 가능성, 다운로드 승인 필요. 크기 미확인 | 데이터 목록 |
| 불가 | ACT 학습 (562개, 학습 스텝 수는 논문 미기재) | 공식 설정은 CUDA 전제라고 판단. MPS에서 시간을 재지 않았음 | 공식 README와 설정 |
| 불가 | π0.5 미세 조정 | 같은 이유. 대형 모델 | 논문 §3.2 |
| 불가 | PiPER 실로봇 towel folding (20회 × 모델) | 로봇 필요 | — |
| 불가 | Quest 3S 기록 검증 | 기기와 `docs/quest-hardware-validation.md` 검증 필요. 저장소도 검증 전 교체품으로 보지 말라고 씀 | `handumi-quest-app/README.md` |

### 4.3 노력 추정 (측정 없음)

- 설치와 시뮬 재타깃팅 1개 에피소드 확인: **약 0.5–1일** (추정, 설치 시간 미측정).
- 표 3 전체 7개 로봇 재계산 (현재 배치): **1–2일** (추정, 계산 시간 미측정).
- 원본 데이터 기록 검사: 다운로드 시간과 디스크 크기에 따라 달라짐. 측정 전 판단 보류.
- 학습과 실로봇 평가는 이 저장소의 범위 밖입니다.

## 5. 확인하지 못한 항목 (정리)

- D4 센서 끊김 1 s의 실제 상수 값.
- D8 자기 충돌 거부 여부.
- D10 내부 파라미터 0.8 px 게이트의 위치.
- D11, D12의 임계값 상수 (5 mm, 10 mm, 66 mm).
- 논문 §2의 "14차원 상태" 구성과 저장소 `robots/` 정의의 대조.
- 학습 설정 (논문 §3.2의 배치와 학습률). 공식 학습 설정 파일은 이번 조사에서 읽지 않았음.
