# 04. 공식 코드 대조와 구현 가능성: YUBI (arXiv:2606.10244 v1)

공식 저장소를 **정적으로 읽기만** 했습니다. 실행, 설치, import, 가중치·데이터 다운로드는 하지 않았고, 복제한 내용은 신뢰하지 않는 입력으로 다뤘습니다. 대조 기준은 `02` §2의 커밋입니다(yubi-sw `1eccb04`, yubi-core `620a069`, rebake `1ad9453`, yubi-hw `dd8bd13`, yubi-app `dc1362d`, yubi-real2sim `c1d5d41`, umi-arena-evaluation `f60d4e6`). 번호 D는 논문과 코드·공개 자료의 차이입니다.

## 1. 저장소 구조와 논문 대응

| 논문 요소 | 공개물 | 확인 |
|---|---|---|
| 그리퍼 설계 (§3.1, 그림 20) | yubi-hw: `STEP/`, `STL/`, `docs/BOM/`, `docs/AssemblyInstruction/` (수집용 "Glove", 로봇용 DYNAMIXEL "Gripper", 책상 틀, 휴대 틀, 플랜지) | 파일 목록과 BOM, README 확인. CAD 내용과 치수는 열지 않음 |
| ROS 2 수집 (§3.2) | yubi-sw `yubi_bringup`(엔코더 노드, 카메라 구성, 발판, 더블클릭 감지), `airoa_quest_bridge`(Quest TCP/JSON 브리지), `firmware/ESP32C6_AS5601`, `yubi_description`(URDF) | 읽음 |
| 녹화, 업로드, 기록 게이트 | yubi-core (`record_manager`, `recording_gate`, `recording_uploader`, `data-backend`) | 설정과 문서 읽음 |
| 수집 관리 웹 앱 (부록 D.2) | yubi-app (Go 백엔드 125개 내부 파일, TypeScript 프런트) | 구조만 확인 |
| LeRobot 변환, 데이터 처리 (§4, A.2) | rebake: `config/pipeline/yubi.yaml`, `config/robot_model/yubi.yaml`, 보강기 코드 | 읽음 |
| 정책 학습과 배치 (§6, C.3) | **없음** (`02` §2, D12) | - |

## 2. 논문과 코드의 차이 (D1–D17)

### D1. 회전 표현: 오일러각 대 쿼터니언

§4는 상대 포즈를 번역 t ∈ ℝ³와 회전 r ∈ ℝ³(오일러각)으로 저장한다고 씁니다. rebake의 `robot_model/yubi.yaml`은 모든 포즈를 `[x, y, z, qx, qy, qz, qw]` 7차원(절대)과 `[dx, dy, dz, dqx, dqy, dqz, dqw]` 7차원(델타)으로 내보냅니다. 오일러각은 코드 어디에도 없습니다(rebake `python/rebake`에서 euler, rpy를 찾지 못함). 논문 본문이 데이터 스키마를 단순화했거나 공개 변환 구성이 내부 학습용 변환과 다릅니다.

### D2. 행동의 정의

논문은 "손목 프레임 기준 상대 포즈"(§4)와 "상대 말단 포즈 청크"(§6)입니다. 공개 파이프라인(`yubi.yaml`)은 `DeltaTransformEnricher`를 `delta_reference_frame: previous_target_frame`으로 돌려 **30 Hz 한 스텝 전 대비 델타**(몸체 프레임)를 만들고, 행동 특징은 `action.left_controller.relative`, `action.right_controller.relative`, `action.pose.{left,right}_hand_root.relative`로 둡니다. 조리개 행동은 `/joint_states`를 `ShiftEnricher`로 1스텝 당긴 값입니다(`shift_steps: 1`). 학습에 쓴 "청크 시작 대비 상대"로 바꾸는 코드는 공개물에 없습니다. 델타를 청크로 누적하는 방식(오차 누적)과 청크 첫 프레임 대비 방식은 정책 성능과 폐루프 안정성에 영향이 큰 선택인데 논문에도 코드에도 없습니다.

### D3. 손목 카메라 사양

논문: 손목 카메라 100 Hz(§3.2). `yubi_bringup/config/common/yubi_devices.yaml`: 두 손목 카메라 모두 `framerate: 120.0`, 640×480, `mjpeg2rgb`. 기록 게이트는 60 Hz 미만 경고, 30 Hz 미만 정지(`recording_gate.yaml`). 논문은 해상도를 적지 않았고 대회 페이지는 480×640을 적습니다. 100 대 120 Hz는 실측 대 설정값 차이일 수 있습니다. LeRobot 30 Hz로 줄이므로 정책 입력에는 영향이 작습니다.

### D4. 타임스탬프

논문: 모든 스트림이 "timestamped at the source under a shared clock"(§3.2). `encoder_node.py`의 `on_timer`는 100 Hz 타이머에서 `self.get_clock().now()`로 스탬프를 찍어 마지막으로 읽은 값을 발행합니다(원천 시각이 아니라 PC 발행 시각, 직렬 지연은 보정하지 않음). 펌웨어는 `delay(5)` 루프로 라디안 값을 115200 baud로 보내고 장치 시각이 없습니다. 조리개 값과 영상·포즈 사이의 정렬 정확도는 논문도 측정하지 않았습니다.

### D5. 필터 단계가 공개물에 없음

부록 A.2는 (i) 최소 길이, (ii) 정지 신호 세 종류, (iii) 운동학 타당성, (iv) 추적 품질 플래그로 구성된 필터를 "Rust 데이터 처리 파이프라인"에서 LeRobot 변환과 함께 한다고 씁니다(부록 D.2). rebake는 Rust 핵심과 Python 단계를 가졌지만 `yubi.yaml`의 단계는 수집 → TF 버퍼 → 0차 유지 동기화(30 fps) → TF 체인 → 델타 → 이동 → LeRobot 변환뿐이고, 필터 단계가 없습니다(rebake 소스와 문서에서 stuck, freeze, plausibility, drop, reject에 해당하는 구현을 찾지 못함). 임계값이 논문에도 숫자로 없어서 이 필터를 재구현해 같은 데이터를 얻을 방법이 없습니다.

### D6. 추적 품질 처리: 사후 제거 대 실시간 정지

논문: Quest가 ROS에 내는 프레임별 추적 품질 플래그가 "작은 비율" 넘게 불량한 에피소드를 사후에 버린다(A.2 (iv)). 코드: `/quest/controller/{left,right}/{tracked,valid}` Bool을 기록 토픽에 포함하고(`robot_config.yaml`), **기록 게이트가 실시간으로** 추적 소실·무효를 0.5 s 지속하면 새 녹화를 막고(BLOCK_START), 1.0 s 지속하면 진행 중 녹화를 멈춥니다(HARD_STOP, `recording_gate.yaml`의 `vr_tracking`). 게이트는 시작 시 fail-closed입니다(`lifecycle_recording_gate.md`). 즉 코드의 1차 방어는 온라인 정지이고, 논문이 말하는 사후 비율 기준은 공개물에서 찾을 수 없습니다. 두 장치가 모두 쓰였을 가능성이 크지만 데이터에 남은 에피소드의 추적 불량 비율은 알 수 없습니다.

### D7. 외부 보정(ChArUco)이 공개물에 없음

부록 A.2와 그림 12는 ChArUco 판으로 Quest와 책상 좌표계를 맞춘다고 설명합니다. yubi-sw, yubi-core, rebake, yubi-app 어디에서도 charuco, aruco를 찾지 못했습니다. "calibrat"은 엔코더 최솟값 캡처(`tools/yubi_udev_setup.sh`, `yubi_device_select_gui.py`)에만 나옵니다. rebake의 `TfChainEnricher`는 `quest_origin`을 기준으로 포즈를 체인하지만 책상 정렬 변환은 구성에 없습니다. 책상 22개의 Quest 원점이 서로 다르다는 논문 설명이 맞다면 공개 변환 결과는 책상 좌표계가 아니라 `quest_origin` 기준일 수 있습니다.

### D8. 데이터셋 라이선스와 공개 상태

논문 D.2: LeRobot 데이터를 "Hugging Face에서 CC-BY-NC-SA-4.0으로 공개할 것"이고 §D 서두는 "논문과 함께" 공개한다고 씁니다. 2026-11-22 확인 결과:

- 프로젝트 페이지 링크 줄에 "Dataset (Coming Soon!)".
- Hugging Face `airoa-org/yubi-corl2026-umi-arena`와 `-01-6k6`, `-02-2k1`, `-03-2k0`, `-04-3k9`, `-05-3k2`(총 6개)가 있고 모두 `gated: manual`, 라이선스 `other`(`airoa-limited-access-dataset-terms`)입니다. 접근 신청은 UMI Arena 등록 팀 전용이고 등록은 2026-10-09에 닫혔습니다(UMI Arena 페이지). 저장소 내용은 열지 못했습니다.
- UMI Arena 페이지: "initial ~580 hours ... on Hugging Face already, and the full 20,000 hours follows. The repository is not public ... because the episodes are recordings of real workplaces and homes." 현재 릴리스는 약 2.7 TB.

CC-BY-NC-SA와 제한 접근 약관은 서로 다른 조건입니다. 논문이 말한 개방 공개는 아직 이뤄지지 않았습니다.

### D9. 규모 수치가 다른 자료와 다름

논문은 22책상, 8,434 시간, 119 과제입니다(§4, 표 2). UMI Arena 페이지는 "forty rigs", "full 20,000 hours", "manufacturing, retail, and home tasks"를 적습니다. 같은 시스템의 이후 시점이라면 모순이 아니지만 논문의 수치는 6월 시점이고 장치 대수는 거의 2배입니다. HF 메타데이터의 저장 용량(`usedStorage`, 이전 버전이 들어갔을 수 있음)은 초기 저장소 2.68 TB, 분할 저장소 25.17, 8.26, 8.12, 13.55, 12.74 TB로 합 67.8 TB입니다. 용량 대비 시간은 확인하지 못했습니다.

### D10. 비용 주장을 BOM에서 확인할 수 없음

논문: 그리퍼 하나 USD 200 미만(Quest 3S 제외, §1, §3.1). yubi-hw의 BOM 네 개(`Glove`, `Gripper_DYNAMIXEL`, `Stationary`, `Portable`)는 부품, 수량, 재질, 판매처 링크만 있고 가격이 없습니다. 수집용 글러브 BOM에는 Quest 터치 플러스 컨트롤러 좌우(품목 16, 17), ELP 어안 카메라, ESP32-C6, AS5601, 3D 프린트 부품, 미스미 볼트와 인서트, 고무 시트 등이 있습니다. 컨트롤러가 "$200" 안인지 밖인지 논문이 정의하지 않습니다. 책상 틀은 알루미늄 프레임, 스탠딩 데스크(Flexispot E7), 노트북이 포함되어 별도 비용입니다.

### D11. 로봇 플랜지가 논문의 세 로봇과 같지 않음

논문 C.2는 UR, Franka, ELEY 각각에 어댑터 플랜지를 썼다고 씁니다(그림 19는 Franka와 ELEY). yubi-hw README의 지원 표는 **Franka, OpenArm, Unitree G1** 플랜지를 줍니다. UR은 로봇 장착형 그리퍼 BOM의 가공 부품(`UR5e_FLANGE`, `BRACKET_GRIPPER`)으로만 있고 ELEY 플랜지는 없습니다. Toyota 내부 로봇이라 이해되지만 "논문의 모든 배치를 재현 가능한 경로"라는 §D.1의 문장과는 맞지 않습니다.

### D12. 정책 학습·배치 코드 없음

논문 §6, 부록 C.3: π0.5 후학습(UR 30K 반복·배치 256·청크 16, Franka·ELEY 150K·배치 64·청크 32), 작업 균형 샘플러, 10 Hz 실행, 로봇별 IK, 조리개의 모터 명령 선형 사상. 공개물에서 이에 해당하는 학습 설정이나 IK, 실행 코드는 찾을 수 없습니다. 정책 쪽 코드는 후속 두 저장소뿐입니다.

- `yubi-real2sim`: OpenPI π0.5를 Franka FR3 + 양손 YUBI MuJoCo 시뮬에서 평가하고 3DGS 배경을 씁니다. x86-64 Linux, NVIDIA 드라이버 570.26 이상, CUDA 12.8, GPU 한 장(16 GB급), 디스크 약 26–55 GB를 요구합니다. 학습 코드가 아닙니다.
- `umi-arena-evaluation`: `adapters/openpi_pi05`가 "YUBI transforms"로 π0.5 체크포인트를 서빙하고 `replay.py`가 LeRobot v3 에피소드와 예측을 오프라인 비교합니다. 데이터와 체크포인트는 팀이 따로 구해야 합니다.

### D13. LeRobot 버전과 영상 압축

논문은 "LeRobot 형식 30 Hz"입니다(§4). rebake는 `LeRobotV21Transformer`(v2.1)를 쓰고, 영상은 AV1(CRF 30, GOP 2, 30 fps), `separate_per_primitive: true`로 하위 행동별 에피소드를 만듭니다(`yubi.yaml`). 대회 데이터는 LeRobot v3.0입니다(UMI Arena 페이지). 코드의 변환기와 공개 데이터 버전이 다릅니다. RealSense 깊이는 기록은 하지만 `robot_model/yubi.yaml`에서 주석 처리되어 LeRobot에는 RGB 중앙 영상만 들어갑니다(논문은 RGB+depth를 쓴다고 쓰지만 "전처리에만 쓴다", §3.2). 손실 압축이 정책 성능에 미치는 영향은 논문이 다루지 않았습니다.

### D14. 하위 행동 길이 상한

`recording_gate.yaml`의 `duration_limits`: 하위 행동이 90 s를 넘으면 경고(블록 시작), 120 s를 넘으면 정지. 논문에 이 상한이 없습니다. 평균 에피소드 25–40 s(`03` V5)와 그림 13–15의 에피소드(최장 약 50 s)는 이 안에 있지만, 더 긴 하위 행동은 수집 단계에서 잘렸습니다. 긴 호라이즌 과제 분포에 대한 해석 시 기억할 구속입니다.

### D15. 조리개 신호의 단위

논문 그림 2는 "Gripper Aperture (m)"이고 §4는 "fingertip jaw angle d ∈ ℝ"입니다. 코드는 AS5601 절대 각도(rad, 펌웨어 `getAngleRadians`)를 한 손가락 관절에만 읽고(URDF `finger_lower_limit` 0.00, `finger_upper_limit` 0.94 rad), 반대쪽 턱은 `mimic` 관절(배수 −1)로 대칭 처리합니다. LeRobot 구성의 조리개 특징은 `[left_right_finger_joint, right_right_finger_joint]` 두 개(손마다 하나)입니다. 논문은 두 턱이 각자의 손가락에 양보한다고 쓰는데(§3.1) URDF는 두 턱이 대칭이라고 가정합니다. 기어 연동(그림 20)이 있으면 모순이 아니지만 코드만으로는 확인할 수 없습니다. 미터 단위 조리개로 바꾸는 계산은 공개물에 없고 로봇 배치에서의 "선형 사상"(C.2)도 코드가 없습니다.

### D16. 기록 게이트 설정의 주석과 값 불일치 (코드 내부)

`recording_gate.yaml`의 `tf` 그룹은 주석과 이유 문구가 "> 0.5 s 경과 시 경고, > 1 s 경과 시 정지"인데 `max_age_sec`는 경고 1.0, 정지 5.0으로 다릅니다(`timeout_sec`는 0.5, 1.0). 의도한 임계가 두 값 중 어느 쪽인지 코드만으로 판정할 수 없습니다. 논문 수치에는 영향이 없고 수집 품질 해석에 쓰입니다.

### D17. Quest 쪽 앱은 소스가 아닌 바이너리

수집 소프트웨어의 핵심인 컨트롤러 포즈 스트리밍은 Quest 헤드셋에서 도는 "YubiQuestApp" APK입니다. yubi-sw README는 `releases.dev.airoa.io/yubi/quest-app/yubi-quest-app-v0.1.0.apk`를 받아 `adb install`하라고 안내합니다. 저장소에는 PC 쪽 브리지(`quest_bridge_node`, TCP/JSON)만 있고 Quest 앱 소스는 없습니다. 이 APK는 받지 않았습니다. 논문 부록 D.2가 말하는 "complete software stack"에서 이 부분은 닫혀 있습니다.

## 3. 라이선스와 공개 범위

| 항목 | 라이선스 | 논문 서술과 비교 |
|---|---|---|
| 하드웨어 yubi-hw | CERN-OHL-W-2.0 (`COMMERCIAL.md`가 상용 제조·판매를 허용하되 약한 상호성 조건 설명) | 일치 |
| 소프트웨어 yubi-sw, yubi-core, yubi-app, rebake | Apache-2.0 | 일치 |
| Quest 앱 APK | 라이선스 미확인, 소스 없음 | 논문은 구분하지 않음 (D17) |
| 데이터셋 | HF 접근 제한, `airoa-limited-access-dataset-terms` | 불일치 (D8) |
| yubi-real2sim | Apache-2.0, 3DGS 렌더러는 비상업이라 전체 파이프라인 상업 이용 불가 | 논문 외 |
| 논문 | CC BY-SA 4.0 | `01` §0 |

이 저장소에는 공식 코드와 데이터를 넣지 않았습니다(복제는 `.gitignore`).

## 4. 구현 가능성 판정 (Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 이 맥에서 논문의 실험(사용자 연구, 로봇 배치, 데이터 수집)은 재현할 수 없고, 정책 학습도 현실적이지 않습니다. 하드웨어 제작과 수집 스택은 이 맥 밖의 장비가 있으면 따라 만들 수 있지만, 가장 큰 자산인 8,434 시간 데이터는 지금 공개 접근이 안 됩니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 수치 재계산과 일관성 검사 | **완료** | `03`, `recompute.py` (수 초, CPU) |
| 수집용 글러브 제작 | 이 맥에서 불가 (장비 필요) | 3D 프린터, 미스미 부품, ESP32-C6, AS5601, Quest 3S와 터치 컨트롤러 2개, 어안 USB 카메라. 부품 가격 미확인 |
| ROS 2 수집 스택 구동 | 불가 (이 맥에서) | Docker Compose, Linux udev 규칙, `/dev/video` 장치 전달, Quest APK 사이드로딩이 전제. 맥의 Docker는 USB 장치를 컨테이너에 넘기지 못함 |
| rebake로 ROS bag → LeRobot 변환 | 조건부 | Rust 1.88+와 Python 필요(설치 승인 필요), 입력 rosbag와 `meta.json`을 직접 만들어야 함. 샘플 rosbag는 저장소에 없음(테스트 데이터 56 KB, 메시지 정의와 메타데이터뿐) |
| 공개 데이터로 정책 학습(π0.5 후학습) | 불가 | 제한 접근 데이터(신청 마감), π0.5 학습은 CUDA GPU와 대용량 전제, 학습 설정 비공개(D12) |
| yubi-real2sim 평가 | 불가 | CUDA 12.8, NVIDIA 570+, Linux x86-64 (D12) |
| umi-arena-evaluation의 오프라인 재생 비교 | 조건부 | 데이터 접근 승인과 설치가 선행 |
| 그림 5식 사용자 연구 반복 | 불가 | 장치 2종, 작업자 10명 이상, 너트 세트 |
| 데이터 스키마 변환 토이(쿼터니언 델타 ↔ 청크 상대 행동) | **시작 가능** | numpy/torch만으로 CPU에서 몇 초. 실데이터 없이 합성 궤적 사용 |
| 정밀도 실험의 검출력 설계 | **시작 가능** | `recompute.py`의 식으로 CPU에서 계산 |

- **노력 추정 (측정 없음):** 글러브 한 쌍 제작과 Quest 연결까지 소프트웨어 설치 포함 2–4주(부품 조달과 조립 시간 포함, 추정). rebake 변환 파이프라인 구동은 샘플 rosbag를 만드는 일이 지배적이라 며칠(추정). 정책 학습과 로봇 배치는 이 맥 밖의 자원이 필요해 추정하지 않았습니다.
- **가장 큰 위험:** 공개 데이터 접근 제한(D8)과 필터, 보정 코드 부재(D5, D7). 하드웨어를 만들어도 논문 데이터와 같은 품질 기준을 맞추려면 필터 임계값을 직접 정해야 합니다.
