# 04. 공식 코드 대조 (논문 ↔ 저장소)

공식 저장소 두 개를 얕은 복제해 **정적으로만** 읽었습니다(`repo/droid_main/`, `repo/policy_learning/`). 실행, 설치, import, 가중치와 데이터 다운로드는 하지 않았습니다. 코드는 신뢰하지 않는 외부 데이터로 취급했습니다.

- 수집 스택: `droid-dataset/droid` @ `33ae6a6` (2025-09-15)
- 정책 학습: `droid-dataset/droid_policy_learning` @ `9a29c83` (2025-04-22)
- 대조 기준: 논문 arXiv v2 (`paper/paper.txt`) 본문, 표 II, 부록 B–F. 차이 번호는 D1–Dn입니다.

## 1. 라이선스

| 대상 | 라이선스 | 근거 |
|---|---|---|
| `droid_policy_learning` | MIT | LICENSE 파일, GitHub API `mit` |
| `droid` (수집 스택) | **명시 없음** | LICENSE 파일 없음, `pyproject.toml`과 `setup.py`에 라이선스 필드 없음, API `license: null` |
| 데이터 (RLDS, 원시) | CC BY 4.0 (논문 주장) | 논문 I, III. 저장소 밖 버킷의 라이선스 파일은 확인하지 않음 |
| 보정 행렬과 언어 주석 (HF) | 확인 안 함 | 페이지를 열지 않음 |

판단: 수집 스택 코드는 이 저장소에 옮기지 않습니다. 재사용하려면 저작권자에게 라이선스를 확인해야 합니다.

## 2. 논문과 코드의 차이 (D 항목)

| 번호 | 논문 | 코드 (파일:줄) | 판정 |
|---|---|---|---|
| D1 | 학습률 스케줄 "Linear" (표 II) | 기본 `scheduler_type`은 `multistep`이고 `epoch_schedule`이 `[]`라서 스케줄이 없습니다. 학습률은 1e-4로 고정됩니다 (`robomimic/utils/torch_utils.py:142–160`, `robomimic/exps/templates/diffusion_policy.json:72`). RLDS 설정 파일은 스케줄을 바꾸지 않습니다 | **차이.** 논문 표기와 실행 설정이 다릅니다 |
| D2 | 학습 스텝 25,000 (Cook Lentils OOD는 50,000) (표 II) | `train.num_epochs = 100000` (`droid_runs_language_conditioned_rlds.py:50–52`). 에폭 길이 기본값은 100 스텝 (`robomimic/config/base_config.py:108`). 총 스텝을 고정하는 값은 없습니다 | **확인 불가.** 실행 시점에 정해지는 값입니다 |
| D3 | DROID와 과제 데이터를 배치 50/50으로 섞음 (V-b) | 샘플 가중치 기본값은 DROID 하나에 `[1]`입니다 (`droid_runs_language_conditioned_rlds.py:15–21`). 공동 학습 설정 예시는 없습니다 | **재현 시 직접 설정 필요.** 저장소만으로는 50/50을 만들 수 없습니다 |
| D4 | 매 에피소드 작업을 목록에서 무작위로 제시 (III-B) | 복합 지시 확률 10% (`droid/user_interface/gui_parameters.py:3`, 복합 문구 `text.py:51–56`). 자유 입력 작업에는 `new_task_prob = 0.1`의 가중치를 줍니다 (`gui_parameters.py:2`, `gui.py:933–943`). 고정 목록 안에서는 `task_weights`가 모두 1이라 균등합니다 | **부분 차이.** 복합 지시는 논문에 없습니다 |
| D5 | 장면 변화를 "주기적으로" 냅니다. 장면당 최대 100궤적 또는 약 20분 (III-B) | 궤적마다 20% 확률로 장면 변화 프롬프트 (`gui_parameters.py:4`, `gui.py:1221–1222`). 100궤적마다 이동 프롬프트 (`gui_parameters.py:5`, `gui.py:1013`). **20분 상한을 구현한 코드는 찾지 못했습니다** (`gui.py:1193`의 분 표시는 경과 시간 표시용) | **부분 차이.** 100궤적 규칙만 구현돼 있습니다 |
| D6 | 체커보드와 OpenCV 보정 (III-B) | ChArUco 보드를 씁니다 (`droid/calibration/calibration_utils.py:9, 16–20`) | 경미. 마커가 있는 체커보드 변형입니다 |
| D7 | 절대 말단 위치, 회전, 그리퍼 (V-b). 회전 표현은 쓰지 않음 | 회전은 6D 표현 `abs_rot_6d`입니다 (`droid_runs_language_conditioned_rlds.py:172–189`) | 경미. 논문에 표현 선택이 없어 재현 시 정해야 합니다 |
| D8 | 전체 DROID 중 실패 제외 (V-b) 대 "처음 40K 성공 궤적 중 언어 주석이 있는 것" (부록 F-B) | 성공 필터는 경로에 `/success/`가 있는지만 봅니다 (`robomimic/utils/rlds_utils.py:8–12`). 40K 선택 코드는 없습니다 | **논문 내부 불일치** (`03` K7). 코드에서는 확인할 수 없습니다 |
| D9 | 과제마다 10회 롤아웃 (V-B) | 평가 스크립트는 대화형입니다 (`scripts/evaluation/evaluate_policy.py`). 시도별 결과나 로그가 저장소에 없습니다. `droid_main/scripts/evaluation/results.ipynb`는 다른 실험(Bridge 계열 환경 이름, 로컬 경로)이라 논문 표와 대응하지 않습니다 | **검증 불가.** 논문 수치의 원자료가 공개되지 않았습니다 |
| D10 | 1280×720 스테레오 RGB (App. B) | HD720 리더가 있습니다 (`zed_camera.py:32`). 같은 파일의 기본 fps는 60이고 HD2K 리더는 15입니다 (`zed_camera.py:32, 36`) | 해상도 일치. **카메라 fps와 15 Hz 기록의 관계는 확인하지 않음** |
| D11 | 고정된 DistilBERT 언어 임베딩 (V-b) | `torch.no_grad()` 아래에서 계산합니다 (`diffusion_policy.py:163–167`). 모듈 import 시 `.to('cuda')`와 float16을 씁니다 (`diffusion_policy.py:34–35`) | 고정은 일치. **장치 하드코딩.** 이 맥에서 바로 돌지 않습니다 (§4) |

## 3. 일치한 항목 (M)

| 항목 | 논문 | 코드 (파일:줄) |
|---|---|---|
| 제어 주파수 | 15 Hz (III-A, App. B) | `droid/robot_env.py:31` `control_hz = 15` |
| 로봇 제어 | Polymetis (III-A) | `droid/franka/robot.py`에서 Polymetis 사용 |
| 입력 해상도 | 128×128 (V-b, 표 II) | `droid_runs_language_conditioned_rlds.py:199–203` |
| 자르기 | 116 × 116 (표 II) | `droid_runs_language_conditioned_rlds.py:244` |
| 색 지터 + 자르기 | 색 지터, 무작위 자르기 (V-b) | `droid_runs_language_conditioned_rlds.py:234` (`ColorRandomizer`, `CropRandomizer`), `obs_core.py:587` (밝기 0.3) |
| 시점 | 외부 카메라 두 대 (V-b) | `varied_camera_1_left`, `varied_camera_2_left` (`droid_runs_language_conditioned_rlds.py`의 `cams`) |
| 인코더 | ImageNet 사전학습 ResNet-50 (V-b, 표 II) | `ResNet50Conv`, `pretrained = True` (`droid_runs_language_conditioned_rlds.py:292–296, 311`) |
| 배치 크기 | 128 (표 II) | `droid_runs_language_conditioned_rlds.py:74` |
| 학습률 | 1e-4 (표 II) | `diffusion_policy.json` `learning_rate.initial = 0.0001` |
| 관측 2, 예측 16, 실행 8 | V-b, 표 II | `diffusion_policy.json` `horizon` |
| U-Net 채널 | [256, 512, 1024] (표 II) | `diffusion_policy.json` `unet.down_dims` |
| EMA | 0.75 (표 II) | `diffusion_policy.json` `ema.power` |
| 확산 샘플러 | DDIM (표 II) | `droid_runs_language_conditioned_rlds.py`의 `ddim.enabled` = True |
| 실패 제외 | 실패 궤적 제외 (V-b) | `robomimic/utils/rlds_utils.py:8–12` |
| 언어 주석 | 에피소드당 최대 3개 (III-B) | `policy_learning/README.md` (95%의 성공 에피소드, 75k) |

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

| 구성 요소 | 이 맥에서 | 근거와 노력 (추정, 측정 없음) |
|---|---|---|
| 수집 스택 (Franka, Zed, Quest, Polymetis) | **불가** | 하드웨어가 필요합니다 |
| 정책 학습 파이프라인 (RLDS, Octo 의존) | **조건부** | 설치 전에 승인이 필요합니다. `.to('cuda')`를 바꾸고 MPS로 옮겨야 합니다 (D11). 노력 1–2일 (추정) |
| 예제 데이터로 짧은 학습 (droid_100, 2 GB) | **조건부** | 다운로드와 설치 승인이 필요합니다. 25,000 스텝은 시간이 수일로 추정되므로 스모크 테스트에 그쳐야 합니다 |
| 논문 설정으로 전체 학습 (25k 스텝, 전체 DROID) | **불가** | GPU 전제. 이 맥에서 수 주로 추정 (측정 없음) |
| 카메라 보정 재실행 (DUSt3R, CtRNet-X) | **불가** | 논문이 8×A100에서 약 6일이라고 보고합니다 (App. G-A, G-B). 저장소에 파이프라인이 없습니다 |
| 공개 보정 행렬 검산 (재투영 계산) | **조건부** | HF 파일 다운로드 승인이 필요합니다. 계산 자체는 CPU로 가능합니다 |
| 장면 유형 분류 (GPT-4V) | **불가** | 외부 API가 필요합니다. 프롬프트는 논문 Listing C.1에만 있습니다 |
| 장면 중복 제거 (App. D) | **구현 가능 (새로 짜야 함)** | CPU와 메타데이터만 있으면 됩니다. 저장소에 구현이 없습니다 |
| 정책 실험 재현 (그림 8, 10) | **불가** | 실로봇 평가가 필요합니다. 공개 로그도 없습니다 (D9) |

**판정:** 이 맥에서는 수집과 실로봇 평가를 재현할 수 없습니다. 정책 학습 코드는 CUDA 전제를 고치면 짧은 스모크 테스트까지 가능하지만, 논문 규모의 학습은 불가입니다. 데이터 메타데이터 수준의 검산(장면 중복 제거, 에피소드 집계)은 CPU에서 할 수 있습니다.

## 5. 논문에만 있고 저장소에는 없는 것

- GPT-4V 장면 분류 프롬프트 (App. C, Listing C.1). 저장소에서 "Industrial office", "scene type" 문자열을 찾지 못했습니다.
- 장면 중복 제거 절차 (App. D). 구현이 없습니다.
- 카메라 보정 파이프라인의 DUSt3R 수정과 CtRNet-X 설정 (App. G). 저장소에서 `dust3r`, `ctrnet` 문자열을 찾지 못했습니다.
- 작업 목록 라벨링 (tasq.ai)과 크라우드소싱 지시문 작성 방식.
- 정책 실험의 시도별 결과와 로그 (D9).
- 학습 시드 (`03` K9).
- 정책 체크포인트: 그림 1 캡션과 III절은 "pre-trained model checkpoints"를 공개한다고 씁니다. 두 저장소의 README에서는 링크를 찾지 못했습니다. 프로젝트 페이지 링크는 HF `KarlP/droid` 하나뿐이어서, 체크포인트 위치는 확인하지 못했습니다.

## 6. 확인하지 못한 것

- 데이터 버킷과 HF 페이지의 실제 라이선스 파일.
- 카메라 fps 60과 15 Hz 기록의 관계 (D10).
- `droid_main`의 나머지 모듈(후처리 `postprocessing/`, 데이터 로더 `data_loading/`)의 세부 스키마. 이 문서는 정책 학습 쪽 데이터 경로를 중심으로 봤습니다.
- 정책 학습 코드를 실제로 실행했을 때의 동작. 실행하지 않았습니다.
