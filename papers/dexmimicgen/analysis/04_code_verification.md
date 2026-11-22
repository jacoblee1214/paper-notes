# 논문과 공식 코드 대조 (04)


## 0. 결론 요약

- 공개 저장소는 **시뮬레이션 환경 9개, 재생 스크립트, BC-RNN 학습 설정 생성기**입니다. 논문의 핵심인 **데이터 생성 시스템(§IV: 팔별 비동기 큐, 조정 동기화, 순서 제약, Transform·Replay)은 코드가 없습니다.** NVlabs/mimicgen의 `main`에도 양팔 전용 코드가 없습니다.
- 공개 코드로 **검증할 수 있는 것**: 과제 9개와 구성 3종(로봇 매핑), 환경의 성공 판정 로직, 학습 설정의 BC-RNN 기본 틀.
- 공개 코드로 **검증할 수 없는 것**: 생성 방법 전체, GMM과 DP 학습 설정, 평가 절차(최댓값 선택, 시도 수), 제어기(OSC와 mink IK), 텔레오퍼레이션, 실로봇, BiGym 생성, 데이터 수.
- 라이선스: 코드는 NVIDIA 라이선스(비상업 연구·평가용), 데이터셋은 카드 기준 CC BY-NC-SA 4.0입니다. README의 "CC-BY 4.0" 표기와 어긋납니다 (D7).
- 버그 수준의 발견: 재생 스크립트의 문자열 비교 오류 (D8). 스테일 TODO 주석 (D5). 정책 설정 기본값은 확인되지 않음.

## 1. 공개 범위

| 논문 구성 요소 | 공개 저장소 | 상태 |
|---|---|---|
| 과제 변형 (초기 분포 D1, D2) | 과제별 초기화 인자 (코드 전반) | 부분 (논문의 D1·D2 설정값과 대조 안 함) |
| 로봇 구성 (Panda, Panda+다지 손, GR-1) | `scripts/demo_random_action.py`의 `ENV_ROBOTS` | 있음 (`D3`에서 확인) |
| OSC·IK 제어기 | 없음 (robosuite 설정에 의존) | 없음 (D3) |
| 텔레오퍼레이션, 리타게팅 | 없음 | 없음 (외부 VisionProTeleop, OmniH2O) |
| 데이터 생성 (§IV) | **없음** | **없음 (D1)** |
| 데이터셋 | HF `MimicGen/dexmimicgen_datasets` (과제별 HDF5 9개) | 있음, 내려받지 않음 |
| BC-RNN 학습 설정 | `scripts/generate_training_config.py` (`bc_rnn_image`) | 있음 (`D2`) |
| BC-RNN-GMM, DP 학습 설정 | 없음 | 없음 (D2) |
| 평가 절차 (체크포인트 최댓값, 3 시드) | 없음 | 없음 (D2) |
| 실로봇 (GR-1, Inspire, 디지털 트윈) | 없음 | 없음 (D10) |
| BiGym 과제 | 없음 | 없음 (D10) |

## 2. 일치한 것

| 항목 | 논문 | 코드 | 근거 |
|---|---|---|---|
| 과제 수 9 | §V, Fig. 4 | 과제 클래스 9개 (Threading, ThreePieceAssembly, Transport, LiftTray, BoxCleanup, DrawerCleanup, Coffee, Pouring, CanSortRandom) | `environments/two_arm_*.py` |
| 구성 3종 | 세 구성 (§V) | Panda 두 팔 (Threading, ThreePieceAssembly, Transport), PandaDexRH·LH (LiftTray, BoxCleanup, DrawerCleanup), GR1 (Coffee, Pouring, CanSort) | `scripts/demo_random_action.py` 25–34줄의 `ENV_ROBOTS` |
| 다지 손 과제에 다지 손 로봇 | §V | PandaDexRH·LH, GR1 | 같은 곳 |
| 성공 판정은 최종 상태 기준 | 부록 IX | Pouring: 공 접촉, xy 거리, 그릇 수직, 패드 접촉. Box: 상자와 뚜껑 위치 정렬. Drawer: 객체가 서랍 안에 똑바로 있고 서랍이 닫힘 | `_check_success` 함수들 |
| BC-RNN 기본 틀 | §VI-A (robomimic [1]) | robomimic `bc.json` 템플릿, `bc_rnn_image`, `seq_length=10`, 이미지 키 (agentview, 손목 카메라) | `scripts/generate_training_config.py` 39, 279–300줄 |
| 원천 시연 재생 | §IV | `playback_datasets.py`로 HDF5 재생 | `scripts/playback_datasets.py` |

## 3. 차이와 확인 항목 (D1–D11)

### D1. 데이터 생성 파이프라인이 공개되지 않음 (핵심 공백)

- 논문 §IV의 팔별 행동 큐, 조정 하위과제의 동기화(남은 스텝 수 맞추기), Transform·Replay 두 방식, 순서 제약은 공개 저장소에 없습니다. 전체 저장소(에셋 제외)에서 `interpol`, `subtask`, `sync`, `ordering`, `coordinat`, `Replay`, `action queue`, `queue`는 한 번도 나오지 않습니다. `sequential`과 `Transform`의 검색 결과는 `SequentialCompositeSampler`(배치 샘플러)와 `transform_utils`(자세 변환 유틸)이며 데이터 생성과 무관합니다.
- NVlabs/mimicgen의 기본 브랜치(`main`)와 브랜치 목록(`experimental/robocasa`, `main`)에서도 양팔 전용 코드를 찾지 못했습니다.
- **결과**: 논문의 핵심 방법(§IV)은 코드로 확인할 수 없습니다. 재현하려면 논문 설명으로 다시 구현해야 하고, 그러면 논문과 같은 수치가 나온다는 보장이 없습니다. 보간 시작점, 조정 구간 종료 맞춤, 충돌 처리 같은 구현 세부가 논문에 없기 때문입니다.

### D2. 학습과 평가 설정이 BC-RNN 하나뿐임

- 생성기는 `bc_rnn_image` 하나를 만듭니다(`generate_training_config.py` 279–300줄). BC-RNN-GMM 설정과 Diffusion Policy 설정은 없습니다. GMM을 켜는 인자는 이 스크립트에서 보이지 않습니다.
- 에폭 수, 평가 롤아웃 수, 체크포인트 주기, 시드 목록도 이 스크립트에 없습니다. robomimic 템플릿의 기본값에 맡겨져 있습니다.
- 논문의 평가 절차(3 시드, 체크포인트 최댓값, §VI-A)는 공개 설정에서 재현할 수 없습니다.

### D3. 제어기와 IK는 코드에 없음

- 논문: Panda는 OSC, GR-1은 mink 기반 IK 제어기를 직접 구현했다고 함(§V). 공개 저장소에는 IK나 mink 코드가 없습니다(`mink`는 README와 requirements에 없음).
- 제어기 설정은 robosuite의 composite controller 설정을 불러와 쓰는 방식입니다(`demo_random_action.py` 61줄). GR1 로봇 모델(`GR1FixedLowerBody`, `GR1ArmsOnly`)도 외부 패키지에서 옵니다.
- 결과: 논문의 "직접 구현한 IK 제어기"는 확인할 수 없고, 재현하려면 외부 구현이 필요합니다.

### D4. 과제 이름이 논문 본문, 표, 코드에서 다름

| 본문 (§V) | 표 (Table I–III) | 코드 |
|---|---|---|
| Box Packing | Box Cleanup | `TwoArmBoxCleanup` |
| Tray Lift | Tray Lift | `TwoArmLiftTray` |
| Can Sorting | Can Sorting | `TwoArmCanSortRandom` (`TwoArmCanSortRed`, `TwoArmCanSortBlue` 변형도 있음) |

본문의 "Box Packing"은 표와 코드의 "Box Cleanup"과 같은 과제인 것으로 보입니다. 이름만 다르고 실험 정의는 같은지는 확인하지 못했습니다.

### D5. 성공 판정 주석과 로직

- `two_arm_pouring.py` 368줄: "TODO: implement success check by checking sphere contained within bowl". 그러나 바로 아래에 접촉, xy 거리, 그릇 수직 조건이 구현돼 있습니다. **주석이 스테일입니다.**
- `two_arm_box_cleanup.py` 274줄과 `two_arm_can_sort.py` 254줄도 같은 종류의 TODO입니다. 두 함수 모두 실제 판정 로직이 이어집니다. Can Sort는 `is_red`가 None이면 항상 False를 돌려줍니다.
- `two_arm_threading.py` 205–227줄: 성공은 **바늘 기하 중심과 고리 중심의 3차원 거리가 고리 반지름보다 작음**으로 정의합니다. 바늘이 고리 구멍을 실제로 통과했는지(축 방향, 가림)는 보지 않습니다. 논문이 Threading 실패 원인을 가림으로 든 것(부록 X)과 맞물려, 판정이 느슨하면 성공률이 부풀 수 있습니다. 재생 검사로 확인해야 하는 항목입니다.

### D6. 의존성 버전이 고정되지 않음

- `requirements.txt`는 h5py, imageio, numpy==1.23.3, setuptools, termcolor, huggingface_hub, numba를 고정하지만 **robosuite와 mujoco는 없습니다.**
- README는 "최신 robosuite"를 설치하라고 씁니다. 논문의 실험은 어떤 robosuite 버전인지 명시하지 않습니다(논문 [70]은 arXiv 2009.12293).
- 결과: robosuite 버전이 바뀌면 물리와 렌더링이 바뀌고, 수치가 달라질 수 있습니다. 재현 전에 버전을 맞춰야 합니다.

### D7. 데이터셋 라이선스가 README와 카드에서 다름

- GitHub README: "datasets are released under CC-BY 4.0".
- HF 데이터셋 카드(메타데이터와 README 본문): `license: cc-by-nc-sa-4.0`.
- 저장소 `LICENSE`(NVIDIA Source Code License)는 코드에 적용됩니다. 데이터셋 `LICENSE` 파일(4 KB)은 내려받지 않아 내용을 확인하지 못했습니다.
- 보수적 판단: NC-SA로 취급하고, 상업적 사용과 변경 재배포를 피합니다.

### D8. 재생 스크립트의 문자열 비교 오류

- `scripts/playback_datasets.py` 369줄: `if "env_name" == "TwoArnCanSortRandom" and args.use_current_model:`
- 왼쪽이 변수 값이 아니라 **문자열 리터럴 `"env_name"`**입니다. 오른쪽은 `TwoArn`(철자 오류)입니다. 조건은 항상 거짓이며, 본래 의도(Can Sort에서 `use_current_model`일 때 경고)는 실행되지 않습니다. 기능에 큰 영향은 없지만 실제 버그입니다.

### D9. 데이터셋 크기와 과제별 개수

- HF 저장소: `generated/` 아래 과제별 HDF5 9개(Box 5.2 GB, Can Sort 4.4 GB, Coffee 5.7 GB, Drawer 5.1 GB, Lift Tray 9.8 GB, Pouring 6.3 GB, Threading 3.9 GB, Piece Assembly 4.3 GB, Transport 15.2 GB), GIF 1개. 합계 약 **59.9 GB**(메타데이터 기준).
- 과제별 데모 수(1000개 기본)와 총 21K는 내려받지 않아 확인하지 못했습니다. 다운로드 없이는 K4(`03`)의 3K 차이를 해결할 수 없습니다.

### D10. 논문에서 공개 코드로 확인할 수 없는 부분

| 항목 | 논문 | 공개 상태 |
|---|---|---|
| 실로봇 (GR-1, Inspire, 디지털 트윈, GroundingDINO, Ditto) | §VI-D | 없음. 가중치와 하드웨어 필요 |
| BiGym 3과제 생성 | §VI-B | 없음 (BiGym 외부 저장소) |
| Vision Pro 텔레오퍼레이션, OmniH2O 리타게팅 | §V | 없음 (외부) |
| 원천 시연 수집 | §V | 없음 (사람 작업) |

### D11. 코드에만 있고 논문에 없는 변형

- `TwoArmCoffeeLidClosed`(뚜껑이 닫힌 상태에서 시작하는 어려운 Coffee, `two_arm_coffee.py` 540줄). `TwoArmCanSortRed`, `TwoArmCanSortBlue`(색 고정 변형, `two_arm_can_sort.py` 328–335줄). 논문의 9개 과제 정의에는 없습니다. 논문 표와 대조하면 사용된 과제인지 확인이 필요합니다.

## 4. 구현 가능성 (이 맥 기준)

**환경: Apple M4 Pro, 약 24 GiB, MPS 있음, CUDA 없음.** 이 분석에서는 아무것도 설치하지 않았습니다. 아래는 코드와 문서를 읽은 결과에 따른 판단입니다.

| 작업 | 자원 | 이 맥에서 | 판정 |
|---|---|---|---|
| 공개 환경 설치와 한 과제 재생 | robosuite(버전 고정 필요), mujoco, CPU 렌더링 | CPU로 가능할 가능성이 큼. macOS 렌더링 경로(GLFW, CGL)는 설치 후 확인 필요 | **조건부 가능** (설치 승인 필요) |
| 데이터셋 한 과제 다운로드와 재생 | 약 4–15 GB 과제 파일 | 디스크와 네트워크로 가능. 재생은 CPU | **조건부 가능** (다운로드 승인 필요) |
| 공개 코드로 생성 재현 | 생성 코드 없음 | 불가 | **불가** (D1) |
| 논문 설명으로 생성기 재구현 후 한 과제 생성 | 환경 + 원천 시연 + 자체 구현 | 가능 (CPU 생성, 시도당 시뮬레이션 시간은 측정 전) | **조건부 가능** (재구현 노력 큼) |
| BC-RNN·DP 학습 (9과제, 이미지, 3 시드) | GPU 수 시간~수일 (추정) | 제한적 (MPS로 작은 모델만) | **불가** (GPU 전제, 사용자 정책상 재현 안 함) |
| 실로봇 재현 | GR-1, Inspire 손, 카메라, 디지털 트윈 | 불가 | **불가** |
| BiGym 생성 | BiGym 설치, 생성 코드 없음 | 불가 | **불가** |

**노력 추정 (추정치, 측정 전).** 환경 설치와 재생 확인 0.5–1일. 생성기 재구현(팔별 큐, 동기화, 순서 제약, 두 변환 방식) 1–2주. 생성 한 번의 시간은 측정 전이라 추정하지 않습니다. 학습과 실로봇은 맥에서 할 수 없습니다.

**판정 요약.** 논문 방법을 맥에서 **구현**할 수는 있지만, 공개된 코드로는 **재현**할 수 없습니다. 생성기를 다시 짜는 것이 가장 큰 일이고, 그 결과를 논문 표와 비교하려면 GPU 학습이 필요합니다. 따라서 "구현할 수 있는지"의 답은 "환경과 생성기는 맥에서 조건부로 가능하고, 학습과 실로봇은 불가"입니다.

## 5. 저자에게 물을 질문

1. **v2에서 표 I의 BC-RNN 열이 바뀐 이유.** 9개 셀 중 8개의 평균과 9개 모두의 ±가 달라졌습니다. 표 II·III과 본문은 v1 값을 씁니다(`03` K1–K3).
2. **"±"의 정의**(SD인지 SEM인지), **시도 수**, **체크포인트 선택 규칙**(최댓값 방식의 세부).
3. **순서 제약 실험의 기본 조건.** 본문의 Drawer 50.7, Pouring 88.7이 표 I의 기본 값(80.0, 79.3)과 같은 조건인지.
4. **21K 데모의 구성.** 블록으로 설명되는 18K를 넘는 약 3K를 어떤 실험에서 만들었는지.
5. **생성 파이프라인 코드 공개 계획.** `dexmimicgen` 브랜치나 별도 저장소가 있는지.
6. **데이터셋 라이선스.** 카드는 CC BY-NC-SA 4.0, README는 CC BY 4.0입니다.
7. **Threading 성공 판정.** 3차원 중심 거리만으로 충분한지, 축과 구멍 통과를 따로 봤는지.
8. **GMM 설정.** 공개 생성기에 없습니다. 어떤 robomimic 설정으로 BC-RNN-GMM을 학습했는지.
9. **robosuite와 mujoco 버전.** 실험에 쓴 버전.

## 6. 재현 관련 메모

- 과제 파일 중 `two_arm_threading.py` 179줄에는 "swap this in to be consistent with D0 in MimicGen"이라는 TODO가 있습니다. 초기 분포 D0 정의가 MimicGen과 다를 수 있습니다. 확인하지 않았습니다.
