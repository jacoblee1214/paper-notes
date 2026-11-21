# 04 공식 코드 정적 대조와 구현 가능성

공식 저장소 `umersiddique94/ReGround`를 얕게 클론했습니다(커밋 16a98486f560, 2026-11-15). 파일을 읽기만 했고, 실행, import, 설치, 가중치 다운로드는 하지 않았습니다. 가져온 내용은 신뢰하지 않은 자료로 취급했습니다. 대조 항목은 D1–D20입니다.

## 1. 저장소 개요

- 파이썬 3.12, 약 3,500줄(`main.py` 116줄, `reground/` 패키지, 테스트 2개). `reground/method/`에 추적 MPC(`mpc.py`), 제어기 기반(`controller.py`), 앵커(`anchors.py`, `build.py`), 지연 추정(`delay.py`), 모니터(`monitor.py`), 의도 읽기(`reference.py`)가 있습니다.
- 설정은 `configs/paper.json`(셀별 선택값)과 `configs/model_select/*.json`(모델 구조)입니다. `configs/speed_envelope.json`에 π0.5 속도 상한 0.2757이 있습니다.
- `README.md`는 논문 재현 순서(바빙, 수집, 맞춤, 평가, 요약)를 적었습니다.
- 학습과 추론은 CUDA를 전제합니다(README는 "CUDA-capable NVIDIA GPU", `torch.cuda.is_available()` 검사). 저장소 안에 실행 결과(런 기록, 모니터 보정 파일, 에피소드 trace)는 없습니다.

## 2. 논문 ↔ 코드 대조 (D1–D20)

| 번호 | 논문 | 코드 (`configs/paper.json`, `reground/`) | 차이와 영향 |
|---|---|---|---|
| D1 | 신뢰 영역의 앵커 `ū`는 "VLA 청크를 d̂만큼 밀고 M으로 사상한 것, 또는 한 스텝 IDM"(§3). 둘 중 무엇을 쓰는지 셀별로 밝히지 않음 | 셀마다 다름. lat/5, lat/8: `vla_shift`(청크를 d̂만큼 밀기). combo/1: `lin_shift`(M 적용). rot_z/90, gain/3: `idm`. SmolVLA rot_z/30, gain/2는 키가 없어 기본값 `idm` (`mpc.py` DEFAULTS) | 셀별 앵커 선택이 표 1 결과에 영향을 줍니다. 선택 절차(검증 분할 기준인지)는 저장소에 없습니다. 논문은 "hyperparameters are selected on validation states 0–14"라고만 적음 |
| D2 | "H=1이 더 긴 horizon과 성공률이 같다"(§3). 수용 horizon H 값은 본문에 없음 | `paper.json`: H=1(lat/5, lat/8, rot_z/90), H=2(combo/1, gain/3, gain/2), H=8(rot_z/30). λ=100(lat/5, rot_z/90, gain/3), λ=10(나머지) | 주장 "H=1이 충분하다"는 π0.5 다섯 셀 중 두 셀(combo, gain/3)에서 H=2를 씀. 설정 근거가 저장소에 없음 |
| D3 | 주 실험 "a minute of babbling"(1k 전이 ≈ 50 s), 결론에서도 "a minute" | `build.py` DEFAULT_BUDGET = 10,000, DEFAULT_EPS = 0.6. 적응 데이터 = 바빙 6,000 + VLA 구동 4,000 (`adaptation_data`). combo/1, gain/2는 eps=1로 바빙 10,000 | 주 실험의 데이터는 약 5–8분 로봇 시간입니다. "1분"은 그림 3c(250 ms, 1k 전이)의 한 점입니다. 본문의 혼합 비율(VLA 구동 40%)도 밝히지 않음 |
| D4 | 학습 몸 모델은 "nominal babbling으로 한 번 맞춤"(§3). 전이 수는 없음 | `cache.py`: NOM_BUDGET = 100,000, NOM_EPS = 0.6 → 바빙 60,000 + VLA 구동 40,000 | 명목 모델은 약 83분 로봇 시간(20 Hz)입니다. 논문의 "1분" 강조에 포함되지 않음 |
| D5 | Oracle은 "analytic inverse"(§4 표 1) | 코드의 해석적 역변환 `oracle_inv`가 이에 해당하는 것으로 보임. 복합 시프트의 `Combo.inverse`는 회전만 역변환합니다(`shifts.py` 187–188행, "the stiffness part has no analytic inverse; left as is"). `chunk_shift`로 지연만 다시 맞춥니다 | **Oracle이 복합 시프트에서 부분적입니다.** 강성 ×0.33을 보정하지 않으므로 "ReGround 0.97 > Oracle 0.92"는 완전한 역변환과의 비교가 아닙니다. 논문 표 1의 Oracle 열 복합 값 0.92가 이 `oracle_inv`에서 나온 것이라면 부분 역변환의 결과입니다(논문이 이를 확인해 주지 않음) |
| D6 | "0.91–1.00 회복"(§4 질문 B). 회복의 정의 없음 | `stats.py` `recovery`: (방법 − 시프트 기준) / (명목 기준 − 시프트 기준). 명목 기준은 검증 또는 시험의 명목 성공률 | 논문 정의와 일치하는 것으로 보이나 논문에 적지 않음. 우리 재계산에서 명목 0.99를 쓰면 하한이 0.89(400 ms)입니다 (`03` §3) |
| D7 | "task-seed IQM with stratified bootstrap intervals"(§4). 표와 그림에는 구간 없음 | `stats.py`: 런 = (과제, 적응 시드)의 평균, IQM은 런 단위(trim 25%), 계층 부트스트랩은 과제 안에서 재표집 | 논문은 방법을 밝혔지만 구간을 싣지 않음. 저장소의 `summarize`가 구간을 내지만 그 출력은 저장소에 없음 (`03` V5) |
| D8 | "임계값은 명목 검증 에피소드로 정함"(§3 Shift monitor) | `monitor.py`: 잔차 `e_t`는 Σ로 정규화한 norm/√22. μ = 명목 잔차 평균, κ = 0.5, 임계값 = 1.1 × 명목 검증 최대 CUSUM. 첫 전이는 워밍업으로 무시 | 세부 값은 코드에 있지만 논문에 없음. 보정 산출물 `configs/monitor/`는 저장소에 없어 **오경보 2.1%와 탐지 지연을 정적으로 검증할 수 없음** (`03` V7) |
| D9 | "task-agnostic motor babbling으로 d̂와 M을 식별"(§3). 혼합 여부는 "optionally mixing in VLA-driven transitions" | `delay.py`: 상관 곡선 정렬로 d̂ 추정. `anchors.py`: 지연 보정 회귀로 M 추정. 입력은 기본적으로 바빙과 VLA 구동 전이의 혼합 (`adapt_eps` 0.6) | 논문은 "바빙만 쓴다"는 인상을 주지만, 기본 설정은 40%가 VLA 구동 전이입니다. "task-agnostic"은 과제 라벨이 없다는 뜻으로는 맞습니다. 지연 추정 방법(상관 곡선)은 논문에 없음 |
| D10 | 불안전 기준에 "speed-envelope violation" 포함(§4) | `configs/speed_envelope.json`: π0.5 = 0.2757. 단위와 유도 방법 없음 (`jobs.py` `speed_limit`) | 임계값이 불안전률을 직접 바꿉니다. 논문은 값도 유도 방법도 적지 않음 |
| D11 | 안전 지표: 비대상 접촉, 물체 변위, 낙하 (§4) | `safety.py`: 변위 > 3 cm, 낙하 = 높이 < 테이블 − 2 cm 또는 기울기 > 45°, 접촉 = 로봇 기하가 비대상 물체 기하와 접촉. 테이블 접촉은 불안전에 넣지 않음 (`rollout.py` 212행) | 수치 임계값은 논문에 없음. 테이블 접촉을 뺀 점은 논문 문장과 맞음 |
| D12 | "SmolVLA (not retuned)"(표 1) | SmolVLA 셀도 `paper.json`에 따로 있음: rot_z/30은 H=8이고 앵커 키 없음, gain/2는 eps=1. π0.5 셀과 설정이 다름 | "재조정 없음"의 뜻이 불분명. 그렇다고 해서 SmolVLA 셀이 π0.5 설정을 그대로 쓴 것은 아님 |
| D13 | 배치 CEM (§3) | `mpc.py`: N=256 후보, I=4 반복, 엘리트 비율 0.1, 초기 표준편차 0.3, 앙상블 5, 이력 L=8, 네트워크 256 폭 × 4 깊이 (`model_select/lat.json`) | 구현 세부가 논문에 없음. 재현하려면 필요 |
| D14 | 공개 코드 | 라이선스 파일 없음 (GitHub API license null). `pyproject.toml`에도 라이선스 필드 없음 | **재사용 권리가 없습니다.** 이 분석은 읽기만 했습니다(`04` §3) |
| D15 | 논문 PDF (arXiv v1) | 저장소의 `paper/Right_Goal_Wrong_Body.pdf`는 사전 빌드 (`01` §0). 수치는 같고 그림 1 도식과 표 1의 † 표시가 다름 | 논문 판본을 명시해야 함 |
| D16 | "20 Hz" (§2, §4) | `mpc.py` CONTROL_DT = 0.05 s. lat/5 = 5 스텝 = 250 ms, lat/8 = 400 ms | 일치 |
| D17 | 기준 경로 `x̂`는 청크를 명목 모델로 한 번 굴린 것 (§3) | `reference.py`: 기본 `abs`(한 번 굴림). `rel`, `path` 모드도 있음. `paper.json`은 모두 `abs` | 일치. 논문에 없는 모드는 실험에 쓰이지 않음 |
| D18 | 복합 시프트 정의 없음 (표 1, 그림 2) | `shifts.py`: `Combo` = 회전 45° + 지연 2 스텝(100 ms) + 강성 ×0.33. 프로젝트 페이지도 같은 정의 | **논문 본문에 정의가 없음.** 프로젝트 페이지와 코드는 일치 |
| D19 | 시도 수: 시드 3–5 (§4) | README `reproduce`: π0.5 시드 5, SmolVLA 시드 3, 시험 초기상태 15–49 (`jobs.py` SPLITS: 검증 0–14, 시험 15–49) | 일치. 과제당 시도 수는 시드 수에 따라 105 또는 175 (`03` §2) |
| D20 | 하드웨어 언급 없음 | README: Python 3.12, CUDA GPU, `torch==2.11.0`, `lerobot[libero,smolvla,pi]==0.6.1`, π0.5는 게이트된 PaliGemma 라이선스 필요 | 재현 조건 (§4) |

## 3. 라이선스와 재사용

- 공식 저장소에 `LICENSE` 파일이 없고 GitHub API의 license가 null입니다. 저작자 허락 없이 코드를 옮기거나 재배포할 수 없습니다.
- 프로젝트 페이지 템플릿(Nerfies)은 CC BY-SA 4.0입니다. 이것은 페이지 디자인의 라이선스이며 코드나 논문의 라이선스가 아닙니다.
- 논문 라이선스는 arXiv non-exclusive입니다(`01` §0). 공개본에는 논문 그림을 넣지 않습니다.
- 이 분석은 코드를 공개본에 넣지 않았고, 실행하지도 않았습니다.

## 4. 구현 가능성 (이 맥)

**판정: 이 맥에서 논문 실험을 재현하는 것은 불가능합니다.** 학습과 평가가 CUDA를 전제하고(README와 코드의 `cuda:` 장치 지정), 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)은 그 조건을 채우지 못합니다. π0.5는 3B 규모이고 게이트 라이선스가 필요하며, LIBERO와 MuJoCo 환경 설정도 필요합니다. 이 분석 범위에서 GPU가 필요한 재현은 하지 않았습니다.

| 작업 | 판정 | 이유 |
|---|---|---|
| 표 1 재현 (π0.5, 5셀 × 5시드 × 8과제 × 35 초기상태, 공식 코드) | **불가** | CUDA 필수, 3B 모델, 수천 에피소드 |
| 명목 모델 (100k 전이, 약 83분 로봇 시간) | **조건부** | 바빙 시뮬레이션은 렌더링이 없어 CPU로 가능할 수 있으나 LIBERO 설치와 승인이 필요 |
| SmolVLA 행 | **불가** | 같은 이유 |
| 지연 추정, CUSUM 모니터, 추적 MPC의 CPU 토이 (numpy) | **가능** | 직접 짠 numpy 토이는 허용 범위 |
| 공식 코드 실행, 새 가상환경 설치 | **보류** | 공식 코드 실행과 새 가상환경 설치는 사용자 승인 후 진행 |

**노력 추정 (측정 전, 추정치).** 토이 구현과 검증은 1–2일입니다. 공식 코드 재현은 CUDA 기기에서 설치와 LIBERO, π0.5 가중치 포함 1–2주이고, 시뮬레이션 시간은 별도입니다.

**구현 가능성에 대한 한 줄.** 방법의 핵심 구성(의도 읽기, 신뢰 영역이 있는 추적, 모니터)은 수식으로 명확하고 코드도 짧아 옮기기 쉽습니다. 반면 결과를 판단하려면 셀별 앵커와 horizon 선택(D1, D2), 복합 Oracle(D5), 데이터 규모(D3, D4)를 같은 조건으로 맞춰야 하며, 이것은 논문만으로는 알 수 없습니다.

## 5. 정적으로 확인한 것과 못 한 것

- 확인: 설정 값과 논문 수치의 대응, 불안전 정의, IQM과 회복 공식, 모니터 수식, 복합 시프트 정의, 셀과 지연의 단위.
- 못 함: 실행 결과 대조(저장소에 로그나 에피소드 trace가 없음), 모니터 보정 값 확인, 학습된 모델의 품질 확인, 테스트 파일(`test_method.py`, `test_features.py`)은 읽지도 실행하지도 않음.
- 저장소에 `__pycache__` 파일이 같이 올라와 있습니다(cpython-312). 실행하지 않았고 영향은 없습니다.
