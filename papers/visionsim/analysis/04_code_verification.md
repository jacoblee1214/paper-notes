# 04. 논문과 공식 코드 대조: VisionSIM (arXiv:2611.08368)

## 1. 대조 대상과 방법

- **저장소:** `WISION-Lab/visionsim`, 얕은 복제 `repo/` (gitignore). 커밋 `eda2b4bb3f578562cefa7e449f6845375d58e41f` (2026-11-21, "Merge pull request #41 ... Add transient heat simulation for moving geometry"). 패키지 버전 0.2.0 (`visionsim/__init__.py`).
- **시점 차이:** 논문 v1은 2026-11-05, 코드는 그보다 16일 뒤의 main입니다. 논문 이후에 합쳐진 변경(예: 열 과도 모드, PR #41, 2026-11-21 병합)이 섞여 있습니다. 반대로 논문이 쓴 기능이 아직 main에 없는 경우는 열린 PR(#39)로 확인했습니다.
- **방법:** 정적 읽기만 했습니다. 코드를 실행, import, 설치하지 않았고 의존성도 받지 않았습니다. 복제한 내용은 신뢰하지 않는 데이터로 다뤘습니다. 문서(`docs/source`), `docs/source/roadmap.rst`, 예제, 테스트 목록, 열린 이슈와 PR 설명을 함께 읽었습니다.
- **판정:** 일치(논문과 코드가 같다), 차이(코드가 논문과 다르다), 없음(코드에 해당 기능이 없다), 부분(일부만 있다).

## 2. 논문 ↔ 코드 차이 (D1–D19)

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 능동 SPAD (dToF LiDAR) 모델 | 중심 모델. 식 1, S10–S11, 비동기 EWH, EDH, FOV 마스크, 미분 가능 PyTorch (Sec. 4.1, S.5.2) | `visionsim/emulate/`에 dToF 모듈이 없음(`dvs`, `itof`, `imu.py`, `rgb.py`, `spc.py`뿐). `docs/source/sections/sensors/spcs/active_spc.rst:33`의 경고: "Current version of VisionSim uses SPCSim library ... Future versions ... will integrate active SPC modules". 예제 `examples/sensors/ewh_spc.py`, `edh_spc.py`는 외부 `SPCSim`을 import하고 저자 개인 홈 디렉터리 경로가 하드코딩됨. 열린 PR #39(`active-spc`, 49개 파일)가 "단일 픽셀, 단일 프레임 반복 방식만, EDH 미검증"이라 설명 | 없음 (main에는) |
| D2 | 데드타임, 파일업, 거리 걸음, 글레어(PSF), IRF 학습, 상관 손실(식 S21) | Sec. 4.1, 5, S.5.2.1, S.6.2–S.6.4 | `dead.?time`, `pileup`, `range.?walk`, `glare`, `bloom`, `IRF`, `retroreflect`, `ADS6311`을 `.py`, `.rst`, `.md`, `.ipynb`에서 검색해도 `docs/source/tutorials/dvs.rst:71`의 무관한 한 줄뿐. 디지털 트윈(Adaps ADS6311, 49개 PSF, 1000 epoch 학습)을 만드는 코드와 데이터가 없음 | 없음 |
| D3 | iToF 잡음, 포화, 양자화, 탭 간 움직임 | 식 2의 Q(·), η_i, 모션은 "시간 오프셋 보간 프레임 통합"(Sec. 4.2, 그림 2, S.5.3) | `docs/source/sections/sensors/itof.rst:52`: "The emulator produces noiseless measurements. Shot noise ..., read noise, quantization, and motion or blur effects are not simulated". `emulate/itof/simulation.py` `simulate_measurements`는 `(T_exp/T_period)(β C_k(d) + P_amb κ_k α)`만 계산. 주변광과 위상 접힘(`d % max_depth`)은 구현. `roadmap.rst:15`가 "Add noise modeling to iToF emulation" 계획. CLI 기본 변조 주파수 120 MHz (논문 그림은 20, 35, 40 MHz) | 부분 (이상형만) |
| D4 | 열 복사 모델 | L_out = ε U(T) + (1−ε) L_in, U는 카메라 분광 응답으로 적분한 흑체, Sakuma-Hattori 근사 (식 S15, S.5.4.1) | `simulate/heatsim/thermal_shader.py`는 Cycles 노드로 ε σ T⁴ x `radiance_scale` + (1−ε) 확산 반사. `radiance_scale`은 "magnitude knob"이라고 주석. 대역 적분이나 Sakuma-Hattori 없음. 선택적 글로시 로브(`specular_fraction`)가 추가되어 있고 논문은 람베르트만 | 차이 |
| D5 | 마이크로볼로미터 열 관성 | 1차 IIR y[n] = a y[n−1] + (1−a) x[n], a = e^{−Δt/τ}, τ = 5–15 ms (식 S16–S20, S.5.4.2) | `bolometer`, `inertia`(센서 의미), `NETD`, `fixed.?pattern`을 코드에서 찾을 수 없음. 열 문서의 "thermal inertia"는 재료의 밀도와 비열을 뜻함. `roadmap.rst:12`가 "Add thermal camera modeling" 계획 | 없음 |
| D6 | 열 솔버 세부 | 대류 항 h_c를 노출(식 S4, S5, S7, 표 2의 "Thermal parameters"). 주변 온도 T_amb 사용자 지정. 접선 행렬을 한 번 prefactor하고 matrix-free CG를 warm start (S.2.5.2) | `solver.py:213` `h = 0.0` 하드코딩이라 대류 항은 항상 0. 주변 온도는 `physics.py`의 고정 295 K (문서 "fixed 295 K ambient reference"). prefactor 없이 Jacobi 전처리 CG(`pcg_solve`)를 매 스텝 실행, float32 기본, Tikhonov 정규화 1e-8 (`solver.py:116, 178`). 기본 모드는 `STATIC`(`config.py:161`), `TRANSIENT`는 PR #41(2026-11-21 병합, 논문 이후)로 들어옴. `roadmap.rst:13`의 "only steady state has shipped so far"는 낡은 문장 | 차이 |
| D7 | 재료 열확산도 | 그림 3: 세라믹 2, 강철 18.8, 알루미늄 97 mm²/s (Sec. 4.3). 유리 0.5 (S.2.5.2) | `heatsim/materials.py` 프리셋: 알루미늄 97.0, 강철 15.4, 세라믹 0.6, 도자기 0.7, 유리 0.43 mm²/s. 알루미늄/유리 = 225배. 세라믹은 "검증된 출처 없는 추정"이라고 주석 | 차이 (그림 3 값 ≠ 프리셋) |
| D8 | 이벤트 모델 | 임계값이 사건마다 확률적으로 뽑힘(L1552). 불응기 모델 (Table 2의 "refractory drop-out") | v2e 포팅. 임계값은 `_init()`에서 픽셀당 한 번 `np.random.normal(pos_thres, sigma_thres)`로 뽑는 고정 불일치(`emulator.py:547`). 불응기 기본값 0(꺼짐, `emulate.py:116`, 클래스도 0.0). 광수용체 컷오프 CLI 기본 200 Hz, 클래스 기본 30 Hz가 서로 다름 | 차이 |
| D9 | RGB 모델과 ISP | 식 S9, ISO 이득, 양자화, "baseline ISP 구현 (see methods)" (S.3) | `emulate/rgb.py`: 프레임 합 → 푸아송 → 가우시안 읽기 잡음(`readout_std x burst/255`) → fwc 정규화 → 양자화 → sRGB. ISP(디모자이킹, 노이즈 제거, 톤 매핑)는 없음. fwc는 전자 단위가 아닌 상대 단위(문서 note). `roadmap.rst:11`이 ISP 계획. 열린 PR #23(병합 안 됨)이 모자이킹과 디노이즈 추가 제안 | 부분 |
| D10 | 보간 배율 | 160배(SPAD), 40배(이벤트), 16배 (Sec. 7, S.2.7) | `cli/interpolate.py:17, 76`: 배율은 2의 거듭제곱이어야 하고 RIFE만 지원. 160과 40은 불가. (참조: 그 밖에 코드는 장면 렌더 fps를 `keyframe_multiplier`로 올릴 수 있어 렌더로 보완할 수도 있음) | 차이 |
| D11 | 키프레임 배수 탐색 (식 S8) | α = 0.95 감쇠, 구간 클램프, 95퍼센타일 광류, K0 = 5, 목표 1 px | `cli/blender.py:411 optimize_rate`: 기본 `percentile=95`, `target=1.0`, `init_k=5.0`, `scale_decay=0.95`, 갱신 `k += k (factor − 1) scale_decay**i`(`:607`), 비율 클램프 [1/15, 15], tolerance 0.1 px, 정체 감시 추가 | 일치 (보강됨) |
| D12 | 데이터셋 생성 CLI | `scripts/mkdataset.py create-datasets --render-config.width=... --render-config.depths --render-config.normals ...` (S.7) | 스크립트는 `config: RenderConfig`라서 플래그가 `--config.width`, `--config.include-depths`. 저장소 튜토리얼(`large-dataset.rst`)은 `--config.*`로 적음. 논문 예시는 그대로는 실행되지 않음 | 차이 |
| D13 | 데이터셋과 자산 배포 | 100개 리깅 장면, 체크섬, 재생성 스크립트, 자산 CC BY 4.0 (Sec. 7, S.7) | 저장소 장면은 `tests/test_files/scenes/cube.blend` 하나. 체크섬, 장면 목록, 센서 설정 파일 없음. 프로젝트 페이지는 "Coming soon. The prerelease is available in the meantime." HTCondor 제출 스크립트도 없음(Docker 이미지와 CI는 `.github/workflows/`에 있음) | 없음 |
| D14 | IMU | Crassidis 랜덤 워크, Leutenegger 표 I 값, 250 Hz 데이터셋 (S.3) | `emulate/imu.py`: 같은 모델과 같은 값(`std_acc=8e-3`, `std_gyro=1.2e-3`, `std_bias_acc=5.5e-5`, `std_bias_gyro=2e-5`). 기본 `dt = 1/800`(`:106`)이고 250 Hz는 인자로 지정. 각가속도 미구현(`imu_integration_step` 주석 `:65`, `roadmap.rst:16`) | 일치 (기본값만 다름) |
| D15 | 구현 언어 | "implemented entirely in PyTorch", GPU 벡터화 (S.5.2.1, L1708) | 센서 에뮬레이션(`rgb`, `spc`, `imu`, `itof`, `dvs`)은 numpy와 scipy. torch는 열 솔버, RIFE, `Dataset`(`torch.utils.data.Dataset`), `spc_avg_to_rgb`에만. 해당 문장은 dToF(외부 SPCSim, PR #39)에만 맞을 수 있음 | 차이 |
| D16 | 수동 SPAD 파라미터 | Table 2: 노출 파라미터에 "Exposure, dark count rate, fill factor", 본문은 양자 효율, 암계수 | `emulate/spc.py` `emulate_spc(img, factor)` = `Binomial(1, 1 − exp(−img x factor))`. 암계수, 충전율, 양자 효율은 `factor` 하나에 흡수(논문 S.5.1도 f에 흡수한다고 함) | 일치 (Table 2 표기만 과장) |
| D17 | 저장과 로더 | 비트팩 numpy 8배, `transforms.json`, 센서별 SQLite, PyTorch와 TensorFlow용 로더 (S.4, S.7) | `Data.bitpack_dim`과 `_slice_bitpacked_array`(`dataset.py:208`) 구현, `transforms.json`과 peewee SQLite(`transforms.db`) 있음. `Dataset`은 `torch.utils.data.Dataset`이고 TensorFlow 로더는 없음 | 일치 (TensorFlow 제외) |
| D18 | 분산 렌더링 | RPC 클라이언트/서버, UDP 레지스트리, 풀 (S.2.2) | `simulate/blender.py`: `rpyc`, `rpyc.utils.registry.UDPRegistryClient`, `BlenderClients.pool` 구현 | 일치 |
| D19 | 라이선스 | "free and open-source library" (S.2.2) | `LICENSE.md`는 GPL-3.0-or-later이며 "A commercial license is available for those who cannot comply with the GPL" 문구가 붙은 이중 라이선스 형태. 기여는 MIT로 받음 | 일치 (GPL 전염성 주의) |

## 3. 일치하는 부분과 판단

- **구조(분리 설계)는 일치합니다.** Blender 렌더(`blender.render-animation`) → 정답 패스(깊이, 법선, 광류, 분할, 재질, 점, 확산, 정반사, 열) → 센서 에뮬(`emulate.spad`, `events`, `rgb`, `imu`, `itof`) → `transforms.json` 데이터셋이라는 흐름이 코드에 그대로 있습니다. 스테레오 문서(`tutorials/stereo.rst`), HDR, 가우시안 스플랫 튜토리얼도 있습니다.
- **가장 큰 간격은 dToF입니다.** 논문의 제일 큰 사례(디지털 트윈)와 초록의 "pileup and glare for LiDAR"가 공개 main에 없습니다(D1, D2). 이슈 트래커와 로드맵이 이를 인정하고 있어 숨기는 것은 아니지만, 논문이 "플랫폼을 공개한다"고 쓴 문장은 이 모달리티에 대해서는 아직 성립하지 않습니다.
- **열은 정적 모드만 논문 시점에 있었고 모델이 단순합니다.** 열 솔버는 0.2.0(2026-11-03)에 들어 있었고 PR #40(2026-11-04 병합)이 굽기와 렌더링을 고쳤으며, 과도 모드는 PR #41(2026-11-21 병합)로 논문 뒤에 왔습니다(D6). 방출 모델은 단순 σT⁴이며(D4), 마이크로볼로미터 관성이 없습니다(D5). 논문의 "history-dependent emission"은 솔버의 과도 열수송에는 맞지만 센서 응답에는 아직 해당하지 않습니다.
- **iToF는 이상형입니다.** 논문 그림 2의 다섯 실패 모드 중 위상 접힘과 주변광만 코드가 다룹니다(D3).
- **문서 사이에 낡은 문장이 있습니다.** `roadmap.rst`의 "Add dynamic thermal simulation; only steady state has shipped so far"는 PR #41 이후 사실과 다릅니다(D6).

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 전체 재현은 불가능합니다. 이유는 GPU가 아니라 코드, 데이터, 하드웨어의 부재입니다. 공개된 센서 모델 중 수동 SPAD, 이벤트, RGB, IMU, 이상형 iToF, 열(정적)은 이 맥의 CPU에서 구현할 수 있어 보이지만, 설치와 코드 실행 승인이 먼저 필요하고 이번에는 하지 않았습니다.**

| 수준 | 판정 | 근거와 자원 |
|---|---|---|
| dToF 디지털 트윈 (그림 4 위, Sec. 5) | 불가 | 코드가 main에 없음(D1, D2), Adaps ADS6311 실기와 실측 히스토그램, 49개 PSF 측정, 보정된 Blender 장면이 필요 |
| 100장면 데이터셋 재생성 (Sec. 7) | 불가 | 장면과 체크섬 미공개(D13). 저장만 SPAD 비트팩 4.8 TB(`03` V2). 논문 생성은 수천 CPU와 수백 GPU(HTCondor, S.2.2) |
| 실측 대조 (그림 5: FLIR, Chronoptics, EVK4, Pi-Imaging) | 불가 | 실기와 촬영 데이터 필요. 장면 자료 미공개 |
| 센서 묶음 비교 (그림 4 아래) | 불가 | RF-DETR, 재구성 모델 가중치 필요, 비교 조건 미공개 |
| 한 장면 렌더 후 수동 SPAD, 이벤트, RGB, IMU, iToF 에뮬 | 조건부 가능 | `pip install visionsim` + Blender 3.6 이상(arm64 맥 빌드). 에뮬 부분은 numpy라 CPU로 충분. 새 설치 승인과 코드 실행 승인이 필요. Cycles 렌더 속도(CPU 또는 Metal)는 측정 전 |
| 열 정적 한 장면 (`--config.thermal.device cpu`) | 조건부 가능 | torch CPU 열 솔버와 Cycles 굽기. 속도 측정 전. 과도 모드는 점 수에 따라 느릴 수 있음(측정 전) |
| 마이크로볼로미터 필터(식 S20) 직접 추가 | 가능 | 한 줄짜리 IIR. 우리가 `03` V7에서 산술은 이미 확인 |
| iToF에 잡음, 포화, 양자화 추가 | 가능 | 식 2 그대로. 우리가 `analysis/toy_itof_failures.py`(0.3초)로 구현해 봄 |
| dToF 전방 모델 + IRF 경사 학습을 합성 데이터로 직접 재구현 | 조건부 가능 | torch(CPU/MPS)로 충분. 데드타임 루프는 파이썬이라 느릴 수 있으나 토이(`analysis/toy_dtof_pileup.py`)는 0.3초. 실기 보정은 불가 |

**노력 추정(측정 전, 추정):**
- 설치 + 한 장면 다중 센서 에뮬 재현: 반나절–하루(승인 후)
- 열 방출과 마이크로볼로미터를 논문 식대로 보강: 1–2일
- iToF 잡음, 포화, 양자화, 모션 모델 코드화: 1–2일
- dToF 전방 모델 + 미분 가능 IRF 학습(합성 데이터): 1–2주
- 논문 사례 재현(실기, 데이터, 코드 필요): 외부 자료 없이는 불가

## 5. 라이선스와 의존성

- 코드는 GPL-3.0-or-later이고 상업용은 별도 문의입니다(`LICENSE.md`). 이 저장소에 코드를 가져와 파생물을 만들면 GPL 조건을 따라야 합니다. 이 폴더는 코드를 읽기만 했고 `repo/`는 gitignore입니다.
- 의존성(`pyproject.toml`): numpy, torch, scipy, matplotlib, OpenEXR, rpyc, peewee, pydantic, tyro, robust-laplacian 1.1.0(비 Windows) 등. Blender는 PATH에 있어야 합니다(`visionsim post-install`이 Blender의 Python에 필요한 것을 넣음). 설치는 하지 않았습니다.
- 외부 SPCSimLib은 저작권 문구만 있고 SPDX 라이선스가 인식되지 않습니다(GitHub API NOASSERTION). 의존하려면 사용 조건을 따로 확인해야 합니다.
- 테스트(`tests/`, 95개 파일)와 CI(`.github/workflows`)가 있지만 실행하지 않았습니다.
