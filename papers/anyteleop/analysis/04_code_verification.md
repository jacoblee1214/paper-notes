# 04. 공식 코드 대조와 구현 가능성

공식 코드가 **부분적으로** 있습니다. 리타게팅 라이브러리(`dex-retargeting`)와 시뮬 웹 뷰어(`sim-web-visualizer`)는 공개되어 있고, 논문의 핵심 서버(손 검출, 융합, 동작 생성, 클라이언트)는 공개되어 있지 않습니다. 그래서 이 문서는 **논문 식 (1)과 웹 뷰어 서술 ↔ 공개 코드**를 대조하고, 나머지는 "코드 없음"으로 표시합니다. 두 저장소는 얕은 복제(`--depth 1`)해서 `repo/`에 두었고, 정적으로만 읽었습니다. 실행, 설치, import를 하지 않았습니다.

## 1. 저장소와 라이선스

| 저장소 | 역할 | 라이선스 (파일) | 별 | 마지막 push | 복제 커밋 | 크기 (복제본) | 상태 |
|---|---|---|---|---|---|---|---|
| [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 손 리타게팅 라이브러리, README가 "AnyTeleop 프로젝트에서 시작" (v0.5.0) | MIT (`LICENSE`: Copyright 2023 Yuzhe Qin) | 1321 | 2025-08-21 | `3f56141` (커밋 날짜 2025-04-04) | 56 MB | 정적 읽기 (`repo/dex-retargeting/`) |
| [NVlabs/sim-web-visualizer](https://github.com/NVlabs/sim-web-visualizer) | 웹 기반 시뮬 뷰어 (meshcat), README가 "AnyTeleop 프로젝트의 일부" | MIT (`LICENSE`: NVIDIA 2022–2023) | 429 | 2024-05-26 | `20b02f3` (2024-05-26) | 88 MB | 정적 읽기 (`repo/sim-web-visualizer/`) |
| 프로젝트 페이지 `https://yzqin.github.io/anyteleop/` | 논문 제목 쪽 주소 (v3) | — | — | — | — | — | **현재 GitHub Pages 404.** 보관본 2025-11-18 (`web.archive.org/web/20251118211139/`)에 두 코드 링크와 RSS 2023 BibTeX |
| `http://anyteleop.com` | 논문 부록 A의 프로젝트 주소 (v1–v3) | — | — | — | — | — | **무관한 콘텐츠 사이트로 연결됨**(2026-11-22 확인). 공식 페이지가 아님 |
| `yzqin/*` 저장소 목록 | 저자 계정 | — | — | — | — | — | `anyteleop` 저장소 없음 (2026-11-22 목록 확인) |
| RalphFH/AnyTeleop | 제3자 구현 (단안 RGB로 손 제어 주장) | MIT | 17 | 2026-09-30 | 복제하지 않음 | — | 비공식. 참고만 |

- 공식 코드 라이선스는 두 저장소 모두 MIT이고 LICENSE 파일이 있습니다. 논문 자체는 arXiv CC BY 4.0입니다(§5).
- 논문의 "open-sourcing" 약속(§IX)은 위 두 저장소로만 이행된 것으로 보입니다. 서버 본체 저장소는 확인되지 않았습니다(GitHub 검색으로 확인, 저자 계정 목록 대조).

## 2. 논문 ↔ 공개 코드 대조 (D1–D9)

| 번호 | 논문 | 공개 코드 (파일:위치) | 차이와 영향 |
|---|---|---|---|
| D1 | 서버 전체: 손 검출(MediaPipe, 신경망 위치 추정), 검출 융합(자동 보정 N = 50, SMPL-X 신뢰도), 동작 생성(CuRobo, 25 → 120 Hz), 클라이언트 (§IV) | 해당 코드 없음. `dex-retargeting/example/vector_retargeting/detect_from_video.py`는 **단일 영상**에서 MediaPipe 검출 후 리타게팅하는 예시입니다. 융합, CuRobo, 실시간 루프는 없습니다 | 논문의 주요 모듈 대부분을 검증할 수 없습니다. 검출 정확도, 융합 효과, 120 Hz 동작 생성은 논문 서술만 있습니다 |
| D2 | 식 (1)의 첫 항은 **제곱** 합 ‖α v_t^i − f_i(q_t)‖² | `optimizer.py` `VectorOptimizer.get_objective_function`(약 249–273행): `vec_dist = norm(robot_vec − α·target)`, 그 다음 `SmoothL1Loss(beta=huber_delta)`(기본 δ = 0.02) | **Huber 손실**로 바뀌었습니다. 작은 오차에서는 제곱에 가깝고 큰 오차에서는 선형입니다. 스케일 α는 `torch_target_vec * self.scaling`으로 들어가 논문과 같습니다. 논문에 δ 값은 없습니다 |
| D3 | 시간 항 β ‖q_t − q_{t−1}‖² (식 (1)의 일부) | `optimizer.py` 194, 300, 571행: 목적함수 **값**(`result`)에는 시간 항이 없고, **기울기**에만 `grad_qpos += 2 * norm_delta * (x − last_qpos)`가 더해집니다. 기본 `normal_delta = 4e-3`(`retargeting_config.py` 49행) | 기울기와 목적함수 값이 같은 함수에서 나오지 않습니다. SLSQP의 선형 탐색은 값을 쓰므로 수렴점이 식 (1)의 최소점과 다를 수 있습니다. **코드 읽기에 따른 추정이며 실행해 확인하지 않았습니다.** β의 논문 값도 없습니다 |
| D4 | 출력 필터는 논문에 없음 | `retargeting_config.py` 63행 `low_pass_alpha = 0.1`, `seq_retarget.py`의 `filter.next`: y ← y + 0.1 (x − y) (지수 평활) | **논문에 없는 평활화**가 기본으로 켜져 있습니다. 논문의 "매끄러운 궤적" 주장에 이 필터가 얼마나 기여하는지 분리할 수 없습니다 |
| D5 | 관절 한계 q_l ≤ q_t ≤ q_u | `optimizer.py` `set_joint_limit`: 한계에 ε = 1e-3 여유를 두고 `set_lower_bounds`/`set_upper_bounds` | 여유 1e-3은 사실상 같습니다. 차이가 생기는 것은 한계 밖을 허용하는 경우뿐입니다 |
| D6 | 최적화 방법 미기재 | `optimizer.py` 41행 `nlopt.LD_SLSQP`, 시작점은 이전 자세(warm start, `last_qpos`를 한계로 clip) | 논문 재현에 필요한 정보가 빠져 있습니다. 해의 수렴 조건과 허용 오차도 미기재 |
| D7 | 사람 키포인트와 로봇 링크의 대응을 수동으로 지정 (§IV-C) | `retargeting_config.py` 28행 `target_link_human_indices`, 38행 `target_origin_link_names`, 74–97행: 구성 파일에서 지정하고 형상을 검사 | **일치합니다.** 구성 파일은 42개(`configs/teleop/`)이며 Allegro, Shadow, Schunk SVH, Inspire, Ability, LEAP, Panda 그리퍼 계열을 포함합니다. 논문 Fig. 10의 DLR 손에 맞는 파일은 없습니다 |
| D8 | 웹 뷰어: meshcat 기반, Three.js 렌더링, IsaacGym·SAPIEN 지원 (§V, 부록 C) | `sim-web-visualizer/sim_web_visualizer/`: `base_visualizer_client.py`(MeshCat), `isaac_visualizer_client.py`, `sapien_visualizer_client.py`. 웹소켓 언급은 README와 예제 파일에 있습니다 | **일치합니다.** 지연, 대역폭, 여러 창 동기화는 코드에 측정 도구가 없습니다. 측정 결과도 논문에 없습니다 |
| D9 | (논문과 무관) 의존성 선언 | `pyproject.toml`의 핵심 의존성은 numpy ≥ 2, pytransform3d, pin, nlopt, anytree, pyyaml, lxml입니다. 그런데 `optimizer.py` 6행이 `import torch`를 하고 torch는 선언되어 있지 않습니다 | 설치 후 실행에서 torch 오류가 날 수 있습니다. 맥 CPU 빌드 설치로 해결 가능하다고 보지만 확인하지 않았습니다 |

**논문에 값이 없는 설정 (재현 불가 항목).** α(손 크기 배율), β(시간 가중치), Huber δ, 최적화 허용 오차, 시뮬 학습의 DAPG 하이퍼파라미터, 시드 3개의 선택 방법은 논문에 없습니다. 공개 코드 기본값(δ = 0.02, norm_delta = 4e-3, low_pass_alpha = 0.1, scaling_factor = 1.0)은 논문 설정이라고 가정할 근거가 없습니다.

## 3. 공식 코드에 없는 것 (확인 불가 목록)

- CuRobo 사용 코드, 동작 생성 루프, 25 Hz → 120 Hz 전환 (논문 §IV-D). 저장소에서 `curobo` 문자열 0건.
- SMPL-X 신뢰도 계산, 첫 N = 50 프레임 보정, 다중 카메라 융합 (논문 §IV-B). 저장소에서 `smplx`, `fusion` 0건.
- FrankMocap 방식의 약한 원근 배율 신경망 (§IV-A). 저장소에서 `frankmocap` 0건.
- 손 검출 서버와 클라이언트의 네트워크 통신, 도커 이미지 정의 (§III-A.6). 저장소에서 `docker` 0건.
- 협업 실험 설정 (§VII-B), 표 III 시뮬 학습 설정, 실로봇 설정 (§VI-B). 학습과 평가 스크립트가 없습니다.
- 표 V, 표 II의 프로파일링 코드 (`dex-retargeting/example/profiling` 디렉터리는 있으나 이 논문의 모듈 전체를 재는 코드는 아님. 이 문서에서는 열지 않았습니다).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 전체 시스템은 이 맥에서 재현할 수 없습니다. 리타게팅과 뷰어까지는 CPU로 가능성이 있지만, 설치와 승인 뒤에 판정해야 합니다. 실로봇과 시뮬 학습은 불가능합니다.**

| 구성 요소 | 맥에서 가능? | 근거 |
|---|---|---|
| 손 검출 (MediaPipe, RGB) | **조건부 가능** | 논문은 CPU 실시간이라고 씀(§IV-A). 설치 필요 (승인 대상). 논문 표 II의 26 ms는 RTX 3090 측정이라 직접 비교가 아님 |
| 검출 융합, SMPL-X 신뢰도 | 조건부 | 공식 코드 없음. 논문 식만 있어 재구현해야 함 |
| 리타게팅 (dex-retargeting) | **가능 (CPU)** | 핵심 의존성은 numpy, pytransform3d, pin, nlopt 등이고 소스에서 CUDA 호출은 찾지 못했습니다. torch는 선언되지 않았지만 import됩니다(D9). 설치 승인 필요. 리타게팅 시간 9–10 ms (논문 표 II, RTX 기기)를 맥 CPU에서 재볼 수 있음 |
| 동작 생성 (CuRobo) | **불가** | CUDA 가속 전제(§IV-D). 맥에서 대체 구현은 재구현 항목 (수 주 추정) |
| 웹 뷰어 | 조건부 | `python_requires >=3.6,<3.11`이라 맥 기본 Python 3.12와 맞지 않음. 별도 환경 필요 (승인 대상) |
| 시뮬 모방학습 (SAPIEN, IsaacGym) | **불가 또는 미확인** | IsaacGym은 CUDA 전제. SAPIEN의 맥 지원은 확인하지 않음. DAPG 학습 3시드 × 6조건은 GPU 학습 전제 |
| 실로봇 (XArm6, Allegro, RealSense) | **불가** | 하드웨어 필요 |

- **노력 추정 (측정 전, 추정):** 리타게팅 목적함수 재구현과 D2–D3 검증 1–2일. 검출, 융합, 동작 생성까지 포함한 서버 재구현은 2–4주(동작 생성의 CUDA 대체 포함). 실로봇은 저장소 밖 장비가 필요합니다.
- 이 맥에서 바로 할 수 있는 것: 식 (1)의 목적함수 값과 기울기를 numpy로 비교하는 검사(설치 없이 가능). 이것이 D3에 대한 실행 기반 확인이 됩니다.

## 5. 라이선스 (논문과 코드)

- 논문: arXiv 초록 페이지의 "view license" 링크가 `creativecommons.org/licenses/by/4.0/`를 가리킵니다(2026-11-22 확인). 원문 PDF(v1, v2, v3, RSS 2023 판)에서 ©, copyright, All rights reserved 같은 고지를 찾지 못했습니다. 그림 게시 판정은 `01` §0과 `README.md`를 봅니다.
- 코드: 두 저장소 모두 MIT. 복제본은 이 저장소 밖으로 내보내지 않습니다(`repo/`는 gitignore).
