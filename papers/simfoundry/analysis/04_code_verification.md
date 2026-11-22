# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `NVlabs/SimFoundry` (공식, 프로젝트 페이지의 GitHub 링크, Apache-2.0). 커밋 `9e34ebefcd020583fbb755a8b57268dce78eca26` (2026-08-27, "Refinement (#7)"). 얕은 복제를 `repo/`에 두었습니다(344개 파일, 108 MB, 파이썬 약 8.3만 줄, gitignore).
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았고 모델 가중치, 예시 자산, VLM API 호출도 하지 않았습니다. 복제한 내용은 신뢰할 수 없는 데이터로 다뤘습니다.
- 읽은 파일: `README.md`, `docs/INSTALL.md`, `scripts/pipeline/README.md`, `scripts/cfg/real2sim_cfg.yaml`(784줄 전체 구조), `scripts/cfg/sample_init_states/example.yaml`, `scripts/cfg/task/droid/droid_desk_stack_dishware.yaml`, `scripts/pipeline/A_reconstruction/stages/12_stabilize_physics.py`, `scripts/pipeline/B_augmentation/stages/1_prompt_object_cousins.py`, `7_propose_scene_tasks.py`, `scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py`, `5_generate_demos.py`의 머리말과 설정, `simfoundry/pipeline/frame_selection.py`, `simfoundry/utils/placement_utils.py`, `scene_sampling_utils.py`, `simfoundry/domain_randomization/`, `PATCH_PROVENANCE.md`, `THIRD_PARTY_LICENSES.md`, `third_party_notices/Hunyuan3D-2.1-LICENSE.txt`.
- 코드의 스테이지는 A 재구성 13단계, B 증강 8단계, C 적용 6단계입니다(`scripts/pipeline/README.md`). 이 분석은 논문 표와 직접 대응하는 설정을 중심으로 읽었고 전체 8만 줄을 줄 단위로 읽지는 않았습니다.

## 1. 설치 요구사항과 라이선스

- **요구사항(README, INSTALL.md).** Linux + NVIDIA GPU(CUDA), mamba/conda, ffmpeg, **약 250 GB 디스크**(conda 환경 약 100 GB, `deps/` 약 82 GB 중 VOID 모델만 41 GB), Hugging Face 계정(SAM3 등 게이트 모델), **Google Cloud Vertex AI 또는 Gemini API 키**(재구성, 관절, 증강의 VLM 단계가 모두 원격 호출). 최소 VRAM 16 GiB, 24 GiB 카드는 `s7_mesh.low_vram=true` 필요(기본은 메시 형상 생성에 약 29 GiB), 관절 단계는 18 GiB 이상. 시뮬은 OmniGibson(BEHAVIOR-1K), Isaac Sim 위에서 돌고 설치 스크립트가 NVIDIA Omniverse 라이선스 동의(`--accept-nvidia-eula`)를 요구합니다.
- **저장소 코드 라이선스.** Apache-2.0(NVIDIA 소유 코드). 일부 파일은 ACDC/digital-cousins(Apache-2.0)에서 파생되었고 머리말에 표시됩니다.
- **의존성 라이선스는 제각각입니다**(`THIRD_PARTY_LICENSES.md`, `PATCH_PROVENANCE.md`). 패치 9개 중 3개는 비오픈소스 조각을 담고 있습니다: `Any6D.patch`(비상업·학술 전용), `FoundationPose.patch`(NVIDIA Source Code License, 비상업), `Hunyuan3D-2.1.patch`(Tencent 커뮤니티 라이선스). **기본 메시 생성기인 Hunyuan3D-2.1의 라이선스는 "유럽연합, 영국, 한국에는 적용되지 않으며 그 영토로 제한된다"고 명시합니다**(`third_party_notices/Hunyuan3D-2.1-LICENSE.txt` 3행, "Territory" 정의). 이 지역에서는 기본 경로를 라이선스상 쓸 수 없으므로 TRELLIS.2(MIT) 등 대체 백엔드가 필요합니다.
- 논문 PDF의 라이선스(arXiv CC BY-SA 4.0, 본문 © NVIDIA)와 코드 라이선스(Apache-2.0)는 별개입니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**, **차이**(값이 다르거나 논문에 없는 기능), **부분**(일부만 구현), **미공개**(저장소에서 찾지 못함), **미확인**.

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | 공개 범위 | 파이프라인, 실제-시뮬 평가(§5.1), 시뮬→실제 학습(§5.2), 재구성 벤치마크(§5.3) 전부를 보고 | README 뉴스: "V0 rigid-body and articulation generation" 공개(2026-08-14), 예시 자산(08-26). "Coming Soon: Automated background generation, **Robotics data generation, training, and evaluation**". 그런데 `C_application/stages/`에는 평가(1), 원격조작(2), 주석(3), 웨이포인트(4), 시연 생성(5), 재생(6), sweep 평가가 이미 있고 자동 배경 폴더(7개 스크립트)도 있어 README가 코드보다 뒤쳐짐. **없는 것**: 정책 학습 코드(H.2의 배치 256, 학습률 1e-5, 10k/40k 스텝), Pearson/MMRV 계산, PolaRiS 비교, SAM3D 비교와 준-GT 재구성 벤치마크(표 L.2), cousin 절제 실험 설정 | **부분** | 표 G.1–G.8, L.2의 수치는 공개 코드로 재현할 수 없고, 평가 대상 정책(π0, π0.5, GR00T, DreamZero)은 서버 클라이언트만 있음(`simfoundry/policies/`) |
| D2 | 시뮬레이터와 내보내기 | §4 "export ... to downstream robotics simulators such as IsaacLab [59]". OmniGibson은 부록 F의 프롬프트에만 나옴 | 코드는 **OmniGibson** 장면(USD/JSON)만 내보냄(`14_create_og_scene.py`, `s14_og/reconstructed_og_scene.json`). Isaac Lab 내보내기 코드를 찾지 못함. `domain_randomization/configs.py` 머리말에 "no IsaacLab dependency" | **차이** | 논문 실험이 어느 시뮬에서 돌았는지 본문에 없음. 공개 코드 경로는 OmniGibson |
| D3 | VLM·모델 이름 | 부록 E.2: V_scene, V_articulation = "Gemini-Pro-3", V_image = "Gemini-Pro-3-Image-Preview" | `gemini-3.1-pro-preview`(감지, 대표 프레임, 정면 선택, 관절 3종, 물성), `gemini-3-pro-image`(제거·업샘플). **객체 cousin 텍스트 제안과 작업 제안은 `gemini-2.5-pro`**(`real2sim_cfg.yaml` 690, 765줄) | **차이** | 모델 세대가 단계마다 다르고 논문 서술과도 다름. 원격 비결정 출력이라(부록 C) 결과 재현성이 낮음 |
| D4 | 대표 프레임 | 부록 E.1 "frame 0 as the default representative frame" | `s3_ground.img_idx: auto`(109줄). `frame_selection.py`가 후보 프레임을 휴리스틱(물체 점유율, 분리도, 선명도, 평면 적합)으로 점수화하고 **VLM이 최종 선택**(`mode: hybrid`). 머리말에 "used to be hard-coded `img_idx: 0`" | **차이** | 기본 동작이 논문 서술과 다름. 논문 실험이 어느 방식이었는지 불명 |
| D5 | 포즈 정렬 | §4 "FoundationPose [89]로 정제" | `use_foundationpose: true`, `est_refine_iter: 5`(275줄)로 일치. **추가**: VLM이 메시 정면을 고르고(`front_pick_model`, 7.5도 간격 미세 탐색, 요 스냅) 기울기를 메시에 굽는 로직(`bake_fitted_tilt`, `min_tilt_deg` 10 / `max_tilt_deg` 80) | **일치 + 논문에 없는 단계** | 정렬 품질은 이 보정에 의존. 논문 부록 K.2는 등방 스케일 포함 유사 변환만 서술 |
| D6 | 물리 안정화 | 부록 E.4: PyBullet, 속도를 매 스텝 0으로, 포즈가 안정될 때까지 | `12_stabilize_physics.py`: PyBullet DIRECT, `resetBaseVelocity(0)` 매 스텝, 위치 임계 1e-4, 자세 임계 1e-3, `load_at_once: true`일 때 최대 10000 스텝(한 개씩 모드는 상한 없음), 평면은 `plane.urdf`, 기본 z 오프셋 0.05 | **일치** | 임계값과 스텝 상한은 논문에 없는 세부. 단일 평면 가정(부록 C)과 일치 |
| D7 | 물성 | §4 "질량, 마찰을 V_scene에 물어서" | 재구성 경로는 `s11_sim.vlm_model`로 일치. 충돌은 CoACD, `hull_count: 32`, `n_submesh: 10`(361–363줄). **cousin 경로는 `keep_original_mass_friction: true`**(710줄): 원본 물체의 질량·마찰을 복사하고 VLM에 묻지 않음 | **일치 + 차이** | cousin은 형상이 달라져도 질량이 원본과 같아 현실성이 낮아질 수 있음. 논문에 없음 |
| D8 | 시연 증식 | 부록 H.1, F.3.1: 사람 시연 10–15개를 **MimicGen [58]**으로 증식(작업당 10개→100개), 재질·카메라·(DROID) 탁자 높이 무작위화 | MimicGen 패키지를 쓰지 않음. `3_annotate_src_demo.py`("pick/place signals for MimicGen") + `4_extract_waypoints.py` + `5_generate_demos.py`가 **객체 중심 웨이포인트 재생**을 자체 구현(k 최근접 `k_nearest_neighbors [3,3]`, 위치 0.5/자세 2.0 가중, 선형·스플라인 보간, **CuRobo 기본 꺼짐** `use_curobo: false`, 기본 `n_demos: 1000`, `max_attempts: 2000`, `grasping_type: assisted`). 장면 재샘플링 `xy_range 0.03`, `z_rotation_deg 20`(기본 꺼짐). 재질·조명 무작위화는 `domain_randomization/`에 있고 카메라 구면 섭동은 `pick_place_task.py`에 있음. **탁자 높이 무작위화는 찾지 못함** | **차이 + 부분** | 논문 "MimicGen"과 구현이 같은지 불명. 탁자 높이 무작위화(DROID)는 공개 코드에 없음 |
| D9 | 장면 cousin | 부록 F.2: 앵커 물체를 VLM 또는 사용자가 정하고 술어 {LeftOf, RightOf, InFrontOf, Behind, OnTopOf, Inside}를 **샘플링**, 방해 물체는 BEHAVIOR에서 질량·부피·밀도·범주로 필터 | `placement_utils.py`: 술어 9종(`on_top, left_of, right_of, behind, in_front_of, inside, near, between, inside_link`), 간격 범위는 **작업 YAML에서 사용자가 지정**(`group_predicate_placement`). VLM 앵커 선택은 찾지 못함. 방해 물체 필터는 `query_assets`(부피, 질량, 밀도, bbox, 능력). 별도 `6b_sample_asset_scene.py`는 OnTop 관계를 보존한 **무작위 XY 이동(0.03 m)·Z 회전(50도)** | **차이 (더 넓지만 자동은 아님)** | "의미 있는 대안 배치를 자동 제안"이라기보다 사용자가 술어를 지정하는 구성 방식 |
| D10 | 작업 cousin | 부록 F.3: VLM이 작업 제안, 예시 13개 | 프롬프트는 부록 그림 F.2와 거의 같음(OmniGibson 문구, YAML 형식). 기본 **`num_tasks: 5`**, `gemini-2.5-pro`. 프롬프트에 논문에 없는 "State usage guidance" 단락 추가 | **일치 + 차이** | 13개는 설정으로 바꿔야 함. 프롬프트가 논문 그림보다 길어짐 |
| D11 | 객체 cousin | 부록 F.1 프롬프트 A–E, 부품별 기하/위상/외형, VLM 검증, 제한된 재시도 | `1_prompt_object_cousins.py`에 프롬프트가 거의 그대로, 부품마다 2/2/2 변형(`num_*_variation: 2`), `check_reasonableness: true`, `max_reasonableness_attempts: 3`. `num_obj_to_swap: 9`, `max_combinations: 4` | **일치** | 실험의 "+9 cousins"와 설정값 9가 대응하는 것으로 보임 |
| D12 | 평가 프로토콜과 시뮬 맞춤 | 부록 J.1: 과제당 25회, 5×5 격자 25개 위치를 물체마다 독립 비복원 추출, 위치는 모든 체크포인트에 고정 | `sample_init_states/example.yaml`: `grid_resolution: [5, 5]`, `randomize_pairings: true`(물체마다 섞인 격자 매핑), 안정성 검사와 재시도(섭동 0.02 m). **그러나** `s15_eval.n_episodes: 20`이 기본(478줄), `timeout_s: 90`, `execute_horizon: 8`, `action_freq: 15`, 작업 `max_steps: 1000`(환경 스텝이 15 Hz라면 약 67초라 90초 제한보다 먼저 걸림, 스텝 단위는 확인하지 못함)은 논문에 없음. 로봇 기본 **`grasping_mode: assisted`**(394줄). 평가 스크립트 머리말의 작성자 메모는 "렌더 색 보정, 실제 데이터 액션 재생으로 궤적 정렬 확인, 3DGS 배경을 메시로 교체, **assisted를 physical 파지로 변경**, 모든 물체에 마찰 추가, 물체 포즈·스케일·로봇 포즈 조정, 손가락 최대 속도 낮추기"를 열거 | **부분 + 차이** | 실제-시뮬 상관은 이 수작업 맞춤에 의존할 수 있음(C9). 보고 수치가 assisted인지 physical 파지인지 논문에 없음. 코드의 `grasping_mode` 덮어쓰기는 기본이 `null` |
| D13 | 성공 판정 | 이진 성공, 과제 기준 rubric(부록 I.2) | 작업 YAML이 목표 술어(예: Stack Dishware = cup OnTop bowl, bowl OnTop plate, **robot-cup Touching false**)와 milestone을 가짐. 논문 rubric의 "Place cup in bowl"은 코드에서 OnTop | **일치 (의미 근사)** | 접촉 해제 조건이 성공 판정에 들어감. 논문 rubric에 명시 없음 |
| D14 | 정책 서버 | π0, π0.5, GR00T N1.6/N1.7, DreamZero 평가 | `simfoundry/policies/`에 openpi, gr00t, dreamzero 클라이언트와 `eval_sweeps/` 예시(체크포인트 간격 1000스텝). 가중치와 서버는 저장소 밖 | **일치 (클라이언트만)** | 모델 호출 환경은 별도 구성 |
| D15 | 하드웨어 | 부록 L.4: RTX 3090 24 GB, 물체당 약 5분 | INSTALL: 16 GiB 최소, 24 GiB에서 `low_vram` 필요. README: "short real-world video into a physics-ready simulation scene in **under an hour**" | **대체로 일치** | 자동 배경(약 90분, 부록 C)은 별도 옵션(`2c`, `auto_bg_reconstruction`) |
| D16 | 설정 파일의 옛 흔적 | — | `simfoundry/configs/default.yaml`에 GPT-4o 설정(`gpt_version: 4o`), DinoV2, Grounded SAM v2(ACDC 유래)가 남음 | **죽은 설정으로 보임** | 실행 경로는 `scripts/cfg/real2sim_cfg.yaml` |

요약: 파이프라인 구조(추출 → 생성 → 증강, 객체 cousin 프롬프트, PyBullet 안정화, CoACD, FoundationPose)는 논문과 잘 맞습니다. 차이는 (1) 공개 범위(학습·지표·벤치마크 코드 없음), (2) 사용 모델 세대(Gemini 3.1과 2.5가 혼재, 논문은 Gemini-Pro-3), (3) 기본 동작(대표 프레임 자동 선택, assisted 파지, 작업 5개), (4) 논문에 없는 수작업 맞춤 목록에 몰려 있습니다. 시뮬→실제 학습 수치(표 G.4–G.8)와 실제-시뮬 상관 수치(표 G.1)는 이 저장소만으로 재현할 수 없습니다.

## 3. 테스트

`tests/`에 39개 파일(파이프라인 오케스트레이터, 프레임 선택, 술어 배치, 안정화 관련 등)이 있으나 실행하지 않았습니다. `AGENTS.md`는 OmniGibson 편집기에 테스트가 없다고 밝힙니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**공식 파이프라인은 이 맥에서 실행할 수 없습니다. 논문의 평가 수치 재계산과 구조 일부의 재구현(CPU)만 가능하고, 재계산은 끝냈습니다.**

| 무엇 | 판정 | 이유 |
|---|---|---|
| 공식 파이프라인 A(영상 → 장면) | **불가** | Linux + NVIDIA GPU(CUDA) 전제, SAM3, Hunyuan3D, FoundationPose, DA3, VOID 등 CUDA 모델, 디스크 약 250 GB. 거기에 Gemini(Vertex AI) 원격 호출이 필수라 API 비용과 승인이 필요 |
| 시뮬 평가·시연 생성(C) | **불가** | OmniGibson은 Isaac Sim 기반(Linux GPU). 맥 미지원 |
| 정책 학습(π0.5 미세조정, YAM flow-matching) | **불가** | GPU. 학습 코드도 저장소에 없음 |
| 실로봇 평가(DROID, YAM) | **불가** | 로봇과 현장 |
| 표 재계산, 검정, 시행 잡음 토이 | **완료** | numpy, scipy (`analysis/recompute.py`, 수 초) |
| 파이프라인 일부의 개념 재구현 | **가능 (CPU)** | 안정화(MuJoCo 등으로), 술어 기반 배치 샘플러, 재구성 지표(챔퍼, F1@1 cm, 위치 오차), MMRV/r 평가 라이브러리. mujoco 4.0, torch 2.15 설치됨 |
| 공개 예시 자산(Hugging Face, 약 0.36 GB) 구조 읽기 | **조건부** | 다운로드 승인이 필요. OmniGibson 없이 JSON/USD 구조만 볼 수 있음 |

**노력 추정(측정 없음).** 위 개념 재구현 세 가지는 각 1–3일. 같은 구조를 Linux GPU 머신에서 돌리려면 환경 구성만 2–4일(설치 스크립트 13개, 게이트 모델 승인 대기 포함)이고, VLM 호출 비용과 약관(Hunyuan3D 지역 제한, Any6D·FoundationPose 비상업)을 먼저 확인해야 합니다. 논문의 실로봇 평가를 따라 하려면 DROID 또는 YAM 장비가 필요합니다.
