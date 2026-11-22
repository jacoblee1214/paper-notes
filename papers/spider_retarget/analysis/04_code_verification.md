# SPIDER 공식 코드 대조 (정적 읽기)

대상: `facebookresearch/spider` (GitHub), 얕은 복제 `repo/` (커밋 `1d607af226dcb5aeb8c1013335732ab513fff125`, 2026-11-17 커밋 "docs: seed check of the CPU recipe"). 저장소는 2026-11-21에 마지막으로 push되었습니다. 복제는 정적으로만 읽었습니다. 실행, 설치, import, 데이터 다운로드는 하지 않았습니다. 복제 폴더는 gitignore 처리했습니다.

## 0. 저장소 요약

| 항목 | 값 | 확인 방법 |
|---|---|---|
| 저장소 | `github.com/facebookresearch/spider` | GitHub API |
| 별 / 포크 | 607 / 70 | GitHub API (2026-11-22 기준) |
| 마지막 push | 2026-11-21 | GitHub API |
| 생성 | 2025-12-09 | GitHub API |
| 라이선스 | `LICENSE` 파일 본문: **CC-BY-NC 4.0** (Meta 저작권). README 배지도 CC BY-NC 4.0. GitHub API는 "Other (NOASSERTION)"으로 표시 | `repo/LICENSE`, README |
| 설명 | "A general physic-based retargeting framework." | GitHub API |
| 복제 크기 | 약 454 MB (`.git` 129 MB 포함, `docs` 44 MB, `figs` 37 MB) | `du` |
| 문서 사이트 | `facebookresearch.github.io/spider` (README 링크) | README |
| 프로젝트 페이지 | `jc-bao.github.io/spider-project`, 페이지 바닥 CC BY-SA 4.0 (웹 템플릿 라이선스, 논문과 별개) | `paper/project_page.html` |
| 데이터 | HF `retarget/retarget_full` (README 공지: 2026-09-22, **7,876 성공 궤적, 2,885 원천 에피소드, 4개 데이터셋, 4개 로봇**), HF `retarget/retarget_example` | README, 받지 않음 |

README 배지의 arXiv 번호는 `2406.12345`로 잘못되어 있습니다(실제 2511.09484). 기록만 하고 고치지 않았습니다.

## 1. 논문과 코드의 대조 (D1–D11)

| 번호 | 논문 주장 | 공식 코드 | 판정 |
|---|---|---|---|
| **D1** | ADD-AUC는 물체 표면 512점을 프레임마다 기준 자세와 달라진 자세로 변환해 평균 L2를 쓴다(식 6, 부록 A.4, A.1에서 회전 포함) | `spider/postprocess/add_auc.py`의 `compute_add_auc`는 **물체 중심 위치**만 씁니다(`achieved_position`, `target_position`, 3차원). `get_success_rate.py`도 "raw world-frame positions"를 씁니다. 512점 표면 샘플링 구현을 grep에서 찾지 못했습니다. 논문 자신도 위치만 쓰면 63.25%, 표면점은 61.48%라고 씁니다(부록 A.4). | **차이.** 공개 코드로는 표 3·7의 ADD-AUC를 재현할 수 없고, 위치 전용 값으로만 확인할 수 있습니다. |
| **D2** | 샘플링 온도 τ = 1.0 (표 5, 표 6; 부록 A.1의 CEM 온도) | `examples/config/default.yaml`: `temperature: 0.1`. `spider/config.py` 기본값 0.3. `examples/config/cpu.yaml`: 0.3. `dexmachina.yaml`, `maniptrans.yaml`: 1.0 | **부분 차이.** 비교 실험용 설정(DexMachina, ManipTrans)은 1.0으로 맞췄지만, 기본 MJWP와 CPU 설정은 다릅니다. |
| **D3** | 대규모 생성: 샘플 1,024, 반복 16 (표 6). 표 5: 샘플 1,024, 반복 32, 호라이즌 2.0 s(Oakink) | `default.yaml`: 샘플 1,024, 반복 32, 호라이즌 1.6 s. **`cpu.yaml`(기본 경로, 2026-11-17 추가): 샘플 128, 반복 4, 온도 0.3, 호라이즌 0.4 s, `final_noise_scale: 1.0`(주석 "no annealing over the iterations")** | **차이.** 논문 설정이 아닌 **훨씬 가벼운 기본 경로**가 공식 기본값이 되었습니다. 이 맥 CPU에서 돌릴 때 쓸 설정으로 적합하지만, 논문 수치 재현 설정은 아닙니다. |
| **D4** | 전체 데이터 6,520 궤적(초록, 표 7), 6,532(서론 §1 기여 목록), 191 식별자 | 프로젝트 페이지: **6,523** 시연, 191 물체. 공개 저장소 README: HF `retarget_full`은 **7,876** 성공 궤적, 2,885 원천 에피소드(2026-09-22 공지). | **불일치 (세 값 + 공개 데이터셋 값).** 논문 내부 6,520과 6,532, 페이지 6,523, HF 7,876은 시점이나 필터 기준이 다른 것으로 보입니다. 필터 기준은 논문 부록 A.4에 있지만 HF 쪽 필터는 확인하지 못했습니다. |
| **D5** | 표 16(ManipTrans 비교): 시뮬 시간 간격 **1/120 s**, 호라이즌 0.6 s = 36 시뮬 스텝 | `examples/config/maniptrans.yaml`: `sim_dt: 0.0166667` (1/60 s), `horizon: 0.6`. 0.6 s ÷ (1/60) = 36 스텝으로 **코드와 맞고 논문 표의 1/120 s와 어긋납니다** | **논문 내부 불일치.** 코드는 1/60 s입니다. |
| **D6** | 식 3은 두 어닐링 파라미터 β1(반복), β2(호라이즌)를 씀. 표 5와 6에는 "noise decay β" 하나만 있음 | `spider/config.py` 408행: `beta_traj = final_noise_scale ** (1/max_num_iterations)`. `spider/optimizers/sampling.py` 384행: `global_noise_scale = beta_traj ** i`. 호라이즌 방향은 `first/last_ctrl_noise_scale`로 선형 보간. 즉 **β1에 해당하는 것만 구현**되고 β2 항은 없습니다(보간 방식으로 대신함). | **차이.** 코드 구조는 논문 식 3의 β1 항과 맞고, β2 항은 호라이즌 선형 보간으로 바꿔 쓴 것으로 보입니다. 논문 식 3과 코드의 대응을 명시하지 않았습니다. |
| **D7** | 속도 시험과 절제는 RTX 4090, 대규모 생성은 L40S (부록 A.3). 표 1 캡션은 FPS를 "one NVIDIA L40S"로 적음 | 코드로 확인할 수 없습니다(장비 설정 파일 없음). | **논문 내부 불일치.** 표 1 FPS가 어느 GPU인지 불분명합니다. |
| **D8** | 강건화(식 4)의 파라미터 집합 `D`: 접촉 여유, 마찰, 질량. 범위 명시 없음 | `spider/config.py` 73–80행: DR 그룹 수 기본 1, `perturb_force/torque` 기본 0. `spider/optimizers/sampling.py` 278행: "domain randomization: pick the minimum reward across all DR parameter sets" 경로가 있으나 `spider/simulators/mjwp.py` 11행은 "no DR groups here; see legacy script for advanced features"라고 씁니다. 범위 값은 찾지 못했습니다. | **차이.** 강건화 경로는 있으나 기본 MJWP에서 꺼져 있고, 논문에 쓴 범위는 코드에 없습니다. |
| **D9** | 접촉 필터: 지속 시간 `t_c,min`, 드리프트 `d_c,max`가 넘으면 제약을 끈다(부록 C, 32쪽) | 키워드 검색(`contact_filter`, `unstable`, `drift`)으로 해당 로직을 찾지 못했습니다(`cpu_test.py`의 `drift`는 테스트 코드). | **미확인.** 찾지 못했다는 뜻이고 구현이 없다는 증명은 아닙니다. |
| **D10** | 접촉 안내 감쇠 γ_g: Oakink 0.8, GigaHands 0.5 (표 5) | `spider/config.py` `guidance_decay_ratio: float = 0.5`. `examples/config/`에서 Oakink 값은 확인 못함 | **대체로 일치.** GigaHands 값은 기본값과 같습니다. Oakink 값은 코드 기본값과 다릅니다. |
| **D11** | 표 8 보상 가중치(손가락 `0.065, 0.065, 0.039, ...`, 합 0.53) "weights are normalized" | 저장소에서 PPO 학습 코드(키워드 PPO, rsl_rl, rl_games, stable_baselines 검색)를 찾지 못했습니다. 보상 가중치 표는 확인할 수 없습니다. | **범위 밖.** 논문 내부에서 합이 1이 아닌 점만 기록합니다(`03` V9). |

## 2. 코드에 있고 논문에 설명된 것

- **샘플링 MPC 핵심:** `spider/optimizers/sampling.py`의 `optimize_once`. 가중치는 `_compute_weights_impl`에서 상위 샘플의 softmax로 구하고(온도 λ), 업데이트는 가중 평균(`ctrls_mean = (weights * ctrls_samples).sum`)입니다. 논문 식 2와 맞습니다.
- **조기 종료:** `terminate_resample`과 "last n steps improvement below threshold". 논문 Algorithm 1 10–12행과 맞습니다.
- **접촉 안내:** `config.contact_guidance`가 켜지면 `scene_act.xml`과 `trajectory_kinematic_act.npz`를 쓰고, 물체 액추에이터 12(양손) 또는 6(한손)을 붙입니다(`spider/config.py` 371–470행). 논문 §3.3의 "가상 제약" 방식과 맞습니다.
- **ADD-AUC 임계값:** 0–10 cm, 100점, 정규화 후 사다리꼴 적분(`add_auc.py`). 논문의 AUC 정의와 맞지만 거리 정의(D1)가 다릅니다.
- **데이터셋 처리:** `spider/process_datasets/`(ARCTIC, GigaHands, OakInk, OakInk v2, HOT3D, FAIR 계열)가 있어 원천 데이터를 불러오는 경로는 논문 A.4와 대응합니다. 사용 방법은 문서 `docs/`에 있습니다.

## 3. 새 CPU 워크플로 (2026-11-17)

- `simulator=cpu`가 기본이고, MuJoCo 3.17 이상의 `mujoco.batch` API로 돕니다. GPU가 필요 없다고 씁니다(`docs/workflows/workflow-cpu.md`).
- 설치는 `uv venv --python 3.12`와 `mujoco>=3.17 torch numpy scipy loguru hydra-core tyro ...` 목록입니다. 설치 목록은 이 분석에서 **설치하지 않았습니다**.
- 장면은 HF `retarget/retarget_example`의 **ARCTIC 277 시퀀스**입니다(다운로드하지 않음). README 공지는 그중 267개가 리타게팅된다고 씁니다.
- 속도: 문서 표 기준 "30 s 클립은 16 스레드에서 약 5분"(유휴 64코어 기준). 이 맥에서 측정하지 않았습니다.
- 손: Sharpa 두 손 양손 과제만 예시로 씁니다(`docs/workflows/workflow-cpu.md`).

## 4. 구현 가능성 판정

**판정: 논문의 수치 재현은 이 맥에서 불가능합니다. 공식 CPU 워크플로로 장면 한 개를 돌리는 것은 조건부로 가능합니다. 설치와 공식 코드 실행 승인이 필요합니다.**

| 무엇 | 이 맥에서 | 근거 |
|---|---|---|
| 표 1·2·3·7 (IsaacGym, Genesis, MuJoCo Warp, L40S 대규모 생성) | **불가.** CUDA·GPU 전제 | 시뮬레이터 설정, 부록 A.3, README |
| 표 17·19 (ManipTrans, DexMachina 비교 재실행) | **불가.** IsaacGym·Genesis GPU 전제 | `maniptrans.yaml`, `dexmachina.yaml` |
| 정책 학습 (PPO 4096 환경, 0.99 대 0.07) | **불가.** GPU 학습, 학습 코드는 이 저장소 밖 | 부록 A.6 (L40 GPU 3–4시간) |
| 실로봇 (Franka + Allegro) | **불가.** 로봇 필요 | §4.3 |
| 공식 CPU 워크플로로 ARCTIC 장면 1개 재생 | **조건부.** MuJoCo 3.17 이상 설치, 장면 다운로드 승인, 실행 승인 필요 | `docs/workflows/workflow-cpu.md` |
| 논문 알고리즘을 numpy/torch로 다시 구현 (Alg. 1, 식 2–3, 접촉 안내) | **가능.** 물리는 MuJoCo CPU 배치로 | 식과 알고리즘이 명확함 (D1, D6, D9는 빈칸 있음) |

노력 추정(측정 전): 공식 CPU 워크플로로 장면 한 개 재생은 설치와 승인 포함 반나절에서 하루. 알고리즘 재구현은 1–2주(추정). 빈칸(β1, β2, D 범위, 접촉 필터 임계값)은 공식 저장소에서 찾을 수 없으므로 그만큼 추가 탐색이 필요합니다.

## 5. 라이선스와 권리 (코드)

- 코드 라이선스: CC-BY-NC 4.0 (`LICENSE` 본문). **비상업 전용**입니다. 논문의 CC BY 4.0과 다릅니다.
- 이 분석의 공개 자료에는 코드를 넣지 않았습니다. 재구현 코드를 공개한다면 원 코드를 옮기지 않은 것으로 확인해야 합니다.
- 데이터셋(HF `retarget/retarget_full`)의 라이선스는 확인하지 않았습니다(받지 않음).
