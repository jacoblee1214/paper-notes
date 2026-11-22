# 04. 공식 코드 대조 (정적 읽기)

- 대상: `github.com/brave-eai/dexjoco`, MIT, HEAD `8d23b0f` (2026-05-29), 얕은 복제는 `repo/`에 있습니다(gitignore).
- 실행, 설치, import 없이 파일만 읽었습니다. `repo/`의 `openpi/`와 `GeoRT/`도 같은 방식입니다. 내려받은 내용은 신뢰하지 않는 데이터로 취급했습니다.
- 대조 기준: 논문 v2 본문과 부록 (§III, §IV-E, 부록 VI–VII).
- 결과: 환경·과제·무작위화 범위·행동 차원은 논문과 대부분 일치합니다. **차이는 두 가지입니다.** (1) GeoRT 학습 코드의 충돌 손실이 스텁입니다(D6). (2) 기준선 학습 설정이 공개되어 있지 않습니다(D1, D10).

## 1. 저장소 구성

| 경로 | 내용 |
|---|---|
| `dexjoco/dexjoco/` | MuJoCo 환경(`sim/envs/panda_*_env.py`, 11개), 과제 상태 복원(`tasks/state_restorers.py`), 자산 XML(Franka, Allegro, 과제별 물체) |
| `dexjoco/dexjoco_openpi_client/` | openpi 서버에 붙는 평가 클라이언트 (`eval_dexjoco_openpi.py`, 기본 에피소드 50, seed 0) |
| `openpi/` | π0.5 학습 설정 (`src/openpi/training/dexjoco_configs.py`, `config.yaml`) |
| `docs/reference_code/dexjoco_dp_eval/` | Diffusion Policy 평가 참고 코드 (학습 설정 없음) |
| `docs/reference_code/dexjoco_lerobot_client/` | LeRobot 비동기 추론 클라이언트. `policy_type ∈ {act, groot}` |
| `scripts/record_demos_zarr.py`, `replay_demos_zarr.py` | 시연 수집과 재생(시각 무작위화 포함) |
| `dexjoco-data-converter/` | LeRobot·Zarr 변환 |
| `teleoperation/` | Rokoko, Vive 브리지, GeoRT(서브모듈) |
| `configs/{rand_obj, rand_full, multi_task, ipad_reasoning}/` | 평가 설정 YAML (과제 11개 × 조건) |

## 2. 대조 결과 (D1–D13)

| 번호 | 논문 주장 (위치) | 코드에서 본 것 | 판정 |
|---|---|---|---|
| D1 | 다섯 기준선(DP-T, DP-C, ACT, π0.5, GR00T N1.5)과 "other hyperparameters are provided in the released codebase" (§IV-E) | **π0.5만 학습과 평가가 모두 있음**(openpi, 행동 헤드 44차원 사전학습 경로 포함). ACT와 GR00T는 LeRobot 추론 클라이언트만 있고 학습 설정이 없음. DP는 평가 참고 코드만 있음 (`n_obs_steps == 2` 가정) | **차이.** "released codebase"가 가리키는 범위보다 좁음 |
| D2 | 과제 11개, 이름 (§III-C, 표 I) | `README.md` 표와 `configs/`의 과제 11개 일치 | **일치** |
| D3 | 행동 22/44차원, 상태 23/46차원, 30 Hz (§IV-E) | README "Policy Training Action and State Layout": `action_rotvec` 22/44, 정책 상태 23/46. `data_fps` 기본 30 | **일치** |
| D4 | 시각 무작위화: 카메라 50포즈, 조명 범위, 텍스처 (§III-D, 부록 VII) | `sim/envs/replay_cameras.npy`, `replay_cameras_2.npy` 모두 형상 (50, 4). 조명 ±0.3, ±0.4, diffuse 0.3–0.8 등이 `panda_fold_glasses_env.py`와 `panda_click_mouse_env.py`에 부록과 같은 값으로 있음. 텍스처는 `randomize_desktop_texture`에서 무작위 선택 | **일치** |
| D5 | 테이블 높이 Δh ~ U(0, 0.05) m (§III-D, 부록 VII) | `uniform(0.0, 0.05)` 문장이 10개 env 파일에 있음. 예: `panda_bimanual_assembly_env.py:363` | **일치** |
| D6 | 동역학 무작위화: 질량, 마찰, 강성 곱 (부록 Table 6) | 질량 U(0.75, 1.25), 집게 마찰 U(0, 0.05), 강성 U(0.75, 1.25), 접이 유리 강성 U(1, 1.5)가 과제 env에 있음. 11개 env 파일 모두 동역학 코드가 있음(water plant는 `randomize_dynamics` 인자와 스프레이 캐시) | **일치.** 범위도 부록과 같음 |
| D7 | GeoRT 재타깃팅, 식 (1): L = L_dir + λ1 L_cover + λ2 L_flat + λ3 L_pinch + **λ4 L_col**, "자기 충돌 회피" (§III-B) | `teleoperation/GeoRT/geort/trainer.py`: 방향 손실, 핀치 손실, 곡률(평탄성) 손실, chamfer 손실(`w_chamfer` 80.0), 충돌 손실(`w_collision` 0.0). 논문 식의 L_cover는 chamfer로 보는 것이 자연스럽지만 **코드에 대응 주석이 없음**(추정) | **부분 차이.** 항의 대응은 추정 |
| **D8** | 충돌 항 λ4 L_col을 식 (1)에 포함 (§III-B) | `trainer.py:326–329`: `# Collision Loss integration pending.` 그리고 `collision_loss = torch.tensor([0.0]).cuda()`. 기본 가중치 `--w_collision` 0.0 (`trainer.py:372`) | **차이 (중요).** 공개 학습 코드에서 충돌 항은 항상 0입니다. 논문 식 (1)을 그대로 재현하면 이 항이 빠집니다 |
| D9 | GeoRT 학습은 짝지은 데이터 없이 가능 (§III-B) | 학습 루프는 무작위 샘플(`torch.randn_like`)로 방향과 곡률을 만듭니다. 의존 장치: `torch.tensor(...).cuda()` 하드코딩 (`trainer.py:329`) | **일치 (개념).** 단, CUDA 없이 실행하려면 코드 수정이 필요 |
| D10 | 청크 길이 DP-T 45, DP-C 43, ACT 100, π0.5 30, GR00T 50 (§IV-E) | π0.5 `action_horizon=30` 일치. DP 참고 코드는 `n_action_steps = horizon − n_obs_steps + 1`로 계산하지만 `horizon` 값은 체크포인트 설정에 있고 저장소에 없음. ACT·GR00T 청크 설정 없음 | **미확인** (π0.5만 일치) |
| D11 | 기준선 학습 하이퍼파라미터는 "released codebase"에 있음 (§IV-E) | π0.5: 배치 32, 단팔 30k 스텝, 양팔 60k 스텝, LoRA (`config.yaml`, `dexjoco_configs.py`). 다른 기준선은 없음 | **부분 일치** (D1과 같은 원인) |
| D12 | 평가: 시드 3개 × 과제당 50 에피소드 (§IV-E) | 클라이언트 기본 `episodes=50`, 기본 `seed=0`. 시드 3개를 도는 스크립트와 결과 집계 코드는 없음 | **미확인.** 실제로 3시드를 돌렸는지는 저장소로 확인 불가 |
| D13 | 시연 과제당 100개, 총 1.1K (§IV-E) | `record_demos_zarr.py`의 `--successes_needed` 기본 2. 데이터셋(HF LeRobot)은 받지 않음 | **미확인** (데이터셋 개수는 HF 카드에서 확인 필요) |
| D14 | 지연 시간 측정, RTX 4090 (표 III) | 측정 스크립트 없음 | **미확인** |
| D15 | 집게 등 과제 설정과 성공 조건 (§III-C) | 상태 복원(`state_restorers.py`)과 env의 성공 판정 코드 존재. 세부 판정은 과제마다 달라서 이번에 전부 읽지 않음 | **부분 확인** |

**D8의 해석.** 논문 식 (1)은 충돌 회피 항을 포함하지만, 공개 GeoRT 학습 코드는 그 항을 계산하지 않습니다. 체크포인트가 이 손실로 학습되었는지는 알 수 없습니다. 기본 체크포인트는 `teleoperation/GeoRT/checkpoint/`에 있는데, 이번에는 열어 보지 않았습니다. 논문 식과 공개 구현이 다르다는 것만 확정할 수 있습니다.

## 3. 공개 데이터와 기타 자료 (정보)

- **π0.5 학습 설정:** 배치 32, 스텝 30k(단팔)·60k(양팔), 행동 헤드 44차원 사전학습 경로. 논문 본문에는 없는 세부 사항입니다.
- **DexJoCo2 (별도 데이터셋):** Isaac Sim 기반 촉각 데이터(2,200 궤적, 10과제, 프로젝트 페이지 기준). 이 논문의 MuJoCo 벤치마크와 다릅니다. 혼동하지 않도록 주의해야 합니다.
- **`feat/lerobot` 브랜치:** `main`과 별도입니다. 이번에는 `main` HEAD만 읽었습니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 수치 재현은 이 맥에서 불가능합니다. 환경 실행과 재계산은 승인 후 가능합니다. 정책 학습은 GPU가 필요하고, 수집은 하드웨어가 필요합니다.**

| 작업 | 판정 | 근거 |
|---|---|---|
| π0.5 학습 (단팔 30k, 양팔 60k 스텝) | **불가** | openpi 학습 설정은 JAX와 CUDA를 전제합니다. 맥에는 CUDA가 없습니다 |
| π0.5 평가 (서버 추론 + MuJoCo 롤아웃) | **조건부 (미검증)** | 체크포인트 다운로드(HF 승인 필요), openpi 서버 추론의 MPS 지원 여부 미확인, 오프스크린 렌더링(`rgb_array`) 미확인. 공식 코드 실행은 승인 필요 |
| MuJoCo 환경 기동과 과제 성공 판정 (공식 코드) | **조건부 (승인 후)** | MuJoCo 3.4.0 설치가 필요합니다. 공식 코드 실행은 현재 승인 대상입니다. 맥에서 헤드리스 렌더링이 되는지는 실행 전까지 모릅니다 |
| 시연 수집 (Rokoko 장갑, Vive 트래커) | **불가** | 하드웨어 필요 |
| 데이터 재변환과 통계 확인 (HF 1.1K 시연) | **조건부** | HF 다운로드 승인 필요. 데이터 크기는 확인하지 않았습니다 |
| GeoRT 재학습 | **조건부 (코드 수정 필요)** | `.cuda()` 하드코딩과 충돌 항 스텁(D8, D9) 때문에 맥에서 돌리려면 수정이 필요합니다. 학습은 CPU나 MPS에서 소규모로 가능할 수 있음(추정) |
| 자체 축소판 (MuJoCo Panda+Allegro 기본 모델 위, 한 과제, CPU에서 작은 정책 학습, 시드 3개) | **가능 (추정)** | 자체 코드로 작성합니다. 공식 코드를 실행하지 않습니다. 노력 추정 1–2주 |

**노력 추정 (추정치, 측정 전):** 공식 환경 실행 검증 1–2일(승인 후), 자체 축소판 1–2주, 공식 파이프라인 재현 GPU 여러 장 기준 수일에서 수주.

## 5. 라이선스와 재사용 조건

| 구성 요소 | 라이선스 | 메모 |
|---|---|---|
| DexJoCo 코드 (`brave-eai/dexjoco`) | MIT | `LICENSE` 파일과 API 표기 일치 |
| GeoRT (`facebookresearch/GeoRT`, `teleoperation/GeoRT`) | **비상업 (업스트림 라이선스)** | README가 번들 구성요소로 명시. 재사용하면 이 조건을 따라야 함 |
| Franka 자산 (`franka_emika_panda`) | Apache-2.0 | DexJoCo README 표 |
| Allegro 자산 (`wonik_allegro`) | BSD-2-Clause | DexJoCo README 표 |
| openpi (`repo/openpi/`) | Apache-2.0 + Gemma 모델 조건 | 가중치 사용 시 Gemma 조건 |
| MuJoCo Menagerie (자산 출처) | 자산별 (NOASSERTION) | 개별 확인 필요 |
| 논문 | arXiv 비독점 배포 | 그림 재사용 불가 (README 라이선스 절 참조) |

## 6. 이 논문의 저장소와 논문의 차이 (요약)

- 맞는 것: 과제 목록, 행동·상태 차원, 제어와 기록 주파수, 무작위화 범위(카메라 50포즈, 조명, 텍스처, 테이블 높이, 동역학), 재생 기반 시각 증강, GeoRT 사용, π0.5 청크 길이와 LoRA.
- 다른 것: GeoRT 충돌 항(D8, 스텁), 기준선 공개 범위(D1, D11).
- 확인할 수 없는 것: 3시드 집계(D12), 1.1K 시연 수(D13), 지연 시간 측정(D14), 기준선 청크 길이(D10).
