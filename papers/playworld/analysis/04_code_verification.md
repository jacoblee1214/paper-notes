# 04. 코드 대조와 구현 가능성 (PlayWorld)

공식 코드로 보이는 저장소는 `irom-princeton/open-world`(프로젝트 페이지 Code 링크)입니다. 커밋 `457942696e322cb24b00450d854fbf9875e01888` (2026-07-20)을 `repo/`에 얕게 clone했고, **정적으로만 읽었습니다.** 실행, import, 설치, 가중치와 데이터 다운로드는 하지 않았습니다. 저장소 안 파일은 모두 신뢰하지 않는 입력으로 취급했습니다. 파일은 664개, 약 36 MB(.git 포함)이고 LICENSE 파일이 없습니다. 이 저장소를 다시 받으려면 `git clone --depth 1 https://github.com/irom-princeton/open-world repo`를 합니다.

## 1. 대조 범위와 첫 판정

저장소 README는 스스로를 "OpenWorld: 세계모델로 로봇 정책을 만들고, 미세조정하고, 평가하는 오픈소스 플랫폼"이라 부르며 PlayWorld라는 이름을 어디에도 쓰지 않습니다(`README.md`). 저장소 전체에서 `curriculum`, `kmeans`, `k-means`, `playworld`, `autonomous play`, `t-SNE`를 찾으면 0건입니다. **논문의 새 기여(놀이 데이터 수집, 곡선 학습, 접촉 벤치마크)에 해당하는 코드는 저장소에서 찾지 못했습니다.** 논문과 대조할 수 있는 부분은 세계모델 구조·학습 틀, 정책 평가 틀, RL 틀입니다.

| 대상 | 파일 |
|---|---|
| 학습 | `openworld/training/world_model/{config,dataset,train_wm}.py`, `configs/training/libero_wm*.py` |
| 정책 평가 | `scripts/run_evaluation.py`, `openworld/runners/evaluator.py`, `openworld/envs/world_model_env.py`, `configs/evaluation/*.yaml` (133개) |
| RL | `openworld/runners/rl_finetune_runner.py`, `openworld/training/{ppo,reward_scorer}.py`, `configs/rl/rl_finetune_pi0_droid.yaml` |
| 데이터 수집 | `scripts/run_data_collection.py`, `openworld/scenegen/` |
| 문서 | `README.md`, `docs/*.md` |

## 2. 논문 ↔ 코드 차이 (D1–D16)

| 번호 | 논문 | 코드 | 위치 | 영향 |
|---|---|---|---|---|
| D1 | 놀이 데이터 수집: VLM 과제 제안기, π0.5-DROID 실행기, 관절 한계 안전 필터, VLM 리셋 지시 (§3.2, 부록 A) | 해당 코드 없음. `scripts/run_data_collection.py`는 LIBERO **시뮬레이터**에서 `pi05_libero` 롤아웃을 저장하는 스크립트이고 VLM 제안기와 리셋이 없습니다 | `scripts/run_data_collection.py` (docstring) | 논문의 핵심 기여인 수집 시스템을 코드로 확인할 수 없습니다. 프롬프트 예시는 논문 부록 A에만 있음 |
| D2 | 곡선 학습: CLIP 중심(K = 5), 거리 순위(R = 5), 순위 분포 스케줄 (식 4–6, 부록 A) | 해당 코드 없음. 학습 데이터 샘플링은 데이터셋 혼합 확률 `prob`(정적)뿐입니다 | `openworld/training/world_model/{dataset,config}.py` | 표 1의 곡선 행을 코드로 만들 수 없습니다. δ_r, 스케줄 보간, 총 스텝 수도 문서에 없음 (`03` §5) |
| D4 | 표 2: 학습률 5e-6, 웜업 100, 코사인 스케줄, 배치 64, 과거 7프레임 | `LiberoWMArgs` 기본 학습률 1e-5(LIBERO 설정 파일도 1e-5), `train_wm.py`에 스케줄러 없음(`torch.optim.AdamW`만, 웜업·코사인 코드 검색 0건), 배치는 GPU당 4, 과거 프레임 `num_history = 6`, 최대 스텝 500,000. 기울기 클립 1.0은 일치 | `training/world_model/train_wm.py` 222, `configs/training/libero_wm.py` | LIBERO 설정이라 DROID 학습 설정과 같다고 볼 수 없지만, 같은 틀의 DROID 설정 파일은 저장소에 없습니다(주석이 외부 Fast-Control-World 저장소를 가리킴) |
| D5 | 추론: 50 디노이징 스텝, guidance 2.0 (표 2) | `ctrlworld.yaml`은 50스텝에 guidance 1.0, `vidwm.yaml`은 4스텝(shortcut)에 guidance 2.0, RL 설정은 10스텝 | `configs/world_models/*.yaml`, `configs/rl/*.yaml` | 설정마다 다름. 표 2와 같은 조합(50스텝, 2.0)을 한 파일에서 찾지 못함 |
| D6 | 정책 평가에서 세계모델 롤아웃의 성공 판정 방법이 본문에 없음 (§4.4) | 평가 스크립트는 Robometer(4B) 성공 확률의 **최댓값 ≥ 0.5**를 성공으로 둡니다(`success_threshold = 0.5`). 평가 설정의 보상 모델은 `dummy`이고 영상 생성 뒤 2단계로 점수화 | `scripts/run_evaluation.py` 133, 265–267, `scripts/score_videos_robometer.py` | 그림 7의 세계모델 성공률이 같은 방식인지는 문서에 없습니다. 임계값 변화에 대한 민감도 분석 없음 |
| D8 | 지표: LPIPS, SSIM (표 1), PSNR, MSE (표 4) | LPIPS와 SSIM 계산 코드 없음(`lpips`, `ssim` 문자열은 `pyproject.toml`과 잠금 파일에만 있고 `.py`에는 없음). 재생 스크립트는 픽셀 MSE와 PSNR(255 대신 [0, 1] 범위)만 출력 | `scripts/replay_libero_wm_traj.py` 308–311 | 표 1을 저장소로 다시 만들 수 없음. 평가 클립 533개의 목록도 없음 |
| D9 | RL: DSRL로 확산 정책의 **잠재 잡음 정책**만 학습 (§4.5, 부록 D 식 8–12, 표 5: UTD 10, τ 0.001, 배치 64 등) | 저장소의 RL은 **PPO**: π0 LoRA(rank 16과 32) + 가치 헤드, flow-SDE 잡음 0.5, 클립 0.2, GAE λ 0.95, 학습률 1e-5, 반복 100회 × 에피소드 1개(주석에 4개). DSRL 구현(잡음 정책, 오프폴리시 SAC류 갱신)은 없음. `dsrl` 이름은 확산 정책 사전학습용 포크(`dppo`) 경로에만 쓰임 | `configs/rl/rl_finetune_pi0_droid.yaml`, `runners/rl_finetune_runner.py`, `scripts/train_dppo.py` | 논문의 RL 실험(그림 9)은 이 저장소로 재현할 수 없음 |
| D10 | 보상: 시연 소수로 학습한 진행도 모델, 인접 프레임 차이 (§4.5, 부록 D) | 보상은 외부 사전학습 Robometer(4B)(`topreward`, `dummy` 대안). 진행도 모델 학습 코드 없음 | `configs/rewards/*.yaml`, `openworld/rewards/` | 보상 설정이 다름 |
| D11 | 행동 조건: 말단 위치(Cartesian) 프레임별 조건 (부록 A). 정책 출력과 데이터 수집의 제어 방식 불일치를 한계로 인정 (§6) | 행동 7차원(말단 6 + 그리퍼 1). 정책 출력(관절 속도 또는 위치)은 `action_adapter`(`model2_15_9.pth`)로 변환하는데 **파일이 저장소에 없음**(설정 파일에서 경로만 참조). 어댑터 학습 코드(`training/action_adapter/`)는 있음 | `configs/evaluation/*.yaml`, `openworld/training/action_adapter/` | 논문 §6의 제어 방식 불일치가 이 어댑터와 관련될 수 있으나 문서에 연결 설명은 없음 |
| D12 | 세계모델은 SVD 백본 한 가지 (§3.3) | 저장소가 **주 트랙**이라 부르는 것은 Wan 1.3B 기반 자기회귀 세계모델(`ar_wan`)이고 SVD는 병행 트랙. 평가 설정 133개 중 `ctrlworld` 124, `vidwm` 6, `ar_wan` 2, `weaver` 1 | `docs/MODELS.md`, `configs/evaluation/*.yaml` | 논문 결과가 어느 설정에서 나왔는지는 저장소 문서에서 확인할 수 없음 |
| D13 | 평가: 과제 3개(그릇·당근, 블록, 수건), 정책당 50회 시뮬 (§4.1, §4.4) | 번들 초기화 세트는 4개(`irom_carrot_pnp` 22, `irom_towel_fold` 24, `rank_carrot` 11, `rank_towel` 10개 초기 장면)이고 **당근과 수건 과제만** 있음(블록 쌓기 없음). 행동 청크 5(`chunk_size: 5`), 기간 20, 5 Hz. 정책 체크포인트는 외부 경로(`checkpoints/policy/...`) | `data/benchmark/`, `configs/evaluation/dp_ctrlworld_v0_10demo_base.yaml` | 초기 장면 수(10–24)가 50회보다 적어 반복 사용했는지 불명. 정책 18개의 체크포인트는 없음 |
| D14 | 데이터: 로봇 놀이 30시간, 사람 시연 6시간 (§4.1) | 프로젝트 페이지 Data 링크(Hugging Face `tennyyyin/playworld_dataset_preview`)의 파일 목록 메타데이터: 폴더 3개(`v0_...2000`, `v1_...2000`, `v2_...2600`)에 주석 2,260 / 1,993 / 2,615개(합 6,868 에피소드), 시점 3개의 영상·잠재, 별도 체크포인트 파일 1개(9.28 GB), 총 약 39.3 GB. 받지 않음 | HF API 메타데이터 | 논문의 30시간 구성(표 1의 6h와 30h 혼합)과 폴더의 대응은 문서에서 확인하지 못함 |
| D15 | 정책 평가 비교 대상(사람 시연, 사람 놀이 세계모델) | 저장소에 해당 학습 데이터 구성, 체크포인트가 없음 | 저장소 전체 | 비교 실험은 저장소 밖 자료가 필요 |
| D16 | 논문에 없는 기능 | 장면 생성(`scenegen`, 이미지 편집 모델 호출), 레드팀 설정, 텔레오퍼레이션, LIBERO 어댑터, 증류(DMD)가 저장소에 있음 | `openworld/scenegen/`, `configs/redteam/`, `docs/TELEOPERATION.md` | 논문의 실험과 관계가 불분명한 범위 |

일치하는 항목: 3시점(`view_order` 3개), 해상도 320 × 192, 예측 5프레임, 5 Hz, 프레임별 행동 조건(`frame_level_cond`), 기울기 클립 1.0, 현재 프레임 + 과거 프레임 조건, ctrlworld 설정의 50스텝 추론.

## 3. 라이선스와 의존성

- 저장소: LICENSE 파일이 없고 GitHub API 라이선스도 null입니다. 사용 권리가 명시되어 있지 않습니다. 서브모듈은 `tenny-yinyijun/openpi`(Apache-2.0 포크)와 `arnavkj1995/WEAVER`(MIT)입니다.
- 기반 코드: Ctrl-World(MIT), dsrl, dppo(MIT), openpi(Apache-2.0), robometer (README 감사의 글).
- 의존성(`pyproject.toml`): Python ≥ 3.11, `torch`, `diffusers==0.34.0`, `jax[cuda12]==0.5.3`(RL 쪽), 정책별 추가 패키지는 서로 충돌하는 두 묶음(openpi 대 molmoact2). 스크립트 상당수가 SLURM 전제(`bash_scripts/*.slurm`)이고 `.cuda()`와 `"cuda"` 문자열이 `.py`에 69곳 있습니다(`grep`).

## 4. 구현 가능성 판정 (구현할 수 있는지)

이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음) 기준입니다. 용량과 시간은 추정이고 측정하지 않았습니다.

**논문 수치의 재현은 불가능하고, 논문의 아이디어를 작은 규모로 다시 만드는 일과 표 수치 재계산은 가능합니다.** 저장소에는 논문의 새 기여 부분 코드가 없어 공식 코드를 돌려도 PlayWorld가 되지 않습니다.

| 수준 | 판정 | 필요한 것과 근거 |
|---|---|---|
| 놀이 데이터 수집 재현 (30시간) | 불가 | DROID 로봇, π0.5-DROID 체크포인트, VLM 호출, 안전 장치. 코드도 없음 (D1) |
| 세계모델 학습 (표 1, 곡선 포함) | 불가 | H200 8장 이틀(§3.3). 모델 약 수 B 파라미터 + AdamW 상태만으로 이 맥 메모리(24 GiB)를 넘음(추정). 곡선 코드와 데이터 목록도 없음 (D2) |
| 정책 평가 (그림 7) | 불가 | 정책 18개, 실로봇 20회씩, 세계모델 체크포인트. 이 맥에서는 통계 재계산(완료)과 잡음 토이(완료)만 가능 |
| RL 미세조정 (그림 9) | 불가 | 실로봇 평가 20회 × 체크포인트, DSRL 코드 없음 (D9) |
| 공식 코드 추론 (공개 체크포인트로 롤아웃) | 조건부 | 체크포인트 다운로드(수 GB), 새 환경 설치, CUDA 전제 코드의 MPS 포팅. 이 목록은 실행 전에 승인이 필요하고, 가중치가 없는 상태에서는 시도하지 않음 |
| 표 1 평가 루틴 (LPIPS, SSIM) 직접 작성 | 조건부 | SSIM은 numpy로 가능. LPIPS는 외부 패키지와 사전학습 가중치 설치 승인 필요 (D8). 평가 클립이 없어 합성 입력으로만 시험 |
| 곡선 알고리즘(CLIP 중심 K-means, 순위 샘플러) | 시작 가능 | 아이디어는 작은 특징 공간 토이로 numpy/torch에서 가능. 실제 CLIP 임베딩은 가중치 다운로드(약 600 MB) 승인 필요 |
| 접촉 다양성이 환각 성공을 줄이는지의 토이 | 시작 가능 | MuJoCo 접촉 장면 + 작은 torch 모델 (설치됨). 규모는 논문과 무관 |

노력 추정(추정, 측정 전): 토이 수준 곡선·커버리지 실험은 1–2일, 접촉 토이와 작은 세계모델 학습은 약 1주. 논문 규모 재현은 실로봇과 H200 급 자원이 필요해 이 맥 계획에는 포함하지 않습니다.
