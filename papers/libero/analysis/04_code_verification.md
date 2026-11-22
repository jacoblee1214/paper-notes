# LIBERO 공식 코드 대조 (정적 읽기)

대상은 `Lifelong-Robot-Learning/LIBERO`의 `master` 얕은 복제(커밋 `8f1084e`, 2025-03-15, 저장소 `repo/`)입니다. 설치하지 않았고, 실행하거나 import하지 않았습니다. 파일은 읽기만 했습니다. 아래 줄 번호는 이 커밋 기준입니다.

## 1. 저장소와 라이선스

| 항목 | 값 | 근거 |
|---|---|---|
| 저장소 | Lifelong-Robot-Learning/LIBERO (공식) | 논문 v2 각주 2, 저장소 README |
| 기본 브랜치, 최신 커밋 | master, `8f1084e` ("Add support for dataset download from huggingface"), 2025-03-15 | `git log -1` |
| 마지막 push | 2025-03-15 (조회 시점 2026-11-22 기준 1년 넘게 갱신 없음) | GitHub API |
| 스타 | 2,563 | GitHub API (조회 2026-11-22) |
| 코드 라이선스 | MIT (`LICENSE`, 저작권 2023 Lifelong Robot Learning) | GitHub API, 파일 확인 |
| 데이터 라이선스 | README 표는 CC BY 4.0 | `repo/README.md` License 표 |
| HF 데이터 카드 라이선스 | apache-2.0 (`yifengzhu-hf/LIBERO-datasets`, 2025-05-18 갱신) | HF API 메타데이터 |
| 얕은 복제 크기 | `repo/` 650 MB (`.git` 225 MB, `libero/libero/assets` 418 MB) | `du` |
| 프로젝트 페이지 | libero-project.github.io (논문·README에 명시) | 내용은 확인하지 않음 |
| 문서 | lifelong-robot-learning.github.io/LIBERO/ (README 링크) | 확인하지 않음 |

**라이선스 문제 하나.** 데이터 라이선스가 README(CC BY 4.0)와 HF 카드(apache-2.0)에서 다릅니다. 어느 쪽이 기준인지 저장소에서 알 수 없습니다. 재배포 전에 확인해야 합니다. 이 분석은 데이터를 받지 않았습니다.

**데이터 크기 (HF API 파일 목록, 다운로드 안 함).** 합계 약 100.4 GB입니다. `libero_90` 66.7 GB, `libero_10` 13.7 GB, `libero_object` 7.4 GB, `libero_goal` 6.4 GB, `libero_spatial` 6.2 GB입니다. 시연 파일은 과제당 hdf5 한 개입니다.

## 2. 논문 ↔ 코드 대조 (D1–D17)

판정: **일치**, **차이**(논문과 코드가 다름), **미기재**(논문에 없고 코드에 있음), **미확인**(코드 근거를 찾지 못함).

| 번호 | 항목 | 논문 | 코드 | 판정 | 근거 |
|---|---|---|---|---|---|
| D1 | 옵티마이저 | "Adam" (부록 D) | `AdamW`, lr 1e-4, betas (0.9, 0.999), weight_decay 1e-4 | 차이 (가중치 감쇠 있음) | `configs/train/optimizer/adam_w.yaml` |
| D2 | ER 메모리 단위 | "과제당 최대 1,000개 **궤적**" (§4.3 주석, 부록 B.1) | `TruncatedSequenceDataset(dataset, n_memories=1000)`: 길이 1,000, 인덱스 0–999의 **시퀀스 창**(길이 10)을 그대로 돌려줌 | 차이 (궤적 대 창, 선택 방식 불명) | `lifelong/datasets.py` 130–139, `algos/er.py` |
| D3 | ER 재생 배치 | "배치마다 32개 궤적 재생" (부록 B.1) | `DataLoader(buf, batch_size=cfg.train.batch_size=32)`를 현재 배치와 합침(총 64) | 차이 (단위는 창, 합치면 배치 64) | `algos/er.py` `start_task`, `observe` |
| D4 | 데이터 증강 | 논문 본문과 부록에 언급 없음 | 기본 `use_augmentation: true`. 색 지터(`batch_wise_img_color_jitter_group_aug`)와 이동 증강(`translation_aug`, 4 픽셀)을 학습 때 씀 | 미기재 | `configs/train/default.yaml`, `configs/data/default.yaml`, `configs/policy/bc_*.yaml`, `lifelong/models/base_policy.py` 67–73, 90–105 |
| D5 | 학습 데이터 분할 키 | 과제당 시연 50개를 학습에 씀 | `data.train_dataset_ratio: 0.8`이 있으나 코드 어디에서도 읽지 않음 (`grep`) | 미확인 (죽은 설정 키) | `configs/data/default.yaml` 39행 |
| D6 | 체크포인트 선택 집합 | "Robomimic 관례에 따라 성공률 최고 체크포인트" (부록 D) | 과제마다 같은 20개 초기 상태로 5 epoch마다 평가하고, 그 최댓값 체크포인트를 저장. 같은 평가 값을 보고함. 별도 검증 집합 없음 | 미기재 (선택과 보고가 같은 집합) | `algos/base.py` 185–231, `metric.py` `evaluate_one_task_success` |
| D7 | 평가 초기 상태 | "20 test rollouts" (부록 D). 어느 초기 상태인지 없음 | 과제별 고정 파일 `init_files/<suite>/<task>.init`(또는 `.pruned_init`)에서 인덱스 `(i·n + k) % N`으로 앞 20개를 씀. 파일 개수 250 (13 MB) | 미기재 | `metric.py` 74–110 |
| D8 | FWT 정의 | 식 (3): 학습 중 11회 평가의 평균, e = 0 포함 | `algos/base.py`가 `successes.sum()/cumulated_counter`를 반환하고, 최적 이후 값을 최댓값으로 채움 | 일치 (정의는 논문과 같음), 표준 FWT와는 다름 | `base.py` 252–257 |
| D9 | NBT, AUC 계산 | 식 (3)의 NBT와 AUC | 저장소에 NBT, AUC 집계 코드가 없음. `result.pt`에 `S_conf_mat`(성공률 행렬)과 `S_fwd`만 저장 | 미기재 (집계는 직접 해야 함) | `lifelong/main.py` 139–142, 212, `grep -rn nbt\|auc` |
| D10 | 시드 | {100, 200, 300} (부록 D) | 기본 `seed: 10000`. 실험 시 덮어씀 | 일치 (실험 설정 문제) | `configs/config.yaml` |
| D11 | 사전학습 체크포인트 선택 | "최고 성능 체크포인트 선택" (§5.2, 그림 5). 기준은 없음 | `pretrain_model_path`로 가중치를 불러오는 코드는 있음. 선택 규칙은 코드에 없음 (수동) | 미기재 | `lifelong/main.py` 160–166 |
| D12 | PackNet | 상위 25% 유지, 미세조정 50 epoch | `prune_perc: 0.75`(제거 비율), `post_prune_epochs: 50`, `post_eval_every: 5` | 일치 | `configs/lifelong/packnet.yaml` |
| D13 | EWC | 온라인 EWC, γ = 0.9, λ = 5·10^4 | `e_lambda: 50000`, `gamma: 0.9` | 일치 | `configs/lifelong/ewc.yaml` |
| D14 | 과제 순서 | Q5: "다섯 가지 과제 순서" (그림 4). 어느 순서인지 없음 | `task_orders`에 순열 21개(인덱스 0은 항등). `task_order_index` 기본 0 | 미기재 (논문이 쓴 다섯 개를 알 수 없음) | `libero/libero/benchmark/__init__.py` 83–106 |
| D15 | 과제 목록과 이름 | 130개, LIBERO-LONG (논문) | suite별 10·10·10·10과 `libero_90` 90개. 이름은 `libero_10`. 과제 BDDL 130개 | 일치 (이름만 다름) | `libero_suite_task_map.py`, `find ... *.bddl` |
| D16 | 아키텍처 하이퍼파라미터 | 부록 표 5–7 | RNN: hidden 1024, 2층, dropout 0. ResNet-T: 임베딩 64, 트랜스포머 4층·6헤드·64·MLP 256·dropout 0.1·최대 길이 10. ViT-T: 임베딩 128, 공간 7층·8헤드·120, 다운샘플 64 | 일치 | `configs/policy/bc_*.yaml` |
| D17 | 학습·평가 설정 | 50 epoch, 배치 32, 코사인 1e-4 → 1e-5, 평가 5 epoch마다, 600 스텝, 20회 | `n_epochs: 50`, `batch_size: 32`, `CosineAnnealingLR eta_min 1e-5`, `eval_every: 5`, `max_steps: 600`, `n_eval: 20` | 일치 | `configs/train/default.yaml`, `configs/eval/default.yaml` |

**정책 이름 대응.** 논문의 ResNet-RNN, ResNet-T, ViT-T는 저장소의 `bc_rnn_policy`, `bc_transformer_policy`, `bc_vilt_policy`입니다(README와 config 확인). 기본 정책은 `bc_transformer_policy`(ResNet-T)입니다. 논문 §5.2가 ResNet-T를 기본으로 쓴다는 설명과 맞습니다.

**계산 환경.** 논문은 CUDA 11.7이라고 씁니다(부록 B). README는 `torch==1.11.0+cu113`을 요구합니다(cu113 대 11.7 차이는 소규모입니다). 학습 시 `device: "cuda"`가 기본입니다.

## 3. 추가로 확인한 것

- **평가가 끝나는 조건.** `metric.py`는 한 번이라도 `done`이면 성공으로 셉니다(`dones[k] = dones[k] or done[k]`). 성공 판정은 BDDL 술어 기반 환경의 `done`에 있습니다. 환경 쪽 판정 코드는 읽지 않았습니다.
- **과제 임베딩.** `task_embedding_format: "bert"`(기본). 논문 §4.4와 맞습니다.
- **FLOPS 맞춤.** 논문은 약 13.5G FLOPS로 맞췄다고 씁니다. 코드에 `compute_flops`가 있지만(`algos/base.py`, `lifelong/utils.py`), 숫자 확인은 하지 않았습니다.
- **재현 명령.** README의 학습 명령은 `python libero/lifelong/main.py seed=SEED benchmark_name=BENCHMARK policy=POLICY lifelong=ALGO`입니다. 평가만 하는 `evaluate.py`도 있습니다.
- **EGL 가정.** 학습과 평가 명령은 `MUJOCO_EGL_DEVICE_ID`를 설정합니다. EGL은 Linux의 헤드리스 렌더링 경로라서 macOS에서 그대로 쓰일지 불확실합니다(§4).

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 벤치마크의 정적 분석과 통계 재계산은 할 수 있습니다(완료). 시뮬레이터 실행은 조건부입니다. 논문 실험의 재현은 이 맥에서 할 수 없습니다.**

| 구성 요소 | 가능 여부 | 근거와 노력 추정 |
|---|---|---|
| 표 재계산, 검정, 선택 편향 몬테카를로 | 가능, 완료 | `recompute.py`, `toy_max_selection.py`, 수 초 |
| 과제 목록, BDDL 파일, 설정 검사 | 가능, 완료 (정적) | 130개 BDDL, 설정 파일 |
| 시뮬레이터 설치와 렌더링 1회 | 조건부 | Python 3.8.13, robosuite 1.4.0, robomimic 0.2.0, torch 1.11(cu113) 고정. Apple Silicon에서 Python 3.8 설치와 EGL 대체 경로는 미확인. 설치 1–3일 (추정, 측정 전). 설치 승인 필요 |
| 데이터 다운로드 | 조건부 | `libero_spatial` 6.2 GB 정도, 전체 약 100 GB. 디스크와 다운로드 승인 필요 |
| 학습 (ResNet-T, 과제 10개 × 시드 3 × 알고리즘 3 × 아키텍처 3, 과제마다 50 epoch) | 불가 | 논문은 A100·A40 전제. 이 맥은 CUDA가 없고, 이 분석의 범위는 GPU 재현을 제외함. 맥 MPS 학습 속도는 측정하지 않음 |
| 평가만 (공개 체크포인트) | 불가 | 저장소에 학습된 체크포인트가 없음. 공개 체크포인트 링크 확인하지 않음 |

**노력 추정 (측정 없음).** 설치와 렌더링 검증 1–3일, 데이터 1개 suite 다운로드 0.5–1일, 맥에서 BC 한 과제 학습 실험은 측정 전이라 수치를 말하기 어렵습니다.

## 5. 결론

- 공식 코드는 논문 설계를 대부분 따릅니다(D8, D10, D12, D13, D15–D17 일치). 차이와 미기재는 D1–D7, D9, D11, D14에 몰려 있습니다.
- 가장 중요한 차이는 D6과 D7입니다. 체크포인트 선택과 보고가 같은 20개 초기 상태의 평가를 쓰고, 이 최댓값이 보고 지표(c_{k,k})가 됩니다(`03` §3).
- D2와 D3은 ER 설명이 코드와 다릅니다. 궤적 대 창, 1,000 대 "처음 1,000개 창"입니다.
- D9는 NBT와 AUC를 저장소에서 재계산할 수 없다는 뜻입니다.
