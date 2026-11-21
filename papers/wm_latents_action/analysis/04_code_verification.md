# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **저장소.** `github.com/k1seul/corl2026-action-relevant-latents` (GitHub 저자 계정 k1seul). 기본 분기 main, 복제한 커밋 `28abafe` (2026-11-04 17:42 +0900). 별 0, 포크 0, 생성 2026-10-10, 마지막 push 2026-11-04.
- **결과 파일.** 저장소의 `results/` CSV 세 개가 원자료입니다. 재계산은 `03` §1에 있습니다.

## 1. 설치 요구사항과 라이선스

- **라이선스.** MIT (`LICENSE`; GitHub API가 `mit`으로 보고). 저장소 README가 `closed_loop/perturbation.py`와 `closed_loop/libero_ood/`는 LIBERO-PRO(MIT)에서 가져왔다고 적었습니다. `closed_loop/scripts/convert_pi0_weights.py`는 openpi(Apache-2.0)의 체크포인트 변환을 감쌉니다.
- **외부 자원.** Hugging Face 백본(Web-DINO, SigLIP 2, VideoMAE, SDXL VAE, Cosmos 토크나이저)은 자동 다운로드입니다. V-JEPA 2, LAPA, CAIP, Dreamer 4는 `external_models/`에서 읽습니다. Dreamer 4 패치는 저장소에 없습니다(README Notes).
- **공개 체크포인트.** Hugging Face의 `Dreamer-VLA` 조직에 네 개가 있습니다(V-JEPA 2 + ID, LAPA + ID, 기타 +ID 백본, Dreamer 4). 이번 분석에서는 받지 않았습니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 논문 | 코드 | 판정 |
|---|---|---|---|
| D1 | ID 손실은 L2 MSE, 평균 풀링 특징에 적용 (§3, 부록 A) | `aux_invdyn_loss`의 기본 `loss_type="l2"`, 풀링은 토큰 평균 (`src/training/aux_loss.py`) | 일치. L1 변형도 코드에 있음(부록 A 표 6과 일치) |
| D2 | 순방향 동역학: (z_t, a_t) → z_{t+1}, 목표는 정지 그래디언트 (부록 M) | `ForwardDynAuxHead`: 행동을 64차원으로 선형 임베드한 뒤 MLP. 목표는 `.detach()` | 일치. 행동 임베드 차원 64는 논문에 없음 |
| D3 | 과제 안 행동 섞기 (derangement, 부록 M) | `train_matched_control.py`의 기본 `within_task_cross_episode_derangement`. 순환 이동(`circular_shift`)은 민감도 조건으로만 씀 | 일치. 순환 이동 코드의 주석도 같은 기본값을 밝힘 |
| D4 | 프로브 MLP [D→256→128→7], 3000 스텝, AdamW lr 3e-4, 배치 512 (§3, 부록 M) | `eval_pipeline.py`: `probe_steps=3000`, `hidden_dims=(256,128)`, `batch_size=512`, `lr=3e-4` | 일치. **가중치 감쇠 1e-4는 논문에 없음** |
| D6 | ID 헤드 구조와 클립 길이 (논문에 없음) | 헤드 은닉층 (512, 256), GELU. 클립 길이 32 (`--clip-len` 기본) | 논문 누락. 재현하려면 필요 |
| D7 | DIFF + ID: λ = 0.05, 배치 8, lr 1.5e-4, 100k 스텝, 91M (부록 G) | `configs/diff_id.yaml`: `aux_invdyn_lambda 0.05`, `batch_size 8`, `lr 0.00015`, `max_steps 100000`, EMA 0.9995, 해상도 128 | 일치. 워밍업 5000과 EMA 0.9995는 논문 본문에 없음 |
| D8 | 폐루프 목표: 클립 결합 가중치 0.1, 토큰 결합 0.01, 목표 클립 o_t..o_{t+16}, 머리 2048 (§4.6, 부록 N) | `closed_loop/train.py`: `--aux-weight` 기본 0.1 (클립) 또는 0.01 (토큰), `--clip-horizon 16`, `--head-hidden 2048` | 일치 |
| D9 | π0.5 미세조정: 10k 스텝, 배치 32, 최대 lr 1.2e-4, 워밍업 200, bf16 (부록 N) | `--num-train-steps 10000`, `--batch-size 32`, `--peak-lr 1.2e-4`, `--warmup-steps 200` | 일치. **가중치 감쇠 기본 0.01은 논문에 없음** |
| D10 | 평가: 과제 10개 x 10회, 스위트당 (부록 N) | `eval_libero_pro.py`: `num_trials_per_task=10`, 재계획 5스텝 | 일치. 재계획 주기는 논문에 없음 |
| D11 | 평가 중 로드 실패 (논문에 없음) | 환경 로드 실패는 해당 시도를 **실패로 셈** (`logging.warning` 뒤 `successes 0`) | 논문 누락. 실패 횟수가 결과에 섞였을 수 있음. 로그가 공개되지 않아 크기를 알 수 없음 |
| D12 | 매칭 대조의 유효 배치 16 (부록 M) 대 표 13의 V-JEPA 유효 배치 4 (부록 G) | `train_matched_control.py`는 마이크로 배치 4와 그래디언트 누적으로 유효 16을 씀 | 일치 (논문 두 곳에 각각 적힘). 표 1과 표 2의 조건 차이로 `03` K12 |
| D13 | 프리즈 V-JEPA 프로브 값 0.400 (표 1) 과 0.455 (표 2) | README Notes: 데이터 로더 워커 수, 인코드 배치 크기, 환경에 따라 값이 바뀜. 다른 기기에서 0.455 | 설명됨. 절대 프로브 값의 환경 민감도(약 ±0.06) 확인 |
| D14 | 실로봇, 기준선 SysID와 PhysPred 없음 | 해당 없음 (이 논문 범위 밖) | 해당 없음 |
| D15 | CALVIN과 MetaWorld 분할 (부록 D) | `scripts/build_{calvin,metaworld}_{splits,manifest}.py`와 `build_*_splits.py` | 스크립트 존재. 데이터는 공개 다운로드 필요 |
| D16 | 표 2 원자료 재측정 (GCP VM, 제출 평가기) | `results/matched_controls_per_cell.csv` (63행) | 일치 (`03` K3) |
| D17 | 표 4와 표 20 원자료 | `results/closed_loop_success_by_suite.csv`, `..._by_task.csv` | 일치 (`03` K1, K2) |
| D18 | 표 6, 표 5, 표 9의 L2 λ = 0.05 값 | 코드로 확인 불가. 원자료에 λ 스윕 결과 없음 (`paper_figures/data` 참조) | 논문 내부 불일치 (`03` K9) |

### 2.1 논문 주장의 코드 쪽 확인

- **"짝짓기가 이득을 만든다" (표 2)의 대조 구현.** 공식 대조는 과제 안 교차 에피소드 derangement입니다(D3). 논문 설명과 같고, 순환 이동은 보조 조건입니다. 따라서 표 2의 "섞은 ID"는 논문이 말한 조건과 같습니다.
- **폐루프 대조.** `closed_loop/train.py`의 `--shuffle-target`은 배치 안에서 목표를 섞고, `--random-init-target`은 같은 구조의 무작위 초기화 인코더를 씁니다. 논문 설명과 같습니다.
- **재현 가능성.** 논문 주장의 대부분은 저장소에 설정과 원자료가 있어 정적으로 확인 가능합니다. 다만 Dreamer 4 패치, 실로봇 자료, 체크포인트 원본 파일 해시(표 2 주석은 "byte-level weight hash"를 확인하지 않았다고 적음)는 없습니다.

## 3. 안정성과 누락 점검

- **시드 처리.** 학습 스크립트 기본 시드는 42이고, 매칭 대조는 시드 0과 1을 씁니다. 폐루프 시드 1 실행은 DIFF + ID에만 있고 LIBERO 표준 평가는 없습니다(README와 RESULTS.md 일치).
- **결과 파일 존재.** RESULTS.md의 표 1, 표 2, 표 3, 표 4는 논문 v2 값과 일치합니다. RESULTS.md는 "카메라 준비판 논문에서 옮긴 값"이라고 적었습니다. 저장소의 `paper_figures/data/results.jsonl`과 `decoder_results.jsonl`은 이번 분석에서 다시 대조하지 않았습니다.
- **데이터 분할.** `data/SPLITS.md`에 과제 분할 설명이 있습니다. 분할 파일은 `runs/corl2026_rebuttal/splits/`에 해시와 함께 있습니다(`*.sha256`). 해시는 확인하지 않았습니다.

## 4. 구현 가능성 (이 맥)

판정 **부분 가능.** 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 논문의 학습과 폐루프 결과를 재현하는 것은 불가능합니다. 원자료 재계산과 정적 대조는 이미 끝났습니다. 프로브의 일부는 조건부로 가능합니다.

| 항목 | 필요 자원 | 이 맥에서 | 근거 |
|---|---|---|---|
| 표 재계산과 검정 | CSV, numpy, scipy | 가능 (완료) | `03` §1 |
| 정적 코드 대조 | 저장소 텍스트 | 가능 (완료) | §2 |
| 프리즈 인코더 특징 추출과 프로브 재학습 (LIBERO 4개 과제) | V-JEPA 2 ViT-L 체크포인트, LIBERO HDF5 데모 | 조건부. 데이터와 가중치 다운로드 승인 필요. MPS 속도 미측정 | 저장소가 특징 파일을 제공하지 않음. 특징 추출은 순전파만 |
| 26개 과제 프리즈 커버리지 점검 (표 19) | 1300 보류 에피소드의 특징 추출 | 조건부 (위와 같음). 특징 저장 용량이 커질 수 있음 | 표 19의 "26 tasks" 열 |
| DIFF와 DIFF + ID 학습 (100k 스텝) | GPU | 불가 | `configs/diff_id.yaml` |
| 폐루프 π0.5 목표 교체 | GPU, openpi, LIBERO 시뮬레이터 | 불가 | `closed_loop/README.md` |
| CALVIN, MetaWorld 프로브 | 데이터 다운로드, 특징 추출 | 조건부 (데이터 크기 미확인) | 부록 D |

**노력 추정(근거 없는 추정, 측정 전).**

- 프리즈 프로브 재구현 (LIBERO 4개 과제, 특징 추출 포함): 3–5일.
- 26개 과제 커버리지 점검: 추가 3–5일 (특징 저장과 디스크 용량이 병목).
- 매칭 대조 한 행을 재학습: GPU 한 장 기준 수 일에서 1–2주 (추정).
- 폐루프 재현: 공식 저장소 기준으로 GPU 여러 장과 시뮬레이터 설정이 필요합니다. 이 맥에서는 현실적이지 않습니다.

## 5. 재실행

이번 분석에서는 공식 코드를 **실행하지 않았습니다.** 재계산은 저장소의 CSV만 읽습니다.

```bash
git clone --depth 1 https://github.com/k1seul/corl2026-action-relevant-latents repo
python3 analysis/recompute.py
```
