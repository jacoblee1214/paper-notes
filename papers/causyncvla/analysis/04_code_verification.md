# 04. 공식 코드 대조 (정적 읽기)

**방법.** 공식 저장소 `TwSphinx54/CauSyncVLA`를 얕게 복제해 `repo/`에 두고(커밋 `22336bf8d1a4`, 2026-11-20), 파일을 **읽기만** 했습니다. 설치, 실행, import, 가중치 다운로드를 하지 않았습니다. 저장소 내용은 신뢰하지 않는 데이터로 취급했습니다. 대조 대상은 논문 본문 식 (1)–(5), Table 7(구조와 학습 설정), §3.3(PFWS), §B.9(교사 누설), §C.9(실로봇 프로토콜)입니다.

**전체 판정.** 설계 골격(앵커 잔차 게이트, 0으로 시작하는 투영, 이력 조건화, 두 손실의 가중합, 교사에 `no_grad`, 전문가 동결, ShiftRepeat와 ρ = 0.5 재노이즈, 한 스텝 추론, H/C = 10/5, 이력 48, 나이 1–45 샘플링)은 논문과 맞습니다. 불일치는 **문서화와 선택 규칙**에 있습니다(D4, D5). 실로봇 쪽은 가중치가 비공개라서 대조할 수 없습니다(D9, D10).

## 1. 저장소 구성

- `src/openpi/dual/`: 본 방법(`model.py`, `history.py`, `rose_tfws.py`, `c1_engine.py`, `train.py`, `data.py`, `eval_libero.py`, `realtime.py`, `deadline_dispatch.py`, `action_metrics.py` 등). 약 4,700행.
- `franka/overlay/openpi/`: 실로봇 실행 경로(FR3 정책, 주파수 조정, 서버). 가중치는 없음.
- `configs/`: `libero.json`(추론), `train_libero.json`/`init_libero.json`(학습 인자), `franka.json`(실로봇).
- `docs/EXPERIMENTS.md`, `docs/PAPER_RESULTS.md`, `README.md`, `franka/README.md`.
- `tests/test_contracts.py`: 계약 테스트(실행하지 않음).

## 2. 대조표 (D)

| 번호 | 항목 | 논문 | 저장소 (파일:위치) | 판정 |
|---|---|---|---|---|
| D1 | 식 (2) 잔차 게이트와 항등 초기화 | §3.2, 부록 A.1: 0 초기화 투영, I[k>0] 게이트 | `model.py` L58–59(AdaLN 변조 0 초기화), L95–96(출력 투영 0 초기화), L143(`condition[:, :1] > 0` 마스크 게이트) | **일치.** 게이트의 조건 성분 인코딩은 실행하지 않아 세부는 미확인 |
| D2 | 식 (3) PFWS: ShiftRepeat와 재노이즈 | §3.3, ρ = 0.5, NFE = 1 | `rose_tfws.py` L32–44(`shift_repeat_endpoint`: 왼쪽 시프트, 마지막 행동 반복), L48–70(`resolve_initial_latent`: 전 위치 재노이즈, 시작 시각 `restart_time`), `configs/libero.json` `restart_time: 0.5`, `warm_nfe: 1` | **일치** |
| D3 | PFWS의 상태 상대 재기준화 | §3.3: Q_t를 s_t로 재기준화 | `history.py` L23, L56(이력 상태는 s[t0]에 대한 상대 변위). Q_t 자체의 재기준화를 하는 명시 함수는 찾지 못함 | **확인 못함.** 구현 위치를 더 찾아야 함 (`c1_engine.py`, `realtime.py`) |
| D4 | 학습 프레임워크 버전 | Table 7: PyTorch 2.8.0/CUDA 12.8. Table 8: 체크포인트 스윕과 모듈 대조는 Torch 2.6/CUDA 12.4, 실로봇 학습은 RTX 5090 | `pyproject.toml`: `torch==2.7.1`. README: CUDA GPU와 Python 3.11 | **불일치(문서).** 세 가지 버전이 나옴. 결과에 영향이 있을 수 있음 (§C.1은 백엔드 차이를 적음) |
| D5 | 체크포인트 선택 규칙 | §A.6, §B.5, Fig. 8b: 12k 선택 (native 성공률이 최고) | `train_libero.json`: `--selection-metric val/action_weighted`, `--milestone-steps 6000,12000,18000`, `--keep-best-checkpoint`. 같은 설정으로 고르면 Table 11 기준 18k(0.00648 < 0.00686) | **불일치.** 공개된 규칙으로는 18k가 뽑힘. 보고는 12k. 선택이 시험 세트에서 이뤄졌다는 점을 저자도 인정 (§B.5). 재현 시 주의 |
| D6 | 손실 가중 | Eq. 4, Table 7: L_KV 1.0, L_act 2.0, 그리퍼 0.1 | `train.py` L134–172 `compute_losses`: `kv_weight*kv + action_weight*act`, 기본 1.0, 2.0, `grip_weight` 0.1. `train_libero.json`: `--kv-weight 1.0 --action-weight 2.0 --grip-weight 0.1` | **일치** |
| D7 | 교사와 전문가 동결 | §3.4: 교사 A★ = sg[·], 전문가 동결 | `train.py` L104, L139(`with torch.no_grad()`, "Teacher labels use no_grad; the student hand MUST remain in this graph"), L166 주석(전문가 파라미터는 동결, KV 입력은 기울기 경로) | **일치** |
| D8 | 나이 샘플링과 이력 상한 | Table 7: 1–10 (80%), 11–45 (20%), K = 48 | `train_libero.json`: `--age-lo 1 --age-hi 45 --age-core-lo 1 --age-core-hi 10 --age-core-prob 0.8`, `--history 48`. `history.py` L24, L40: 이력 밖의 나이는 자르지 않고 거부(실시간 경로는 재갱신). `configs/libero.json`: `max_age: 48` | **일치** |
| D9 | 실로봇 기반 백본과 전문가 | §C.9: 원래 행동 전문가 동결, 시간 조건 업데이터 적응 | `docs/EXPERIMENTS.md`: FR3 기반은 "Lab-finetuned π0.5, state tokens enabled". `franka/README.md`: "privately distributed FR3 backbone". 업데이터·대응 전문가·학습 기록은 "privately held" | **재현 불가 (비공개).** 논문 설명과 저장소 설명이 같은 것을 가리키는지는 공개 자료로 확인되지 않음 |
| D10 | 실로봇 가중치와 데이터 | §C.9: 205 demonstrations, 35,198 학습 행, 4k 적응 갱신 | `docs/EXPERIMENTS.md` "Data access": 실로봇 업데이터와 녹화 파일 비공개. 공개 체크포인트는 LIBERO만 (`hf download … --include 'libero/*'`) | **재현 불가** (가중치 비공개) |
| D11 | 구조 설정 수치 | Table 7: 6블록, 폭 384, 헤드 4, 18층 × 256, 텍스트 32, 상태 8, 행동 7 | `configs/libero.json` `updater`: layers 18, head_dim 256, hidden 384, depth 6, heads 4, text_tokens 32, state_dim 8, action_dim 7. `train_libero.json`: `--hidden-dim 384 --depth 6 --heads 4` | **일치.** 파라미터 수 24,273,024는 모델을 만들어야 확인 가능하므로 미검증 |
| D12 | 배치와 갱신 구성 | Table 7: 쌍 8/GPU 마이크로배치, 전체 32, 앵커 2 × 목표 4, 누적 2 | `docs/EXPERIMENTS.md`: DDP 2개 랭크, 앵커 스냅샷 2, 목표 4, 누적 2, 유효 배치 32. `train_libero.json`: `--batch-snapshots 2 --targets-per-snapshot 4 --grad-accum-steps 2` | **일치** |
| D13 | 학습률 과 갱신 수 | Table 7: 최대 5e-5, 워밍업 200, 코사인, 선택 12k, 완료 18k | `train_libero.json`: `--lr 5e-05 --warmup 200 --lr-cycle-steps 6000 --steps 18000 --milestone-steps 6000,12000,18000` | **일치** (선택 규칙은 D5) |
| D14 | 평가 설정 | §4.1: LIBERO-10, 500 starts, 시드 7, 520 행동 예산 | `README.md` 평가 명령: `--benchmark native --tasks all --trials 50 --seed 7 --max-steps 520`. `eval_libero.py` | **일치** |
| D15 | LIBERO-Plus 설정 | §C.5, Table 15: 2,519 변형 | `README.md` 문장이 "For LIBERO-Plus, select its cor…"로 중간에서 끊겨 있음. 해당 설정 파일 없음 | **문서 미완** |
| D16 | 시간 고정 20 Hz 평가 | §4.2, Table 2, §C.10 | `docs/EXPERIMENTS.md`: "The timed protocol uses a separate scheduler" (측정 기준 기록 필요). `deadline_dispatch.py`, `c1_clock.py` | **구현 있음.** 결과 재현은 하지 않음 |
| D17 | 행동 경계와 그리퍼 뒤집힘 지표 | Table 4, 14: 경계·내부 뒤집힘 | `action_metrics.py`: `GRIPPER_DEADBAND = 1e-6`, 중립 전이는 뒤집힘으로 세지 않음, 경계와 내부 분리(`_transitions`). 논문 정의와 완전히 같은지는 본문에서 확인 못 함 | **구현 있음**, 정의 대조는 부분 |
| D18 | 라이선스 | 논문: 공식 코드 공개 | `LICENSE`: Apache-2.0. `NOTICE`, `LICENSE_GEMMA.txt`(Gemma 약관), OpenPI는 상위 약관. HF 모델은 라이선스 메타데이터 없음 | **확인됨.** HF 가중치는 조건 불명 |

## 3. 중요 세부

- **교사 누설(§B.9).** 학습 쌍에는 나중 기록 상태의 실제 목표 인코딩이 들어 있지만, 교사 레이블로만 쓰이고 배포 시 쓰이지 않습니다. 코드는 `no_grad`로 교사를 막아 이 주장과 맞습니다. 다만 학습 시 PFWS 사전 분포가 "이전 신선 교사 끝점"에서 오므로 배포의 재귀 상황과 다르다고 저자가 인정합니다(§B.9). 저장소의 `cache_boundary_context` 설명(`train.py` L646)과 일치합니다.
- **평가와 학습 정보.** 논문은 "검증 분할과 정규화는 평가와 분리된다"고 적었습니다(본문 부록 B.13 주변). 그러나 학습 기록(expert 600/300/300/300 궤적)의 초기 상태가 평가 500 starts와 겹치지 않는지는 명시되어 있지 않습니다. 분할 파일 `data/libero/split.json`은 저장소에 포함되어 있지 않아 확인할 수 없습니다.
- **실시간 경로.** `realtime.py`와 `c1_engine.py`는 "최신 완료 실제 prefix"와 "이미 실행된 명령만 입력"을 강제합니다(`c1_engine.py` L42: `frame.step − snap.frame.step > max_age`이면 실제 prefix로 대체). 이는 §B.9의 "배포 시 미래 정보 없음" 주장과 맞습니다.

## 4. 구현 가능성 판정 (이 맥)

**판정: 논문 수치의 재현은 이 맥에서 할 수 없습니다. 구조 재구현은 가능하지만 CUDA 학습과 공개 데이터가 필요합니다.**

| 항목 | 이 맥에서 | 이유 |
|---|---|---|
| 공식 코드 설치 | 불가 (현재 환경) | `pyproject.toml`이 `torch==2.7.1`을 고정하고 README가 Linux, CUDA GPU, Python 3.11을 요구. `transformers_replace` 파일을 site-packages에 복사하는 단계가 있음. 설치는 이 분석에서 하지 않음(승인 필요) |
| LIBERO 평가 | 불가 | `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` 실행 명령, `torch.compile` 모드 `max-autotune-no-cudagraphs` 설정. CUDA 전제 |
| π0.5-LIBERO 체크포인트 변환 | 조건부 | 체크포인트 다운로드와 변환 필요. 저장소의 `convert_jax_model_to_pytorch.py`는 OpenPI 체크포인트 입력을 받음. 용량과 승인 필요 |
| 업데이터 학습 (LIBERO) | 불가 | 2 GPU DDP 전제, 체크포인트 18k 갱신, 공개되지 않은 expert 기록(600/300/300/300 궤적) 필요 |
| 실로봇 재현 | 불가 | FR3, Robotiq, D405 두 대, 비공개 백본·업데이터(D9, D10) |
| 구조 재구현 (PFWS 식 (3), ShiftRepeat, 예측기 구조) | 가능 (작은 규모) | torch CPU 또는 MPS로 작은 flow 정책 위에 붙이는 방식. 노력 추정 1–2주. 논문 수치와 직접 비교는 불가 |
| 재계산과 검정 | 완료 | `analysis/recompute.py` |

**노력 추정 (근거 없음, 추정).** 구조 재구현 1–2주. 공식 코드를 CUDA 환경에서 LIBERO 평가까지 돌리는 데 3–5일(환경과 체크포인트 준비 포함). 논문 수준 학습은 2 GPU에서 수일 규모로 보이나 측정하지 않았습니다.
