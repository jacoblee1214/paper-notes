# 04. 공식 코드 대조와 구현 가능성

**대상.** `Shengjiewang-Jason/EfficientZeroV2`, 커밋 `12b9e775df1f8afb4a1dc854cf722ea84fbfb61d`(2024-08-09, "Update INSTALL.md"), GPL-3.0. 얕은 복제본은 `repo/`에 있고 저장소에는 올리지 않습니다.
**방법.** **정적으로만 읽었습니다.** 코드를 실행하거나 import하지 않았고, 패키지를 설치하지 않았고, 가중치나 데이터를 받지 않았습니다. 읽은 파일은 `README.md`, `INSTALL.md`, `requirements.txt`, `scripts/*.sh`, `ez/config/exp/{atari,dmc_state,dmc_image}.yaml`, `ez/agents/base.py`(손실), `ez/agents/models/base_model.py`(정책 출력), `ez/mcts/cy_mcts.py`(행동 샘플링), `ez/utils/loss.py`(연속 정책 손실), `ez/worker/batch_worker.py`와 `ez/worker/data_worker.py`(혼합 목표, 우선순위), `ez/agents/ez_dmc_state.py`(구조)입니다.

---

## 1. 저장소와 관련 자료 표

| 자료 | 역할 | 라이선스 | 별 | 마지막 push | 메모 |
|---|---|---|---|---|---|
| [Shengjiewang-Jason/EfficientZeroV2](https://github.com/Shengjiewang-Jason/EfficientZeroV2) | **공식 코드** (본 논문) | GPL-3.0 | 126 | 2024-08-09 | README 소개: ICML 2024 Spotlight. 복제본 `repo/`(정적 읽기만) |
| 프로젝트 페이지 | 별도 없음 | — | — | — | README의 "Project Page" 링크가 arXiv 초록 페이지를 가리킴 |
| [YeWR/EfficientZero](https://github.com/YeWR/EfficientZero) | 기반 코드 (README에 명시) | GPL-3.0 | 944 | 2023-12-20 | 미복제, 메타데이터만 확인 |
| [danijar/dreamerv3](https://github.com/danijar/dreamerv3) | DreamerV3 기준선 (공식 데이터 출처) | MIT | 4,001 | 2026-10-04 | 미복제 |
| [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) | TD-MPC2 기준선 (저자가 재실행) | MIT | 1,004 | 2026-10-20 | 미복제 |
| [google-research/google-research](https://github.com/google-research/google-research) | BBF 공식 결과 경로 (논문 부록 I.2.6) | Apache-2.0 | 39,245 | 2026-11-20 | 저장소 전체 메타데이터. BBF 폴더만 해당 |
| [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2) | DrQ-v2 기준선 | MIT | 439 | 2022-05-31 (보관됨) | 보관 저장소 |
| [denisyarats/pytorch_sac](https://github.com/denisyarats/pytorch_sac) | SAC 기준선 | MIT | 602 | 2021-12-05 | 논문 부록 I.2.3이 가리킴 |

- 별점과 날짜는 2026-11-21 조회 기준입니다.
- 공식 저장소는 2024년 8월 이후 바뀌지 않았습니다. README 소식란의 마지막 항목도 2024-05-30(코드 공개)입니다.

---

## 2. 저장소 개요 (정적 확인)

- **지원 영역.** 세 설정 파일(`atari`, `dmc_state`, `dmc_image`)과 에이전트 세 개(`ez_atari.py`, `ez_dmc_state.py`, `ez_dmc_image.py`)가 있습니다. 연속 행동 DMC 학습 경로는 `dmc_state`와 `dmc_image`입니다.
- **탐색.** `ez/mcts/cy_mcts.py`의 `sample_actions`는 정책 분포에서 `policy_action_num`=4개, 평평한 분포(표준편차에 `std_magnification`=3을 곱함)에서 `random_action_num`=12개를 뽑습니다. 합계 16이며 논문의 K=16과 맞습니다.
- **손실.** `ez/utils/loss.py`의 `continuous_loss`는 행동 차원이 1이면 논문 식 (8)에 해당하는 항만, 2 이상이면 식 (9)에 해당하는 항만 씁니다(D8).
- **가치 목표.** `ez/worker/batch_worker.py`는 `value_target == 'mixed'`일 때 `start_use_mix_training_steps`와 `mixed_value_threshold`로 TD 목표와 탐색 목표를 고릅니다. `ez/agents/base.py`의 가치 손실 분기도 같은 구조입니다.
- **우선순위.** `ez/worker/data_worker.py`는 `use_priority`가 켜지면 새 궤적의 우선순위를 예측 가치와 목표 가치의 L1 오차로 계산합니다(논문 부록 A의 "우선순위 사전 계산"과 맞음).
- **정책 출력.** `ez/agents/models/base_model.py`는 평균을 `5·tanh(μ/5)`로 자르고, 표준편차는 `softplus(σ+1)+0.1`로 만듭니다(논문 부록 G의 "5배 tanh, softplus"와 맞음. 0.1과 1.0은 논문에 없음, G6).

---

## 3. 논문과 코드의 차이 (D 항목)

표 값은 논문 Table 3과 저장소 설정 파일을 비교합니다. 설정 파일 주석 `# prev`는 저자가 이전 값을 기록한 것이라서, 현재 값은 설정 파일의 값을 씁니다.

| 번호 | 항목 | 논문 | 저장소 | 판정 |
|---|---|---|---|---|
| D1 | 가치 손실 계수 λ3 | 0.25 (Table 3) | `dmc_state` 0.5, `dmc_image` 0.5, `atari` 0.5 (주석 "prev 0.25") | **차이.** 모든 설정에서 바뀌어 있음 |
| D2 | 엔트로피 손실 계수 | 5×10⁻³ (Table 3) | `dmc_state` 5×10⁻² (주석 "prev 5e-3"), `dmc_image`·`atari` 5×10⁻³ | **차이(DMC 상태 기반만)** |
| D3 | 할인율 γ | 0.997 (Table 3, 모든 영역) | `dmc_image` 0.99, `dmc_state`·`atari` 0.997 | **차이(이미지 DMC)** |
| D4 | 재생 버퍼 용량 | 10⁶ (Table 3) | `dmc_state` 10⁵, `dmc_image` 2×10⁵, `atari` 10⁶ | **차이(DMC 두 설정)** |
| D5 | 혼합 목표 스케줄 T1, T2 | T1 = 4×10⁴, T2 = 2×10⁴ (Table 3) | `dmc_state`·`dmc_image` 동일, `atari` T1 = 3×10⁴, T2 = 5×10³ | **차이(Atari)** |
| D6 | 표현 잔차 블록 수(1차원 입력) | 3 블록, 폭 128 (부록 G) | `dmc_state` `num_blocks: 2` (주석 "prev 3") | **차이.** 설정 파일 기준이며, 다른 경로가 같은 값을 쓰는지는 확인하지 않음 |
| D7 | 목표 네트워크 갱신 간격 | 400 (Table 3) | 설정 파일에 같은 이름의 키 없음. 재분석 갱신 200, 자기 대전 갱신 100 | **대응 관계 불명확.** 어느 키가 Table 3의 항목인지 논문과 코드 모두에서 확정하지 못함 |
| D8 | 연속 정책 손실 | 식 (8)과 (9)를 함께 쓰는 것으로 읽힘 (§4.2.2의 "Additionally") | 행동 차원 1: 식 (8)형만, 2 이상: 식 (9)형만 (`loss.py` 79–88행) | **논문이 결합 방식을 명시하지 않음.** 코드는 한쪽만 사용. 코드 주석은 "Eq. 6/7"로 적혀 있어 논문 식 번호와 어긋남 |
| D9 | Atari 재현 | 표 1의 수치 | README: "일부 Atari 환경의 성능이 논문 결과와 맞지 않는다. 계속 확인 중" | **저자 스스로 보고한 불일치** |
| D10 | 기준선 튜닝 | EZ-V2는 튜닝함(부록 A 마지막 항목). 기준선은 저자 권장값, 없으면 튜닝(부록 I.1) | — (코드 차이 아님) | **비교 프로토콜 차이.** 튜닝 예산이 같지 않음 |

**논문에 없는 설정(G 항목, 재현에 필요한 빈칸).**

- G1. 평평한 분포의 `std_magnification`=3. 논문은 "평평하게 만든 분포"라고만 씁니다(§4.1).
- G2. 정책 샘플과 평평한 분포 샘플의 비율(4 : 12). 논문은 "일부"라고만 씁니다.
- G3. 보상 지원 범위의 스케일(`dmc_state`에서 보상 0.01, 가치 0.5). 논문은 51개 구간과 범위만 씁니다.
- G4. 정책 표준편차의 초기 이동 1.0과 최소값 0.1.
- G5. 혼합 목표에서 "최근 전이"의 정확한 정의(`top_value_masks`). 논문 부록 F는 말로만 씁니다.

**D1과 D8의 의미.** D1은 Table 3과 설정이 확실히 다릅니다. 저자가 "prev 0.25"라고 적었으므로, 최종 실험이 0.5로 돌았을 가능성이 큽니다. 논문 Table 3을 그대로 믿고 재구현하면 결과가 달라질 수 있습니다. D8은 연속 행동 정책 목표의 핵심이라서, 재구현할 때 가장 먼저 확인해야 할 부분입니다.

---

## 4. 구현 가능성 판정 (이 맥)

**판정: 공식 코드를 이 맥에서 그대로 돌리는 것은 불가능하고, 알고리즘을 MPS나 CPU용으로 옮겨 쓰는 것은 조건부로 가능합니다. 전체 논문 재현은 이 맥으로는 실질적으로 불가능합니다.**

근거와 세부는 다음과 같습니다.

1. **CUDA 종속.** 77곳에서 `.cuda()`를 직접 호출합니다(9개 파일). 장치 추상화가 없어서, MPS나 CPU에서 돌리려면 호출을 모두 바꿔야 합니다.
2. **오래된 환경.** `requirements.txt`는 `numpy==1.20.0`, `gym==0.22.0`, `ray==1.0.0`, `torchrl==0.1.0`, `wandb==0.14.2`, `protobuf==3.20.1`이고, `INSTALL.md`는 Python 3.8과 PyTorch 2.0.1을 씁니다. 신규 맥 환경과 충돌이 큽니다. 설치 전에는 설치 목록을 먼저 확인해야 합니다.
3. **외부 의존.** `requirements.txt`의 `dmc2gym`은 `git+https://gitclone.com/github.com/...`라는 미러 주소를 씁니다. 원래 저장소(denisyarats/dmc2gym)와 같은지 확인하지 않았습니다. 설치하기 전에 확인할 항목입니다.
4. **C++ 확장.** `ez/mcts/ctree*`는 Cython과 C++로 빌드해야 합니다. 여러 버전(`ctree`, `ctree_v2`, `ori_ctree`)이 함께 있습니다.
5. **계산 비용(논문 Table 7 근거).** 10만 학습 스텝에 2.7시간이 걸린다고 합니다(8×RTX 3090 환경으로 측정). DMC 과제 20개 × 시드 3개 + Atari 26게임 × 시드 수를 합치면 수백 시간 이상의 GPU 시간입니다. 이 맥의 Apple Silicon(MPS)에서는 단일 실행이 더 오래 걸릴 것입니다(추정: 2–4배, 측정 안 함).
6. **로봇과 데이터.** 로봇이나 실제 데이터는 필요 없습니다(시뮬레이션 벤치마크만 씀).

**이 맥에서 가능한 것 (CPU 또는 MPS, 조건부).**

- 탐색 한 단계(샘플링 Gumbel + SVE)만 떼어 작은 MLP 모델로 옮기고, Pendulum 같은 1차원 행동 과제에서 확인하는 것(가능, 수 시간 규모로 추정). 논문의 핵심 식 (6)–(7), (10)의 정책 개선과 편향 주장을 확인할 수 있습니다.
- 공식 코드의 `loss.py`, `cy_mcts.py`를 읽는 것은 이미 했습니다(정적).

**이 맥에서 불가능한 것.**

- 논문 Table 2, Table 1의 전체 재현(DMC 20과제, Atari 26게임, 시드 3개).
- 공식 설정 그대로의 학습(CUDA 필요, 예산 부족).

**노력 추정(분석자 판단, 측정 안 함).**
- 공식 코드를 MPS로 옮겨 DMC 과제 하나를 돌리는 데 약 1–2주(환경 정리 포함).
- 논문 방법을 처음부터 옮겨 쓰기(연속 Gumbel + SVE, 1차원 DMC) 약 2–4주.
- 논문 전체 재현은 GPU 클러스터 기준 수백 시간 이상, 이 맥 기준 사실상 불가.

---

## 5. 라이선스와 사용 조건

- 공식 저장소는 **GPL-3.0**입니다(LICENSE 파일 확인). 기반인 EfficientZero도 GPL-3.0입니다.
- 이 분석 저장소는 코드를 복사하지 않습니다. 공개본에도 코드 조각을 넣지 않습니다.
- 재구현을 공개하려면 GPL 조건(소스 공개 등)을 따를지 결정해야 합니다. 법적 판단은 하지 않았습니다.
- 논문 자체의 라이선스(arXiv non-exclusive)는 `01_paper_analysis.md`와 `README.md`에 적었습니다.

---

## 5. 확인하지 못한 것

- D4의 `dmc_state` 버퍼 10⁵ 설정이 논문 실험의 실제 값인지(설정 파일의 `total_transitions`도 10⁵).
- D6 블록 수가 실제로 어느 모듈에 적용되는지(`ez_dmc_state.py`의 표현 네트워크는 `num_blocks`를 그대로 씀. 다른 경로는 확인하지 않음).
- D7의 두 키 중 어느 것이 Table 3의 "목표 네트워크" 항목인지.
- `scripts/train.sh`에는 특정 GPU 번호(6, 7)와 포트(12399)가 하드코딩돼 있습니다(환경 종속). 이것은 개인 설정 문제가 아니라 저자 환경의 흔적이므로 공개 문서에서는 다루지 않습니다.
- `ez/mcts/ctree*` C++ 확장의 정확성. 읽지 않았습니다.
- 학습 로그, 체크포인트, 시드별 결과. 저장소에 없습니다.
