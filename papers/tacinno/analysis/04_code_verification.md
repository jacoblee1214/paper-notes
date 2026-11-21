# 04. 공식 코드 대조: TacInno

공식 저장소 `LisavilaLee/TacInno`를 얕게 복제해 **정적으로만** 읽었습니다 (`repo/`, gitignore). 실행, 설치, import, 테스트 실행, 체크포인트와 데이터 다운로드를 하지 않았습니다. 복제 커밋은 `e3c81f9`(2026-11-20, "policy.md: state the EMA and the demonstrations of the released checkpoints precisely")입니다. 저장소 안의 파일 경로와 줄 번호로 인용합니다. 코드와 설정은 신뢰하지 않는 외부 자료로 취급했습니다.

## 1. 라이선스와 출처

- 저장소 LICENSE: Apache-2.0 (GitHub API `apache-2.0`). `NOTICE`에 제3자 구성요소가 나열되어 있습니다. 주요 항목은 openpi(Apache-2.0, `third_party/openpi/LICENSE`)와 openpi_client입니다.
- 체크포인트는 π0.5에서 미세조정했으므로 README는 "π0.5와 PaliGemma 구성요소의 라이선스, 특히 Gemma 이용 약관"을 따른다고 적습니다. 가중치는 받지 않았습니다.
- 시연 데이터(약 112 GB)와 체크포인트(약 53 GB, 정책 한 개당 약 7.5 GB)는 Hugging Face에 있습니다. 라이선스는 확인하지 않았습니다 (받지 않음).
- 결론: 코드는 Apache-2.0이므로 이 저장소 안에서 읽고 비교하는 데 문제가 없습니다. 실행 승인과 가중치 다운로드 승인은 별도로 필요합니다 (§4).

## 2. 논문 항목과 코드의 대조

| 항목 | 논문 | 코드 (위치) | 판정 |
|---|---|---|---|
| 토크나이저 시점 | t, t+5, t+10, t+15 (§3.1) | `configs/_base_/models/tacinno_tokenizer.py` `offsets=(0, 5, 10, 15)` | 일치 |
| 16비트 코드 | 4 슬롯 × 16값 (§3.2) | `num_slots=4, codebook_size=16` | 일치 |
| 인코더·디코더 조건 | 둘 다 영상과 행동 조건 (§3.2, 식 3) | `context='both'` (`tacinno/models/tokenizer/tokenizer.py`) | 일치 |
| 디코더 목표 | ΔX_{t+5k} = X_{t+5k} − X_t (식 1, 3) | `target='cumulative'` (`losses.py`의 `change_targets`) | 일치 |
| 접촉 가중 L1 | "contact-weighted L1" (§3.2) | 가중 1 + 4 × 접촉 (`losses.py`, `contact_weight=4.0`) | 일치 (가중치 값은 논문에 없음, D6) |
| VQ 슬롯별 코드북 | 슬롯마다 코드북 16개 (§3.2) | `quantizer.py` 슬롯별 코드북, `commit_weight=0.25`, 주기적 리셋 | 일치 (리셋과 커밋 가중치는 논문에 없음, D4) |
| 기반 모델 | π0.5, PaliGemma 2B + Gemma 300M 행동 전문가 (§3.3) | `paligemma_variant="gemma_2b"`, `action_expert_variant="gemma_300m"` (`configs/_base_/models/tacinno_vla_pi05.py`) | 일치 |
| 촉각 토큰 60개 | 세 프레임 t−8, t−4, t (§3.3) | `tactile_history=(8, 4)`, `tactile_tokens_per_frame=10`, 젤 2개 → 3 × 2 × 10 = 60 (`condition_tokens.py` docstring) | 일치 |
| 코드 쿼리 4개 | 학습 가능 쿼리 4개 (§3.3) | `use_code_queries=True`, `num_code_slots=4` | 일치 |
| 주의 마스크 | 코드 쿼리는 아무도 보지 않음 (§3.3) | `code_to_action=False` (`tacinno_vla.py` 설명, `condition_tokens.py` 주의 설명) | 일치 |
| 코드 손실 | 슬롯별 분류 교차엔트로피 평균 (식 4) | `code_ce_loss` (`condition_tokens.py`), 프레임 유효성 마스크 포함 | 일치 (유효 프레임 마스크는 논문에 없음) |
| 접촉 상태 손실 | 접촉 면적, 전단, 압입 깊이 등 (§3.3) | 14개 표준화 속성, MSE와 BCE 혼합 (`condition_tokens.py` L40) | 일치 방향, 세부 다름 (D7) |
| 전체 손실 | L = L_flow + λ_contact L_contact + λ_code L_code (식 5) | `tacinno_vla.py` 설명, `forward`의 `aux_loss`, `code_ce` | 일치 |
| λ 보정 | 시작 시 보조 손실이 L_flow의 15% (§3.3) | `code_loss_weight=0.119`, `contact_loss_weight=0.467` (Lift Can 보정 설정). 기준 시점이 다름 | 방향 일치, 기준 다름 (D3) |
| 배포 시 제거 | 토크나이저와 두 보조 헤드 제거 (§3.3) | `docs/en/policy.md`와 `tacinno/apis/server.py` (배포 모델에 촉각 토큰 60개만) | 일치 |
| 흐름 매칭 샘플링 | 명시 없음 | 10 Euler 단계 (`tacinno_vla.py` 설명) | 논문에 없음 (D12) |
| 행동 청크 길이 | H 단계 (§3.1) | `action_horizon=32` | 값은 코드에만 있음 (D9) |
| 고유감각 토큰 | q_t 하나 (§3.3) | 설정 `proprio_history=8` (Lift Can 시뮬만), 나머지 0 | **다름** (D1) |
| 학습 절차 | 모든 정책은 같은 50 시연에서 학습 (§4.1, §4.6) | 두 가지 절차가 섞여 있음 (D2) | **다름** (D2) |
| 촉각 가리기 검사 | 촉각 입력을 고정된 접촉 없음 프레임으로 교체 (§4.4) | `tacinno/apis/server.py` L17–18, `rest_frame` | 일치 |
| 실로봇 시연과 학습 분할 | 앞 45개가 정책과 토크나이저 학습 (§4.1) | `configs/real_robot/…`의 데이터 설정과 README | 일치 |
| 시뮬 시연 분할 | 0–49 정책, 0–89 토크나이저, 90–99 분석 (§4.1) | README "Train from scratch", `data/univtac` 레이아웃 | 일치 |
| 시뮬 평가 시드 | 같은 시드, 100 에피소드 (§4.1) | `PAPER_SEEDS = range(1000000, 1000100)` (`tacinno/evaluation/results.py` L36) | 일치 (시드 값은 논문에 없음) |
| 시뮬 평가 행동 한계 | 논문에 없음 | 기본 2배 (`tools/eval_univtac.py` L156), README "twice the default action limit" | **논문에 없음** (D8) |
| 실로봇 제어 주파수 | 수집 30 Hz (§4.1), 제어 주파수 언급 없음 | 배포 10 Hz (`deploy … control_hz=10`) | 논문에 없음 (D9) |
| 실로봇 모델 수 | 과제별 비교 (§4.6) | 체크포인트 하나가 4과제 처리 (README "Deploy on a real robot") | 불명확 (D10) |
| Put Bottle 카메라 | 그림 3: head와 wrist 영상 | `deploy = dict(cameras=("head",))` (`configs/tacinno_vla/tacinno-vla_univtac-put-bottle.py`) | **다름** (D11) |
| 기준선 재학습 | 공식 설정으로 재학습 (§4.1) | 코드에는 기준선 구현이 없음 (UniVTAC-ACT, FTP-1, SimpleTouch 등은 외부 저장소) | 확인 불가 |

**재현 가능성 검사 (README 주장).** 저장소 README는 공개 체크포인트로 표 1의 과제별 값을 재현할 수 있다고 적고, 값은 논문과 같습니다 (Lift Can 88, Insert Hole 95, Insert Tube 95, Lift Bottle 78, Pull-out Key 94, Put Bottle 75, 평균 87.5). 이는 `recompute.py`의 표 1 재계산과 같습니다. 체크포인트 실행은 하지 않아 이 주장은 확인하지 못했습니다.

## 3. 차이 D1–D12

논문과 코드가 다른 부분입니다. 영향은 **높음**(결론에 영향), **중간**(재현 결과에 영향), **낮음**(세부)으로 적었습니다.

**D1. 고유감각 토큰 (영향: 중간, 논문 §3.3과 다름).**
논문은 고유감각 토큰을 q_t 하나로 만든다고 씁니다. 공식 설정은 Lift Can만 `proprio_history=8`(관절 t−7..t, `configs/tacinno_vla/tacinno-vla_univtac-lift-can.py`와 `init/…lift-can_init.py`)입니다. 나머지 시뮬 과제와 실로봇은 0입니다. 표 2와 표 3의 시뮬 Lift Can 결과는 H=8 설정으로 얻었습니다. 표 4의 실로봇 Lift Can과 나머지 시뮬 과제는 H=0입니다. 논문이 설명한 구조(q_t 하나)와 다른 설정이 절제 결과의 기준 모델에 들어 있습니다.

**D2. 학습 절차가 과제마다 다름 (영향: 높음, 논문 침묵).**
- Insert Hole, Lift Can: 여섯 과제를 함께 학습한 17000 스텝(`init/…_init.py`, `joint_17k.py`, 배치 64, 학습률 5e-5) 다음, 과제별 17000 스텝 미세조정(`finetune_17k.py`, 학습률 5e-6, EMA 0.99).
- Insert Tube, Lift Bottle, Pull-out Key, Put Bottle: π0.5 기반 가중치에서 과제별 20000 스텝(`single_task_20k.py`, 배치 64, 학습률 5e-5, EMA 0.99). 여섯 과제 공동 학습 단계가 없습니다.
논문은 "모든 정책은 같은 50 시연에서 학습"이라고만 씁니다 (§4.1). 여섯 과제 공동 학습은 다른 과제의 시연을 씁니다. 이 차이는 같은 시연 조건이라는 주장을 흐립니다. 특히 π0.5 기준선은 "다중 과제 미세조정"이라고 씁니다 (§4.1). TacInno와 기준선이 같은 절차인지 논문은 밝히지 않습니다.

**D3. λ 보정 기준 (영향: 낮음~중간).**
논문은 "학습 시작 시 보조 손실이 L_flow의 15%"라고 씁니다 (§3.3). 코드는 Lift Can 보정 설정에서 "두 가중치를 1로 둔 200 스텝 실행의 100–199 스텝 구간"을 기준으로 했다고 적습니다 (`init/…lift-can_init.py` 주석과 `docs/en/policy.md`). 실로봇은 자체 데이터에서 같은 기준으로 보정했다고 적습니다 (`real_robot` 설정 주석). 보정 값은 Lift Can 0.119와 0.467, 실로봇 0.118과 0.406입니다.

**D4. 토크나이저 학습 세부 (영향: 중간, 논문 침묵).**
12000 스텝, 배치 256, AdamW(β 0.9 / 0.95), 학습률 3e-4, 1000 스텝 워밍업 후 0.1배까지 코사인 감소, 가중치 감쇠 0.05 (`tokenizer_12k.py`). 코드북 리셋 주기 2000 스텝, 커밋 가중치 0.25 (`tokenizer.py` L32, `tokenizer_12k.py`). 논문은 이 값을 적지 않았습니다. 저자들은 README에서 H200 한 장으로 약 1.7 시간이라고 적습니다.

**D5. 실로봇 학습 설정 (영향: 중간, 논문 침묵).**
실로봇 설정은 `single_task_20k` 일정을 쓰고, 2 젤의 입력 중 오른쪽 젤은 왼쪽 젤의 복사입니다 (`configs/real_robot/tacinno-vla_real-4task.py` 주석). 논문은 학습 설정을 적지 않았습니다.

**D6. 접촉 가중 L1의 세부 (영향: 낮음).**
접촉 임계값은 압입 깊이 28.3 mm 미만입니다 (`tacinno/datasets/tactile_fields.py` L48). 접촉이 있는 픽셀의 가중치는 1 + 4 = 5이고, 두 프레임 중 하나라도 접촉이면 적용됩니다 (`losses.py` L37–38). 논문은 "contact-weighted"만 씁니다.

**D7. 접촉 상태 헤드 (영향: 낮음).**
논문은 접촉 면적, 전단, 압입 깊이를 예로 듭니다 (§3.3). 코드는 14개의 표준화 속성을 MSE와 BCE의 혼합으로 학습합니다 (`condition_tokens.py` L40–41). 14개가 무엇인지 논문에 열거되어 있지 않습니다.

**D8. 시뮬레이션 평가 프로토콜 (영향: 중간, 논문 침묵).**
공식 `eval_univtac.py run`의 기본값은 행동 한계 2배입니다 (`tools/eval_univtac.py` L156, 주석 "2 in the paper"). 논문 본문에는 "2배"라는 말이 없습니다. README는 "episodes interrupted by simulator errors are discarded until 100 valid episodes are collected"라고 적습니다 (`repo/README.md`). 시뮬레이터 오류 에피소드를 버리는 규칙은 논문에 없습니다. 기준선도 같은 규칙을 썼는지는 확인할 수 없습니다.

**D9. 실로봇 배포와 행동 청크 (영향: 중간, 논문 침묵).**
행동 청크는 32 단계입니다. 배포 때 각 청크의 1–10행만 10 Hz로 실행하고, 다시 질의합니다. 관절 오프셋은 추론 시점의 목표 관절 기준으로 더합니다 (`repo/README.md` "Deploy on a real robot"). 논문은 "execution horizon"이라고 쓰기만 하고 값이 없습니다 (§4.2).

**D10. 실로봇 체크포인트 구조 (영향: 낮음~중간).**
실로봇 체크포인트 하나가 4과제를 함께 처리하고, 언어 지시로 과제를 고릅니다 (`repo/README.md`, `docs/en/model_zoo.md`). 논문 표 4는 과제별 성공률을 적었지만 학습 구조를 적지 않았습니다. 또 모델은 두 젤 입력을 받지만 실제로는 한 젤만 쓰고 나머지는 복사합니다 (D5 참조).

**D11. Put Bottle 카메라 (영향: 낮음).**
Put Bottle은 머리 카메라만 씁니다 (`configs/tacinno_vla/tacinno-vla_univtac-put-bottle.py`의 `data` 와 `deploy`). 논문은 그림 3에서 "head + wrist images"를 기본으로 보여줍니다. 이 과제만 예외라는 설명이 논문에 없습니다.

**D12. 기타 구현 세부 (영향: 낮음).**
- 조건 토큰은 tanh(g)로 곱해지며 g는 학습합니다. 초기값은 설정에서 0.1입니다 (`tacinno_vla_pi05.py`의 `gate_init=0.1`, 코드 기본값은 0.0; `condition_tokens.py` L110, L24). 논문에 없습니다.
- 학습은 bf16 자동 혼합 정밀도를 씁니다 (`runtime`의 `precision="bfloat16"`, `torch.compile` 사용).
- 시간 주의는 인과적이고 프레임 내 주의는 공간적입니다. 논문과 같습니다.

## 4. 이 맥에서의 구현 가능성 (코드 기준)

**하드웨어와 소프트웨어 제약.**
- 코드는 CUDA를 전제합니다. `tools/train_tokenizer.py` L77은 `device = 'cuda'`로 고정하고, `tools/serve_policy.py` L57은 `--device` 기본값이 `cuda`입니다. `tools/export_codes.py` L51은 CUDA가 없으면 CPU로 떨어지지만, 학습과 서빙은 그렇지 않습니다. MPS 경로는 찾지 못했습니다 (`grep -rn mps`).
- bf16 자동 혼합 정밀도(`tacinno/models/tokenizer/tokenizer.py` `autocast`)와 TF32 설정(`train_tokenizer.py` L80)은 CUDA 전용입니다.
- 학습 환경은 `torch==2.7.1`과 CUDA 12.6 휠입니다 (`repo/README.md`, `requirements/runtime.txt`).
- 시뮬레이션 평가는 UniVTAC의 Isaac Sim 환경을 쓰고, README는 "one RTX-class GPU per simulator process"라고 적습니다. 이 맥의 M4 Pro에서 Isaac Sim을 돌릴 수 있는지는 확인하지 않았습니다 (UniVTAC 문서 미확인, 추정으로만 판단).

**자원 요구 (README와 설정 기준).**

| 단계 | 요구 | 이 맥에서 |
|---|---|---|
| 토크나이저 학습 | GPU 24 GB 이상, H200 1장 약 1.7시간 | 불가 (CUDA 하드코딩, 학습 시간 미측정) |
| TacInno-VLA 학습 | GPU 한 장당 80 GB 이상 (π0.5 전체 미세조정) | 불가 (메모리와 CUDA) |
| 정책 서버 추론 | GPU 10 GB 이상 (bf16) | 조건부 (MPS 포팅 필요, 추론 속도 미측정) |
| 시뮬 평가 | RTX급 GPU, UniVTAC, Isaac Sim | 불가 (추정, 확인 필요) |
| 데이터 | 시연 약 112 GB | 디스크와 다운로드 승인 필요 |
| 가중치 | 약 53 GB (정책 한 개 약 7.5 GB) | 승인 필요 |

**노력 추정 (측정 전, 추정).**
- 정적 읽기와 대조: 완료.
- 추론 경로의 MPS 포팅과 체크포인트 로드 검사: 가중치 한 개(약 7.5 GB) 다운로드와 `openpi` 의존성 설치 승인이 먼저 필요합니다. 포팅은 `cuda` 고정 제거와 bf16 경로 교체가 필요합니다. 추정으로 1–2주입니다.
- 토크나이저 학습을 맥에서 재현: 메모리 24 GB와 학습 시간 때문에 비현실적입니다. 작게 줄인 토이 토크나이저로 대체하는 것이 현실적입니다.

**판정.** 이 저장소는 **Apache-2.0 공식 구현이고 구조는 논문과 대체로 일치**합니다. 다만 **이 맥에서 공식 코드를 그대로 재현하는 것은 불가**합니다. 추론 검사는 조건부입니다 (MPS 포팅과 승인 필요).

## 5. 소결

1. 논문과 코드가 가장 크게 다른 곳은 D2(학습 절차)와 D1(고유감각 토큰)입니다. 두 가지 모두 표 2와 표 3의 Lift Can 결과에 직접 관련됩니다.
2. 시뮬 체크포인트의 결과(표 1)는 README의 표와 논문 표 1이 같습니다. 체크포인트 실행은 하지 않았습니다.
3. 논문이 빼놓은 평가 세부(D8, D9)는 기준선과 비교할 때 공정성을 확인하는 데 필요합니다.
