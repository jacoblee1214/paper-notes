# TipDex-VTLA: 논문 ↔ 공식 코드 대조 (정적 읽기)

대상: `tiplab/TipDex-VTLA` (Apache-2.0), 얕은 복제 HEAD `7e68bc5`, 커밋 13개, 릴리스 `0.2.1` (`validation/release_summary.json`, 2026-11-20). 복제본은 `repo/`에 있고 **실행, 설치, import를 하지 않았습니다.** 복제본 안의 내용은 신뢰하지 않는 자료로 읽었습니다. 파일 경로는 모두 `repo/` 기준입니다.

범위: 저장소 README는 "official inference example"이라고 말합니다. 학습 루프, 데이터 파이프라인, 평가 스크립트, UniVTAC 실행 코드, 실로봇 드라이버는 없습니다. 따라서 대조는 **추론 쪽 식과 구조**가 중심이고, 학습은 손실 정의 일부만 볼 수 있습니다.

## 1. 저장소 구성 요약

| 영역 | 파일 | 역할 |
|---|---|---|
| VASE 헤드 (추론) | `src/tipdex/VASE/head.py` | 2048→128→1 MLP, sigmoid. 학습 타깃 함수 `paper_vase_targets` (Eq. 1)와 페어 MSE 함수 |
| TACE 게이트·잔차 (추론) | `src/tipdex/TACE/refiner.py` (`_correct_features`) | Eq. (4) 구현 |
| 실행 점수 G | `src/tipdex/TACE/score.py`, `src/tipdex/forecast/verifier.py` | 임계값 기반 감소 |
| 촉각 예측과 검증 | `src/tipdex/forecast/*.py` | Eq. (3), 적격 예측, 명령 일관성, 신선도 |
| 느린-빠른 실행 | `src/openpi/fastslow/model.py`, `src/openpi/vtla/model.py` | 6단계 느린, 4단계 빠른 흐름, 캐시 |
| 학습 손실 일부 | `src/openpi/fastslow/model.py::_paired_loss`, `src/openpi/models/pi0.py::compute_loss_and_metrics`, `src/openpi/models/tactile_contact.py` | 흐름 브리지 손실과 접촉 BCE (§3 참고) |
| 백본 | `src/openpi/models/pi0.py`, `pi0_config.py`, `src/openpi/models/siglip.py`, `tactile_encoder.py`, `tactile_adapter.py` | π0.5 계열 (PaliGemma 2B + Gemma 300M expert) |
| 설정 | `configs/paper.json`, `configs/tipdex.json`, `src/tipdex/config.py` | `paper`와 `tipdex` 두 프리셋 |
| 모델 메타데이터 | `src/tipdex/static/backbone_9309.json` | 백본 설정 (에그 과제 체크포인트) |
| 테스트 | `tests/*.py` (49개 통과, 릴리스 기록) | 프로토콜과 수식 단위 테스트 (`validation/`) |

## 2. 논문과 코드의 대조

### 2.1 논문과 일치하는 부분

| 번호 | 논문 | 코드 | 근거 |
|---|---|---|---|
| M1 | Eq. (2) $q_v=\sigma(h_\omega(\mathrm{Pool}(c_t)))$, 추론 때 짝 이미지 없음 | 마지막 접두부 문맥의 마스크 평균 풀링 후 헤드, sigmoid | `VASE/head.py` (`POOLING = masked_mean_final_prefix_context`, `NumpyVASEHead.predict`) |
| M2 | Eq. (1) $q'_v=\exp(-\mathrm{MSE}/\lambda)$ | `paper_vase_targets`가 $\exp(-\mathrm{mse}/\lambda)$를 계산, 바닥값과 재매핑 없음 | `VASE/head.py::paper_vase_targets` |
| M3 | Eq. (4) $g=\sigma(S-\kappa\,\mathrm{logit}(q))$, $\tilde U = U + g\,b\tanh(R)$ | 같은 식. $q$에 정지 기울기, $b=0.1$, $\kappa=1$ | `TACE/refiner.py::_correct_features` (line 151–159) |
| M4 | 게이트는 보정 위치에만 적용, 실행된 행동과 시각은 고정 | 매 흐름 단계에서 보호된 위치를 복원 | `TACE/refiner.py` 주석, `docs/METHOD.md` 표 마지막 행 |
| M5 | Eq. (3) 예측 $\hat T = P(c_t,\xi_{t'})$, 캐시 후 나중에 검증 | 적격 예측 선택, 명령 일관성 검사 | `forecast/verifier.py`, `forecast/execution.py` |
| M6 | $e$는 같은 정규화에서 MAE. 적격 예측이 없으면 $e=0$과 시각 점수로 복귀 | 364개 값의 현재 프레임 MAE, 없으면 `q_exec=q_base` | `forecast/verifier.py::score` (line 99–154), `TACE/score.py` |
| M7 | $G(q_v,0)=q_v$, $G$는 $q_v$에 단조 증가, $e$에 단조 감소 | 문서화된 `paper` 프리셋에서 항등, 단조 (테스트 있음) | `TACE/score.py`, `tests/test_method.py::test_paper_score_identity_monotonicity_and_large_error_only` |
| M8 | 느린 6단계 + 빠른 4단계, $\tau_s=0.6$, 10단계 적분 | `slow_steps 6`, `num_flow_steps 10`, `fast` 4단계 | `backbone_9309.json`, `docs/METHOD.md` |
| M9 | 최근 촉각 이력 $T_{\le t}$ (논문은 프레임 수를 밝히지 않음) | `tactile_num_frames 16`, 30 Hz. 논문 값이 아니므로 참고용 | `backbone_9309.json` |
| M10 | 촉각 364 값, 30 Hz | `tactile_layout f1_364`, `F1` 입력 364 uint16 | `pi0_config.py`, `vtla/model.py`, `sensors.py` |
| M11 | 다시점 RGB (논문 §4.1은 카메라 3대), 언어 | `Three RGB views` | `README.md` §2. 19차원 상태는 논문에 수치가 없음 |
| M12 | 행동 chunk의 길이 $H$ (논문은 값을 밝히지 않음) | `action_horizon 30`, 행동 차원 48 중 13 활성 | `backbone_9309.json` (`loss_action_dims`) |

### 2.2 차이 (D-항목)

**D1. 학습 파이프라인과 2단계 학습이 없음.** 
- 논문 §3.4: Stage 1은 흐름 매칭 $\mathcal{L}_1$과 촉각 Huber $\mathcal{L}_T$, Stage 2는 얼린 백본에서 $\mathcal{L}_Q$(BCE)와 $\mathcal{L}_R$(MSE$_M$)로 VASE와 TACE만 학습. 20,000 + 10,000 스텝, AdamW, 배치 128, 학습률 $5\times10^{-5}$, A100 8장.
- 코드: 옵티마이저, 학습률, 배치, 스텝 수, 학습 진입점, 데이터 로더가 없습니다 (`pyproject.toml`의 스크립트는 `tipdex`와 `tipdex-export`뿐). `scripts/`에는 `check_http.py` 하나입니다. 손실 정의의 일부만 있습니다.
- VASE의 BCE 학습 코드와 TACE의 $\mathcal{L}_R$은 없습니다. `TACE/refiner.py` 주석은 "stop_gradient prevents L_R from updating VASE/G"라고 쓰지만 $\mathcal{L}_R$ 구현은 없습니다.
- **판정: 학습 절차는 코드로 검증할 수 없습니다 (C11).**

**D2. 학습 손실이 논문 식과 다릅니다.** 릴리스의 `_paired_loss` (`src/openpi/fastslow/model.py` line 167–199)는 다음을 씁니다.
- 느린 흐름 손실 (`slow_loss`) + 빠른 흐름 "브리지" 손실 (`fast_loss`): 느린 롤아웃 끝점 $x_{\text{split}}$에서 시연 chunk까지 선형 보간, 속도 목표 $(x_{\text{split}} - a)/\tau$.
- 접촉 BCE (`contact_loss_weight`, 공개 백본은 0.0, 기본값은 0.01, `pi0_config.py:63`).
- 논문의 $\mathcal{L}_1$(Eq. 6)과 $\mathcal{L}_T$(Huber 촉각 예측 손실), $\mathcal{L}_Q$, $\mathcal{L}_R$(Eq. 8)과 대응하지 않습니다. $\lambda_T, \lambda_Q, \lambda_R$은 코드에 없습니다.
- 이 코드가 논문 학습의 어느 단계에 쓰였는지 불명합니다. 논문의 학습 목적과 코드의 목적이 다른 것으로 보입니다.
- **판정: 학습 목적함수는 논문과 다릅니다 (논문 수식 (5)–(8)과 대응 없음).**

**D3. VASE 타깃의 λ와 짝 샘플링이 공개되지 않음.**
- 식 (1)의 $\lambda$는 모델 메타데이터 `lambda_scale`에서 읽습니다 (`VASE/head.py`). 값은 저장소에 없고 HF 모델 패키지에 있을 것입니다. 이 분석에서는 받지 않았습니다.
- 코드의 `masked_paired_action_mse`는 여러 노이즈 시드 $S$에 걸쳐 MSE를 평균합니다 ("equal mean over paired noise seeds"). 논문 식 (1)에는 시드 평균이 없습니다.
- 짝 샘플링 확률 20%씩 다섯 조건(§3.2)을 만드는 코드는 없습니다.
- **판정: 식 (1)의 형태는 일치하지만 $\lambda$, 시드 수 $S$, 짝 분포가 논문에 없고 코드에도 없습니다.**

**D4. 실행 점수 $G$의 구체 형태가 논문에 없음.** 논문은 $G$의 성질(단조, 항등)만 씁니다. 코드는 다음 형태입니다 (`TACE/score.py`, `docs/METHOD.md` "분수 융합").
$$
q_{\text{base}} = o + s\,q_v,\quad x=\mathrm{clamp}\!\left(\frac{e-L}{F-L},0,1\right),\quad q = q_{\text{base}} - \min\!\big(\delta_{\max}\,x^2(3-2x),\; q_{\text{base}}/2\big)
$$
- $L$, $F$ (`error_threshold_low/full`)는 모델 계약에서 읽습니다. 저장소의 테스트는 $L=0.005$, $F=0.01$을 쓰고, `verifier.py`의 기본값은 $L=1.0$, $F=2.0$입니다. 실제 값이 무엇인지 저장소만으로 알 수 없습니다.
- 감소 상한 $\delta_{\max}=0.1$과 $q_{\text{base}}/2$ 상한은 논문에 없습니다.
- **판정: $G$는 논문 성질을 만족하지만 구체 형태(smoothstep, 임계값, 상한)는 논문에 없습니다. 논문 결과를 재현하려면 이 값들이 필요합니다.**

**D5. 검증 창 (적격 예측 선택과 신선도).** 논문은 세부를 보충 자료에 맡깁니다. 코드에서 확인된 것은 다음입니다.
- 현재 프레임의 측정만 검증합니다 (`verifier.py`). 프레임 단위 MAE는 364 값의 평균입니다.
- 측정의 최대 나이 $0.05$ s (`measurement_max_age_seconds`), 허용 오차 창 (`alignment_tolerance_seconds`).
- 명령이 바뀌면 적격이 아닙니다. 중간 명령 변화가 있으면 "command_history_not_verified"로 되돌립니다 (`verifier.py` line 136). 이 조건은 논문 §3.3의 "under commands consistent with the intervening executed actions"와 일치합니다.
- **판정: 방향은 일치합니다. 시간 창과 신선도 값은 논문에 없습니다.**

**D6. 두 프리셋이 논문 설명과 다릅니다.** `configs/tipdex.json` (배포 프리셋)과 `configs/paper.json` (논문 프리셋)의 차이는 다음과 같습니다.

| 설정 | `paper` (논문) | `tipdex` (배포) |
|---|---:|---:|
| 시각 점수 $q_{\text{base}}$ | $q_v$ | $0.6 + 0.4q_v$ |
| 잔차 상한 $b$ | 0.1 | 0.01 |
| 예측 H | 30 | 5 |
| 촉각 융합 | `context_concat` | `action_adapter` |

- 논문 결과는 `paper` 프리셋에 해당하는 것으로 보입니다. 하지만 어떤 프리셋이 Table 1–3을 만들었는지 논문은 말하지 않습니다.
- **판정: 배포 프리셋은 논문과 다른 수치를 씁니다. 논문 결과는 `paper` 프리셋으로 봐야 합니다.**

**D7. 촉각 융합 방식이 학습 메타데이터와 다를 수 있음 (확인 필요).**
- 공개 백본 메타데이터 `backbone_9309.json`은 `tactile_fusion: "action_adapter"`입니다. 학습 기록의 값이 이것입니다.
- 논문 프리셋은 `context_concat`입니다 (`configs/paper.json`). 논문 §3.1은 촉각 특징을 문맥 $c_t$와 이어 붙인다고 씁니다.
- `docs/METHOD.md`는 두 프리셋이 "같은 모델 파라미터를 로드"하고, `context_concat`은 "이미 로드된 촉각 투영 파라미터"로 구현한다고 설명합니다.
- 학습 때 쓴 것이 `action_adapter`라면, `context_concat` 경로가 학습된 가중치와 같은 함수인지는 이 저장소로 확인할 수 없습니다. 논문 결과가 `context_concat`으로 얻어진 것이라면 학습과 추론 사이에 불일치가 있을 수 있습니다.
- **판정: 확인 필요. 실행하지 않고는 판단할 수 없습니다.**

**D8. 백본이 논문에 없습니다.** 논문은 "pretrained VLM"과 "flow-matching Action Expert [22]"라고만 씁니다. 메타데이터는 `pi05: true`, PaliGemma `gemma_2b`, Action Expert `gemma_300m`, `max_token_len 200`, 행동 차원 48 (활성 13)입니다. 기반은 π0.5 계열입니다.
- 논문 표 1–2에서 π0.5는 기준선입니다. 따라서 TipDex의 기여(VASE, TACE, 촉각 인코더)는 π0.5에 더해진 것입니다. 논문이 이 사실을 본문에 쓰지 않는 것은 독자가 기여를 판단하기 어렵게 합니다.
- **판정: 불일치라기보다 누락입니다. 백본 이름을 논문에 명시하는 것이 필요합니다.**

**D9. 촉각 배열 구조.** 논문 §3.1과 §4.1은 촉각을 "5개 손가락 영역, 3개 손바닥 영역, 8개 배열 364 taxel"로 설명합니다. 코드는 364개 uint16 레지스터 값을 14×26 행 우선 격자로 표시하고, "레지스터 순서이며 해부학적 영역이나 보정된 힘이 아님"이라고 명시합니다 (`docs/METHOD.md`, 촉각 표현 절).
- 손가락과 손바닥 영역의 매핑은 코드에 없습니다.
- **판정: 논문의 영역 설명과 코드의 레지스터 설명이 다릅니다. 영역 구조가 학습에 어떻게 쓰였는지 저장소로 확인할 수 없습니다.**

**D10. 공개 데이터 규모.** 논문은 300 궤적 (200 + 100 쌍)을 씁니다. HF `TipDex-Data`는 원본 50 + 쌍 50입니다 (`README.md` §3). 공개 데이터는 논문 학습 데이터의 일부로 보입니다. 메타데이터만 확인했습니다. 다운로드하지 않았습니다.

**D11. 공개 체크포인트는 에그 과제 하나.** `backbone_9309.json`과 README의 "Egg-transfer backbone, checkpoint 9309"는 한 과제입니다. 논문 표 2의 4과제 결과가 과제별 모델인지 한 모델인지 논문은 말하지 않습니다. 공개된 것으로는 4과제 결과를 재현할 수 없습니다.

**D12. 평가 코드가 없습니다.** 실세계 시험 프로토콜, UniVTAC 통합, 성공률 집계 코드는 없습니다. README는 "replay supplies simulated execution acknowledgements and does not send commands to a robot"라고 말합니다. `validation/release_summary.json`은 scope를 "inference functionality; no tactile accuracy or task success claims"로 적습니다. **저장소는 논문의 성공률을 재현하지 않습니다 (C10–C12와 연결).**

**D13. 추가로 공개된 것 (논문에 없음).** 2 Hz 시각 분기와 30 Hz 촉각 분기 (프로젝트 페이지), 두 프리셋, 실행 확인 프로토콜 (`docs/DEPLOYMENT.md`), 보정 페이지와 대시보드, HTTP 서비스. 논문 재현과는 무관하지만 배포 세부는 공개되어 있습니다.

## 3. 학습 손실의 비교 (참고)

| 항목 | 논문 | 코드 (`_paired_loss` 등) |
|---|---|---|
| 기본 흐름 손실 | Eq. (6) $\mathcal{L}_1$, $\|v_\theta - (a^\star-\epsilon)\|^2$ | 느린 롤아웃 흐름 손실 (`slow_loss`), 흐름 브리지 |
| 촉각 예측 | Huber $\mathcal{L}_T$ (Eq. 6) | 없음. 접촉 BCE만 있음 |
| VASE | BCE $\mathcal{L}_Q$ (Eq. 8), 타깃 Eq. (1) | 추론 헤드만. 학습 코드 없음 |
| TACE | MSE$_M$ $\mathcal{L}_R$ (Eq. 8), 타깃 $\Delta a$ (Eq. 7) | 추론 게이트·잔차만. 학습 코드 없음 |
| 단계 | 2단계 (Stage 1 얼림 후 Stage 2) | 단일 목적함수 (`paired`), 얼림 단계 코드 없음 |

## 4. 구현 가능성 판정

**결론: 논문 결과를 재현하는 데 필요한 것의 절반 이하가 공개되어 있습니다. 추론 코드는 있으나 CUDA 전제이고, 학습과 평가는 공개되지 않았습니다.**

| 항목 | 이 맥(M4 Pro, 24 GiB, MPS, CUDA 없음) | 필요 자원 | 판정 |
|---|---|---|---|
| 공식 추론 (`replay`, `serve`) | CUDA 전제 (`pyproject.toml`의 `jax[cuda12]`, README "NVIDIA GPU with CUDA 12"). 맥에 CUDA 없음. JAX CPU 설치는 가능할 수 있으나 설치하지 않았음. CI는 CPU에서 프로토콜 테스트만 실행 (`.github/workflows/test.yml`) | JAX CUDA 또는 CPU 포팅 | **조건부**. 모델 크기 (2B + 300M, bf16)는 메모리에 맞을 것으로 보이나 CPU 추론 속도는 측정하지 않음. 포팅 노력 추정 1–2주 (측정 전) |
| 학습 재현 (Stage 1, 2) | 학습 루프와 VASE·TACE 학습 코드가 없음. 8 × A100, 2만+1만 스텝 | GPU 여러 장, 학습 파이프라인 직접 구현 | **불가** (맥 자원과 학습 코드 부재) |
| 실로봇 (RM75 2대, Inspire 손, 촉각 배열) | 하드웨어 없음 | 하드웨어 | **불가** |
| UniVTAC 시뮬 평가 (Table 1) | 평가 코드 없음. 시뮬 설치 자체를 확인하지 않음 | 시뮬 환경, 통합 코드 (직접 작성) | **불가 (현재 공개 자료로)** |
| VASE/TACE 식 (추론) 검증 | 저장소 테스트 49개 통과 (릴리스 기록). 맥 CPU에서 실행하면 확인 가능하지만 설치는 하지 않음 | numpy, jax CPU | 조건부 (설치 승인 후) |
| 표 재계산 (수치 검정) | 완료 (`analysis/recompute.py`) | numpy, scipy | **가능, 완료** |
| 공개 데이터 50 + 50 재현 | 메타데이터만 확인, 다운로드 안 함 | 디스크, 시간 | 조건부 (데이터 다운로드 승인 필요) |

**노력 추정 (측정 전):** 공식 추론 CPU 포팅 1–2주. 학습 재구현 (Stage 1, 2, 손실 포함) 3–6주, GPU 필수. 실로봇 재현 불가.

## 5. 라이선스

- 코드: Apache-2.0 (`LICENSE`). 원 OpenPI 코드의 Apache-2.0 고지 유지 (`licenses/OPENPI_LICENSE`).
- 모델: 백본 Gemma/PaliGemma 파생 파라미터. Gemma 조건 (`licenses/LICENSE_GEMMA.txt`, `src/tipdex/static/LICENSE_GEMMA.txt`). 모델 카드와 데이터셋 카드 확인 필요 (HF 메타데이터만 확인).
- `licenses/TIPDEX_REFERENCE_MIT.txt`: "동작 계획 보조 클래스"에 MIT 고지가 붙어 있습니다 (`NOTICE.md`).
- 저장소 어디에도 학회 채택이나 venue 표기가 없습니다.
