# 04. 공식 코드 대조 (정적 읽기)

**대상:** `NamelessCrew/FreeBar`, 커밋 `e24dd74`(2026-09-19, 단일 커밋 "Initial release"), 얕은 복제를 `repo/`에 두었습니다. **실행, 설치, 가져오기(import)는 하지 않았습니다.** 모든 판단은 파일을 읽어서 얻었습니다. 논문 값과 코드 기본값의 차이는 `formula_check.py`로 수치 확인했습니다.

**읽은 파일:**
- `Isaac-GR00T/gr00t/model/action_head/freebar.py` (190줄, 핵심 알고리즘)
- `Isaac-GR00T/scripts/run_eval_async.py` (808줄, 비동기 실행, 캐리, 인수)
- `Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py` (`get_action_freebar` 래퍼)
- `kinetix/src/model.py` (`freebar_action`, RTC `realtime_action`, `get_prefix_weights`)
- `kinetix/src/eval_flow_rtc.py` (실행 루프, 격자, 기본값)
- `README.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE`

## 0. 요약

- **알고리즘은 충실히 구현되어 있습니다.** 닫힌형 λ, 수평선 가중치, 1−t 계열 τ가 논문 Algorithm 1과 같은 순서로 나옵니다. 역전파는 없습니다(`torch.no_grad`).
- **세부는 논문과 다릅니다.** 에너지의 가중치 제곱(D1), τ 하한(D2), 가중치 분모(D3), λ 상한의 대상(D4)이 다릅니다. 보고된 디노이징 스텝 수에서는 D2의 효과가 없지만 D1, D3는 있습니다.
- **Kinetix 경로에 잠재적 결함이 있습니다(D6).** 참조 청크의 0 패딩 구간에 안내 가중치가 실립니다. GR00T 경로는 `2h ≤ H`를 강제해서 이를 피합니다. 정적으로 확인한 것이고 실행으로 확인하지 않았습니다.
- **재현에 필요한 것이 빠져 있습니다.** 실로봇 코드, 체크포인트, RoboCasa 자산과 `models/`, GR00T에서의 RTC와 표 2 기준선, 비계층 CBF 절제가 없습니다. 논문의 α, β 값도 없습니다.

## 1. 저장소 구성과 빠진 것

| 항목 | 상태 |
|---|---|
| Kinetix 평가(FreeBAR, naive, RTC) | 있음 (`kinetix/src/eval_flow_rtc.py`, `model.py`) |
| Kinetix 학습 (`train_flow.py`, `train_expert.py`) | 있음. 학습된 BC 체크포인트는 없음 (`pretrained/bc/README.md`만 있음) |
| GR00T 포크의 FreeBAR 액션 헤드 | 있음 (`freebar.py`, 약 190줄) |
| GR00T 비동기 평가 | 있음 (`run_eval_async.py`). 공개 메서드는 `naive`, `freebar`뿐 (`METHOD_CHOICES`) |
| GR00T에서의 RTC 기준선 | **없음** |
| 표 2의 가벼운 기준선 (시간 앙상블, 행동 평활화, 청크 경계 보간, 균일 강제 정렬) | **없음** |
| 표 1의 비계층 CBF 절제 | **없음** (균일 가중치로 바꾸는 옵션이 없음) |
| 실로봇 코드 (듀얼암 휴머노이드 컨베이어, Franka 두더지 잡기) | **없음** |
| RoboCasa 자산 (약 10 GB), `robocasa/models/` | 제외됨. 설치 절차는 README에 있음 |
| 정책 체크포인트 (GR00T 60k, π0) | 없음. `--model_path`로 사용자가 넣음 |
| 실험 로그, 시드별 결과, 표 생성 스크립트 | 없음 |

## 2. 논문과 코드의 차이 (D1–D9)

| 번호 | 논문 | 코드 | 영향 |
|---|---|---|---|
| **D1** | 식 (2): B = −½ Σ w̃_i‖x̂_{1,i}−ā_i‖² (w̃ 선형). Algorithm 1 6행도 같음 | `freebar.py` 142–144행: 중간 구간에서 `weighted_error_mid = w·e` 후 `h_mid = −½ Σ (w·e)²`. 즉 w²을 씀. Kinetix `model.py` 260–261행도 같음 | b = αE/τ가 w²로 계산되어 식 (2)와 다른 에너지에서 나옴. g는 `−w·e`로 선형 w를 써서 식 (3)과 일치하지만 ∇(에너지)와는 맞지 않음. 단위 오차 기준 코드 에너지는 식 (2)의 0.92배 (`formula_check_results.json`) |
| **D2** | τ(t) = 1−t | `freebar.py` 150행과 Kinetix `model.py` 264행: `τ = max(1−t, 1/max_gw)` | 보고된 설정(GR00T K=4, Kinetix K=5, 실로봇 K=10)에서는 1−t ≥ 1/max_gw라서 작동하지 않음. K가 커지면 끝 스텝의 b 상한으로 작동 |
| **D3** | 식 (11): 중간 구간 exp(−β(i−d+1)/(h_s−d)) | `freebar.py` 101행: `safe_span = h−d+1`. Kinetix `model.py` 244행도 같음 | RoboCasa 설정(d=3, h=6, β=3)에서 중간 가중치 최대 차 0.10 (0.37 대 0.47). 감쇠가 더 느림 |
| **D4** | §3.2: "보정 크기에 상한 10 (GR00T)" | `freebar.py` 156–157행: **λ 자체**를 [0, max_gw]로 자름. `clip_lambda=False` 옵션도 있음 (`run_eval_async.py` `--freebar_no_clip`) | 보정 크기 λ‖g‖의 상한은 ‖g‖에 따라 달라짐. 논문의 "상한" 기술과 대상이 다름. 공식 설명이 없음 |
| **D5** | 식 (2)의 참조 유효 마스크 m_i (§3.1) | 명시적 마스크 없음. 캐리는 뒤쪽 h칸을 0으로 채움 (`run_eval_async.py` 395–404행). GR00T 경로는 `2·h > H`이면 오류를 냄 (252–262행) | `2h ≤ H`이면 0으로 채운 칸이 가중치 밖에 놓여 마스크가 사실상 필요 없음. 이 조건은 논문에 없음 |
| **D6** | (논문은 언급 없음) | Kinetix 경로: `eval_flow_rtc.py`는 h를 max(1,d)…8−d까지 돌리고(H=8이라 h≥5 포함), `freebar_action`은 0 패딩 칸에 가중치를 그대로 둠. RTC는 `get_prefix_weights(..., end=H−h)`로 같은 칸을 0으로 막음 (`model.py` 196–199행) | 설명은 §5 (`03_analysis.md`). d=1, h=7에서 안내 가중치의 63%가 0 목표에 실림. **Kinetix의 h≥5 결과 전체가 영향을 받을 수 있음.** 실행 확인 필요 |
| **D7** | (언급 없음) | 에피소드 경계에서 끝난 환경 칸의 캐리를 naive 청크로 다시 씀 (`reset_carry`, 265–300행). 첫 추론도 naive 청크를 기준으로 함 | 각 에피소드 첫 청크는 FreeBar의 이전 청크 정렬을 받지 않음. 논문 Algorithm 1의 입력 설명과 다름 |
| **D8** | 본문에 α, β 값 없음. 부록 A.5는 "전체 이산 탐색 공간은 부록에 있다"고 하지만 부록에는 결과 표만 있음 | 기본값: `alpha=10`, `max_guidance_weight=5`, `exp_decay_rate=3` (`freebar.py`, Kinetix `eval_flow_rtc.py` 36–39행). RoboCasa 예시는 `--cbf_alpha 20.0 --cbf_max_guidance_weight 10.0` (README). GR00T 실행 기본값은 `cbf_alpha=10`, `max=5` (`run_eval_async.py` 181–182행) | 표 1의 RoboCasa 주결과가 α=20인지 10인지 논문만으로는 알 수 없음. README는 α=20을 예시로 씀 |
| **D9** | GR00T N1 [1] 인용 | 코드는 `GR00T_N1_5` 클래스 (`gr00t_n1.py`), README는 GR00T N1.5 | 백본 버전 표기 불일치. 체크포인트 버전이 결과에 영향 줄 수 있음 |

추가로 확인한 것 (일치):

- **Kinetix 구조:** MLP-Mixer 4층, 청크 길이 8 (`model.py` 16–17행) → 논문 A 절 "4-layer MLP-Mixer, H=8"과 일치.
- **Kinetix 격자:** d ∈ {0..4}, h ∈ [max(1,d), 8−d] (`eval_flow_rtc.py` 260–264행) → 그림 3의 horizon 스윕 (d=1, h=1–7)과 delay 스윕 (h=max(d,1))과 일치.
- **λ 닫힌형:** `(b − g·v)/(‖g‖²+1e-8)`의 max(·, 0)과 클램프 → 식 (5)와 같음. 임의 입력 1000개에서 제약 g^T ṽ ≥ b가 모두 성립 (`formula_check.py`).
- **추론 시간 0 역전파:** `run_eval_async.py`의 `grad_ctx` (`torch.no_grad`) → 논문 "역전파 없음" 주장과 일치.
- **RoboCasa 실행 인수:** `--n_action_steps 6 --inference_delay 3 --override_action_horizon 16` (README 예시) → 논문 §4.1.2 (H=16, d=3, h=6)과 일치. 부록 A.3의 chunk size 10은 불일치 (`03_analysis.md` V1).

### 2.1 D6 자세히 (정적 판독)

`eval_flow_rtc.py`의 `execute_chunk`는 실행한 h개를 버리고 뒤를 0으로 채웁니다.

```
next_action_chunk = concat(next_action_chunk[:, h:], zeros(h))
```

다음 추론의 `prev_action_chunk`는 이 값입니다. `freebar_action`은 가중치를 인덱스 0..h−1에 두고, 목표 `prev_action_chunk`의 인덱스 H−h..H−1은 0입니다. 그래서 H−h < h인 경우, 즉 h > H/2인 경우에 가중치가 0 목표에 걸립니다. RTC는 `prefix_attention_horizon = H − h`를 넘기고 `get_prefix_weights`가 그 뒤를 0으로 만들므로 이 문제를 피합니다.

`formula_check.py`의 계산(H=8):

| (d, h) | 0 목표에 실린 가중치 | 칸별 최대 |
|---|---|---|
| (1, 7) | 63% | 0.65 |
| (0, 7) | 66% | 0.47 |
| (2, 6) | 36% | 0.55 |
| (1, 5) | 12% | 0.17 |

이것은 코드 읽기에서 나온 추론입니다. 실행 결과 (논문 그림 3의 h=5–7 값)가 실제로 얼마나 바뀌는지는 Kinetix 평가를 돌려야 알 수 있습니다. 확인 방법은 아래 §5에 있습니다.

## 3. 라이선스와 공개 범위

- **FreeBar 저장소 (최상위):** Apache-2.0 (`LICENSE`).
- **Isaac-GR00T 포크:** Apache-2.0 (NVIDIA 저작권 헤더, `freebar.py` 상단 SPDX).
- **RoboCasa 사본:** MIT (`robocasa/LICENSE`). GitHub API는 원 저장소를 NOASSERTION으로 표시합니다.
- **robomimic:** MIT (`THIRD_PARTY_NOTICES.md`).
- **RTC (Kinetix 원본):** MIT (원 저장소 기준).
- **논문:** arXiv non-exclusive distribution license 1.0. CC BY가 아니므로 논문 그림은 공개 산출물에 넣지 않습니다.
- **공개 산출물에서의 코드 취급:** 코드는 복제하거나 재배포하지 않습니다. 우리 문서에는 파일 경로와 짧은 인용만 있습니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 알고리즘은 구현할 수 있습니다. 논문 수치의 재현은 이 맥에서 RoboCasa와 실로봇은 불가, Kinetix는 조건부입니다.**

| 대상 | 필요 자원 | 이 맥에서 | 노력 (추정, 측정 안 함) |
|---|---|---|---|
| FreeBAR 알고리즘을 임의의 흐름 정책에 붙이기 (PyTorch, MPS 또는 CPU) | 흐름 정책 한 개 (작은 MLP나 MLP-Mixer로 충분) | **가능.** 핵심은 `freebar.py`의 `get_action_freebar` (약 140줄, 주석 포함) | 0.5–1일 (추정). 우리 구현으로 D1–D5를 골라 비교하는 것까지 포함하면 2–3일 |
| Kinetix 평가 (JAX, 12 과제, 격자 전체) | JAX (`jax[cuda12]` 고정, macOS 휠 없음 → 의존성 변경 필요), Kinetix 환경, BC 정책 학습 (`train_flow.py`) | **조건부.** 의존성 변경과 CPU JAX 학습 시간이 측정 전 | 1–2주 (추정). 학습 시간은 측정하지 않음. 승인 필요 (설치, 학습) |
| GR00T N1.5 기반 RoboCasa 평가 | CUDA (`torch.autocast(device_type="cuda")`, `run_eval_async.py` 351행), MuJoCo EGL (`MUJOCO_GL=egl`), RoboCasa 자산 약 10 GB, GR00T 체크포인트 (미공개) | **불가.** CUDA 경로와 EGL, 체크포인트가 없음 | 측정 불가 (GPU 필요) |
| 실로봇 (듀얼암 컨베이어, Franka 두더지 잡기) | 로봇, 코드 (없음), 체크포인트 (없음) | **불가** | 코드가 없어 외부에서는 재구현만 가능 |
| 논문 표 재계산과 통계 검사 | 표 값 | **가능, 완료** (`recompute.py`) | 완료 |

- **전체 재현의 장벽:** 공개된 코드로는 실로봇 표를 재현할 수 없고, RoboCasa 표는 GPU와 체크포인트가 필요합니다. 표 2의 기준선 4종과 RTC(GR00T)는 코드가 없어 같은 비교를 할 수 없습니다.
- **새로 짜기의 장벽:** 알고리즘은 짧지만 세부(D1–D7)를 어떻게 고를지 논문이 말하지 않습니다. 재구현 결과가 공식 코드와 다를 수 있습니다.

## 5. 확인 방법 (실행하지 않음)

다음은 승인 없이는 하지 않습니다. 각 항목의 승인 필요 여부를 적었습니다.

1. **D6 검사 (Kinetix, 한 과제):** 참조 0 패딩을 막는 버전(RTC처럼 H−h 뒤를 가중치 0)과 현재 버전을 같은 (d=1, h=7)에서 비교. 필요: JAX 환경과 BC 체크포인트 → **승인 필요**.
2. **D1, D2, D3 비교 (torch, 토이):** 식 (2) 대 코드 에너지를 흐름 정책 토이에 넣어 비교. 필요: torch(이미 있음 여부 확인 필요), 작은 흐름 정책 학습 (CPU 2분 이내) → **새 설치 없이 가능한지 먼저 확인**.
3. **α, β 값 확인:** 논문 저자에게 문의. 공식 저장소의 실험 설정 파일이 있는지 저자가 알려 주면 그 값을 쓴다. 필요: 문의만 (자료 다운로드 없음).
