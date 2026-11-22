# 04. 논문 ↔ 공식 코드 대조 (정적 읽기)

대상: `repo/` = `lucas-maes/le-wm` 얕은 clone, 커밋 `8edfeb3`(2026-05-22, "fix data config"), MIT. 파일: `jepa.py`, `module.py`, `train.py`, `eval.py`, `utils.py`, `config/train/*`, `config/eval/*`. 실행, import, 설치는 하지 않았습니다. 기반 라이브러리(stable-worldmodel)의 `scripts/train/pldm.yaml`과 `lewm.yaml`만 GitHub API로 읽었습니다.

라벨 D1–D18은 이 문서의 대조 항목입니다. 줄 번호는 `repo/` 기준입니다.

## 1. 한눈 요약

- **일치하는 핵심:** 모델 구조(ViT-Tiny [CLS] 인코더, 6층 16헤드 AdaLN 예측기, 폭 192), 두 항 손실(예측 MSE + λ·SIGReg), SIGReg의 매듭 17개와 투영 1024, 계획 비용(마지막 스텝 MSE), 계획 설정(horizon 5, 블록 5, CEM 샘플 300, 엘리트 30), 평가(에피소드 50, 목표 거리 25, 예산 50), v3의 TwoRoom 평가 설정.
- **불일치:** λ 값(0.1 대 0.09), SIGReg 적분 구간과 배치 곱, 손실 정규화(평균 대 합), 예측기 구조 설명(ViT-S 대 폭 192 자체 구현), 에포크 수(10 대 기본 100), TwoRoom 히스토리(1 대 기본 3), CEM 반복 수(10 대 기본 30).
- **공개되지 않은 것:** 표 1·3·4 프로브 코드, 그림 7·9·10의 디코더, VoE(그림 8, 12, 14), 기준선 학습 설정(DINO-WM, PLDM은 stable-worldmodel에 있지만 이 저장소에는 없음), 세 시드 학습 스크립트.
- **하드웨어:** 공식 평가 코드가 CUDA를 **하드코딩**합니다. 맥에서 쓰려면 수정이 필요합니다.

## 2. 대조표 (구조와 손실)

| 항목 | 논문 | 공식 코드 | 판정 |
|---|---|---|---|
| 인코더 | ViT-Tiny, 패치 14, 12층, 헤드 3, 폭 192, [CLS] | `vit_hf(size="tiny", patch_size=14)`, `jepa.py` 의 `last_hidden_state[:, 0]` | 일치. HF ViT의 `last_hidden_state`에는 마지막 LayerNorm이 적용되어 논문의 "LN이 방해한다"는 설명과 맞음 (D18) |
| 투영 | 1층 MLP + BatchNorm (§3.1) | `projector`: Linear(192→2048) + BN + GELU + Linear(2048→192) (`config/train/model/lewm.yaml:33`) | **불일치 D5**: 논문은 "1-layer", 코드는 은닉 2048의 2층 |
| 예측기 | 6층, 헤드 16, 드롭아웃 0.1, 약 10M (§3.1) | `ARPredictor`, depth 6, heads 16, dim_head 64, mlp 2048, dropout 0.1 (`lewm.yaml:17–21`) | 일치 (§3.1 기준). App D와는 불일치 D4 |
| 예측기 기반 | App D: "ViT-S backbone" | 폭 192의 자체 구현 `module.py` `ARPredictor` | **불일치 D4** |
| 행동 주입 | AdaLN, 0 초기화 (§3.1) | `ConditionalBlock`, `adaLN_modulation` 마지막 층 0 초기화 (`module.py`) | 일치 |
| 예측 손실 | `||ẑ − z||²₂` (식 1) | `(pred_emb − tgt_emb).pow(2).mean()` (`train.py:39`) | **불일치 D3**: 평균(÷ D) 대 식의 합 |
| 교사 강요와 정지 그래디언트 | 정지 그래디언트 없음 (§3.1) | `tgt_emb = emb[:, n_preds:]`에 `detach` 없음 (`train.py:31`) | 일치 |
| SIGReg 투영 | M = 1024 무작위 단위 방향 (§3.1, 부록 A) | `A = randn → div_(norm)`, 호출마다 새로 뽑음 (`module.py:29–31`) | 일치 |
| SIGReg 매듭과 구간 | 부록 A: "T nodes uniformly in [0.2, 4]" | `linspace(0, 3, 17)` (`module.py:16`) | **불일치 D2** |
| SIGReg 가중치 함수 | 식 (EP)의 `w(t)`에 λ 인자가 들어 있다고 쓰여 있음 | `window = exp(−t²/2)`로 λ 인자 없음 (`module.py:20`) | 논문 식의 λ 인자는 코드에서 1로 고정된 것으로 보임 (논문 설명 부족) |
| SIGReg 크기 | 식 (EP): 배치 N 곱 없음 | `statistic = (err @ weights) * proj.size(-2)` (`module.py:35`), `proj.size(-2)` = B | **불일치 D2**: 배치 N 곱 |
| SIGReg 적용 축 | 매 시간 스텝마다 (Alg 3, step-wise) | `self.sigreg(emb.transpose(0, 1))`, 입력 (T, B, D) (`train.py:40`) | 일치 (D21) |
| 총 손실 | `L_pred + λ·SIGReg` (식 3) | `loss = pred_loss + lambd * sigreg_loss` (`train.py:41`), `lambd = cfg.loss.sigreg.weight` | 일치, 단 λ 값 D1 |
| λ | 기본 0.1 (§3.1) | `weight: 0.09` (`config/train/lewm.yaml:40`) | **불일치 D1** |
| 계획 비용 | `C = ||ẑ_H − z_g||²` (식 4) | `criterion`: 마지막 스텝의 MSE 합 (`jepa.py`) | 일치 (D18) |
| 계획 설정 | horizon 5, 블록 5, MPC 재계획 5 (App D) | `pusht.yaml`/`tworoom.yaml`/`cube.yaml`/`reacher.yaml`: horizon 5, receding 5, action_block 5 | 일치 |
| CEM | 샘플 300, 엘리트 30, 반복 30(PushT) 또는 10(그 밖) (App D) | `cem.yaml`: num_samples 300, topk 30, **n_steps 30**. 그 밖 환경은 재정의 없음 | **불일치 D8** |
| 프레임 스킵 | 5 (App D) | `frameskip: 5` (`config/train/data/*.yaml`) | 일치 |
| 서브 궤적 길이 | 4 프레임 (App D) | `num_steps = num_preds + history_size = 1 + 3 = 4` | 일치 |
| 히스토리 | 3(PushT, Cube), 1(TwoRoom) (App D) | `history_size: 3` 전역 (`lewm.yaml:15`), TwoRoom은 재정의 없음 | **불일치 D7** |
| 학습 에포크 | 모든 세계모델 10 에포크 (App E) | `max_epochs: 100` (`lewm.yaml:18`), 데이터 설정에 재정의 없음 | **불일치 D6** |
| 배치 | 128 (App D) | `loader.batch_size: 128` | 일치 |
| 드롭아웃 | 0.1 (§4.3, 표 9) | `dropout: 0.1` (`lewm.yaml:21`) | 일치 |
| 학습·검증 분할 | 언급 없음 | 90 / 10, 시드 3072 (`train_split: 0.9`, `seed: 3072`) | 논문 미기재 (D9) |
| 평가 에피소드 | 50 (v3 §4.1, 표 5) | `num_eval: 50`, 시드 42 | 일치 |
| 목표 거리, 예산 | v3: TwoRoom 25 / 50, PushT 25 / 50 (App F.1) | `goal_offset_steps: 25`, `eval_budget: 50` (모든 평가 설정) | 일치 (v3 기준). v1·v2와 다름 D12 |

## 3. 학습 설정 (논문에 없는 값)

| 항목 | 공식 설정 | 논문 |
|---|---|---|
| 옵티마이저 | AdamW, lr 5e-5, weight decay 1e-3 (`lewm.yaml:31–34`) | 기재 없음 (D9) |
| 스케줄 | LinearWarmupCosineAnnealingLR, 에포크 단위 | 기재 없음 |
| 정밀도, 그래디언트 클리핑 | bf16, clip 1.0 (`lewm.yaml:21–22`) | 기재 없음 |
| 입력 | 224 px, 프레임 간 정규화(`ZScoreNormalizer`, `utils.py`) | 224 px만 기재 |
| 행동 정규화 | 열별 z-score (`utils.py`의 `get_column_normalizer`) | 기재 없음 |

재현에 필요한 이 값들은 논문에 빠져 있습니다. 공식 코드에만 있습니다. 이 점은 재현의 장벽입니다.

## 4. 불일치 목록 (D1–D18)

**D1. λ 값.** 논문 §3.1은 기본 λ = 0.1이라고 씁니다. 공식 설정은 0.09입니다(`config/train/lewm.yaml:40`). 그림 16은 피크가 "near λ = 0.09"라고 쓰므로, 저자들은 0.09를 골랐을 가능성이 큽니다. 영향: 손실 정규화(D3)와 함께 봐야 실제 상대 가중치가 결정됩니다.

**D2. SIGReg 수식.** 부록 A는 적분 노드를 [0.2, 4]에 두고, 식 (EP)에는 배치 N 곱이 없습니다. 공식 코드(`module.py:16`)는 [0, 3]을 쓰고, 가중치 exp(−t²/2)에 배치 B를 곱합니다(`module.py:35`). 이 차이는 SIGReg 값의 크기를 바꾸므로 λ 튜닝 결과를 논문 식만으로 옮길 수 없습니다. 독립 검산에서 구현 값은 등방 1.05, 붕괴 51.5로 구분이 큽니다(`03` §7).

**D3. 예측 손실 정규화.** 식 (1)은 `||·||²`입니다. 부록 C.1은 `(1/BT) Σ ||·||²`입니다. 공식 코드는 `.pow(2).mean()`로 **모든 원소(B·T·D)에 대해 평균**합니다(`train.py:39`). 식을 문자 그대로 옮기면(차원 합) 예측 항이 192배 커지고, λ = 0.1의 상대 가중치는 코드의 λ = 0.09보다 약 192배 작아집니다. 논문은 어느 쪽인지 밝히지 않았습니다. 재구현자는 공식 코드를 따라야 λ = 0.09와 맞습니다.

**D4. 예측기 설명.** App D는 예측기를 "ViT-S backbone"이라고 씁니다. ViT-S는 폭 384, 헤드 6, 12층 계열입니다. 공식 코드는 폭 192, 헤드 16, 6층, 자체 `ARPredictor`입니다. §3.1의 "6층, 16헤드, 약 10M"과는 맞지만 App D와는 맞지 않습니다. 파라미터 수는 6층 × (어텐션 약 0.8M + MLP 0.8M + AdaLN 0.2M) ≈ 10M으로, 코드와 §3.1이 맞습니다(추정 계산).

**D5. 투영 층 수.** §3.1은 "1-layer MLP with BN"이라고 씁니다. 공식 설정은 은닉 2048의 두 선형 층 사이에 BN과 GELU를 둡니다(`lewm.yaml:33`). 예측기 뒤에도 같은 MLP(`pred_proj`, `lewm.yaml:42`)가 있습니다. App D는 "predictor is followed by a projector network"라고 씁니다. 이 부분은 맞지만 "1-layer"는 틀렸습니다.

**D6. 에포크 수.** App E는 네 데이터셋 모두 10 에포크라고 씁니다. 공식 설정의 `max_epochs`는 100입니다(`lewm.yaml:18`). 데이터 설정(`config/train/data/*.yaml`)과 학습 스크립트에 재정의가 없습니다. 논문 결과가 10 에포크인지 100 에포크인지 알 수 없습니다. 부록 D는 "10 epochs are sufficient"라고만 씁니다.

**D7. TwoRoom 히스토리.** App D는 TwoRoom에서 히스토리 1이라고 씁니다. 공식 설정은 전역 3입니다(`lewm.yaml:15`), TwoRoom 설정에는 재정의가 없습니다(`config/train/data/tworoom.yaml`). 평가 설정에서 TwoRoom의 히스토리가 무엇인지도 확인되지 않습니다.

**D8. CEM 반복.** App D는 PushT 30, 나머지 10이라고 씁니다. 공식 `cem.yaml`의 `n_steps`는 30이고, 다른 평가 설정에는 재정의가 없습니다. 실제로 10을 쓰려면 실행 인자로 바꿔야 합니다.

**D9. 논문에 없는 학습 값.** 옵티마이저, lr, 감쇠, 정밀도, 클리핑, 분할 비율, 시드는 공식 설정에만 있습니다(§3).

**D10. 하드웨어와 CUDA 하드코딩.** v1과 v2는 "단일 NVIDIA L40S GPU"라고 썼습니다. v3는 이 문장을 뺐습니다. 공식 코드는 `eval.py:89`에서 `model.to("cuda")`를 호출하고, `config/eval/solver/*.yaml`의 `device: "cuda"`(`cem.yaml:8`, `adam.yaml:7`)를 씁니다. `eval.py:3`은 `MUJOCO_GL=egl`을 설정합니다. EGL은 macOS에 없어서 맥에서는 이 설정을 바꿔야 합니다. 학습은 `accelerator: gpu`와 `precision: bf16`입니다(`lewm.yaml:20–21`).

**D11. 데이터 파일 이름.** `config/train/data/pusht.yaml:4`는 `pusht_expert_train.lance`를 씁니다. README는 `.h5`라고 쓰고, HF 데이터 파일은 `pusht_expert_train.h5.zst`(13.1 GB)입니다. 저장소 안에서도 확장자가 맞지 않으므로, `.lance` 경로는 HF 파일과 맞지 않습니다.

**D12. TwoRoom 평가 프로토콜과 그림 6.** v1·v2는 TwoRoom 예산 150, 목표 거리 100이었습니다. v3는 50과 25입니다. 공식 `config/eval/tworoom.yaml`은 v3 값(`goal_offset_steps: 25`, `eval_budget: 50`)입니다. 그런데 그림 6의 TwoRoom 수치(LeWM 87, PLDM 97, DINO-WM 100 등)는 v1과 v3에서 **그대로**입니다(`01` §6). 그림을 새로 그리지 않았거나, 두 설정에서 결과가 같았는지 논문에서 확인할 수 없습니다.

**D13. PLDM 손실 계수.** 논문 표 2의 계수(α = 18, β = 12, γ = 0.2, ζ = 0.7, ν = 0, µ = 0)는 stable-worldmodel의 `pldm.yaml`(std 18, cov 12, temp_align 0.2, std_t 0.7, cov_t 0.0, idm 0.0)과 대응합니다. 즉 IDM 항과 시간 공분산 항은 0입니다. 논문은 PLDM을 "7항 목적함수"라고 하지만, 최적 설정에서는 예측 항을 포함해 **활성 항이 다섯 개**입니다. 이 점은 논문 §2와 부록 C.2에 적혀 있지 않습니다. (확인: `scripts/train/config/pldm.yaml`의 `loss:` 블록.)

**D14. LeJEPA 체크포인트.** 저장소 README 표에 `lejepa` 기준선(네 환경 모두)이 있습니다. 논문 결과에는 LeJEPA가 없습니다. 기준선으로 쓰였는지, 어느 표에 있는지 논문에서 찾지 못했습니다.

**D15. 프로브, 디코더, VoE 코드 없음.** 저장소 파일 목록(`LICENSE, README.md, utils.py, train.py, jepa.py, module.py, eval.py, config/`)에 프로브 학습, 디코더, VoE, t-SNE, 직선화 측정 코드가 없습니다. 따라서 표 1, 3, 4와 그림 7–10, 13, 17, 8, 12, 14는 이 저장소만으로 재현할 수 없습니다.

**D16. 기준선 학습 코드 위치.** DINO-WM, PLDM, GCBC, GCIQL, GCIVL의 학습 스크립트는 stable-worldmodel의 `scripts/train/`에 있습니다(파일 목록 확인: `gcbc.py`, `gciql.py`, `gcivl.py`, `pldm.py`, `lewm.py`, `prejepa.py`, `genie.py`, `hilp.py`, `tdmpc2.py`, 설정 `*.yaml`). 이 분석은 그 본문을 읽지 않았습니다.

**D17. 그림 6의 DINO-WM 두 판.** 공개 체크포인트 표는 DINO-WM(proprio 포함)을 Two-Room과 Push-T에만 줍니다. 그림 6의 회색 DINO-WM+prop 막대도 그 두 환경에만 있습니다. 파란 DINO-WM(noprop)은 네 환경 모두 있습니다. 두 그림이 맞습니다. 다만 표 5의 "DINO-WM 92.0"이 proprio 판인지 본문에 적혀 있지 않습니다(`03` §3).

**D18. 일치한 부분(참고).** 인코더의 [CLS]와 HF 마지막 LN, 식 (4)의 마지막 스텝 비용, 계획 설정, 평가 설정(에피소드 50, 목표 25, 예산 50, 시드 42), 드롭아웃 0.1, 배치 128, 서브 궤적 4, SIGReg 매 스텝 적용, 예측 항의 교사 강요 방식이 맞습니다. `proprio`와 `state`는 학습 설정에서 로드되지만 `encode`와 손실에는 쓰이지 않습니다. 따라서 픽셀 전용 주장과 맞습니다.

추가로 발견한 사소한 것: 평가 스크립트의 docstring이 "dinowm vs random"이라고 복사되어 있습니다(`eval.py`). 에피소드 샘플링은 `choice(len(valid_indices) - 1, …)`이라 마지막 유효 인덱스를 뺍니다(영향 작음). 논문 Alg 3의 의사코드는 `F.mse_loss`의 인자와 괄호가 빠져 있습니다.

## 5. 저자에게 물을 질문

1. λ는 0.1인가 0.09인가? 예측 손실을 차원 평균으로 쓰는가, 차원 합으로 쓰는가?
2. SIGReg 적분 구간은 [0.2, 4]인가 [0, 3]인가? 배치 곱은 식에 있어야 하는가?
3. 모든 세계모델을 10 에포크 학습했는가, 공식 기본 100 에포크를 썼는가?
4. TwoRoom 히스토리는 1인가 3인가? 평가 예산과 목표 거리를 v1 기준에서 v3 기준으로 바꿨는데, 그림 6이 그대로인 이유는?
5. CEM 반복 수는 환경별로 10과 30 중 어느 쪽인가?
6. 표 5, 6, 9, 10의 "±"는 모집단인가 표본인가? 시드 수와 에피소드 수는? 표 10의 PLDM SGD "4.67 ± 0.06"은 어떻게 나왔는가?
7. 표 4 Overall은 어떻게 집계했는가?
8. 그림 3의 고정 FLOPs 설정은 무엇인가? 그림 3의 DINO-WM Cube 48과 그림 6의 86은 왜 다른가?
9. 프로브와 VoE 코드, 디코더를 공개할 계획이 있는가? 기준선 설정을 본 저장소에 넣을 계획이 있는가?
10. LeJEPA 기준선은 논문 어느 곳에 쓰였는가?

## 6. 구현 가능성 (맥 기준, 측정 없음)

이 맥은 Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 사용 가능, CUDA 없음입니다(맥 사양 기준). 아래는 **코드와 파일 크기를 읽은 판단**이며, 실행 시간은 측정하지 않았습니다.

**구현할 수 있는 것 (맥, 조건 없음에 가까움):**
- 모델 구조(인코더, 예측기, 투영), 두 항 손실, SIGReg, CEM 계획. 모두 PyTorch 표준 연산이고 CUDA 전용 커널이 없습니다. 파라미터 약 15M이라 메모리는 충분합니다(가중치 약 72 MB).
- SIGReg 수식의 독립 구현과 검산은 이미 했습니다(`03` §7).

**조건부로 가능한 것:**
- **공개 체크포인트 추론 (PushT, TwoRoom, Reacher):** 필요한 것은 (a) stable-worldmodel과 MuJoCo 설치(설치 승인 필요), (b) `eval.py`의 `cuda` 두 곳을 `mps` 또는 `cpu`로 바꾸고 `MUJOCO_GL`을 맥 설정으로 바꾸는 패치, (c) 가중치 72 MB와 평가용 데이터(Push-T 13.1 GB, TwoRoom 3.4 GB, 압축 상태), (d) 평가 데이터를 읽는 파이프라인입니다. 맥에서 PushT 추론 한 환경 검증까지 노력은 **2–4일 추정**입니다.
- **그림 6 LeWM 열 재현(추론만):** 네 환경(Cube 포함, 46.2 GB 데이터)의 평가를 맥에서 돌리는 것은 시간이 오래 걸릴 수 있습니다. 계획 한 번의 시간은 측정되지 않았습니다. 추정 **1–2주**.
- **프로브, VoE, 디코더 재구현:** 코드가 없으므로 직접 짜야 합니다. 그림 8과 표 1의 방향은 확인할 수 있지만 같은 수치는 기대하기 어렵습니다. 추정 **1–2주**.

**맥에서 어려운 것:**
- **처음부터 학습:** 프레임당 ViT-Tiny(패치 14, 256 토큰)의 순전파와 역전파를 기준으로 계산량을 어림하면, 128 배치의 한 스텝은 약 수 TFLOP입니다. 맥의 실효 처리량은 측정하지 않았으나 수 TFLOPS 이하일 가능성이 큽니다. 따라서 **한 에포크가 수 시간 이상**일 수 있고, 10에포크면 수십 시간, 100 에포크면 비현실적입니다. 이 추정은 측정 전 판단이며 오차가 큽니다. 논문은 "a few hours on a single GPU"라고 쓰지만, 그 GPU 종류와 에포크 수가 적혀 있지 않습니다(D6).
- **기준선 재현(DINO-WM, PLDM, GCIQL):** 체크포인트는 Google Drive에 있고 받지 않았습니다. 재학습은 GPU가 필요합니다. 따라서 그림 6의 기준선 열은 맥에서 재현할 수 없습니다.
- **데이터:** Cube 46.2 GB, Reacher 23.8 GB 압축 파일. 디스크 승인이 필요합니다.

**판정:** **부분 구현 가능.** 모델과 계획은 맥에서 옮길 수 있습니다. 공개 체크포인트로 PushT와 TwoRoom의 추론 수치를 확인하는 것은 조건부로 가능합니다(설치, 패치, 13 GB 데이터 승인이 먼저). 처음부터의 학습과 기준선 재현은 맥에서 어렵습니다. 노력 추정은 PushT 추론 검증 2–4일, 네 환경 추론 1–2주, 프로브·VoE 재구현 1–2주입니다.
