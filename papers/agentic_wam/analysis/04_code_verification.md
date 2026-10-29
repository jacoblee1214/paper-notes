# 04. 논문 ↔ 코드 대조와 주장 검증

- **공식 코드:** https://github.com/ZhuoyangLiu2005/agentic-wam
  - 커밋 `275818e5787d59012eba4c43504342f1201ad25b` (2026-10-27)
  - BSD-2-Clause, Python 약 74k줄
  - `code_official/`에 clone했고, git에서는 제외합니다.
- **공개된 것:**
  - 학습·추론·배포 코드 전체
  - RoboTwin 데이터 준비 도구, 하이브리드 평가 도구, 코드 라벨 파이프라인
  - 사전학습 체크포인트, RoboTwin 사후학습 체크포인트(WAM, hybrid)
- **공개되지 않은 것:**
  - 사전학습 데이터(약 2,900시간)
  - 실제 로봇 사후학습 데이터와 체크포인트
  - 실제 로봇 평가 영상·채점 기록, 시드별 결과

## 1. 논문과 코드가 일치하는 부분

| 항목 | 논문 | 코드 | 위치 |
|---|---|---|---|
| 세 전문가 MoT, C → V → A mask | §2.1 | 코드는 attention 섬(ViT prefix + 지시문 + canvas만), video는 코드를 읽고, action은 모두 읽음 | `agenticwam_model.py` `_build_mot_attention_mask` |
| 코드 확산 목표 | LLaDA식 masked diffusion | `code_objective: joint` (기본), `ar`, `blockdiff` 선택 | `configs/model/agenticwam_cosmos.yaml` |
| video noise | u = 0.999·Beta(7,1), shift 5 | `video_noise_beta: [7, 1, 0.999]`, `sigma_shift: 5` | 같은 파일 |
| Noise 재표본화 | α = 0.5, w_v = p_pre/q, w_g = U/q, clip 4, 배치 평균 정규화 | 같은 식, 층화 표본(배치의 round(αB)개를 p_pre에서) | `_sample_video_t_rw` |
| 손실 가중치 | (v, c, a) = (1, 0.5, 1) | `loss_lambda_video 1.0, code 0.5, action 1.0` | model yaml |
| 코드 전문가 동결 | SDAR 임베딩·LN·FFN·출력층 동결, attention·modulation 학습 | `freeze_backbone=True` 기본, `self_attn`·`modulation`은 항상 학습 | `code_expert.py` |
| 사전학습 | 배치 6 × 48 GPU × 누적 9, lr 1e-4, 1 epoch, 지시문·프로그램 마스크 15% | `batch_size 6`, `gradient_accumulation_steps 9`, `learning_rate 1e-4`, `num_epochs 1`, `AGENTICWAM_CFG_DROPOUT=0.15,0.0` | `pretrain_magic_soup.yaml` |
| 하이브리드 실행에 특권 정보 없음 | 부록 J.2 | 공개 평가 스크립트가 `AGENTICWAM_HYBRID_PERCEPTION=1`(Molmo + SAM 3 + GraspNet)과 `AGENTICWAM_NN_REPAIR=0`으로 설정 | `robotwin/hybrid_eval/launch.py` |

## 2. 논문과 코드가 다른 부분 (D1–D11)

| # | 항목 | 논문 | 공개 코드 | 영향 |
|---|---|---|---|---|
| D1 | **"w/o code" 변형의 시각 입력** | 코드 전문가만 뺀 것으로 서술(§3.3 Q1) | `use_code_expert=false`이면 Qwen3-VL ViT prefix도 붙지 않음(`build.py`: "the vision prefix only feeds the code expert") | 32.9 → 71.1 차이에 ViT 효과가 섞임 (`03` §3) |
| D2 | **메모리 캐시** | 학습형 K개 슬롯, 확률 p로 자기 예측 사용(§2.3) | `code_memory: false`이고, 공개 설정 어디에서도 켜지 않음. 대신 프롬프트에 이전 코드 조각을 텍스트로 넣거나(`# Completed:`), 개수 + 마지막 조각만 넣음. RoboTwin 하이브리드는 이력 없음(`code_slices_no_history: true`) | 논문 결과가 학습형 메모리로 얻은 것인지, 공개 설정이 논문 실험과 같은지 확인 불가 |
| D3 | **joint denoising 단계 수** | 4 (Table 6), 지연 0.23 s | 실물 배포 서버 기본 10, RoboTwin WAM 20, hybrid 40 | 공개 설정으로 RoboTwin을 돌리면 지연이 0.23 s보다 클 것. action이 읽는 video의 선명도도 다름 (`03` §5) |
| D4 | **코드 canvas** | 추론 128 토큰 | 실물 64, RoboTwin WAM 128, hybrid 124(학습 canvas 고정, EOS로 채움) | 설정 주석: "canvas가 크면 분포 밖이라 모델이 filler로 채움" |
| D5 | **RoboTwin 학습 설정** | 2노드 × 8 GPU × 배치 6 = 96, lr 2e-4, 100k step (부록 A) | WAM: GPU당 배치 2, lr 1e-4, 200k step. hybrid: 3노드 × 8 × 4 = 96, lr 2e-4, 15k step | 논문 숫자를 낸 설정이 어느 쪽인지 불명확 |
| D6 | **기본 모드 vs 하이브리드 모드의 차이** | "학습 데이터 수집 방식과 실행 방식이 다르다"(부록 J.2) | 그 외에도 학습 step(200k vs 15k), 시작 체크포인트(`robotwin_wam_init_ckpt` vs `robotwin_hybrid_init_ckpt`), 추론 단계(20 vs 40), 코드 디코딩(신뢰도 임계 0 vs 0.5, 단계당 공개 상한 없음 vs 4), 코드 이력(있음 vs 없음)이 다름 | clean −17.7, random +8.6의 원인을 "하이브리드 실행"에 돌릴 수 없음 |
| D7 | **하이브리드 코드 수리** | "생성된 조각은 수리 없이 검사한다"(부록 I) | arbiter가 실행 전에 규칙 기반 `repair`를 적용. 컴파일을 막는 "잔해" 줄(식별자만 있는 줄 등)을 주석 처리(R1). 코드처럼 보이는 줄을 주석 처리해야 하면(R2) 거부. 최근접 학습 조각으로 바꾸는 `NN_REPAIR`도 구현되어 있으나 공개 평가에서는 끔 | R1 수리가 통과율을 얼마나 올렸는지 논문에 없음 |
| D8 | 실제 로봇 사후학습 길이 | 20 epoch, 시연 100개 | `max_steps: 100000`, 배치 4 | 에피소드 길이를 모르므로 같은지 확인 불가 |
| D9 | RoboTwin 평가 횟수 | 과제당 50회 | 공개 벤치마크 계획 `standard`는 과제당 100회(리더보드와 같음) | Table 11 값이 모두 2% 단위이므로 논문 결과는 50회와 정합 |
| D10 | 실제 로봇 학습 시드 | 3개 (§3.2) | `seed: 42` 하나만 설정, 시드 반복 스크립트 없음 | 시드별 결과·분산을 재현할 도구가 없음 |
| D11 | ViT merger 저장 버그 | merger는 학습 가능(부록 A) | 주석 "FIX 2026-09-03: memory + ViT merger are saved", `AGENTICWAM_FREEZE_VIT_MERGER=1`은 "수정 전 ViT 실행과의 비교 가능성"을 위한 옵션 | 2026-09-03 이전 체크포인트에는 학습된 merger가 저장되지 않았을 수 있음. 어느 결과가 영향을 받았는지는 알 수 없음 |

## 3. 수치 정합성 (V1–V5)

| # | 확인 | 결과 |
|---|---|---|
| V1 | Table 11 → Table 1 평균 12개 | 모두 일치(최대 차 0.04) |
| V2 | Table 11 600칸이 50회 시행 격자에 있는가 | 모두 2% 단위 |
| V3 | 본문 수치·그림 평균 36개 | 모두 일치 |
| V4 | Table 12의 전체 = 성공·실패 가중 합 | step 수와 코드 비율 모두 일치, 역산 성공률 63.12%·29.24% |
| V5 | 실제 로봇 점수의 루브릭 격자 | AW-0 8칸 모두 격자 위, 배점 0.1 과제 72칸 중 54칸(무작위 기대 약 35칸) |

## 4. 주장별 판정

판정: ✅ 지지됨 · ⚠️ 부분적으로 지지되거나 해석에 주의 · ❌ 근거 부족

| # | 주장 (위치) | 판정 | 근거 |
|---|---|---|---|
| C1 | RoboTwin에서 평균 성공률이 가장 높다(50.7%) (§3.1) | ⚠️ | 논문이 재현한 기준선 중에서는 맞음. RoboTwin 공식 리더보드의 π0.5(같은 학습 데이터)는 평균 58.3. 리더보드에 먼저 등록된 27개 항목 중 13위 (`03` §2) |
| C2 | clean에서 WLA보다 16.0점 높다 (§3.1) | ✅ | 과제 단위 부호 검정 p = 0.005, 과제별 Fisher로 21개 과제 유의하게 좋음, 1개 나쁨 |
| C3 | 하이브리드 실행이 분포 이동에 강건하다(random 20.6 → 29.2) (§3.1) | ⚠️ | 이득의 65%가 5개 과제에서 나옴(과제 단위 부호 검정 p = 0.54). 학습 데이터·step·디코딩이 모두 달라 실행 방식의 효과로 분리 불가(D6) |
| C4 | 하이브리드가 random에서 π0.5를 포함한 모든 기준선보다 높다 (§3.1) | ❌ | 논문 π0.5(27.7) 대비 과제 단위로 25승 21패(p = 0.66). 리더보드 π0.5(46.0)보다 16.8점 낮음 |
| C5 | 실제 로봇 8과제에서 기준선보다 25.1점 높다 (§3.2) | ✅ | 8과제 평균 차의 z가 분산 상한 기준 7.7. 과제별로도 6/8에서 표본 오차를 넘음. 다만 시드별 값 미보고 |
| C6 | 코드 기반 계획이 언어 이해·의미 추론을 강화한다 (§3.2, §3.3 Q1) | ⚠️ | w/o code(32.9) 대비 이득의 80%는 자연어 subgoal로도 얻음. w/o code 변형은 ViT도 없음(D1) |
| C7 | C → V → A 순서가 낫다 (§3.3 Q1) | ✅ | 48.3 vs 71.1. 7/8 과제에서 낫고 Replace Lightbulb만 V → C → A가 높음(77.1 vs 75.0). 본문은 이 예외를 언급하지 않음 |
| C8 | ViT가 VAE보다 코드 전문가 입력으로 낫다 (§3.3, Table 3) | ✅ | 8/8, 평균 +15.6 (z 상한 4.7) |
| C9 | 주석·스킬·코드 형식이 모두 기여한다 (§3.3 Q2, Fig. 7) | ⚠️ | 방향은 일관되지만 과제별 차이는 모두 표본 오차 상한 이내. 평균 차 7.5–8.3점(z 상한 2.3–2.5), 시드 변동 미반영 |
| C10 | joint denoising이 AR보다 좋고 2.39배 빠르다 (§3.3 Q3) | ✅ (조건부) | 7/8, 평균 +16.6. 지연은 4단계 설정에서의 값. 공개 설정(10–40단계)의 지연은 미보고(D3) |
| C11 | 코드를 더해도 지연은 0.04 s(21.1%)만 늘어난다 (§3.3) | ✅ (조건부) | 같은 4단계 조건의 측정값. 산술은 맞음 |
| C12 | 실제 로봇에서 하이브리드가 action chunk보다 낫다(90.6 vs 84.4) (§3.2) | ⚠️ | 과제 1개, 시행 수 미기재. n = 48이어도 z 상한 0.9 |
| C13 | 사전학습한 코드 전문가가 시각에 근거해 코드를 만든다 (부록 F.1) | ✅ (정성) | 관측을 섞으면 코드 손실이 1.774 증가. 다만 primitive F1 0.365, 스킬 호출 비율 29.2%로 의미 일치는 부분적 |

## 5. 구현 가능성

### 5.1 공개 체크포인트로 RoboTwin 재평가
- 필요한 것:
  - 공개 RoboTwin 체크포인트(WAM, hybrid)
  - RoboTwin 2.0 시뮬레이터
  - 하이브리드 모드라면 Molmo·SAM 3·Contact-GraspNet 서버
- 문서 기준 1-GPU 시험에 H100 80 GB를 씁니다(`docs/ROBOTWIN.md`). **GPU가 필요해서 이 기기에서는 하지 않습니다.**
- 하면 가장 가치 있는 것은 두 가지입니다.
  - 과제당 100회(리더보드 프로토콜)로 다시 평가해서 리더보드 π0.5와 직접 비교하기
  - 같은 체크포인트로 joint 단계 수 4/20을 바꿔 보기

### 5.2 사후학습 재현
- 모델 규모는 video 2B + 코드 1.7B + 동결 ViT + action DiT입니다. 사후학습도 H100 다중 GPU 설정입니다.
- 실제 로봇 데이터가 공개되지 않아서, 실물 결과를 재현할 수는 없습니다.

### 5.3 코드 라벨 파이프라인
- `codegen/`과 `.claude/skills/`에 VLM 라벨 파이프라인이 있습니다. 다만 Gemini·Claude API 비용이 들고, 데이터를 외부로 보내야 합니다.
- 내 데이터에 적용하는 것은 CPU로 가능하지만, 비용과 외부 전송은 미리 확인해야 합니다.

### 5.4 이 기기에서 한 것
- 표와 그림 수치 재계산(`table_check.py`)
- noise 재표본화 toy(`noise_toy.py`)
- 코드 정적 대조(이 문서)
