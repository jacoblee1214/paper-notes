# π0.5: a Vision-Language-Action Model with Open-World Generalization (arXiv:2504.16054) 논문 분석

- 저자: Physical Intelligence (Black, Brown, … Zhilinsky; 공동 36명)
- 버전: arXiv **v1 하나** (2025-04-22 제출, 14,770 KB). arXiv abs 페이지의 Submission history에 v2·v3는 없음 (2026-11-21 확인).
- 학회: CoRL 2025, PMLR v305, pp. 17–40 (`proceedings.mlr.press/v305/black25a.html`). 자세한 판정은 `02_prior_work.md`와 README 참조.
- 프로젝트 페이지: https://pi.website/blog/pi05 (자동 접근이 막혀 내용은 직접 읽지 못함; 다음 절 참조)
- 공식 코드: openpi (`github.com/Physical-Intelligence/openpi`, Apache-2.0). 정적 대조는 `04_code_verification.md`.
- 원문: `paper/paper.pdf` (19쪽, 16.2 MB, 로컬 보관. 원문 PDF 자체의 재배포 권한은 아래 라이선스 절 참조)

## 0. 읽기 전 확인한 것

- 원문 PDF는 pdftotext로 본문·부록(부록 A–E)을 모두 읽었습니다. 참고문헌은 훑기만 했습니다.
- 프로젝트 페이지(pi.website)는 Vercel 보안 확인(429)에 막혀 내용을 읽지 못했습니다. 따라서 블로그 쪽의 주장은 이 분석에 쓰지 않았습니다.

## 1. 한 문장 요약

π0 위에 **이질적인 데이터 원천 여섯 가지(모바일 조작기 약 400시간, 다른 로봇 비이동형 데이터, 실험실 교차 embodiment, OXE, 고수준 subtask 라벨, 웹 VQA/캡셔닝/검출)** 를 한 transformer에 공동 학습시키고, 후반 post-training에서 flow matching 액션 전문가를 붙여 고수준 추론(subtask 텍스트 생성)과 저수준 액션 추론을 같은 모델이 하게 만든 VLA입니다. 새로운 집 3곳에서 10–15분짜리 청소 과제를 수행한 것이 주된 결과입니다.

## 2. 구성 (Sec. III–IV, Fig. 3)

### 2.1 입력·출력 분해

논문은 분포를 πθ(a_{t:t+H}, ℓ̂ | o_t, ℓ) = πθ(a_{t:t+H} | o_t, ℓ̂) · πθ(ℓ̂ | o_t, ℓ) 로 나눕니다 (Sec. IV-A). ℓ̂는 고수준 subtask 텍스트이고, 액션은 ℓ에 직접 의존하지 않고 ℓ̂에만 의존합니다. 즉 같은 모델이 (1) 고수준 텍스트 추론과 (2) subtask 조건부 저수준 액션 생성을 맡습니다.

### 2.2 두 단계 학습

| 단계 | 목적 | 액션 표현 | 스텝 수 | 데이터 |
|---|---|---|---|---|
| Pre-training | 이질 데이터 공동 학습 | **FAST 이산 토큰** (α = 0, 자기회귀 CE만) | 280k (Sec. IV-D) | MM, ME, CE, HL, WD |
| Post-training | 모바일 조작 특화 + 액션 전문가 추가 | FAST 토큰 CE + **flow matching** (α = 10) | 추가 80k (Sec. IV-D) | MM·ME 중 성공·짧은 에피소드, WD, HL 일부, **VI(언어 시연)** |

손실은 Eq. (1): 텍스트·FAST 토큰에 대한 교차엔트로피 H + α·‖ω − a − f_θa(a^{τ,ω}, o, ℓ)‖² 입니다. 액션 전문가는 smaller transformer(300M, width 1024, MLP 4096)입니다 (Appendix E).

### 2.3 데이터 원천 (Sec. IV-C, Fig. 4)

- **MM (Diverse Mobile Manipulator)**: 약 400시간, 약 100개 가정. 평가 과제와 가장 가까운 데이터.
- **ME (Multi-Environment non-mobile)**: 고정형 단·양팔 로봇을 여러 집에 설치해 수집. 다른 embodiment.
- **CE (Cross-Embodiment lab)**: 실험실 tabletop 다양한 로봇, OXE 포함.
- **HL (High-level subtask)**: 사람이 subtask 텍스트(“pick up the mitten”)와 바운딩 박스를 붙인 라벨. 저수준 액션 데이터와 같은 샘플에서 공동 예측.
- **WD (Web)**: CapsFusion, COCO 캡션; Cambrian-7M, PixMo, VQAv2 QA; 가정 물체 검출(바운딩 박스 추가 주석).
- **VI (Verbal instruction)**: post-training에서만. 전문가가 실시간으로 언어 명령을 내려 학습된 저수준 정책을 “언어로 원격 조종”한 기록. 논문 수치로 전체 HL 모바일 예시의 약 11%.

pre-training에서 전체 예시 중 **97.6%가 모바일 가정 데이터가 아니다**라는 서론의 문장이 핵심 주장입니다 (Sec. I). 이 비율은 학습 예시 수 기준이며, 수치 자체는 재계산할 자료가 없습니다.

### 2.4 아키텍처 (Sec. IV-A, Appendix E)

- 백본: PaliGemma 2B (Gemma 2B 텍스트 + SigLIP 400M 비전). 액션 전문가: Gemma 300M 크기 (width 1024, MLP 4096).
- 입력 상태 q_t는 **이산 텍스트 토큰**으로 프롬프트에 들어갑니다 (π0은 연속 토큰).
- 타임스텝은 π0처럼 sinusoidal → MLP → **adaptive RMSNorm**(각 층)으로 주입 (π0은 액션과 concat).
- 어텐션: 이미지·언어·상태 prefix는 양방향, FAST 토큰은 prefix와 이전 FAST 토큰에만 주의, 액션 전문가는 prefix와 자신에게만 주의, **VLM 임베딩은 액션 전문가를 보지 않음** (Appendix E). 정보 흐름은 VLM → 액션 전문가 단방향입니다. 이 마스크가 무엇을 막는지는 `04` D-항목에서 다룹니다.
- 액션 청크 H = 50 (Appendix E는 “H = 49”로 적혀 있으나 이는 인덱스 표기). 추론 시 텍스트를 자기회귀로 생성한 뒤 10 denoising 스텝으로 액션 생성.
- 타임스텝 샘플링: Beta((s−τ)/s; 1.5, 1), s = 0.999 (Appendix E, π0 따름).
- 이미지 증강: random crop 95%, resize, rotate ±5°, color jitter(brightness 0.3, contrast 0.4, saturation 0.5).

### 2.5 로봇과 제어 (Sec. IV-E, Fig. 5)

두 종류의 모바일 조작기 (양팔 6 DoF + 평행 그리퍼, 손목 카메라 2대, 전·후방 카메라 2대, 홀로노믹 베이스, 토소 리프트). 상태·액션 18 또는 19 차원. 4대 카메라 중 고수준 추론은 4대 모두, 저수준은 손목 2대 + 전방 1대를 씀. 목표 자세·베이스 속도를 50 Hz 액션 청크로 내고 PD로 추종. 궤적 계획이나 충돌 검사는 없음.

## 3. 실험 (Sec. V)

| 질문 | 설정 | 결과 요약 (논문 기술) | 그림 |
|---|---|---|---|
| Q1 실제 집 | 새 집 3곳, 청소 과제 | 여러 단계 과제에서 일관된 성공 (그림 7, 수치는 그림에만) | Fig. 7 |
| Q2 장면 수 | 3, 12, 22, 53, 82, 104 위치 | 장면 수 증가에 따라 평균 개선. 104 위치 모델은 테스트 집을 학습에 넣은 대조군과 비슷 | Fig. 8, 9 |
| Q3 혼합 구성 | no WD / no ME / no CE / no ME·CE | ME 또는 CE 제거 시 크게 악화. WD 제거는 mock 과제 평균에서 비유의 (p = 0.385) | Fig. 10, 11 |
| Q4 π0 비교 | π0, π0-FAST+Flow | π0.5가 모든 과제에서 우위 (Fig. 12). 300k 스텝 π0도 우위 | Fig. 12, 15 |
| Q5 고수준 추론 | GPT-4 HL, implicit HL, no HL, human HL | 전체 모델이 최고, human oracle보다도 높다고 보고. VI 제거와 WD 제거가 큰 손실 | Fig. 13, 17 |

평가 프로토콜 (Appendix B): 4개 과제(접시 넣기 8점, 서랍에 넣기 4점, 빨래 3점, 침대 정리 5점), 과제당 10회, 12개 위치(실제 집 3곳 + 모의 주방 3곳 + 모의 침실 3곳 등, 설명이 일부 모호함). 정책당 40 평가. 검정은 양측 t-test라고만 씀 (`03` V-항목 참조).

## 4. 주장과 근거의 강도

| 주장 | 근거 종류 | 평가 |
|---|---|---|
| 97.6% 예시가 모바일 가정 데이터가 아님 | 학습 예시 집계 (수치만 있음) | 확인 불가 (데이터 미공개) |
| 새 집에서 10–15분 과제 수행 | 실제 집 3곳 시연 (Fig. 2, 7) | 정성적 시연 + 루브릭 점수. 횟수와 분산은 그림에만 있음 |
| 다른 embodiment 데이터가 핵심 | Fig. 10 ablation (mock, 과제당 10회) | 방향은 뚜렷함. 세부 p값·분산 표기 없음 |
| FAST 토큰 사전학습이 flow 단독보다 compute 효율적 | Fig. 12 (π0 300k 대비) | 비교 설정이 다름 (데이터·단계 수가 “comparable”이라고만 함) |
| 고수준 추론 자체가 이득 | Fig. 13 (implicit HL과 비교) | implicit HL도 높아 “추론 자체의 이득”은 작다고 논문도 인정 |
| 장면 수 증가가 일반화를 높임 | Fig. 8, 9 (40k 스텝 고정) | 단일 실행, 신뢰구간 없음 |

## 5. 한계 (논문 Sec. VI과 본문 관찰)

- 논문 스스로 든 한계: 낯선 손잡이와 무거운 캐비닛, 가림(occlusion)으로 보이지 않는 엎질러짐, 서랍을 반복해서 여닫는 식의 고수준 subtask 혼동, 짧은 문맥(memory 부족), 단순한 프롬프트.
- 실험 설계상 한계: 정량 평가가 mock 환경 4개 과제에 집중. 실제 집 평가는 정성적 집계. 정책 간 학습 단계 수가 다름 (π0 80k·300k 대 π0.5 280k+80k).
- 재현성: 데이터·체크포인트(학습 완료본)가 논문에 공개되어 있지 않음. 공개된 것은 openpi의 **flow-only 추론·미세조정 코드와 base 체크포인트**(README 표) 뿐. 이 차이는 `04` 참조.

## 6. 라이선스와 공개 범위

- arXiv abs 페이지의 license 링크: `http://arxiv.org/licenses/nonexclusive-distrib/1.0/` (“view license”; 2026-11-21 확인). 이는 **CC BY 4.0이 아닙니다.**
- 따라서 논문 그림·표를 공개 산출물(공개 리포트, 게시본)에 넣지 않습니다. 이 분석의 리포트와 PPT도 원문 그림을 넣지 않고, 우리가 그린 그래프만 씁니다.
- PDF 중 pdftotext 전체 페이지에서 ©, copyright, All rights reserved, 회사 고지는 확인하지 못했습니다. 그래도 라이선스 판정이 먼저이므로 결과는 같습니다.
- 공개 산출물의 그림은 모두 우리가 그린 것입니다 (`03`의 Fig. 재계산 그림).

## 7. 이 논문이 말하지 않는 것

- 학습 데이터 목록과 분포, 주석 절차 상세 (라벨러 지침, VI 수집 절차의 품질 관리 수치).
- 실제 집 평가의 시도 횟수, 실패 원인 분류, 과제 선정 기준.
- post-training에서 VI 데이터의 정확한 비율과 HL 데이터 비율 (11% 외에는 없음).
- 체크포인트 공개 여부와 학습 하이퍼파라미터 (배치 크기, 학습률). 이 값들은 openpi의 fine-tune 설정에서 확인하며 논문 값과 다를 수 있음 (`04`).
