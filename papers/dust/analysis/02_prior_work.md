# 02. 선행연구: Dust는 어디에서 왔는가

## 1. 계보

```
[가중치 공간 0차 최적화]                         [활성값(node) 섭동]
 Random search / Gaussian smoothing               Widrow & Lehr 1990 (MADALINE, node perturbation)
 (Nesterov & Spokoiny 2017)                       GEMINI (Le Cun, Galland, Hinton 1988): 잡음 주입 + 행렬 역변환
 Evolution Strategies (Salimans et al. 2017)      Werfel, Xie, Seung 2003: node vs weight perturbation의 학습 곡선
 MeZO (Malladi et al. 2023): 미세조정               Scaling Forward Gradient with Local Losses (Ren et al. 2023)
 ES at Scale (Qiu et al. 2026): 수십억 파라미터       Zoop (Hu et al. 2025): 출력 섭동 미세조정
 ZO pretraining, KronZO (Allaire et al. 2025, 2026)  multi-tangent forward gradient (Flügel et al. 2025)
 EGGROLL (Sarkar et al. 2025): 저랭크 ES
        └──────────────────────┬────────────────────────┘
                 Dust: 토큰마다 독립인 활성값 섭동 + 토큰 단위 보상 → "가상 population"
```

- **가중치 공간 ES:** population의 구성원마다 가중치 사본과 forward가 필요합니다. 그래서 population을 늘리는 비용이 큽니다. EGGROLL은 저랭크 섭동으로 사본 비용을 줄였지만, 구성원 하나가 배치의 한 시퀀스입니다.
- **node perturbation:** 출력 차원(d_out)이 가중치 차원(d_out × d_in)보다 작다는 차원 논리가 근거입니다. Transformer에서는 토큰 T개 × d_out의 잡음이 가중치 수보다 많아서, 이 논리가 그대로 성립하지 않습니다(§2.1).
- **Dust의 관점 전환:** 토큰 축을 배치 축과 직교하는 **새 population 축**으로 씁니다. 토큰마다 보상이 따로 있으므로, forward 한 번에 T개 구성원을 평가합니다.

## 2. 구성 요소별 출처

| 구성 요소 | 가져온 곳 | Dust에서 바뀐 점 |
|---|---|---|
| 출력 섭동 → 출력 오차 → 입력과의 외적 | node perturbation (Widrow & Lehr 1990; Werfel et al. 2003), GEMINI (1988) | 토큰별 보상, 층 종류별 분리 pass, 캐시된 forward에서 블록 l부터만 재계산 |
| 미래 토큰 credit (γ) | 강화학습의 할인 return과 같은 형태 | attention 내부(k, v, gate, ve)에만 γ ≈ 0.98–0.99 |
| attention 내부의 국소 목표 | Ren et al. 2023의 local loss와 같은 발상 | 손실 대신 추정된 attention 출력 오차와의 내적으로 보상(Eq. 4) |
| 보상 중심화 | ES의 baseline / fitness shaping | 같은 forward 묶음(n = 1–16 draw)의 평균을 뺌. 크기가 (1 − 1/n)배로 줄어듦(`03` §5) |
| 모델 | modded-nanoGPT 계열(RMSNorm, RoPE, squared-ReLU, value embedding, sliding window) | 출력 투영을 0이 아닌 값으로 초기화(backprop 기준선도 같음) |
| 비교 ES | EGGROLL (Sarkar et al. 2025) | 같은 Transformer 구조로 다시 구현(EGGROLL-Transformer) |

## 3. 가까운 대안과의 차이

| 방법 | 무엇을 흔드나 | 보상 단위 | forward 한 번의 정보 | Dust와의 차이 |
|---|---|---|---|---|
| EGGROLL | 가중치(저랭크) | 시퀀스 손실 | 구성원 1개/시퀀스 | Dust는 토큰마다 구성원 |
| MeZO | 가중치(전체) | 배치 손실 | 방향 1개 | 미세조정용. 사전학습에는 약함 |
| Forward gradient (Ren et al. 2023) | 활성값 + forward-mode AD | 국소 손실 | 방향 도함수(정확) | Dust는 AD 없이 유한 차분 |
| Zoop | 출력 | 토큰 손실 | 출력 오차 추정 | 미세조정, 국소 미분 사용 |
| KronZO | 가중치(Kronecker 구조) | 배치 손실 | | 사전학습. 메모리 절감 중심 |
| 예측 부호화 PC-ALM (Seely & Gould 2026) | 없음 | 국소 동역학 | | 국소 미분 사용, 이미지 분류 |

## 4. 논문이 말하지 않은 관련 연구

- **REINFORCE / likelihood-ratio gradient:** Eq. 2는 가우스 정책의 score function 추정기와 같은 꼴입니다. 분산을 줄이는 기법인 baseline, control variate, antithetic sampling은 그 문헌에 많이 있습니다. Dust는 묶음 평균만 baseline으로 씁니다.
- **SPSA, randomized smoothing:** 유한 σ에서 추정하는 것은 실제 손실이 아니라 가우스로 평활화한 손실의 gradient입니다. 논문이 "backprop보다 낮아지는" 이유로 추측하는 "평평한 영역으로 끌리는 효과"는 이 평활화로 설명될 가능성이 있습니다(`03` §6). 논문은 이 연결을 언급하지 않습니다.
- **noise injection 정규화(Gaussian noise, dropout):** 학습 극초기(6–61 step)에 backprop보다 낮은 손실이 나오는 현상은, gradient의 질보다 잡음이 섞인 업데이트와 튜닝 차이의 영향일 수 있습니다.

## 5. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| DDPG (`ddpg/`) | 같은 "가우스 탐색 잡음 + 보상" 구조입니다. DDPG의 OU 잡음 분석(상관 시간, 정상 표준편차)이 Dust의 σ 선택과 대응합니다. Dust의 Eq. 2는 결정적 정책 gradient가 아니라 확률적 정책의 score function 추정에 해당합니다. |
| GQKAE (`gqkae/`) | 미분하기 어려운 생성기(양자 회로)를 GRPO식 보상으로 학습합니다. Dust가 노리는 "미분 불가능한 구조 학습"의 한 사례입니다. |
| PINN 연구 전반 | PINN의 잔차 손실은 자동미분이 전제입니다. 시뮬레이터나 접촉이 섞여 미분할 수 없는 물리 모듈이 루프에 들어가면, Dust식 활성값 섭동이 대안이 될 수 있습니다. 다만 연산량 문제가 있습니다(`03` §2). |
| GPT-as-Tactile-Policy, AW-0 | 둘 다 외부 프로그램(LLM 도구 호출, Code-as-Policies)이 정책 루프에 들어갑니다. 저자들이 Dust로 학습하고 싶다고 한 "외부 프로그램이 루프에 있는 망"의 예입니다. |
