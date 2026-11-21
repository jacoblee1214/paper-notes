# DreamerV3: Mastering Diverse Domains through World Models (arXiv:2301.04104) 논문 분석

*Mastering Diverse Domains through World Models* (Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap; Google DeepMind, University of Toronto). 분석 대상은 **arXiv v2 (2024-04-17, 40쪽)**입니다. 저널판 제목은 *Mastering diverse control tasks through world models*로 바뀌었고, Nature 640, 647–653 (2025)에 실렸습니다(§0.2). 분석일 2026-11-21.

## 0. 버전과 출처

### 0.1 arXiv 버전 기록 (abs 페이지, 2026-11-21 확인)

| 버전 | 날짜 | 쪽수 | 내용 차이 |
|---|---|---|---|
| v1 | 2023-01-10 | 38쪽 | 제목 동일. 저자 소속 DeepMind, Toronto. 알고리즘 이름은 "DreamerV3"로 통일. Atari 200M 55게임, Crafter 1과제, DMLab, Minecraft(40 seed), 소요 17 GPU일 |
| v2 | 2024-04-17 | 40쪽 | 이름을 "Dreamer"로 줄이고 Atari 57게임, ProcGen 16과제 추가, 벤치마크 표 재정리, Minecraft 10 seed·1 GPU·9일, 모델 크기를 200M 기본값으로 명시 |

- v1과 v2의 차이는 `paper/paper_v1.txt`와 `paper/paper.txt`의 단어 단위 diff로 확인했습니다. 본문 약 600개 변경 블록 중 상당수가 초록과 서론 재작성입니다.
- **Minecraft 결과 주장이 v1에서 v2로 바뀌었습니다.** v1은 40 seed 중 24개가 최고 점수(다이아몬드 포함)에 도달했다고 보고했습니다(v1 부록 G). v2는 10 seed 모두 다이아몬드를 얻었다고 씁니다(Fig. 5). 같은 실험인지, 다시 돌린 것인지는 본문에 없습니다. 이 차이는 `03` §C9에서 다시 다룹니다.
- 이 분석은 v2를 기준으로 합니다. v1 수치는 차이를 설명할 때만 씁니다.

### 0.2 학회·저널 표기

- arXiv 메타데이터에는 학회·저널 표기가 없습니다(Comments: "Website: https://danijar.com/dreamerv3"만 있고 Journal ref 없음).
- **Nature 2025로 확인했습니다.** Nature 페이지(doi 10.1038/s41586-025-08744-2)의 메타데이터는 Nature 640, 647–653 (2025), 온라인 공개 2025-04-02, 제목 "Mastering diverse control tasks through world models", CC BY 4.0 오픈 액세스입니다.
- 프로젝트 페이지(danijar.com/project/dreamerv3)도 "Nature Paper" 링크로 같은 저널판을 가리킵니다. 공식 GitHub README의 BibTeX도 `journal={Nature}`, 제목은 저널판 제목을 씁니다.
- 이 분석은 arXiv v2를 기준으로 하고, 저널판 본문은 읽지 않았습니다. 저널판은 제목과 일부 설명이 달라졌을 수 있습니다(§0.1의 v1·v2 차이를 볼 때 v2 이후 추가 수정 가능성이 있습니다).

### 0.3 라이선스와 권리 표시

- arXiv abs 페이지의 "view license" 링크는 `http://creativecommons.org/licenses/by/4.0/`이고, 로고는 `by-4.0.png`입니다. 즉 **CC BY 4.0**입니다.
- PDF 본문(v1, v2 모두)에서 ©, Copyright, All rights reserved, 기관 고지를 찾지 못했습니다. 1쪽의 "Google DeepMind"는 저자 소속 표기입니다. 권리 주장이 아닙니다.
- 판정: 논문 그림을 게시할 수 있는 조건은 갖춰졌습니다. 다만 이 게시본은 **우리가 만든 그래프만 씁니다**(§6과 `report/`). 제3자 이미지(Fig. 2의 데이터셋 장면, Fig. 4의 예측 영상)를 피하라는 원칙을 따릅니다.

## 1. 연구 질문과 주장

DreamerV3는 **하이퍼파라미터를 바꾸지 않고** 여러 영역(연속·이산 행동, 시각·저차원 입력, 2D·3D, 희소·조밀 보상)에서 전문가 알고리즘과 맞먹거나 넘는 성능을 내는 강화학습 알고리즘입니다. 주장은 네 가지로 정리됩니다.

1. **범용성:** 8개 영역, 150개 이상의 과제(초록)에서 고정 설정으로 전문가 알고리즘을 이깁니다(Fig. 1a, 표 6–13).
2. **Minecraft 다이아몬드:** 사람 데이터와 교육 과정 없이 처음부터 다이아몬드를 얻는 첫 알고리즘입니다(Fig. 1b, Fig. 5, 표 5).
3. **확장성:** 모델 크기(12M–400M)와 replay ratio를 키우면 성능과 데이터 효율이 함께 오릅니다(Fig. 6c, 6d).
4. **안정화 기법의 효과:** 로버스트 기법(symlog, symexp twohot, free bits와 KL 변형, 반환 정규화)이 각각 일부 과제에 기여합니다(Fig. 6a). 학습 신호는 재구성 손실에 크게 기댑니다(Fig. 6b).

## 2. 방법 (Learning algorithm)

세계 모델, 비평자(critic), 행위자(actor) 세 신경망을 동시에 학습합니다. 재생 버퍼의 궤적으로 세계 모델을 학습하고, 세계 모델 안에서 상상한 궤적으로 행위자와 비평자를 학습합니다.

### 2.1 세계 모델 (RSSM, 식 1–3)

- 인코더 `z_t ~ q(z_t | h_t, x_t)`, 순차 모델 `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`, 동역학 예측기 `ẑ_t ~ p(ẑ_t | h_t)`, 보상·계속 예측기, 디코더로 구성됩니다(식 1).
- 손실 식 2–3: `L_pred`(디코더, 보상은 symlog 제곱오차, 계속은 로지스틱), `L_dyn = max(1, KL[sg(q) || p])`, `L_rep = max(1, KL[q || sg(p)])`. 가중치는 β_pred = 1, β_dyn = 1, β_rep = 0.1입니다.
- **free bits 1 nat**으로 KL을 1에서 잘라, 이미 충분히 줄어든 항의 기울기를 끕니다(식 3). 범주 분포는 1% 균등 혼합으로 0 확률을 막습니다(§Methods "Distributions").
- 공식 코드와 대조한 결과(`04` §2)에서 이 항들은 설정값까지 맞았습니다.

### 2.2 비평자와 행위자 (식 4–7)

- 할인율 γ = 0.997(유효 지평 333), 상상 길이 H = 15, 람다 반환 λ = 0.95입니다(식 4–5, 표 4).
- **비평자는 값의 분포를 학습합니다.** 지수 간격 bin에 대한 범주 분포를 twohot 목표로 학습하고, 값은 분포의 기댓값으로 읽습니다. 지수적 bin 경계와 개수는 본문에 없고 코드에서 255개로 확인했습니다(`04` D6).
- 비평자 손실은 상상 궤적(β_val = 1)과 재생 궤적(β_repval = 0.3)에 모두 씁니다. 목표가 자기 예측에 의존하므로 지수이동평균 비평자로 정규화합니다(감쇠 0.98, 표 4).
- **반환 정규화(식 6–7):** 반환의 5–95 백분위 범위 S를 EMA(0.99)로 추정하고 `max(1, S)`로 나눕니다. 엔트로피 계수 η = 3×10⁻⁴를 모든 영역에 똑같이 씁니다. 작은 반환은 건드리지 않고 큰 반환만 줄입니다.

### 2.3 로버스트 예측 (식 8–12)

- **symlog**(식 9): `symlog(x) = sign(x) ln(|x|+1)`. 입력과 재구성 목표에 씁니다.
- **symexp twohot**(식 10–12): 지수 간격 bin 위의 softmax를 두 bin 사이 선형 twohot으로 학습합니다. 손실이 bin 위치 값에 의존하지 않아 기울기 크기와 목표 크기가 분리됩니다.
- 출력층 가중치는 0으로 초기화해, 학습 초기에 보상과 가치를 환각하지 않게 합니다(§Methods).

### 2.4 구현 세부 (Methods, 표 1–4)

- 재생: 균일 재생 버퍼에 온라인 큐를 씁니다(우선순위 재생은 쓰지 않음, 이유는 구현 단순화).
- replay ratio: 환경 스텝당 학습하는 시간 스텝 수입니다. 예를 들어 Atari는 replay ratio 32, 배치 16×64, action repeat 4이면 128 환경 프레임마다 그라디언트 스텝 1회입니다(§Implementation).
- 최적화기: AGC(0.3) 후 LaProp(ε = 10⁻²⁰, β₁ = 0.9, **β₂ = 0.99**라고 본문에 적음). 학습률 4×10⁻⁵(표 4).
- 네트워크: 이미지는 stride-2 합성곱으로 4×4 또는 6×6까지 줄이고, 벡터는 3층 MLP입니다. 순차 모델은 블록 대각 행렬 8개의 GRU입니다. 보상·계속 예측기는 1층입니다(§Implementation).

## 3. 실험 설정

| 벤치마크 | 과제 수 | 예산 | 반복 | 모델 | 비교 대상 (논문 값) | 출처 |
|---|---|---|---|---|---|---|
| Minecraft Diamond | 1 | 100M | 1 | 200M | IMPALA, Rainbow, PPO (직접 실행) | 표 2, 5 |
| DMLab | 30 | 100M | 4 | 200M | IMPALA 10B·1B·100M, R2D2+ 10B, PPO 100M | 표 2, 8 |
| ProcGen | 16 | 50M | 1 | 200M | Original PPO, PPO(고정 설정), PPG | 표 2, 7 |
| Atari 200M | 57 | 200M 프레임 | 4 | 200M | PPO, MuZero (표 6) | 표 2, 6 |
| Atari 100k | 26 | 400K 프레임 | 4 | 200M | SimPLe, SPR, TWM, IRIS, PPO | 표 9, 10 |
| BSuite | 23 (468 설정) | — | 1 | 200M | Boot DQN 등 | 표 2, 13 |
| DMC 고유감각 | 18 | 500K | 2 | 12M | D4PG, DDPG, DMPO, MPO, PPO | 표 2, 11 |
| DMC 시각 | 20 | 1M | 2 | 12M | DrQ-v2, CURL, SAC, PPO | 표 2, 12 |

- 모든 Dreamer 실행은 A100 GPU 한 장으로 학습했습니다(§Benchmarks, 표 2).
- 기본 seed는 Dreamer와 PPO에 5개입니다. 예외는 ProcGen 1 seed와 BSuite·Minecraft 10 seed입니다(§Seeds). 공식 저장소의 실제 실행 수는 이와 다릅니다(`03` §C15).
- **Atari 100k는 Atari 200M과 같은 게임 중 26개를 씁니다.** 과제 수 191은 프로토콜을 세면서 Atari 게임을 두 번 셉니다. 중복을 빼면 165입니다. "150개 이상"은 두 방식 모두에서 맞습니다.

## 4. 결과 (논문 주장 요약)

| 주장 | 본문 위치 | 논문의 수치 |
|---|---|---|
| Atari 200M에서 MuZero와 Rainbow, IQN보다 낫다 | Results, 표 6 | 중앙값 830%, 평균 3381% (MuZero 693%, 3054%) |
| ProcGen에서 튜닝된 PPG와 같거나 낫다 | Results, 표 7 | 정규화 평균 66.0 (PPG 64.9) |
| DMLab에서 100M 스텝으로 IMPALA 1B(10배 데이터)를 넘는다 | Results, 표 8 | 사람 대비 평균 71.4 (IMPALA 1B 66.3) |
| Atari 100k에서 IRIS, TWM, SPR 등을 넘는다 | Results, 표 9 | 평균 125, 중앙값 49 (TWM 중앙값 51) |
| 고유감각 제어에서 D4PG·DMPO·MPO보다 낫다 | Results, 표 11 | 평균 871 (DMPO 801) |
| 시각 제어에서 DrQ-v2와 CURL보다 낫다 | Results, 표 12 | 평균 861 (DrQ-v2 770) |
| 사람 데이터 없이 Minecraft 다이아몬드를 얻는다 | Minecraft, Fig. 5 | 10 seed 모두 100M 안에 다이아몬드 (Fig. 5), 에피소드 기준 0.4% (Fig. 9) |
| 모델 크기와 replay ratio를 키우면 성능과 효율이 오른다 | Scaling, Fig. 6c–d | 12M–400M에서 단조 증가 |

- 재계산 결과(`03`)와 대조하면, 이 중 상당수가 맞습니다. 특히 Atari의 PPO 대비와 ProcGen의 PPG 대비가 강합니다. 반면 Atari 100k의 TWM 중앙값, DMC의 평균·중앙값 표기, Minecraft 시드 수와 시간 환산에는 문제가 있습니다.

## 5. 한계와 논문이 말하지 않는 것

- **재현 비용:** 모든 실험이 A100 한 장으로 최대 9일입니다(Minecraft). 공식 코드는 CUDA를 전제합니다(`04` §4).
- **최종 점수 정의가 없습니다.** 표의 최종 값을 어떻게 집계했는지 본문에 없습니다. 공식 데이터로 시험하니 과제마다 마지막 3–5개 평가 점의 평균과 맞습니다(`03` §V5). 이는 논문의 한계가 아니라 재현성 기술의 빈틈입니다.
- **Atari 100k와 EfficientZero:** 본문은 EfficientZero가 최고 점수를 낸다고 인정하면서, 프레임 재설정을 이유로 비교가 어렵다고 씁니다(표 10: EffMuZero 평균 190, 중앙값 109).
- **Minecraft 기준선 구성:** IMPALA와 Rainbow만 직접 실행했습니다. 나머지(VPT 등)는 문헌값이거나 언급만 합니다.
- **DMC 기준선 측정 차이:** DMC 기준선 값은 문헌에서 가져왔습니다. 표 11·12의 평균과 중앙값 행이 뒤바뀐 것으로 보입니다(`03` §C4, C5).
- **저널판과의 차이:** 본 분석은 arXiv v2입니다. 저널판은 확인하지 않았습니다.

## 6. 그림

- 논문 그림은 게시본에 넣지 않습니다. 권리 표시는 없지만, 원칙에 따라 우리가 그린 그래프 두 장(`report/figures/own_*.png`)만 씁니다.
- 그래프 내용: (a) Atari 57 게임별 HNS, Dreamer 대 PPO. (b) Minecraft 실행별 최종 반환(공식 저장소). DMC 시각·고유감각 과제별 값.
