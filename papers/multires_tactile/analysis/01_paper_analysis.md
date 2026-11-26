# 01. 논문 분석: MiTaS (Multi-Resolution Tactile Imitation Learning)

*Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation* (Rickmer Krohn, Erik Helmut, Niklas Funk, Jan Peters, Vignesh Prasad, Georgia Chalvatzaki; TU Darmstadt 외). arXiv:2606.06281, CoRL 2026 포스터. 쪽·절·표·그림 번호는 **arXiv v2 PDF 기준**입니다(23쪽: 본문 9쪽, 참고문헌 5쪽, 부록 A.1–A.10).

## 0. 판, 날짜, 학회, 라이선스

| 항목 | 내용 |
|---|---|
| arXiv 판 | **v1 2026-06-04**(32,439 KB, comments "20 pages, preprint")와 **v2 2026-10-21**(33,035 KB, comments "20 pages, 5 figures, CoRL 2026"). 분석 기준은 최신 v2이고, v1과의 차이는 아래 표에 정리했습니다 |
| 학회 | **CoRL 2026 포스터**. 근거: corl.org 공식 채택 목록(`https://www.corl.org/program/accepted-papers`)에 포스터 번호 143(세션 Poster post-p2, 11월 10일 화요일 17:00–18:30)으로 실려 있고(OpenReview forum `b2gSINMX3R`), arXiv v2 comments가 "CoRL 2026", PDF 1쪽 각주가 "10th Conference on Robot Learning (CoRL 2026), Austin TX, USA", 프로젝트 페이지가 "Accepted at CoRL 2026". 프로젝트 페이지는 IROS 2026 워크숍 두 곳(산업 응용 로봇 학습, 확장 가능한 촉각 센싱)의 award finalist도 적었습니다 |
| 라이선스 | arXiv abs 페이지(2026-11-26 확인)의 "view license" 링크가 `http://creativecommons.org/licenses/by/4.0/`입니다. **CC BY 4.0**입니다. PDF 1쪽과 마지막 쪽을 `pdftotext`로 읽어 ©, copyright, All rights reserved, 회사 고지를 찾았으나 없었습니다(감사의 글은 연구비 출처만 적음) |
| 코드 | **공개 없음.** 프로젝트 페이지(`mitas-touch.github.io`)에 "Code (Coming Soon)"이 있습니다. 데이터셋은 Hugging Face `RKrohn/MiTaS`(카드 license 태그 cc-by-4.0)에 있습니다(`02` §2) |

### v1 → v2 차이 (본문과 표 비교)

v1(2026-06-04)과 v2(2026-10-21)를 문장 단위로 비교했습니다. 데이터셋 1의 값(v1 표 7 = v2 표 9)은 한 칸도 바뀌지 않았습니다. 바뀐 것은 아래와 같습니다.

| 항목 | v1 | v2 |
|---|---|---|
| 평가 데이터 | 한 데이터셋, 모델·과제당 20회(Fig. 4와 모든 표) | **데이터셋 하나 더 추가**(독립 수집). Fig. 4는 두 데이터셋의 성공 수를 합쳐 과제당 40회, 부록 표 8–10 신설(총 1700회) |
| 초록 수치 | MiTaS 80%, 시각 전용 31%, 시각촉각 54% | MiTaS 80%, 시각 전용 35.5%, 시각촉각 55.5%와 63% |
| 초록 내용 | 학습만 Evetac를 쓰는 co-training이 "일부 과제에서 10% 이상" 향상한다는 문장, 프로젝트 주소 | 해당 문장 삭제, "systematic study" 문구 추가 |
| 기법 이름 | multi-tactile **co-training** (§3.3, 표 2) | multi-tactile **dropout (MTD)** (§3.3, 표 3). 학습 방식은 같고(배치의 절반에만 Evetac) 이름과 설명이 바뀜 |
| 이름 풀이 | 서론에 "Multi-Resolution *Dynamic* Tactile Sensing (MiTaS)" | "Multi-Resolution Tactile Sensing" |
| 융합 기준선 | Sparsh-X 하나 | **PolyTouch(교차 어텐션), Q-Former(학습 쿼리 32개)** 추가(표 1, 본문 "Sensor Fusion Study") |
| 관측 크기 표기 | 비전 R^(2×128×128), GelSight R^(2×120×160) | 채널 3 추가: R^(2×3×128×128), R^(2×3×120×160) |
| 토큰 수 | 그리드(16×16, 12×16)만 서술 | N_vis = 16×16, N_gs = 12×16, N_evt = 12×16과 **N_cond = 644** 명시(§3.1, 표 5) |
| 시각 전용 서술 | "Key in Lock과 Lightbulb를 풀지 못한다", 결론에 "두 과제에서 완전히 실패" | "not able to consistently solve". 결론의 "완전 실패" 문장 삭제 |
| 결론의 기준선 | "best baselines have a success rate of 54%" | "the next best sensor-fusion architecture has a success rate of 63%" |
| 표 3(구 표 2)의 평균 행 | Avg. 행(MiTaS 51→53 +2, Sparsh-X 33→37 +4) 있음 | Avg. 행 삭제 |
| 추가 문장 | - | "MTD 모델에서 추론 때 Evetac를 쓰면 성능이 약간 오른다(부록 A.8)" |
| 저자·표기 | 교신 저자 이메일 각주 | 이메일 삭제, 저자 5번째 소속(DFKI) 추가 |
| 참고문헌 | 53개 | 63개 (Nautilus, TacEx, PolyTouch, BLIP-2 등 추가) |

확인할 점(`04` C4, C5): 같은 내용이 v1에서 "시각 전용은 Key와 Lightbulb에서 0%"이던 데이터셋 1의 값이 v2에서는 데이터셋 2의 값(35–45%, 20–30%)과 섞여 서술이 "not consistently"로 바뀌었고, "Sparsh-X는 키를 여러 번 넣었지만 끝내지 못했다"는 문장은 두 판에 그대로 있습니다.

## 1. 한 줄 요약

Franka 팔의 손목에 **RGB 카메라, GelSight Mini(프레임 방식 촉각, 25 Hz), Evetac(이벤트 방식 촉각, 200 Hz)** 를 달고, 세 센서를 각자의 CNN 스템으로 토큰(총 644개)으로 만든 뒤 트랜스포머로 융합해 **플로 매칭 DiT 정책**(엔드 이펙터 Δx, Δy, Δz, Δψ 4자유도, 로봇 상태 입력 없음)을 조건화합니다. 5개 접촉 과제에서 과제당 시연 30개로 행동 복제한 정책이 평균 성공률 **80%**(두 데이터셋 각각 80/100)이고, Sparsh-X 융합 55.5%, 시각 전용 34.5–35.5%보다 높다고 보고합니다.

## 2. 문제와 기여 주장 (§1)

- 문제: GelSight류 프레임 촉각은 접촉 기하는 잘 주지만 충돌, 미끄러짐 시작, 진동 같은 빠른 변화를 놓칩니다. 이벤트 카메라 촉각(Evetac)은 시간 해상도가 높습니다. 두 센서를 함께 쓰면 접촉이 많은 조작이 나아지는가가 질문입니다(§1).
- 기여 주장 (§1):
  1. 서로 다른 시간 해상도의 촉각 센서를 추출·융합하는 구조(CNN 스템 + 트랜스포머 융합, "novel" 대신 v2는 "effective").
  2. **Multi-Tactile Dropout**: 학습 때만 Evetac를 쓰고 배치의 절반에서 가려, 추론에는 비전 + GelSight만 쓰는 시각촉각 정책을 개선("반응성이 중요한 과제에서").
  3. 다중 해상도 촉각 데이터셋(egocentric RGB + GelSight + Evetac, 5개 과제).
- "to the best of our knowledge" 첫 번째로 프레임 방식과 이벤트 방식 촉각 센서를 결합했다고 주장합니다(§1, §3).

## 3. 방법

### 3.1 관측과 인코더 (§3, §3.1, 그림 2, 부록 A.4)

| 센서 | 입력 | 스템 | 토큰 | 비고 |
|---|---|---|---|---|
| 손목 RGB(RealSense D435) | 2프레임 × 3 × 128 × 128, 25 Hz | 2D 합성곱 4단(strided) | 16 × 16 = 256, D = 256 | 두 프레임을 채널 축으로 쌓음 |
| GelSight Mini | 2프레임 × 3 × 120 × 160, 25 Hz | 비전과 같은 설계 | 12 × 16 = 192 | |
| Evetac | 16프레임 × 120 × 160(200 Hz 센서), [−1/2, 1/2] 정규화 | 3D 합성곱 4단(시간 커널 3–3–3–2, 앞 세 단 시공간 stride 2, 마지막 2×4×5) | 12 × 16 = 192 | 시간을 마지막 층에서 접음 |

- 위치 + 모달리티 임베딩: z′_i = z_i + p_g(i) + e_s(i)(학습 가능, 식 없음).
- 융합: 모든 토큰을 이어 붙인 N_cond = **644**개를 풀링 없이 트랜스포머 인코더에 넣습니다(self-attention이 센서 사이 교차 어텐션을 겸함). 깊이와 헤드 수는 적혀 있지 않습니다(`04` §1).
- 센서 입력은 시각 [0, 1], Evetac는 [−1/2, 1/2]로 정규화해 기본 회색 프레임이 0이 되게 합니다.
- 표 5가 모델별 인코더 파라미터를 줍니다: ViT-CNN 3.08 M, MiTaS V+G 5.07 M, V+E 8.77 M, V+G+E 10.76 M. 모든 모델이 같은 정책 머리 13,701,508개를 씁니다(`03` V8).

### 3.2 정책과 학습 (§3.2, 부록 A.3, 표 7)

- 출력: 길이 H = 16의 델타 명령 열 A_t ∈ R^(H×4)(Δx, Δy, Δz, Δψ. 피치와 롤은 고정). 실행은 첫 n_act = 1–3개 후 재계획. **정책은 로봇 상태를 입력으로 받지 않습니다**(그림 2 캡션, §3, §5).
- 플로 매칭: x_k = k·x_1 + (1 − k)·x_0, 목표 속도 u_k = x_1 − x_0, 손실은 ‖v̂_θ(x_k, k, C) − u_k‖²(유효 마스크 곱, 부록 표 6의 L_FM은 masked MSE). 추론은 명시적 오일러 K = 10 스텝.
- DiT 머리: 깊이 L = 10, 폭 D = 256, 헤드 8, MLP 비율 4, 시간 임베딩 128 사인파, AdaLN-Zero(셀프, 크로스, MLP마다 β·γ·g). 키/값은 융합 토큰 C, 쿼리는 행동 토큰 16개.
- 학습: 배치 64, 정책 lr 1e-4, **인코더 lr 1e-5**, 가중치 감쇠 1e-6, Adam β = (0.95, 0.999), 코사인 스케줄, 기울기 클리핑 1.0, 관측 창 n_obs = 4. **에폭이나 스텝 수는 적혀 있지 않습니다.**
- 제어: 15 Hz 명령, 관절 임피던스 1 kHz(부록 A.1, 표 4: 선형 증분 상한 0.075 m, 회전 0.15 rad, 관절 증분 0.2 rad).

### 3.3 Multi-Tactile Dropout (§3.3)

배치마다 시퀀스의 절반에는 Evetac 관측을 넣고 나머지 절반에는 넣지 않습니다. 학습은 비전 + GelSight + Evetac 구조로 하고, 배포 때는 Evetac 인코더를 빼 비전 + GelSight만 씁니다. 저자는 이것이 Evetac의 고주파 정보가 융합 표현을 정렬하는 효과를 낸다고 설명합니다. 확률 0.5 하나만 시험합니다(확률 격자 없음, `04` C9).

## 4. 실험 (§4)

### 4.1 설정

- 하드웨어: 7자유도 Franka Panda + Robotis RH-P12-RN 그리퍼. GelSight Mini와 Evetac는 두 손가락에 각각 장착. 파이프라인은 Nautilus [57], 텔레오퍼레이션은 SpaceMouse. 시연 **30개**, 제어 15 Hz, 센서 프레임은 0.06 s 창에서 균일 표본화(§4).
- 과제 5개(§4.1, 그림 3, 6): Gear Assembly(IndustReal [61]), Board Wiping, Lamp Installation(FurnitureBench [62]), Key in Lock, Lightbulb Connection(GU10 전구).
- 기준선: Sparsh-X [26](단일 모달 8층 + 융합 4층, attention bottleneck 융합), PolyTouch [46](CNN 스템 + 교차 어텐션, 클래스 토큰), Q-Former [59](학습 쿼리 32개), 시각 전용 ViT와 ViT-CNN. 모든 모델은 같은 DiT 머리를 쓰고 종단 학습합니다.
- 평가: **모델·과제당 20회**. 두 개의 독립 수집 데이터셋(각각 시연 30개로 학습)의 결과를 Fig. 4에서는 합쳐 40회로 보고합니다(§4.2). 전체 1700회는 표 8에 있습니다.

### 4.2 주요 결과

**두 데이터셋을 합친 성공률(표 8, 그림 4, 과제당 40회, 전체 200회)**

| 모델 (센서) | Gear | Board | Lamp | Key | Lightbulb | 합계 | 평균 |
|---|---|---|---|---|---|---|---|
| MiTaS (V+G+E) | 34/40 | 38/40 | 33/40 | 30/40 | 25/40 | 160/200 | **80%** |
| Sparsh-X (V+G+E) | 18/40 | 32/40 | 27/40 | 14/40 | 20/40 | 111/200 | 55.5% |
| ViT (V) | 15/40 | 24/40 | 19/40 | 9/40 | 4/40 | 71/200 | 35.5% |
| ViT-CNN (V) | 15/40 | 28/40 | 13/40 | 7/40 | 6/40 | 69/200 | 34.5% |

**데이터셋별(표 9 = 데이터셋 1, 표 10 = 데이터셋 2, 각 20회, %)**

| 모델 (센서) | 데이터셋 | Gear | Board | Lamp | Key | Lightbulb | 평균 |
|---|---|---|---|---|---|---|---|
| MiTaS (V+G+E) | 1 | 90 | 90 | 80 | 75 | 65 | 80 |
| MiTaS (V+G+E) | 2 | 80 | 100 | 85 | 75 | 60 | 80 |
| Sparsh-X (V+G+E) | 1 | 50 | 85 | 70 | 10 | 55 | 54 |
| Sparsh-X (V+G+E) | 2 | 40 | 75 | 65 | 60 | 45 | 57 |
| Q-Former (V+G+E) | 2 | 50 | 75 | 75 | 60 | 55 | 63 |
| PolyTouch (V+G+E) | 2 | 55 | 70 | 55 | 55 | 55 | 58 |
| ViT-CNN (V) | 1 / 2 | 40 / 35 | 70 / 70 | 20 / 45 | 0 / 35 | 0 / 30 | 26 / 43 |
| ViT (V) | 1 / 2 | 40 / 35 | 60 / 60 | 55 / 40 | 0 / 45 | 0 / 20 | 31 / 40 |

**센서 절제(표 2, 데이터셋 1, 20회)**: MiTaS V+G+E 80%, V+G 51%, V+E 37%. Sparsh-X V+G+E 54%, V+G 33%. V+E는 Gear와 Key에서 0%입니다. 본문은 "Evetac 단독 촉각은 일부 과제에서 유용하지만 손 안 물체 위치가 필요하면 완전히 실패한다"고 서술합니다.

**융합 구조(표 1, 데이터셋 2, 20회)**: MiTaS 80%, Sparsh-X 57%, PolyTouch 58%, Q-Former 63%. 저자는 "융합 방식 말고도 조건 토큰의 모양이 중요하다"고 읽습니다.

**MTD(표 3, 데이터셋 1, V+G)**: MiTaS 51 → 53(Gear +10, Board +15, Lamp −5, Key −35, Lightbulb +25), Sparsh-X 33 → 37(−10, +35, +15, −15, −5). 본문: MiTaS 3/5 과제에서 향상, Sparsh-X 2/5. Key in Lock에서 내려간 것은 "GelSight 특징의 손 안 위치 정보를 교란하기 때문"이라고 설명합니다. 표 10(데이터셋 2)에는 MTD V+G(60%)와 MTD로 학습하고 추론에 Evetac를 쓰는 V+G+E(63%)가 있습니다.

### 4.3 어텐션 분석 (§4.3, 그림 5, 부록 A.9, A.10)

성공한 에피소드 하나에서 "실행된 행동 토큰 → 센서 모달리티" 교차 어텐션을 시간 순으로 그렸습니다(그림 5는 Lamp Installation). 접근(reaching)에서는 비전, 나사 조임(screwing)에서는 GelSight, 삽입(insertion)에서는 세 센서가 고루 쓰인다고 읽고, "Evetac가 표현을 주도하지는 않지만 결정적인 단계에서 관련 특징을 준다"고 결론짓습니다. 부록 그림 10은 Evetac 활성도(기준 영상과의 L1 거리)가 접촉 때 치솟는다고 보입니다. 어텐션은 어느 층과 헤드의 값인지 적혀 있지 않습니다(`04` C13).

### 4.4 파라미터와 추론 (부록 A.2, 표 5)

RTX 4080에서 10스텝 플로 매칭: MiTaS V+G+E 43.5 ms, 총 24.47 M(인코더 10.76 M + 머리 13.70 M), V+G 35.3 ms, Sparsh-X V+G+E 54.6 ms(29.2 M). 표의 Hz 열은 1000 / ms와 다릅니다(`03` V8).

## 5. 한계

- 저자 명시(§5): 센서 한 쌍(GelSight + Evetac)만 시험, SpaceMouse 텔레오퍼레이션이 확장성을 제한, 4자유도 행동이고 6자유도로 가려면 로봇 상태 입력이 필요할 것, 15 Hz 재계획이 반응성의 상한.
- 이 분석에서 본 것(`03`, `04`): 모델·과제 한 칸이 20회이고 학습 실행이 칸마다 한 번이며, 같은 모델의 두 데이터셋 결과가 일부 칸에서 크게 다릅니다(`03` V7). 방법 설명에서 융합 트랜스포머 깊이, 스템 채널 수, 학습 에폭이 빠져 있고 공식 코드가 없어 정확한 재구현은 추정을 포함합니다(`04` §1).

## 6. 이 노트의 읽는 순서

수치 재계산과 일관성 점검은 `03`(V1–V8, K1–K4), 주장별 판정은 `04`(C1–C16), 공개 자료 상태와 구현 가능성은 `04` §1과 §4에 있습니다.
