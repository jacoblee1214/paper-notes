# Open X-Embodiment 논문 분석 (arXiv:2310.08864)

## 0. 메타데이터와 버전

| 항목 | 내용 |
|---|---|
| 제목 | Open X-Embodiment: Robotic Learning Datasets and RT-X Models |
| 저자 | Open X-Embodiment Collaboration (본문 저자 약 290명, 색인 기준 'Open X-Embodiment Collaboration and 293 other authors') |
| 학회 | IEEE ICRA 2024. Crossref에 DOI `10.1109/ICRA57147.2024.10611477`, 발행 2024-05-13으로 등록되어 있습니다. arXiv 초록에는 학회 표기가 없습니다. |
| arXiv 버전 | v1 (2023-10-13) ~ v9 (2025-05-14). 분석 대상은 **v9**(최신)입니다. |
| 라이선스 | arXiv 초록 페이지의 `view license` 링크는 `creativecommons.org/licenses/by/4.0/`(CC BY 4.0)이고, 원문 HTML에도 "License: CC BY 4.0"이 적혀 있습니다. PDF에는 ©, copyright, All rights reserved 표기가 없습니다. |
| 프로젝트 | robotics-transformer-x.github.io (데이터셋 목록은 Google Sheet, 코드는 GitHub) |
| 분량 | 본문 12쪽. 부록은 없습니다. 참고문헌 134편 |

### 버전 차이 (v1 대 v9, 단어 단위 diff)

- **수치는 바뀌지 않았습니다.** 표 I, 표 II, 그림 4 캡션의 50%, "∼3×", 3600회 시행, 527 기술 (160,266 과제), 22 임베디먼트, 21 기관 값이 두 버전에서 같습니다.
- 바뀐 것은 서술입니다. 기여 목표 (1)은 v1 "Demonstrate that"에서 v9 "Evaluate whether"로, 목표 (2)는 "Provide datasets, data formats and models"에서 "Organize large robotic datasets"로 바뀌었습니다.
- 한계 절은 v1에서 "our experiments have a number of limitations: it does not consider ..."였고, v9에서는 "Our experiments do not consider ..."로 같은 내용을 짧게 씁니다. 결론 문장에 "the RT-1-X policy has a 50% higher success rate"가 있는데 v9에도 그대로 있습니다.
- v2–v8은 이 분석에서 diff하지 않았습니다.

## 1. 한 문단 요약

로봇 조작 데이터 60개(22개 임베디먼트, 21개 기관 소속 연구실의 데이터 조합)를 RLDS 형식으로 통일해 공개하고, 그 데이터로 RT-1(35M)과 RT-2(PaLI-X 55B 계열)를 같은 혼합으로 학습한 RT-1-X와 RT-2-X를 보고합니다. 주장은 세 가지입니다. (a) 여러 임베디먼트 데이터를 함께 쓰면 각 데이터셋에서 따로 학습한 정책보다 성능이 좋아진다(긍정 전이). (b) 큰 VLM 백본(RT-2-X)은 새 대상에 대한 일반화와 창발 기술이 좋아진다. (c) 데이터셋과 체크포인트를 공개해 X-embodiment 연구의 기반으로 쓰게 한다.

## 2. 데이터셋 (III-A, III-B, 그림 2)

- **규모:** "1M+ trajectories, 22 embodiments, 60 datasets, 34 robotic research labs"(III-A). 초록과 III는 "21 institutions"라고 씁니다. 같은 논문에서 기관(21), 연구실(34), 저자 소속(약 46 항목)이 서로 다른 수로 나옵니다. 세 값이 무엇을 세는지 정의하지 않습니다.
- **형식:** RLDS(tfrecord). 카메라 수, 깊이, 점군이 다른 데이터를 담을 수 있습니다.
- **행동 공간:** 7차원 엔드이펙터 행동(x, y, z, roll, pitch, yaw, 그리퍼)을 씁니다. 각 데이터셋의 행동을 **데이터셋별로 정규화한 뒤 256 빈으로 이산화**합니다(IV-A, IV-B). 좌표계는 맞추지 않습니다(절대/상대, 위치/속도가 섞여 있음).
- **그림 2(a)–(e):** 임베디먼트별 데이터셋 수, 장면 수, 궤적 수, 기술 빈도(PaLM으로 추출), 물체 빈도. 장면 수와 기술·물체 빈도는 스프레드시트에 없어 재계산할 수 없습니다. 궤적 수는 스프레드시트로 대조할 수 있습니다(03 V6, V8).
- **언어 주석:** PaLM으로 기술과 물체를 추출했다고만 씁니다. 추출 프롬프트, 검증 방법, 오류율은 없습니다. 그 결과로 "527 skills (160,266 tasks)"를 씁니다. 이 수는 재계산할 수 없습니다.

## 3. 방법 (IV)

### 3.1 데이터 형식 통일 (IV-A)
- 입력: 카메라 한 대(표준 해상도), 과거 이미지 이력, 언어 지시문.
- 출력: 7차원 행동을 이산화한 토큰. 한 차원이 에피소드 종료 신호입니다. 논문은 "8 dimensions"라고 씁니다(IV-B).
- 정규화 범위는 본문에 없습니다(데이터셋마다 정규화하지만 범위를 밝히지 않음).

### 3.2 RT-1 (IV-B)
- "35M parameter" Transformer. 이미지 15장 이력, ImageNet EfficientNet에 FiLM으로 USE 언어 임베딩을 결합, 이미지당 81 토큰, 디코더 전용 Transformer가 행동 토큰을 냅니다.
- 공개 코드와 대조한 결과는 04 문서에 있습니다. 공개 추론 예제는 어휘 크기를 512로, 층 너비를 256으로 씁니다. 논문의 256 빈과 다릅니다.

### 3.3 RT-2-X (IV-B)
- RT-2를 PaLI-X 백본(ViT + UL2)으로 구현한 **RT-2-PaLI-X**. 웹 데이터와 로봇 데이터를 약 1:1로 공동 미세조정합니다(IV-C).
- 행동 토큰은 언어 토큰으로 바꿔 쓰며, 파라미터는 5B와 55B 두 가지입니다.

### 3.4 혼합과 학습 (IV-C)
- 혼합(robotics data mixture): **9개 조작기(manipulators)**, **12개 출처**(RT-1, QT-Opt, Bridge, Task Agnostic Robot Play [126, 127], Jaco Play, Cable Routing, RoboTurk, NYU VINN, Austin VIOLA, Berkeley Autolab UR5, TOTO, Language Table).
- 이 혼합은 OXE 전체 22개 임베디먼트보다 적습니다. 저자는 "데이터셋을 계속 늘려 왔고, 실험 시점에 위 목록이 전부였다"고 씁니다.
- RT-1-X는 로봇 혼합만으로 학습했고, RT-2-X는 웹과 로봇 데이터를 공동 미세조정했습니다.
- **빠진 것:** 혼합 가중치와 샘플링 방식, 학습 스텝, 배치 크기, 학습률, 하드웨어, 학습 시간, 시드 수. 본문에 한 줄도 없습니다(본문 전체에서 "weight", "sampling", "epoch", "TPU", "GPU", "seed"를 검색해도 없음).
- 추론: RT-1은 로컬, RT-2는 클라우드에서 3–10 Hz로 돌렸습니다.

## 4. 실험 (V)

총 평가 시행은 **3600회, 6개 로봇**(V 서두)입니다. 셀별 시행 수는 어디에도 없습니다.

### 4.1 소규모 데이터 도메인 (V-A, 그림 4)
- 5개 도메인(Kitchen Manipulation, Cable Routing, NYU Door Opening, Autolab UR5, Task-Agnostic Play). 각 도메인에서 원 데이터셋의 **Original Method**(원 저자 모델)와 RT-1을 비교합니다.
- 그림 4의 평균 막대: Original 41, RT-1 44, RT-1-X 63(라벨 기준).
- 캡션: "RT-1-X mean success rate is 50% higher than that of either the Original Method or RT-1".
- 재계산(03 V1): 평균 대비 +54%(Original), +43%(RT-1). 5개 도메인 중 4개에서 RT-1-X가 Original보다 높고 Autolab UR5만 낮습니다(45 대 53).
- 막대는 정수 라벨이며 n과 오차막대가 없습니다.

### 4.2 큰 데이터 도메인 (V-A, 표 I)

| 모델 | Bridge IRIS (WidowX) | Bridge RAIL (WidowX) | RT-1 논문 6 기술 (Google Robot) |
|---|---|---|---|
| Original (LCBC) | 13% | 13% | - |
| RT-1 | 40% | 30% | 92% |
| RT-1-X | 27% | 27% | 73% |
| RT-2-X 55B | 50% | 30% | 91% |

- 캡션: "RT-1-X underfits and performs worse than the Original Method and RT-1". 표는 두 Bridge 열에서 RT-1-X(27%)가 Original(13%)보다 높다고 보여 줍니다. 캡션과 맞지 않습니다(03 V2).
- 본문: "the larger RT-2-X model outperforms both the Original Method and RT-1". 표는 RAIL 열에서 RT-2-X와 RT-1이 같고(30%) RT-1 논문 열에서는 RT-2-X가 낮습니다(91 대 92). 본문 주장은 IRIS 열에서만 성립합니다.
- Original 값 13%가 두 Bridge 열에 같습니다. 복사 오류일 수 있습니다.

### 4.3 분포 밖 일반화 (V-B, 표 II, 그림 5)
- 미지 물체·배경·환경(표 II 마지막 열, RT-2 평가 프로토콜): RT-2 62%, RT-2-X 61%. 저자는 "거의 같다"고 합니다.
- 창발 기술 (Google Robot에서 Bridge에만 있던 기술): RT-2 27.3%, RT-2-X 55B 75.8%. 본문은 "∼3×"라고 씁니다(75.8 / 27.3 = 2.78배). 프로젝트 페이지는 "3x"라고 씁니다.
- Bridge 제외(행 3): 42.8%. 행 2에서 Bridge를 빼면 창발 성공률이 크게 떨어진다고 해석합니다.

### 4.4 설계 결정 (V-C, 표 II 행 4–7)
- 이력 길이(행 4 대 5, 5B): 창발 44.4 대 14.5, 일반화 52 대 30. 이력이 있으면 좋아진다고 봅니다.
- 웹 사전학습(행 4 대 6): 44.4 대 0(처음부터 학습). 웹 공동 학습의 효과로 읽습니다. 행 6은 웹 사전학습도 없으므로 두 요인이 섞입니다.
- 공동 미세조정 대 미세조정(행 4 대 7): 44.4 대 48.7, 52 대 47. "비슷하다"고 씁니다. RT-2 결과와 다르다고 해석합니다.
- 55B 대 5B(행 2 대 4): 창발 75.8 대 44.4. 저자는 용량 효과로 읽습니다. 그러나 행 2는 이력이 없고 행 4는 이력이 2입니다. 용량과 이력이 섞여 있습니다.

## 5. 주장과 근거 대조 (요약)

| 주장 | 근거 | 판정 |
|---|---|---|
| 여러 임베디먼트 데이터가 긍정 전이를 준다 | 그림 4 평균 63 대 41/44, 5중 4 | 방향은 지지됨. 셀 n과 기준 방법 차이 때문에 정량 주장은 약함 |
| 큰 데이터 도메인에서도 RT-2-X가 낫다 | 표 I | 부분적. IRIS 열에서만 우위, RAIL과 RT-1 논문 열에서는 같거나 낮음 |
| RT-1-X는 큰 데이터에서 과소적합 | 표 I | RT-1 대비 열 2개에서 낮음. 원인이 과소적합인지 이력·정규화인지 구분 못 함 |
| RT-2-X는 창발 기술이 3배 | 표 II 행 1–2 | 방향 지지됨(03 V5: 모든 후보 n에서 p ≤ 0.009). 배수 "3"은 2.78의 올림 |
| Bridge 데이터가 Google Robot의 창발을 만든다 | 표 II 행 2–3 | 방향 지지됨(03 V5: p ≤ 0.001). 학습 한 번씩이라 시드 분산은 없음 |
| 웹 사전학습이 중요하다 | 표 II 행 4 대 6 | 방향은 맞음. 공동 학습 여부와 섞임 |
| 공동 미세조정과 미세조정이 비슷하다 | 표 II 행 4 대 7 | 차이 없음을 증명하지 못함(03 V5: p ≥ 0.52) |
| 데이터 규모 1M+, 60 데이터셋, 22 임베디먼트 | 스프레드시트 v1.0 | 맞음(03 V6). 혼합은 9 임베디먼트, 12 출처 |

## 6. 빠진 정보 (재현과 검토에 필요한 것)

1. 셀별 시행 수(표 I, 표 II, 그림 4). 3600회의 분배.
2. 시드 수와 분산. 모든 결과가 학습 한 번씩으로 보입니다.
3. 혼합 가중치와 샘플링 방식(에피소드 기준인지, 스텝 기준인지). 중요합니다. 스프레드시트로 보면 QT-Opt와 Language Table이 혼합 에피소드의 90%를 차지하고, Bridge는 2.25%입니다(03 V6, V7).
4. 학습 설정(스텝, 배치, 학습률, 하드웨어, 시간).
5. 평가 프로토콜의 세부 항목(성공 판정 기준, 초기 상태 분포, 시간 제한, 지시문 목록).
6. 각 기준 방법(Original Method)의 학습 데이터 범위와 평가 설정이 같은지.
7. 그림 2(a)–(e)의 장면 수와 기술·물체 집계 방법.
8. RT-2-X 55B 가중치와 학습 코드(공개되지 않음).

## 7. 한 줄 판정

긍정 전이라는 **방향**은 실험 설계에서 대체로 지지됩니다. 그러나 셀별 n, 시드 수, 혼합 가중치가 없어서 **크기와 배수**는 확인할 수 없고, 표 I의 캡션과 본문 일부는 표와 맞지 않습니다.
