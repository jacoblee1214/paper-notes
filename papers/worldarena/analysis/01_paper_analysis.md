# 01. 논문 분석: WorldArena

## 0. 기본 정보

| 항목 | 내용 |
|---|---|
| 제목 | WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models |
| 저자 | Yu Shang 외 20인 (Tsinghua, SJTU, HKU, Princeton, CAS, USTC, PKU, NUS). 교신 Yong Li |
| arXiv | 2602.08971, cs.CV (cross-list cs.RO). 제출 2026-02-09 (v1), 개정 2026-02-11 (v2) |
| 분석 버전 | **v2** (최신). 버전 차이는 0.3절 |
| 라이선스 | **CC BY 4.0** (arXiv abs 페이지의 "view license" 링크, 2026-11-20 확인) |
| 학회 상태 | 확인된 학회 표기 없음. 원문은 "Preprint. February 12, 2026"만 있고, arXiv comments와 journal ref 없음. 자세한 근거는 `03`과 README |
| 코드 | 공식 저장소 `tsinghua-fib-lab/WorldArena` (1.0). 자세한 내용은 `02` §3 |
| 프로젝트 페이지 | https://world-arena.ai (현재 2.0 소개, 1.0 페이지는 `/v1/`) |
| 분석 범위 | 본문 22쪽 (참고문헌과 부록 A–C 포함) |

## 1. Abstract 요약

세계 모델(world model)의 평가가 **지각 품질(영상 생성 품질)**에 치우쳐 있고, 의사결정에서 쓸모(functional utility)를 거의 보지 않는다는 문제의식으로 시작한다. WorldArena는 세 축으로 평가한다.

1. 영상 지각 품질: 6개 하위 차원, 16개 지표
2. 임베디드 과제 기능: 데이터 엔진, 정책 평가자, 행동 계획자의 세 역할
3. 사람 평가와 이를 하나로 합친 **EWMScore**(16개 지표의 산술 평균, 0–100)

14개 세계 모델을 평가해 "지각 품질과 기능 사이의 격차(perception–functionality gap)"를 보고한다. 높은 시각 품질이 곧 임베디드 과제 성능으로 이어지지 않는다는 것이 핵심 주장이다.

## 2. 문제 설정과 기여

- **기존 벤치마크의 두 한계** (§1): (a) 임베디드 과제 평가 부족, (b) 평가 모델 수 부족. 대부분 텍스트 조건 영상 생성 모델(Wan, CogVideoX)에 치우쳐 있고 로봇 전용 세계 모델은 거의 평가되지 않았다.
- **Table 1**: 8개 기존 벤치마크와 비교한 표. WorldArena만 열 10개 전부 ✓다. 자기 보고이므로 항목 정의(예: "Human"이 무엇을 포함하는지)가 엄밀하지는 않다.
- **기여 3개** (§1 끝): 첫 번째로 포괄적인 임베디드 세계 모델 벤치마크, EWMScore와 사람 평가 검증, 14개 모델의 체계적 평가.

## 3. 방법

### 3.1 영상 품질 16개 지표 (§3.1, 부록 A, Table 2·3)

| 차원 | 지표 (개수) | 도구 (논문 명시) | 비고 |
|---|---|---|---|
| 시각 품질 (3) | Image Quality (MUSIQ), Aesthetic (LAION predictor), JEPA Similarity | MUSIQ, LAION, V-JEPA | JEPA는 V-JEPA 특징의 MMD (polynomial kernel, α = 40) 식 (3)–(4) |
| 움직임 품질 (3) | Dynamic Degree, Flow Score, Motion Smoothness | RAFT, VFI-Mamba | 본문 §3.1.2는 "세 지표"라고 쓰고 Trajectory Accuracy까지 4개를 나열하지만, Table 3에서는 Trajectory를 Physics에 둔다 |
| 내용 일관성 (3) | Subject, Background, Photometric | DINO, CLIP, AEPE 광류 | 셋 모두 Dynamic Degree로 정적 영상에 페널티 (식 9–13) |
| 물리 (2) | Interaction Quality, Trajectory Accuracy | Qwen3-VL-8B 판정, SAM3 + NDTW | Interaction은 1–5 점수를 [0,1]로 정규화 |
| 3D (2) | Depth Accuracy, Perspectivity | Depth-Anything (median 정렬, AbsRel), Qwen3-VL | |
| 제어 가능성 (3) | Instruction Following, Semantic Alignment, Action Following | Qwen3-VL, Qwen2.5-VL + CLIP 텍스트, CLIP 특징 1−cos | Action Following은 같은 초기 프레임에 지시 3개를 준 영상들의 평균 비유사도 |

**정규화 (§3.4, 부록 A.17, Table 6).** Flow, Trajectory, Photometric, Motion Smoothness는 99/1 백분위 min-max, Depth는 역방향 min-max다. 식 (20)–(21)과 Table 6의 경계값(예: Flow 0.0531–8.9414)이 있다. 나머지 지표는 이미 [0,1]이다. 논문은 백분위 경계를 "8개 모델의 영상"에서 구했다고 쓴다(A.17). 그러나 평가 모델은 14개다. 이 불일치는 `04` D2에서 다룬다.

**동적 게이트 (부록 A.7, A.8, A.9).** 정적 영상이 높은 점수를 받는 "지름길"을 막기 위해 Subject, Background, Photometric 점수에 `min(1, Sdyn/γ)`를 곱한다. 이때 γ 값은 논문에 없다. 코드는 0.1213을 쓰고 규칙도 다르다 (`04` D1, `03` §6).

### 3.2 임베디드 과제 3종 (§3.2, 그림 3)

- **데이터 엔진 (§3.2, 4.2.2)**: 세계 모델을 RoboTwin 2.0 데이터로 미세조정하고, 첫 프레임과 지시에서 영상을 생성한다. 이 영상에서 IDM으로 행동을 뽑아 VPP 방식의 확산 정책 헤드를 붙인다. 그 결과로 π0.5를 합성 데이터 25개 궤적으로 학습해 성공률을 본다.
- **정책 평가자 (§3.2, 4.2.2)**: 데이터 양 10/20/30/50/100%로 학습한 π0.5 정책 5개를, 행동 조건 세계 모델 안에서 롤아웃하고 VLM 판정(부록 B 프롬프트)으로 성공을 본다. 이 값을 RoboTwin 실제 시뮬레이터 성공률과 Pearson으로 비교한다.
- **행동 계획자 (§3.2, 4.2.2)**: 세계 모델이 텍스트와 초기 프레임에서 행동열을 내고, 이를 RoboTwin 시뮬레이터에서 실행해 성공률을 본다.

### 3.3 사람 평가 (§3.3)

70명이 3,500개 영상을 평가했다. 전체 품질, 지시 따르기, 물리 적합성을 1–5점으로 매긴 뒤 0–100으로 정규화했다. 쌍별 비교로 승률(win rate)도 낸다. 평가자 간 일치도와 평가 설계(무작위 순서, 쌍 수)는 보고되지 않았다.

### 3.4 EWMScore (§3.4, 식 없이 설명)

"16개 정규화 지표의 산술 평균 ×100." 이 정의는 `03` §1에서 Table 2·3 값으로 Figure 1(a)을 재현해 확인했다(최대 차이 0.02).

## 4. 실험

### 4.1 설정 (§4.1)

- **데이터**: RoboTwin 2.0, 50 과제, 2,500 영상. 영상 품질 평가는 2,000개로 세계 모델을 학습하고 500개로 시험한다 (`10/40` 분할과 일치하는 것을 저장소 README에서 확인).
- **모델 14개** (§4.1, 각주 1–2): 일반 영상 모델 4개(CogVideoX, Wan 2.2, Wan 2.6, Veo 3.1), 텍스트 조건 로봇 모델 7개(Genie Envisioner, GigaWorld-0, TesserAct, Cosmos-Predict 2.5 text, WoW, RoboMaster, Vidar), 행동 조건 모델 3개(IRASim, Cosmos-Predict 2.5 action, CtrlWorld).
- **학습 여부**: "학습 코드가 있는 모델은 공식 구현을 따라 같은 데이터로 후학습했다"(§4.1). 닫힌 모델(Veo 3.1, Wan 2.6)의 처리는 본문에 없고, 저장소로도 확인할 수 없다 (§5 한계 6).

### 4.2 결과

**4.2.1 영상 품질 (Table 2·3, 부록 C 사례).**

- 시각 품질과 외형: Veo 3.1과 Wan 2.6이 Image Quality와 Aesthetic에서 상위다 (Table 2). 맞는 주장이다.
- 구조와 상호작용: 본문은 "임베디드 모델이 구조와 상호작용 지표에서 강하다"고 쓴다. 그러나 Table 3의 Interaction Quality에서 일반 모델 평균(0.657)이 임베디드 평균(0.507)보다 높다 (`03` §5). 이 주장은 맞지 않는다.
- 개별 주장 중 표와 맞지 않는 것이 여럿 있다 (`03` §5의 C1–C7). 예: "TesserAct가 궤적 정확도에서 높다"(Table 3에서 0.140, 14개 중 9위), "WoW가 행동 따르기가 강하다"(0.043, 8위), "CogVideoX가 시각 품질에서 뛰어나다"(0.358, 9위).

**4.2.2 임베디드 과제 (Table 4·5, Figure 4).**

- **데이터 엔진 (Table 4)**: 실데이터 기준선은 과제 1 77%, 과제 2 66%다. 생성 데이터는 대부분 기준선 아래다. 본문이 "RoboMaster와 WoW가 과제 2에서 실데이터를 넘는다"고 쓴 것은 통계적으로 근거가 없다. 두 차이는 각각 +2pp(p = 0.88), +5pp(p = 0.54)다 (`03` §2).
- **정책 평가자 (Figure 4)**: Ctrl-World의 Pearson r = 0.986, Cosmos-Predict 2.5의 r = 0.483. 정책은 5개뿐이다. 이 n = 5에서 0.986의 95% 구간은 약 [0.80, 0.999], 0.483은 [−0.70, 0.96]이다 (`03` §4).
- **행동 계획자 (Table 5)**: 여섯 모델 모두 π0.5(77%, 66%)보다 낮다. 논문도 이를 인정한다. 다만 TesserAct 행(1%, 35%)이 Table 4의 TesserAct 행과 같다 (`04` D6).

**4.2.3 사람 평가 (Figure 1(b)).** Veo 3.1과 Wan 2.6이 전체 품질, 지시 따르기, 물리 적합성에서 높다. CtrlWorld는 텍스트 전용 모델보다 물리 점수와 승률이 높다. Genie Envisioner는 낮다. 사람 평가의 수치 표는 본문에 없고 그림에만 있다.

### 4.3 지표 간 상관 (Figure 5)

- EWMScore와 사람 평가: r = 0.825. 그림 5의 점은 14개 모델이다(재확인, `03` §3).
- EWMScore와 데이터 엔진: r = 0.600, 6개 모델. 표 4의 두 과제 평균과 정확히 맞는다(`03` §3).
- EWMScore와 행동 계획: r = 0.360, 6개 모델. 표 5의 두 과제 평균과 정확히 맞는다.
- 해석: 저자는 "지각 현실감은 필요조건이지만 과제 성능을 비례로 보장하지 않는다"고 쓴다 (§4.3). 6개 점에서 r = 0.600은 유의하지 않다(임계 0.811, `03` §3).

## 5. 한계 (논문이 스스로 쓴 것과 읽으며 판단한 것)

**논문에 명시된 한계는 없다.** 본문에 한계 절이 없고, 결론 (§5)은 "모델을 더 추가하겠다"는 계획만 쓴다. 다음은 읽으며 판단한 것이다.

1. **통계 보고 부족**: 시드 반복, 신뢰구간, 표준편차가 없다. Table 4·5는 각 100회인데 "averaged"라고만 쓴다. 학습 분산은 보이지 않는다.
2. **표본 크기**: 정책 평가자는 정책 5개, 데이터 엔진과 계획자 상관은 모델 6개다. 그 위에 Pearson을 해석한다.
3. **단일 과제 도메인**: RoboTwin 2.0 양팔 조작 하나, 과제 2개로 과제 기능을 판단한다.
4. **VLM 판정기 의존**: Interaction, Perspectivity, Instruction Following, Semantic Alignment, 정책 성공 판정 모두 Qwen 계열 VLM이다. 판정 불확실성과 편향은 다루지 않는다.
5. **정규화 의존**: EWMScore는 데이터 기반 경계(백분위)에 의존한다. 모델을 추가하면 점수가 바뀐다.
6. **닫힌 모델 공정성**: Veo 3.1과 Wan 2.6은 RoboTwin으로 후학습했는지 본문에 없다. 상위 점수가 학습 여부 차이일 수 있다.

## 6. 읽으며 확인할 질문

- 16개 지표의 경계와 게이트 규칙은 코드와 맞는가 → `04`
- EWMScore 평균의 입력 값은 표 2·3 값 그대로인가 → `03` §1 (맞음)
- 정책 평가자 r = 0.986은 재현되는가 → `03` §4 (그림 값으로 근사, 맞음), 원자료는 없음
- Table 4와 Table 5의 TesserAct 행이 우연히 같은가, 복사 오류인가 → `04` D6 (확정 불가)
- 사람 평가 r = 0.825의 원자료는 공개되었는가 → 미확인 (저장소와 리더보드에 없음)

## 7. 버전 차이 (v1 → v2)

원문 텍스트를 diff했다. 본문 문장은 같다. 바뀐 것은 다음과 같다.

| 항목 | v1 (2026-02-09) | v2 (2026-02-11) |
|---|---|---|
| 저자 목록 | 20인 (Haisheng Su 없음, 소속 번호 순서가 다름) | 21인, Haisheng Su(SJTU) 추가 |
| 주석 | 없음 | "Equal contribution", "Project lead" 주석 |
| 프로젝트 URL | https://worldarena.ai | https://world-arena.ai |
| 날짜 | February 10, 2026 | February 12, 2026 |

v1과 v2의 표 값, 그림, 식은 같다. 따라서 이 분석의 수치는 두 버전 모두에 해당한다.
