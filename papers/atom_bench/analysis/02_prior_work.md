# 02. 선행연구와 관련 저장소

이 문서는 ATOM-Bench의 구성 요소가 어디서 왔는지, 가까운 벤치마크와 무엇이 다른지, 공식 코드와 데이터가 어디에 있는지를 정리합니다. 이름은 논문 참고문헌과 §2 관련 연구 절을 따릅니다(`paper/paper.txt`).

## 1. 선행연구 계보 (구성 요소별 출처)

| ATOM-Bench의 요소 | 출처 | 관계 |
|---|---|---|
| 원자(atom) 분류의 재료 | Meta-World(Yu 2019), RoboCasa(Nasiriany 2024), FMB(Luo 2025), RoboEval(Wang 2025), BEHAVIOR(Srivastava 2022), VLABench(Zhang 2024) | 저자는 이 벤치마크들의 반복 과제 요인을 요약해 원자를 골랐다고 씁니다(§3.1, 부록 E.1). 별도의 선정 기준 실험은 없습니다 |
| 운동 원자의 동작 분류 | Paulius 외 2019, 2020 (조작 동작 분류와 코딩) | 운동 원자 여섯 개의 분류 근거로 인용(부록 E.1) |
| 지시 원자의 언어 요인 | ReferItGame(Kazemzadeh 2014), Mao 2016, Yu 2016(참조 표현), CLEVR(Johnson 2017) | color, shape, size, source relation, count, exclusion의 근거(부록 E.2) |
| 시뮬레이션의 조합 일반화 벤치마크 | VIMA-Bench, LIBERO, VLABench, VLA-Arena, LIBERO-PRO, RoboHiMan | 동기이지만 시뮬레이션이나 디지털 트윈 설정(§2). ATOM-Bench는 같은 질문을 실로봇에 둡니다 |
| 실로봇 평가 인프라 | RoboArena(분산 쌍별 비교), AutoEval(자동 성공 판정과 초기화), RoboChallenge(온라인 대규모 평가), GM-100(100개 세부 과제), SceneReplica, FurnitureBench | 규모, 자동화, 순위를 다루고 "정책이 본 기술의 통제"는 약하다는 것이 저자의 차별점(§1, §2) |
| 평가 대상 정책 | Pi0.5(Physical Intelligence 2025b), Motus(Bi 2025), LingBot-VLA(Wu 2026), GR00T N1.6(GEAR 2025), SmolVLA(Shukor 2025) | 공식 코드와 공개 레시피로 미세조정(부록 D) |
| 로봇 플랫폼과 수집 도구 | Franka Panda + Robotiq 2F-85, SpaceMouse, 손목 카메라 마운트(Taunyazov 2020), Agilex Cobot Magic(Mobile ALOHA 기반, Fu 2024) | 플랫폼 구성(§3.2) |
| 초기 배치 재현 | 마스크 겹침 배치(부록 B) | 새 제안이 아닌 절차 설명. 정밀 측정 없이 "대략 같은" 배치 |
| 지표 | SR, PSR(점수 지점 비율), AS, CFS, TG | AS, CFS, TG가 이 논문의 새 지표입니다. |

### 가까운 대안과의 차이

- **MESA, LIBERO-PRO, VLA-Arena 계열**은 시뮬레이션에서 조합·섭동 일반화를 잰다는 점에서 같은 질문을 던집니다. ATOM-Bench는 실로봇이라 비용이 크고, 대신 시행 수가 작습니다(과제당 10회).
- **RoboArena, RoboChallenge**는 정책 순위를 다룹니다. ATOM-Bench는 순위가 아니라 "어느 원자가 약한가"를 진단합니다.
- 조합을 "원자 훈련 → 새 조합 평가"로 통제한다는 설계는 컴퓨터 비전의 조합 일반화 평가(CLEVR 계열)와 닮았습니다. 훈련 시연에 조합이 한 번도 나오지 않게 한다는 점이 핵심 통제입니다. 다만 값 수준의 새로움이 섞입니다(`01` §5).

## 2. 코드와 저장소 (GitHub 표)

2026-11-26에 확인했습니다. 별 수와 마지막 push는 GitHub API 값입니다.

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 공식 코드 저장소 | 찾지 못함 | 해당 없음 | 해당 없음 | 해당 없음 | 프로젝트 페이지에 코드 링크 없음. arXiv 주석과 논문 본문에도 없음. 평가 하네스, 점수 지점 주석, 학습 스크립트 모두 확인하지 못했습니다 |
| [flageval-baai/AtomBenchPage](https://github.com/flageval-baai/AtomBenchPage) | 프로젝트 페이지 소스(정적 HTML) | 없음(API license null) | 0 | 2026-06-16 | README는 "논문, 코드, 데이터셋, 영상이 공개되면 자리표시 링크를 갱신하라"고 적혀 있고 코드 링크는 없습니다. 최근 커밋 `0a2eb2a1c9827b2e5601b00516662dcf41718b0b` |
| [AtomBench/FrankaPanda](https://huggingface.co/datasets/AtomBench/FrankaPanda) (Hugging Face) | 공식 시연 데이터(Franka, LeRobot 형식) | Apache-2.0(카드) | 좋아요 1 | 수정 2026-06-04 | 과제 15개 × 에피소드 100개, 3 시점 영상. 목록 파일 합계 약 5.6 GB. 받지 않았습니다 |
| [AtomBench/CobotMagic](https://huggingface.co/datasets/AtomBench/CobotMagic) (Hugging Face) | 공식 시연 데이터(Cobot Magic) | Apache-2.0(카드) | 좋아요 1 | 수정 2026-06-04 | 과제 15개 × 에피소드 100개. 목록 파일 합계 약 8.5 GB. 받지 않았습니다 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Pi0.5 구현 | Apache-2.0 | 14,772 | 2026-08-24 | 논문이 쓴 모델 코드(논문은 저장소를 특정하지 않음) |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T N1.6 구현 | Apache-2.0 | 8,423 | 2026-11-17 | 같음 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | SmolVLA 구현 | Apache-2.0 | 29,102 | 2026-11-26 | 같음. 시연 데이터가 LeRobot 형식 |
| [thu-ml/Motus](https://github.com/thu-ml/Motus) | Motus 구현 | Apache-2.0 | 1,321 | 2026-10-09 | 같음 |
| [Robbyant/lingbot-vla](https://github.com/Robbyant/lingbot-vla) | LingBot-VLA 구현 | Apache-2.0 | 1,894 | 2026-06-11 | 같음 |

- 위 모델 저장소는 논문이 가리킨 주소가 아니라 이름으로 찾은 저장소이므로, 실제로 쓴 판과 일치하는지는 확인하지 못했습니다.
- 데이터 쪽 확인(목록 메타데이터만): 두 데이터셋 모두 과제 폴더 15개, 폴더당 parquet 100개와 mp4 300개(3 시점)입니다. 논문의 "3,000 시연"과 과제당 100 시연에 맞습니다(`03` V2). 폴더 이름은 논문 프롬프트와 조금 다릅니다(예: M4가 "빨간 큐브 밀기", 논문 m4는 "큐브 밀기").
- 평가 롤아웃 데이터(초록이 공개한다고 쓴 것)의 공개 위치는 2026-11-26 시점에 확인하지 못했습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "시뮬레이션 성공이 실로봇 일반화의 증거가 못 된다"(§1)는 LIBERO-PRO, VLA-Arena 등의 관찰에 기댑니다. 이 논문은 같은 주장을 직접 시험하지 않고 동기로만 씁니다.
- "Pi0.5가 가장 낫다"는 RoboArena 같은 실로봇 순위 연구의 흐름과 같은 방향입니다. 다만 이 논문의 모델 설정은 모델마다 다르고(표 10) 학습 시드가 하나입니다.
- 사람 원격조작 데이터 100개, 시드 10개라는 규모는 FMB, GM-100 같은 실로봇 과제 모음과 같은 자릿수입니다. 규모의 한계를 논문 §6이 인정합니다.
