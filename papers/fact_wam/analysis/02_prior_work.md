# 02. 선행 연구, 코드 저장소, 관련 분석

FACT(arXiv:2608.10232 v2)의 위치를 구성 요소별로 나눕니다. "논문 인용"은 논문 참고문헌에 있는 것이고, "논문이 인용하지 않은 가까운 연구"는 이 분석 작성자의 지식에서 꺼낸 것이라 이번 세션에서 원문을 다시 확인하지 않았습니다.

## 1. 계보

### 1.1 월드-액션 모델 (논문 인용)

- **영상 우선 계열:** DreamZero [11], LingBot-VA(Causal world modeling for robot control) [5], Motus [7]. 미래 영상을 먼저 만들고 역동역학 모델로 행동을 복원하거나, 영상·행동을 혼합 트랜스포머(MoT)로 함께 다룹니다. 논문은 이 계열이 영상이 충분히 디노이징될 때까지 제어를 못 한다고 지적합니다(1절).
- **미래를 행동 조건으로 쓰는 계열:** Fast-WAM [8], BagelVLA [10], τ0-WM [12], GigaWorld-Policy [9]. 공식 저장소 README의 감사 문구는 FACT가 WAN2.2, giga-world-policy, GigaTrain, GigaDatasets 위에 만들어졌다고 적습니다(`04` §1). 논문 본문은 이 관계를 따로 쓰지 않습니다.
- **FACT의 위치:** "행동 먼저, 미래는 그 행동에 조건부" 순서(식 5)와 교사 강제 마스크(그림 3). 행동이 영상 뒤에 오는 영상 우선 방식과 반대 순서입니다.

### 1.2 실패 데이터와 가치 (논문 인용)

- 실패에서 배우기: REFLECT [45], Grollman & Billard 2012 [46], 내비게이션의 실패 인지 LfD [47], Recover-Discover-Plan [48]. 대부분 기호·언어 수준의 실패 설명이나 안전 제약입니다.
- 가치·보상과 RL: VLA-RFT [42](세계 시뮬레이터 안의 검증 보상), VIVA [43](영상 생성 가치 모델), RISE [44](구성형 세계모델 자기개선).
- 후보 점수화: Qi 외 [50] (예측 세계모델로 생성 정책의 추론 시점을 개선). FACT는 별도 비평가를 두지 않고 가치 가지를 점수로 씁니다(식 10).
- 확산·흐름 학습 도구: 흐름 매칭 [49], AdamW [52], 백본 WAN2.2-5B [51].

### 1.3 논문이 인용하지 않은 가까운 연구 (미확인)

아래는 작성자의 기억에서 가져온 것이고 이번 세션에서 원문을 확인하지 않았습니다.

- **행동 조건 가치는 사실상 Q 함수 회귀입니다.** v_t(a)는 (관측, 지시, 행동 청크)에서 시간 진행을 몬테카를로로 회귀하는 값이므로 Q 함수와 같은 모양입니다. 학습된 Q로 후보를 거르는 방식(가치 유도 정책 조종)이 이미 있어, FACT의 선택 사항 점수화는 그 틀에 가깝습니다.
- **실패와 부정 사례를 함께 쓰는 정책 학습:** 이점 조건부 학습(π0.6\* RECAP 계열)은 성공과 실패를 모두 데이터로 쓰되 이점 라벨로 구분합니다. FACT는 실패의 행동 손실을 끄고 세계 쪽 감독만 남기는 점이 다릅니다.
- **반사실 학습:** 나쁜 행동을 미래의 원인으로 삼아 세계모델을 학습하는 시도(행동 조건 세계모델)는 월드 모델 일반의 오래된 목표입니다. 논문의 새로움은 "WAM의 행동 먼저 순서와 마스크 구성으로 실패를 모방 없이 쓴다"는 설계에 있습니다.

## 2. 코드와 자료 (확인일 2026-11-26)

| 저장소·자료 | 관계 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| [Bariona/FACT](https://github.com/Bariona/FACT) | **공식 구현** (README "official implementation", 설명에 CoRL 2026) | Apache-2.0 | 110 / 2026-09-15 | 대조 기준 커밋 `5b299a2059f00c043fedc5c9dd17bf43fa14cbec`, 얕은 복제로 정적 읽기만 (`04`) |
| [fact-wam.github.io](https://fact-wam.github.io) (저장소 `fact-wam/fact-wam.github.io`) | 프로젝트 페이지 | 라이선스 없음 | 0 / 2026-09-15 | 논문, arXiv, 코드, 영상 링크, BibTeX(`inproceedings`, 학회 표기 없음) |
| [Bariona/fact-wam](https://huggingface.co/Bariona/fact-wam) (HF 모델) | 학습된 가중치 (`transformer/`와 `success_only/transformer/`, 각 .bin 약 10.39 GB) | Apache-2.0 (카드 메타데이터) | 좋아요 1 / 2026-10-27 | 받지 않음 (API 메타데이터의 크기만 확인) |
| [Bariona/robotwin-v2](https://huggingface.co/datasets/Bariona/robotwin-v2) (HF 데이터) | RoboTwin 2.0 시연(LeRobot v2.1) 재배포 | MIT (카드 메타데이터) | - / 2026-08-13 | `robotwin-v2.tar` 약 84.5 GB, 받지 않음 |
| [Bariona/fact-world-action-model](https://huggingface.co/spaces/Bariona/fact-world-action-model) (HF Space) | 체크포인트 데모 | Apache-2.0 | - / 2026-08-13 | 실행하지 않음 |
| [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | 백본 (WAN2.2-TI2V-5B) | Apache-2.0 | 18,931 / 2026-09-21 | |
| [open-gigaai/giga-world-policy](https://github.com/open-gigaai/giga-world-policy) | 코드 계승 대상(README 감사 문구) | 없음(API) | 1,417 / 2026-07-21 | 같은 조직의 `giga-train`(Apache-2.0, 1,074), `giga-datasets`(Apache-2.0, 646) |
| [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | 평가 벤치마크 | MIT | 3,081 / 2026-09-24 | 공식 코드의 평가 패치가 가리키는 커밋 `2eeec32` |
| [thu-ml/Motus](https://github.com/thu-ml/Motus) | 가장 강한 기준선 | Apache-2.0 | 1,321 / 2026-10-09 | 실로봇 기준선 코드는 FACT 저장소에 없음 |

- 학회: corl.org 공식 채택 목록에 포스터로 있습니다(세션 Poster post-p3, 포스터 4번, 2026-11 수집). 공식 코드 설명에는 "[CoRL 2026]"이 적혀 있습니다.

