# 02. 선행연구, 구성 요소, 관련 분석, 자료

번호 [n]은 RARM 논문(v5)의 참고문헌 번호입니다. 논문이 직접 인용한 연구만 계보에 넣었고, 서술은 논문 §1, §2.1–2.2, §4.1과 각 기준선의 제목·초록 수준에서 확인한 범위입니다. 각 기준선 논문의 세부는 읽지 않았습니다.

## 1. 계보: 학습형 보상은 어디서 왔고 RARM은 어디에 놓이는가

| 계열 | 대표 연구 | 핵심 아이디어 | RARM과의 관계 |
|---|---|---|---|
| 가치·표현 기반 보상 | VIP [23], LIV [24], UVD [25], Kim 외 [26](Subtask-aware visual reward) | 사전학습 시각 표현의 목표까지 거리나 하위 목표 분해를 보상으로 | VIP는 RARM 논문 그림 5와 표 7에서만 비교됩니다(§4.1의 기준선 목록에는 없음). 목표 이미지가 아니라 **시연 클립 열**을 기준으로 삼는 점이 다릅니다 |
| 시간 진행 학습 | TimeRewarder [17], PROGRESSOR [10], Rank2Reward [18] | 프레임 간 시간 거리나 순위로 진행을 회귀 | 과제 영상이 필요하고, PROGRESSOR는 자기 지도 온라인 정제를 씁니다. RARM은 **일반 영상으로 한 번만 학습**합니다 |
| VLM 기반 보상·비평 | GVL [12], RoboReward [13], **RoboMeter [14]**, ReWiND [15], SARM [16], RoboDopamine [30], TOPReward [31], VLAC [29], Adapt2Reward [27] | 대형 VLM이 진행이나 성공을 예측 | 의미 사전지식은 크지만 질의 비용이 크고 "그럴듯한 실패"에 높은 보상을 줍니다(§1). RARM의 주 비교 대상(GVL, RoboMeter, RoboDopamine, 실로봇의 RoboMeter) |
| 참조 시연 기반 | RoboCLIP [59], **TemporalOT [33]**, Graph IRL [32], Points2Reward [34], 객체 지향 보상 [35], GenFlowRL [36] | 시연과의 임베딩 유사도, 시간 제약 최적 수송, 점·객체 정렬 | 가장 가까운 계열입니다. 시간 정렬이 어긋나면 깨지는 점(§2.1)을 RARM은 전수 비교와 단조 진행으로 푼다고 주장 |
| 생성형·불확실성 | VIPER [19], Diffusion Reward [20] | 비디오 예측 가능도나 확산 모델로 "전문가다움"을 점수화 | "믿을 수 없을 때 보상을 주지 않는다"는 발상의 선례로 논문이 인용(§1). 과제 영상이 필요 |
| 보상 코드 생성 | Eureka [37], 자기 개선 기반 모델 [38] | LLM이 보상 코드를 짬 | 대안 경로로만 언급 |
| 보상 과최적화 | Gao 외 [39], Shen 외(길이 편향) [40], Leng 외(보상 보정) [11], Tian 외(나쁜 행동 데이터가 필요하다는 입장글) [9] | 학습된 보상은 RL이 악용 | RARM의 동기. 해법을 "신뢰 게이트와 진행 변화 보상"으로 제시 |

## 2. 구성 요소별 출처

| 구성 요소 | RARM이 쓴 것 | 출처 |
|---|---|---|
| 손실 | 소프트 최근접 이웃(식 (1)), τ = 0.1 | Salakhutdinov & Hinton [52] |
| 시각 인코더 | DINOv3 + LoRA | Siméoni 외 [53], Hu 외 [54] |
| 시간 위치 인코딩과 어텐션 | 시간 RoPE, 프레임 간 자기 어텐션, 교차 어텐션 | Su 외 [55], Vaswani 외 [56] |
| 음성 구성 | 시간 역전 클립(동역학만 반대인 음성) | 논문에 직접 출처 없음. 시간 방향 학습의 일반적 아이디어(우리 평가: 새로움은 이 조합이지 개별 요소가 아님) |
| 학습 영상 | DAVIS [60], MOSE [61], MOSEv2 [62] | 영상 객체 분할 벤치마크를 대조 학습 데이터로 재사용 |
| 시뮬레이션 RL | DrQ-v2 | Yarats 외 [57], 공식 구현은 MIT |
| 실로봇 RL | DSRL(고정 확산 정책의 잡음 조종) + π0 | Wagenmaker 외 [58], Black 외 [4] |
| 벤치마크 | LIBERO-10, MetaWorld MT50 | [21], [22] |
| 진행을 "달성한 최고 수준"으로 계산 | 1{p_t > p_{t-1}} | 퍼텐셜 기반 보상 변형과 닮았지만 논문은 PBRS를 언급하지 않습니다(우리 관찰, `03` §4) |

## 4. GitHub와 자료 (확인 2026-11-23, `gh api`와 HF API)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (main)** | https://github.com/pengzhi1998/rarm | **파일 없음**(GitHub 표시 없음, README와 PROVENANCE는 "원 저장소에 루트 라이선스가 없어 만들어 붙이지 않았다"고 설명) | 별 2, 포크 0, 커밋 3개, 마지막 push 2026-11-06, 최신 `55ef3fe` | 보상 모델 학습·추론. `repo/`에 얕은 clone, 정적 읽기만 |
| **공식 코드 (policy_training_RARM 가지)** | https://github.com/pengzhi1998/rarm/tree/policy_training_RARM | 파일 없음. DrQ-v2/DrM 부분은 MIT 고지 포함(`THIRD_PARTY_NOTICES.md`) | 같은 저장소, 최신 `c51695a` (2026-11-06) | LIBERO-10 DrQ-v2 학습·평가. `repo/_policy_training_RARM/`에 `git archive`로 풀어 정적 읽기만 |
| 체크포인트 (HF) | https://huggingface.co/pengzhi1998/RARM | `license: other`(DINOv3 라이선스 전문 동봉) | 다운로드 0, 좋아요 0, 2026-11-06 | `checkpoint_epoch_45_generic.pt` 233 MB. 받지 않음 |
| 프로젝트 페이지 | https://rarm-robotics.github.io/ | - | - | 논문·코드·HF 링크, 개요 영상. 정적 페이지 |
| DINOv3 | https://github.com/facebookresearch/dinov3 | NOASSERTION(DINOv3 전용 라이선스, 상업 이용 조건 있음. 정확한 조건은 원문 확인 필요) | 별 11.8k, 2026-11-02 | 가중치는 별도 신청 방식. 받지 않음 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT | 별 2.6k, 2025-03-15 | 데모 hdf5는 별도 다운로드(받지 않음) |
| DrQ-v2 | https://github.com/facebookresearch/drqv2 | MIT | 별 439, 2022-05-31 | RARM 정책 코드는 Progressor의 DrM 구현에서 이식(`THIRD_PARTY_NOTICES.md`) |
| DrM | https://github.com/XuGW-Kevin/DrM | MIT | 별 79, 2026-08-28 | 위의 상위 구현 |
| Progressor | https://github.com/ripl/progressor | 파일 없음 | 별 11, 2026-04-03 | 논문 [10]. RARM 정책 코드의 직접 출처 |
| DSRL | https://github.com/ajwagen/dsrl | 파일 없음 | 별 244, 2025-08-05 | 논문은 나머지 하이퍼파라미터가 이 공개 구현을 따른다고 밝힙니다(부록 D.1). RARM 저장소에는 DSRL 실로봇 코드가 없습니다(`04` D15) |
| Meta-World | https://github.com/Farama-Foundation/Metaworld | MIT | 별 1.9k, 2026-11-12 | RARM의 MetaWorld 실험 코드는 공개되지 않음 |
| Robometer (기준선) | https://github.com/robometer/robometer | 파일 없음 | 별 284, 2026-08-13 | 실로봇 비교의 기준선. 저장소 별도 `robometer-policy-learning`(MIT) 있음 |
| 코드 출처 저장소 | `daidai189963/Reward-Model-MVP` | 조회 404(비공개 또는 삭제) | - | PROVENANCE: 공개본은 이 저장소의 2026-01-23, 01-25 커밋에서 잘라 낸 것 |
| 공개되지 않은 것 | MetaWorld 정책 코드, 기준선(GVL, RoboDopamine, RoboMeter 래퍼, RoboCLIP, TemporalOT) 구현, 실로봇 DSRL 코드, 표·그림 생성 스크립트, 학습 영상 목록과 전처리 | - | - | `04` D15 |
