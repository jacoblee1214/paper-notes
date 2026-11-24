# 02. 선행연구, 관련 저장소, 자원

## 1. 이 논문의 위치와 계보

논문은 관련 연구를 "(a) 실제 데이터로 키운 VLA, (b) 더 싼 데이터원(손 도구, 사람 영상), (c) 시뮬레이션 데이터"의 세 줄기로 나눕니다(§2). 번호는 원문 참고문헌입니다.

1. **실제 데이터 VLA.** RT-1 [15], RT-2 [16], π0 [1], π0.5 [2], GR00T N1 [3], Gemini Robotics [4, 17]. 데이터 규모와 모델 용량을 키우면 일반화가 좋아진다는 흐름이며, 이 논문이 맞서는 "시뮬만으로는 부족하다"는 통념의 출처입니다(§1). 데이터는 Open X-Embodiment [18], DROID [19] 같은 이질적 실제 시연입니다.
2. **더 싼 실제 데이터.** UMI류 손 도구 [22, 23]와 사람 영상 [24–26]. 이 줄기는 로봇 없이 모으지만 사람과 로봇 사이의 격차가 남는다고 평가합니다.
3. **시뮬 데이터.** 내비게이션은 대규모 시뮬 데이터로 일반 정책이 되었다고 인용합니다([27–29], 같은 연구실의 SPOC [29]와 The One Ring [28]). 조작은 "과제별 후속 학습이 필요"(InternVLA-M1 [30], InternData-A1 [7]) 하거나 "특정 원자 과제에 한정"(GraspVLA [31], ArtiBot [32])된다고 평가합니다. 이 논문의 주장은 그 사이, 즉 **후속 학습 없이 여러 과제로 영샷 전이**하는 자리입니다.
4. **전이 격차를 메우는 전통적 방법.** 시스템 식별(ASID [8]), 반복 적응(AdaptSim [9]), 시뮬·실제 공동 학습(Sim-and-Real Co-training [10]). 이 논문은 이 방법들 대신 "규모와 다양성으로 격차를 완화"한다고 주장합니다(§1).
5. **시뮬 데이터 생성기.** RoboMimic [48], MimicGen [49], RoboCasa-365 [50], InternData-A1 [7]. 표 4가 이들과 에피소드 수, 시간, 환경 수, 과제 수를 비교합니다(MolmoBot-Data는 에피소드 1.7M, 5,704 h, 환경 94.2k, 과제 8).

**표 4 비교의 읽는 법.** 환경 수 94.2k는 MolmoSpaces의 장면 단위이고 과제 수는 8입니다. 과제 다양성이 가장 큰 쪽은 RoboCasa-365(365 과제)이고, 시간은 InternData-A1(7,433 h)이 더 큽니다. 이 논문의 우위는 환경 수와 에피소드 수, 이동 조작 포함 여부입니다. 표 4의 OXE 행은 v3에서 에피소드 1.4M(v1, v2는 "1M+"), 시간 약 2k로 바뀌었습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문에서의 사용 |
|---|---|---|
| 장면과 자산 | MolmoSpaces [33], iTHOR/AI2-THOR [34, 36], Objaverse [35] | 장면 200k 이상 중 94.2k 사용(표 3). 물체는 크기와 수밀성으로 선별 |
| 파지 후보 | MolmoSpaces-Grasp [33] | 사전 계산된 6-DoF 파지를 점수화·충돌·IK로 선택 |
| 전문가 계획 | CuRobo [37](RB-Y1), IK 보간(Franka) | 단계 분해(PREGRASP → … → STOW)와 재시도 규칙 |
| VLM 백본 | Molmo2-4B [39] (SigLIP2 [42] + Qwen3 계열 토크나이저) | 다중 이미지·비디오 입력으로 과거 프레임 수용 |
| 행동 헤드 | DiT [40], π0 [1]의 흐름 매칭과 시각 조건화 | LLM 층마다 교차 주의하는 DiT |
| 행동 청크 | ACT [41] | 16개 예측, 8개 실행 |
| 경량 정책 | SPOC [29] | 분위수 빈 + 병렬 디코더 |
| π0 대조군 | openpi [44], PaliGemma [38] | MolmoBot-Pi0(구조 동일, 시뮬 데이터로만 학습) |
| 평가 플랫폼 | DROID [19], SIMPLER [53], LIBERO [54] | DROID 로봇, DROID로 옮긴 SIMPLER와 LIBERO-Object |
| 시뮬 기준선 | StereoVLA [45], LAP-VLA [21], X-VLA [20], DreamZero [52] | 표 1, 표 11 |

**논문이 인용하지 않았지만 가까운 아이디어.** 행동 잡음을 넣어 상태 커버리지를 넓히는 방식은 시연자 행동에 잡음을 주입해 복구 데이터를 만드는 DART류와 같은 계열입니다. 논문은 재시도 구간 가중과 행동 잡음을 "복구 행동을 주는 장치"로 설명하지만 두 장치를 따로 떼어낸 절제는 없습니다(`03` C16).

## 4. GitHub, 코드, 자원

| 이름 | 위치 | 관계 | 라이선스 | 별 | 최근 push | 상태 |
|---|---|---|---|---|---|---|
| MolmoBot | github.com/allenai/MolmoBot | **공식 코드**(MolmoBot, MolmoBot-Pi0, MolmoBot-SPOC, robot_eval). 프로젝트 페이지 소스도 같은 저장소 | Apache-2.0 | 124 (포크 28) | 2026-11-23 | 얕은 복제(`be82891`, 2026-11-23), 정적 읽기만 |
| MolmoSpaces | github.com/allenai/molmospaces | 환경, 자산, **데이터 생성(MolmoBot-Datagen)**, 벤치마크 평가 | LICENSE 파일은 Apache 2.0 본문(GitHub API는 NOASSERTION) | 529 | 2026-11-23 | 복제하지 않음. 설정 파일 하나만 API로 읽음(`04` D21) |
| MolmoSpaces Policy Zoo | github.com/allenai/molmospaces_policy_zoo | 제3자 정책 모음(README 언급) | Apache-2.0 | 0 | 2026-06-16 | 읽지 않음 |
| 프로젝트 페이지 | allenai.github.io/MolmoBot | 영상, 개요. 논문, 블로그, 코드, 모델, 데이터 링크 | — | — | — | 내용을 읽어 링크와 소개 문구를 확인 |
| 블로그 | allenai.org/blog/molmobot-robot-manipulation | Ai2 블로그 | — | — | — | 읽지 않음 |
| Hugging Face 모델 | huggingface.co/collections/allenai/molmobot-models | 9종: DROID, Img-DROID, Ablation-MF3-DROID, Pi0-DROID(gemma), SPOC-DROID(other), SPOC-RBY1Rigid/Articulated, RBY1DoorOpening, RBY1Multitask | 대부분 Apache-2.0 | — | 2026-03 | 받지 않음 (MolmoBot-DROID 약 20 GB) |
| Hugging Face 데이터 | huggingface.co/datasets/allenai/molmobot-data | MolmoBot-Data. 파일 합 약 10.3 TB, 상용 사용 가능 에피소드 목록 포함 | ODC-BY 1.0 | 좋아요 11 | 2026-07-24 | 받지 않음 |
| openpi | github.com/Physical-Intelligence/openpi | MolmoBot-Pi0의 구현 기반 | Apache-2.0 | 14,745 | 2026-08-24 | 복제하지 않음 |
| CuRobo | github.com/NVlabs/curobo | RB-Y1 전문가 계획 | Apache-2.0 | 2,040 | 2026-11-21 | 복제하지 않음 |

- **논문 라이선스(arXiv 비독점 배포)와 코드·데이터 라이선스는 별개입니다.** 코드는 Apache-2.0이고 모델은 대부분 Apache-2.0이지만, MolmoBot-Pi0는 PaliGemma 기반이라 gemma 라이선스입니다. 데이터는 ODC-BY이며 일부 에피소드는 비상용(NC) 자산을 포함하고 상용 가능한 부분은 `commercial_episodes.parquet`로 따로 표시합니다.
- **논문 발표 이후 변경.** 데이터 카드는 `ObjectBackfill` 설정(추가 물체)을 "공개 모델 학습에 쓰지 않았다"고 적습니다. 표 3의 1.75M 에피소드와 Hugging Face 설정 목록의 대응은 확인하지 않았습니다.
