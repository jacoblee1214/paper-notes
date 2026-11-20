# 02. 선행연구와 관련 연구

## 1. 계보

DreamGen은 세 흐름이 만난 지점입니다.

1. **영상 생성으로 로봇 행동을 뽑는 흐름.** 영상 모델을 계획기로 쓰거나 IDM으로 행동을 복원합니다. Du 외(UniPi, ref [15]), RoboDreamer [16], Ko 외(dense correspondence, [17]), Yang 외(interactive simulators, [18]), Video Language Planning [19]이 여기 속합니다. DreamGen은 이 흐름과 달리 영상 모델을 **실행 중이 아닌 데이터 생성기**로 씁니다(§1).
2. **잠재 행동과 영상에서 행동 배우기.** LAPA [13], Genie [71], Moto [72], LAPA를 쓴 UniVLA [75], AdaWorld [76]. DreamGen은 LAPA를 IDM과 나란히 라벨러로 씁니다(§2.3).
3. **합성 데이터 생성.** 시뮬레이션 기반(MimicGen [30], RoboCasa [20], DexMimicGen [35], SkillMimicGen [38])과 생성 모델 증강(CACTI [40], ROSIE [41], GenAug [42], ROVI-Aug [43], CosmosTransfer1 [44])이 있습니다. 저자들은 생성 증강은 로봇 움직임의 다양성이 낮고 시각적 강건성 용도에만 쓰였다고 비교합니다(§5).

영상 모델 자체는 WAN2.1 [9], CogVideoX [8], Hunyuan [10], Cosmos [7]을 씁니다. 다중 시점 조정과 구조는 Cosmos 계열과 비슷합니다.

## 2. 가까운 대안 (논문이 직접 비교한 것)

| 대안 | 논문에서의 위치 | 차이 |
|---|---|---|
| 시뮬레이션 데이터 (RoboCasa 원본) | Table 4의 30/100/300 traj 기준 | DreamGen은 그 위에 신경 궤적을 더함 |
| 실로봇만 학습 (GR00T N1, π0, DP) | Table 5 기준 | 실험 대상 |
| 영상 모델 제로샷 | Table 2 | 로봇 데이터 미세조정 여부 |
| 사람 영상 기반 방법 | §5 (직접 비교 안 함, Limitation) | 로봇 영상 생성 대 사람 영상 |
| π0.5 [6] "환경 수 증가" | §3.2에서 인용, 비교는 없음 | 저자들은 DreamGen이 실제 수집 없이 새 환경을 다룬다고 강조 |

직접 비교 실험이 있는 대안은 "실로봇만"뿐입니다. 사람 영상 방법과 LAPA 단독 학습과는 정량 비교가 없습니다. 이 점은 저자들도 Limitation에서 인정합니다.

## 4. 구성 요소별 출처

| 요소 | 출처 | 논문 표기 |
|---|---|---|
| 영상 세계모델 | WAN2.1 [9], CogVideoX [8], Hunyuan [10], Cosmos [7] | §2.1, Table 2 |
| 미세조정 방식 LoRA | Hu 외 [21] | §2.1 |
| IDM 백본 | SigLIP-2 + flow matching (인용 없음, "diffusion transformers") | §2.3 |
| IDM 원형 | VPT [14, 23] | §2.3 |
| LAPA | Ye 외 [13] | §2.3 |
| 정책 | Diffusion Policy [24], π0 [2], GR00T N1 [5] | §2.4 |
| 시뮬레이터 | RoboCasa [20] | §3.1 |
| 실로봇 데이터 | DROID [22], LeRobot SO-100 [78] | App. C, I |
| 지시 평가 VLM | Qwen2.5-VL [29], GPT-4o | §4 |
| 물리 평가 | VideoCon-Physics [26], VideoCon [77] | §4, H.2 |

## 5. 해석에 쓸 범위

- 이 논문이 보인 것은 "영상 생성 → IDM 라벨 → 정책 학습" 연쇄가 여러 로봇에서 작동한다는 것입니다.
- 보이지 않은 것은 (a) 신경 궤적의 품질이 정책 성능의 어디까지를 설명하는지(Table 4에 신경 궤적 240k 한 설정만 있음), (b) 평가 점수 차이가 통계적으로 확인되는지(10회 실험, 03 §4), (c) 사람 영상 대안과의 정량 비교입니다.
- 2026년 현재 맥락에서 이 논문은 "세계모델을 데이터 생성기로 쓰는" 초기 방향입니다. 이후 연구(위 목록의 World Action Model과 세계모델 강화학습 계열)는 같은 질문을 다른 축으로 풉니다.
