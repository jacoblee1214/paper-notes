# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

대괄호 번호는 논문 참고문헌 번호입니다.

| 구성 요소 | 이 논문이 쓰는 것 | 직접 선행 |
|---|---|---|
| 합성·최적화 기반 데이터 | 이 논문은 쓰지 않고 비교만 함. 힘 폐쇄·접촉·관통 목적함수로 파지를 직접 최적화 | DexGraspNet [9], BODex [29, ICRA 2025], GenDexGrasp [21], 미분 가능 힘 폐쇄 [20], 생성 + 이중 수준 최적화 [10] |
| RL 롤아웃 주석 | 손마다 PPO 정책(power, precision)으로 부유 물체 파지를 모음. 다양성 보상으로 중복 억제 (§8.1) | UniDexGrasp [27], UniDexGrasp++ [17], GraspXL [11], MultiGripperGrasp [12], Dex1B [14] |
| 어지러운 장면 파지 | 표적 물체 전용 생성 + 장면 점군 조건 (§2.2) | DexGraspNet 2.0 [1, CoRL 2024], AnyDexGrasp [4], ContactDexNet [2, IROS 2025], CADGrasp [7, NeurIPS 2025], ClutterDexGrasp [24], Neural Attention Field [25], HGC-Net [3], DDGC [5] |
| 교차 형태(cross-embodiment) 표현 | 손 링크별 6D 자세와 상대 자세 간선 | UniGrasp [18], ManiFoundation [19], GeoMatch [22], NeuralGrasps [23], D(R,O) Grasp [15, ICRA 2025], T(R,O) Grasp [16, ICRA 2026]. **Grasp++는 T(R,O) Grasp에 환경 노드, 링크 지문, 확장 데이터를 더한 것(§4)** |
| 기하 인코더 | PointNet++ 기반 VQ-VAE를 물체·환경에 따로 사전학습 (§4.1.1) | VQ-VAE [31], PointNet++ [32] |
| 링크 기하 특징 | 공유 기저점 집합(BPS)에서의 거리 (§4.1.2) | BPS [33] |
| 확산 | 코사인 스케줄 노이즈 예측, DDIM 20단계 (§4.2, §9) | DDIM [34] |
| 그래프 어텐션 | 간선 조건부 어텐션 (식 7) | Transformer [35] |
| IK | Pyroki 다중 초기화 IK (§4, §9) | Pyroki [30] |
| 학습 | AdamW [39], bf16, PPO [38] | — |

### 가까운 대안

- **T(R,O) Grasp** [16] (같은 저자, ICRA 2026): 표 4에서 지연 0.210 s, 28.06M으로 더 빠르고 작지만 PPS가 100 초과입니다. 환경과 링크 지문이 없어 어지러운 장면에서 제안 대부분이 쓸 수 없다는 것이 이 논문의 출발점입니다(§1, 표 4).
- **D(R,O) Grasp** [15]: 손과 물체의 상호작용 거리 행렬로 교차 형태 표현을 만든 원형입니다.
- **DexGraspNet 2.0** [1]: 이 논문의 비교 대상이자 가장 가까운 데이터 규모(4.27억 파지, 1,319 물체, 단일 손)입니다.
- **KPGrasp**: 손 키포인트를 흐름 정합으로 생성하는 경쟁 접근입니다(관절 값 대신 유클리드 키포인트). 이 논문은 링크 SE(3)를 쓰므로 회전 표현 문제를 직접 안고 갑니다(`04` D11).
- **CADGrasp** [7]: 접촉·충돌 인지 학습으로 어지러운 장면을 다룹니다. 이 논문은 비교하지 않았습니다.
- **XDex**: 같은 CoRL 2026 포스터 세션(117번, "Learning Cross-Embodiment Dexterous Grasping with 1000 Hands")의 관련 논문입니다. 원문을 보지 못했습니다(OpenReview 목록의 제목만 확인).

## 2. 코드와 저장소 (GitHub 표)

별 수와 마지막 push는 2026-11-22에 `gh api`로 읽었습니다.

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [Barrybarry-Smith/TROGraspV2](https://github.com/Barrybarry-Smith/TROGraspV2) | **공식 구현**(프로젝트 페이지 Code 링크, README가 CoRL 2026 공식 저장소라고 밝힘) | LICENSE 파일 없음(API license: null). `third_party/pyroki`는 MIT | 8 | 2026-10-27 | 정적 읽기만. 커밋 `449db5a16754412f3f31f516dbf4a601897f58ff` (2026-10-27), 287 MB 얕은 복제 |
| [HF 데이터셋 Barrybarry-Smith/TROV2Dataset](https://huggingface.co/datasets/Barrybarry-Smith/TROV2Dataset) | 공식 학습 데이터와 검증 장면 | 데이터 카드가 "라이선스·재배포 조건 미선언"이라고 명시 | — | 2026-10-18 | 받지 않음. Zarr 25,228개 객체(6.19 TiB), 검증 장면 6.9 GB, 객체 점군 1.5 GB, 객체 메시 8.8 GB |
| [NUS-LinS-Lab/TROGrasp2Web](https://github.com/NUS-LinS-Lab/TROGrasp2Web) | 프로젝트 페이지 소스 | 확인 안 함 | 1 | 2026-10-20 | 페이지(`nus-lins-lab.github.io/TROGrasp2Web/`)만 확인 |
| 체크포인트 | 공식 README의 Google Drive 링크(`ckpt`, 8손 `eight_hand.pth`, 9손 `nine_hand_with_wuji.pth`, VQ-VAE `object.ckpt`·`env.ckpt`) | 미표기 | — | — | 받지 않음. 크기 미확인 |
| [Barrybarry-Smith/TRO-Grasp](https://github.com/Barrybarry-Smith/TRO-Grasp) | 전작 T(R,O) Grasp 공식 구현(ICRA 2026) | LICENSE 없음 | 54 | 2026-05-10 | 읽지 않음 |
| [zhenyuwei2003/DRO-Grasp](https://github.com/zhenyuwei2003/DRO-Grasp) | D(R,O) Grasp 공식 | MIT | 309 | 2026-04-17 | 읽지 않음 |
| [PKU-EPIC/DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2) | 비교 대상 [1] | LICENSE 없음 | 165 | 2025-01-23 | 읽지 않음. 이 논문의 베이스라인 평가 코드는 공식 저장소에 없음(`04` D14) |
| [graspnet/AnyDexGrasp](https://github.com/graspnet/AnyDexGrasp) | 비교 대상 [4] | LICENSE 없음 | 231 | 2025-03-22 | 읽지 않음 |
| [JYChen18/BODex](https://github.com/JYChen18/BODex) | 최적화 기반 주석 비교 [29] | NOASSERTION | 156 | 2026-04-21 | 읽지 않음 |
| [PKU-EPIC/DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) | 데이터셋 선행 [9] | LICENSE 없음 | 483 | 2025-01-06 | 읽지 않음 |
| [chungmin99/pyroki](https://github.com/chungmin99/pyroki) | IK 도구 [30] | MIT | 1826 | 2026-04-01 | 공식 저장소가 `third_party`에 복사본을 포함 |

- 공식 저장소에는 **학습·추론·시각화·시뮬 검증 스크립트**가 있고, **데이터 생성(RL 주석, 장면 생성, 난이도 점수), 베이스라인 평가, 절제·규모·다중 시도 실험, 실로봇 코드는 없습니다**(`04` D14).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **"RL 주석이 최적화 주석보다 빠르고 filtering 성공이 높다" (§3.2).** 4090 한 장에서 초당 216개라는 수치와 비교 대상 [21, 29] 사이의 설정 정보(같은 손, 같은 물체)가 논문에 없습니다. 재계산할 수 없는 서술입니다.
- **"링크 지문 없이는 비슷한 링크를 구분하지 못한다" (§4.1, 표 6).** 유사한 접근인 D(R,O) Grasp와 T(R,O) Grasp는 한 모델에 여러 손을 넣을 때 이 문제를 겪었다는 서술이 [16]에 있을 것으로 보이나 이 분석에서 [16]은 읽지 않았습니다.
- **데이터 규모의 비교(표 1).** DexGraspNet 2.0의 4.27억 파지(물체·손당 약 32.4만)와 이 논문의 4.44억(물체·손당 약 1천)은 총량은 비슷하고 다양성 축이 다릅니다(`03` V3).
