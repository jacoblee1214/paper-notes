# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호이고, arXiv ID는 논문 참고문헌 목록의 값을 옮겼습니다. LIBERO-PRO·LIBERO-plus 리더보드와 MolmoAct2 README는 원문을 직접 확인했습니다(§5). 나머지 논문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[VLA]  RT-2 [9] → OpenVLA [41] → π0 [7] → π0.5 [34] → GR00T N1 [6], MolmoAct [43] → MolmoAct2 [25]
          │                                                      (action expert와 YAM 데이터를 ThinkWorld가 그대로 씀)
[중간 추론을 넣은 VLA]
   텍스트 계획: ECoT [79], ThinkAct [33] (같은 그룹, RL로 계획 강화)
   공간 궤적: MolmoAct [43]   잠재 사고: Fast-ThinkAct [32] (같은 그룹)   거친 행동: ACoT-VLA
   추론 지연 없는 추론: TacitVLA [54] arXiv:2610.17540
          │
[사람 영상에서 행동 신호]
   잠재 행동: LAPA [76], UniVLA [10], Moto [18], CoNLA [21]    ← 레이블 없이 확장, 대신 외관·카메라 움직임이 섞임
   명시 감독: EgoMimic [36], EgoScale, EgoVLA [74], EgoWAM [44] ← 자세 추정·대응이 필요
          │
[미래 예측·세계 모델]
   Dreamer [28, 29], V-JEPA 2 [3], Cosmos [1, 2] → VLA-JEPA [64], DreamVLA, LaWAM [15], InternVLA-A1 [12], FoMoVLA [46]
   세계-행동 모델: WorldVLA [13], mimic-video [56], World Action Models [77], Cosmos Policy [42], Hydra-0 [45]
          ▼
★ ThinkWorld: 공유 움직임 토큰(CoD) + CoD 조건부 DINOv3 미래 예측 + 예측-관찰 일치 보상 GRPO.
   추론과 예측은 학습에만 쓰고, 배포는 일반 VLA와 같습니다.
```

**차별점:** 잠재 행동 모델과 달리 CoD는 실제 손목 움직임(사람 손 자세, 로봇 EEF)으로 정답이 정해진 이산 토큰입니다. 미래 예측 VLA와 달리 예측을 계획이나 행동 출력에 쓰지 않습니다. 예측은 학습 중 CoD를 평가하는 데만 씁니다(Sec. 2 "In contrast, ThinkWorld does not use future prediction as the control output").

## 2. 구성 요소별 출처

| 역할 | 출처 | ThinkWorld에서의 쓰임 |
|---|---|---|
| VLM 백본 | Qwen3-VL-2B [4] | 전체 미세조정(SFT, RL), YAM 과제는 LoRA |
| 연속 행동 출력 | MolmoAct2 action expert [25] | 36블록 flow-matching expert. Action SFT는 VLM을 고정하고 학습("knowledge insulating" [23]을 따름) |
| 움직임 토큰화 | FAST [57] | 1초 손목 delta 6D/12D → 어휘 2,048 |
| 미래 표적 공간 | DINOv3 ViT-H/16+ [63] | 224×224 → 14×14 패치 × 1,280차원, 동결 |
| 움직임 가중 유사도 | MV2MAE [61] (motion-aware 가중의 근거로 인용) | Eq. 6의 순위 기반 가중치 |
| RL 알고리즘 | GRPO [62], GDPO [51] (같은 그룹) | K = 4, 그룹 후 배치 표준화 |
| 사람 데이터 | EgoDex [30], EgoVerse [58] | 1,880시간, 손목 자세 주석 사용 |
| 로봇 데이터 | MolmoAct2-BimanualYAM [25], DROID [37], RoboCOIN [72], Bridge [69], RT-1 [55], BC-Z [35] | 2,396시간 |
| VQA 보존 | PixMo-Cap [22], Molmo2 [20], RoboPoint [78], RoboVQA [60] | 언어 능력 망각 방지 |
| 평가 | LIBERO [49], LIBERO-PRO, LIBERO-Plus [26], SimplerEnv [47] | 시뮬레이션 일반화 |
| 기준선 학습 | LeRobot [11] | 실물 기준선 레시피 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **Hydra-0** (`hydra0/`, [45]) | 같은 NVIDIA이고 둘 다 "미래 예측"을 쓰지만 용도가 반대입니다. Hydra-0은 action flow에 조건화된 비디오 world model 자체를 시뮬레이터와 평가기로 씁니다. ThinkWorld는 1초 뒤 DINO 특징만 예측하고, 그 예측을 학습 신호로만 쓴 뒤 배포에서 버립니다. Hydra-0 분석에서 본 "평가 기준이 시뮬레이터"라는 문제가 ThinkWorld의 RL 보상에도 비슷하게 있습니다. 보상의 기준이 실제 결과가 아니라 모델 자신의 예측 일치도입니다. |
| **AW-0** (`agentic_wam/`) | AW-0은 코드·미래 영상·행동을 한 MoT에서 함께 denoise하고, 계획(코드)을 추론 때도 생성합니다. ThinkWorld의 CoD는 손목 움직임 토큰이고 추론 때는 생성하지 않습니다. 두 논문 모두 "계획 스트림"의 기여를 ablation으로 주장하므로, AW-0 분석의 "같이 빠지는 것" 점검이 여기에도 적용됩니다. ThinkWorld의 Table 3에서 미래 예측 손실을 더하면 Env +24.6%p인데, 이 손실만 따로 떼어 본 대조(CoD 없이 미래 예측만)는 있습니다. 반면 "CoD는 있고 미래 예측은 없는" 대조는 없습니다. |
| **FlashDexRetarget, Do as I Do** (`flashdexretarget/`, `do_as_i_do/`) | 사람 시연을 로봇 데이터로 바꾸는 명시적 리타겟팅 계열입니다. ThinkWorld가 "자세 추정과 대응이 필요하다"고 비판하는 쪽입니다. ThinkWorld도 사람 손목 자세 주석을 쓰지만 기구학 대응은 하지 않고 공유 토큰 공간에 넣을 뿐입니다. 실행 가능한 로봇 궤적이 필요하면 리타겟팅이, VLM 표현만 바꾸려면 ThinkWorld 방식이 맞습니다. |
| **DexPolicy, GQKAE** (`dexpolicy/`, `gqkae/`) | 둘 다 GRPO의 advantage 설계 문제를 다뤘습니다. DexPolicy는 궤적 전체에 같은 advantage를 주는 구현이었고, GQKAE는 최선 표본 NLL 항이 있었습니다. ThinkWorld도 후보 하나의 advantage를 모든 토큰이 공유하고, 그룹 표준화 뒤 배치 표준화를 합니다. 여기에 형식 오류 벌칙이 보상 폭보다 커서 표준화에 끼치는 영향이 큽니다(`03` §3). |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **역동역학(inverse dynamics) 모델 기반 의사 레이블링** (예: VPT 계열) | "관찰된 미래를 가장 잘 설명하는 행동 토큰을 고른다"는 RL 목표는 사실상 학습된 순방향 모델을 통한 역동역학 추론입니다. 의사 레이블 + 지도학습이 GRPO보다 단순한 대안입니다. | 내 해석. 서지 확인 필요 |
| **보조 손실로서의 미래 예측** (예: 자기지도 표현 학습의 future prediction) | Table 3에서 이득의 63%가 미래 예측 보조 손실에서 나옵니다. 이 논문의 이득은 RL보다 보조 손실 문헌과 더 가깝습니다. | Table 3 재계산 |
| **TacitVLA [54]** | "추론 지연 없는 추론 VLA"라는 같은 문제를 다룹니다. 학습 때만 추론을 쓰는 설계와 직접 비교할 만합니다. | 논문 인용. 내용 미확인 |

## 5. GitHub 정리 (2026-10-29 `gh api` 확인)

공식 코드와 가중치는 없습니다. 프로젝트 페이지 저장소에도 페이지 HTML·JS와 그림·영상, 그림 빌드 스크립트만 있습니다.

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| thinkworld-project/thinkworld-project.github.io | 프로젝트 페이지 (Code·Model 버튼 비활성) | 공식 | 표기 없음 | 0 | 2026-10-28 | `30b9dab` (파일 목록) |
| allenai/molmoact2 | action expert, BimanualYAM 데이터, 실물 기준선, MolmoAct2-LIBERO 체크포인트 | 구성 요소 공식 | Apache-2.0 | 820 | 2026-10-02 | `6070080` (README) |
| QwenLM/Qwen3-VL | VLM 백본 | 구성 요소 공식 | Apache-2.0 | 20,136 | 2026-01-30 | `9658872` (메타데이터) |
| facebookresearch/dinov3 | 미래 특징 표적 | 구성 요소 공식 | DINOv3 License (SPDX 없음) | 11,643 | 2026-07-15 | `6876159` (메타데이터) |
| Physical-Intelligence/openpi | FAST 토크나이저 | 구성 요소 공식 | Apache-2.0 | 14,442 | 2026-08-24 | `215abfb` (메타데이터) |
| apple-aiml-research/ml-egodex | EgoDex 데이터 | 데이터 공식 | SPDX 없음 | 404 | 2026-09-11 | `718d9eb` (메타데이터) |
| GaTech-RL2/EgoVerse | EgoVerse 데이터 | 데이터 공식 | MIT | 575 | 2026-10-27 | `9a4194b` (메타데이터) |
| Zxy-MLlab/LIBERO-PRO | LIBERO-PRO 평가와 리더보드 | 평가 공식 | MIT | 347 | 2026-07-17 | `eafdb80` (리더보드와 Table 1 대조) |
| sylvestf/LIBERO-plus | LIBERO-Plus 평가와 리더보드 | 평가 공식 | 표기 없음 | 486 | 2026-01-21 | `4976dc3` (리더보드와 Table 8 대조) |
| Lifelong-Robot-Learning/LIBERO | LIBERO | 평가 공식 | MIT | 2,488 | 2025-03-15 | `8f1084e` (메타데이터) |
| simpler-env/SimplerEnv | SimplerEnv | 평가 공식 | MIT | 1,181 | 2026-09-30 | `eddb569` (메타데이터) |
| ginwind/VLA-JEPA | LIBERO 기준선 | 기준선 공식 | Apache-2.0 | 622 | 2026-09-29 | `ec90777` (README) |
| RLinf/LaWAM | 실물 기준선 LaWAM으로 보이는 저장소 | 공식 여부 미확인 | 표기 없음 | 158 | 2026-10-11 | `c12168d` (메타데이터) |
| Zhangwenyao1/DreamVLA | LIBERO-Plus 기준선 | 공식 | 표기 없음 | 373 | 2026-01-06 | `2318020` (메타데이터) |
| huggingface/lerobot | 기준선 학습 레시피 | 의존성 | Apache-2.0 | 28,554 | 2026-10-29 | `72e1d9e` (메타데이터) |
| jasper0314-huang/fast-thinkact | 같은 그룹의 선행 연구 Fast-ThinkAct 페이지 | 관련 | 표기 없음 | 0 | 2026-06-17 | `d8515e4` (README는 페이지 템플릿) |

**메모:**
- MolmoAct2 README의 2026-08-22 공지에 따르면 LeRobot의 MolmoAct2 미세조정 스크립트 문제가 그날 고쳐졌습니다. ThinkWorld의 실물 기준선이 LeRobot 레시피로 학습되었는데, 그 학습이 수정 전인지 후인지는 논문에서 알 수 없습니다.
- VLA-JEPA README에는 LIBERO-Plus 평가 코드가 있습니다. 하지만 ThinkWorld Table 8은 "공개 결과가 있는 방법만"이라며 VLA-JEPA를 넣지 않았습니다. VLA-JEPA가 LIBERO-Plus 수치를 공개했는지는 확인하지 않았습니다.

**한계:** 4절의 서지는 사전 지식에 근거합니다. 게시 전에 원문을 확인해야 합니다.
