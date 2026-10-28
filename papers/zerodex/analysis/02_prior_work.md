# 02. 선행연구와 계보

번호 [n]은 v3 참고문헌 번호이고, arXiv ID는 참고문헌 목록에 적힌 값을 그대로 옮겼습니다. 이 문서의 각 논문을 별도로 열어 내용을 검증하지는 않았습니다. 하이퍼파라미터와 프롬프트는 v1 부록(arXiv:2606.19340v1)에서 직접 확인했습니다.

## 1. 계보도

```
[end-to-end VLA: 로봇 데이터로 학습]
  RT-1 [1] → RT-2 [2] → OpenVLA [3], Octo [4] → π0 [5]
  dexterous·휴머노이드: GR00T N1 [20], DexVLA [15], DexGraspVLA [21], Being-H0 [22] ← 비교 대상(100개 시연 미세조정)

[foundation model로 zero-shot 조작: 중간 표현이 무엇인가]
  코드 생성        Code as Policies [29], ProgPrompt [30], VoxPoser [31] (3D value map)
  visual prompting PIVOT [33], RoboPoint [34], Manipulate-Anything [35]
                   MOKA [6] arXiv:2403.03174   키포인트 + 운동 원시 동작, 단일 RGB-D 역투영   ← 비교 대상
  제약 기반        ReKep [32] arXiv:2409.01652  키포인트 관계 제약을 풀어 엔드이펙터 자세  ← 비교 대상
                   OmniManip [42] (물체 중심 상호작용 원시 동작을 공간 제약으로)
  오케스트레이터   MAESTRO [36], [37] (VLM이 기술을 순서화·검증, 실행은 학습된 정책)

[3D 그라운딩] 다시점 스테레오 [38], SpatialVLM [39], RoboSpatial [40], VLM-Grounder [41](다시점 → 3D 그라운딩)
[실행] 물체 중심 기술 이전: VLMPose [45](같은 그룹), MimicFunc [46]
       dexterous 파지: DexGraspNet [47], BODex [48], Dexonomy [49], RT-Affordance [50]
       모션: cuRobo v2 [52] arXiv:2603.05493, Natural functional gradients [53] arXiv:2605.28202(공저자)
       검증·재계획: Inner Monologue [51]

          ▼
  ★ ZeroDex: VLM 키포인트를 다시점 합의로 metric 3D에 올리고, 물체 중심 6D 궤적 정렬 + 어포던스 파지 + 충돌·IK 검증
```

**차별점 요약:** MOKA와 ReKep은 VLM 출력을 **단일 RGB-D 시점**에서 깊이로 역투영하고, 남는 깊이 모호성을 VLM 추론이나 제약 범위에 맡깁니다. ZeroDex는 같은 VLM 질의를 여러 보정 시점에 반복해서 **기하 합의**로 깊이를 정합니다. 그리고 그 3D 키포인트를 도구 궤적 정렬과 파지 영역 결정에 함께 씁니다. 학습된 정책이 없다는 점은 MOKA, ReKep과 같고, 실행 스택(파지 생성, 모션 계획)은 비교 실험에서 공유합니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| VLM | Gemini Robotics-ER 1.6 (v1: `gemini-robotics-er-1.6-preview`), Gemini Robotics 1.5 [19] | 계획, 기준 시점 선택, 시점별 키포인트, 어포던스 박스, 실행 검증 |
| 키포인트 + 원시 동작 표현 | MOKA [6] | grasp / waypoint / release 구조가 가장 비슷한 선행연구 |
| 강건 합의 | RANSAC [54] | 시점 쌍 삼각측량 후보를 재투영 inlier 수로 고름 (Eq. 5–6) |
| 다시점 기하 | Multi-view stereo tutorial [38] | 삼각측량의 근거 |
| 깊이·자세 | FoundationStereo [43], FoundationPose [44] | 스테레오 쌍 깊이(장면 충돌), 다중 물체 6D 자세 |
| 물체 중심 궤적 생성 | VLMPose [45] arXiv:2604.09781 (Baik, Kim, Choi, Joo) | Bag of Atomic Actions 항목 생성 |
| 파지 생성 | BODex [48] + 어포던스 거리·방향 항 (v1 부록 B.2), Dexonomy [49]식 손가락 닫기 (B.1) | 일반 형상 / 손잡이형 영역 |
| 팔-손 모션 | cuRobo v2 [52], Natural functional gradients [53] | 운반 궤적, 기구학·충돌 제약, η |
| 실행 검증·재계획 | Inner Monologue [51] | 텍스트 메모리 + continue / retry / replan |
| VLA 기준선 | GR00T N1 [20] arXiv:2503.14734, Being-H0 [22] arXiv:2507.15597 | 100개 시연으로 사후학습 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **FlashDexRetarget** (`flashdexretarget/`) | 둘 다 "작업마다 로봇 시연을 모으지 않고" dexterous 손 동작을 만들지만 방향이 반대입니다. FlashDexRetarget은 사람 시연을 RL로 로봇 궤적으로 **변환**하고, ZeroDex는 서론에서 그 경로(리타겟팅의 embodiment 간극)를 피해야 할 문제로 듭니다. 그런데 ZeroDex의 Bag of Atomic Actions는 바로 **물체 중심 6D 도구 궤적**을 저장합니다. FlashDexRetarget이나 아래 Do as I Do가 사람 시연에서 얻는 물체 궤적은 embodiment와 무관하므로 이 라이브러리를 채우는 데 그대로 쓸 수 있습니다. 실행 측면에서는 FlashDexRetarget이 물리적으로 검증된 접촉을, ZeroDex는 파지 시뮬레이션 + 운반 궤적 계획을 씁니다. |
| **Do as I Do** (`do_as_i_do/`) | 같은 "영상·VLM → 3D 기하" 문제를 다른 방식으로 풉니다. Do as I Do는 단안 영상에서 SAM 3D로 메쉬와 4D 궤적을 복원하고, ZeroDex는 보정된 다시점 + FoundationPose로 자세를 얻습니다. ZeroDex가 본문에 드러내지 않은 "물체 메쉬 필요" 가정(01 §4.2)을 Do as I Do식 SAM 3D 복원으로 대체할 수 있는지가 둘을 잇는 질문입니다. Do as I Do가 출력하는 물체 궤적도 Bag of Atomic Actions 항목이 될 수 있습니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **Visual hull / space carving** (Laurentini 1994; Kutulakos & Seitz 2000) | Eq. 9의 어포던스 영역은 시점별 2D 박스 시야뿔의 투표형 교집합입니다. 가시성 검사가 없어서 가려진 뒷면 꼭짓점도 포함됩니다(`03` §4). | 내 사전 지식. 서지 확인 필요 |
| **두 점 정렬과 absolute orientation** (Horn 1987; Umeyama 1991) | Eq. 7은 대응점 2개짜리 정합입니다. 일반 회전은 앵커 축 둘레로 결정되지 않으므로 회전 축 제한이 필수이고, 축척이 다르면 잔차가 남습니다(`03` §3). | 내 사전 지식. 서지 확인 필요 |
| **Self-consistency** (LLM의 다중 샘플 투표) | 시점별 VLM 질의를 합의로 묶는 것은 시점 다양성을 이용한 self-consistency와 같은 구조입니다. 같은 시점에서 여러 번 샘플링해 투표하는 확장도 생각할 수 있습니다. | 내 해석 |
| **과제 지향 파지** (예: GraspGPT, LERF-TOGO) | 언어 조건 어포던스 → 파지라는 같은 문제를 학습 기반으로 다룹니다. | 내 사전 지식. 서지 확인 필요 |

**한계:** 4절의 서지 정보는 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 원문을 직접 확인해야 합니다.
