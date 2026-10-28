# 02. 선행연구, 계보, 관련 GitHub

번호 [n]은 논문의 참고문헌 번호이고, arXiv ID는 논문 참고문헌에 적힌 값입니다. SAM 3D 인용(§2)과 GitHub 표(§5)는 직접 확인했습니다. 나머지 논문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[파지 합성]  해석적 force closure ─→ DexGraspNet [2] (Shadow Hand, 미분 가능 force closure 에너지, 대규모 시뮬레이션 데이터)
                                      └→ UniDexGrasp [4], AnyDexGrasp [5] arXiv:2502.16420 (범용·다른 손)
[후보 순위]  GRaCE [6] arXiv:2309.08887 (안정성·도달성·기능의 다기준 순위, 계층 논리 + 순위 보존 효용)
[과제 지향]  TaskGrasp [18] arXiv:2011.06431 (과제별 파지 데이터 + GCNGrasp)
[언어]       Code as Policies [10], LAN-grasp [11], Grasp as You Say [12], VL 과제 지향 파지 [13]
[힘·순응]    임피던스 제어 Hogan [8], 힘 피드백 기능 파지 [7], FEM 파지 해석 [9]
[인식]       SAM [15], YOLO-World [16], SAM 3D (인용은 [17]이지만 다른 논문, §2)

          ▼ 다기준 순위에 "국소 허용 하중"이라는 기준을 하나 더하고, 같은 값을 제어 강성에도 씀
  ★ GraspSense: 힘 지도 → 재순위화 → 손가락별 임피던스
```

**차별점 요약:** 기존 파지 계획기는 물체 표면을 기계적으로 균일하다고 보고, 순응 제어는 물체 구조와 무관한 일반적인 강성값을 씁니다(Sec. II-E). GraspSense는 둘 사이에 표면별 허용 하중 지도를 끼워 넣었습니다. 새 요소는 학습 모델이 아니라 기하 휴리스틱(벽 두께 → 허용 하중)과 그 값을 쓰는 규칙 두 개입니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 과제 이해 | Qwen [14] arXiv:2309.16609 | 명령에서 (O, a, λ) 추출. Fig. 2는 "Qwen 2.5", 공개 코드는 Qwen2.5-VL-3B-Instruct |
| 검출 | YOLO-World [16] (CVPR 2024) | 텍스트 질의로 물체 검출, 내부 점 하나 |
| 분할 | SAM [15] (ICCV 2023) | 점 프롬프트로 마스크 |
| 3D 복원 | **실제로 쓴 것:** Meta *SAM 3D: 3Dfy Anything in Images* arXiv:2511.16624 (`facebookresearch/sam-3d-objects`). **인용된 것:** [17] *SAM3D: Segment Anything in 3D Scenes* arXiv:2306.03908 | 본문 설명(희소 구조 → 정제, Gaussian/메쉬 디코더), v1 Fig. 2의 Meta 로고, 공개 코드 README가 모두 앞의 것을 가리킵니다. [17]은 SAM으로 3D 장면을 분할하는 다른 연구입니다 |
| 파지 후보 | DexGraspNet [2] (ICRA 2023) | 미분 가능 에너지 최적화, Isaac Gym 검증 |
| 기능 필터 | TaskGrasp [18] (CoRL 2020) | c_F 평가기. 다섯 손가락 파지에 적용한 방법은 설명 없음 |
| 순위 | GRaCE [6] | 기준선 효용 U(g)의 출처로 보이나, 본문에 정의 없음 |
| 순응 제어 | Hogan [8] (1985) | 데카르트 임피던스 Eq. 9 |
| 시뮬레이터 | Isaac Sim (실험), Isaac Gym (후보 검증), Isaac Lab (공개 코드의 USD 변환) | 세 시뮬레이터가 섞여 있음 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **Do as I Do** (`do_as_i_do/`) | 같은 SAM 3D(arXiv:2511.16624)로 물체 메쉬를 복원합니다. Do as I Do 분석에서 본 대로 SAM 3D 메쉬의 스케일은 임의라 별도 정렬이 필요한데, GraspSense 공개 코드에는 실제 크기 정규화가 아직 없습니다(`04` D4). |
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | 둘 다 compliance(임피던스) 제어의 목표를 접촉에 맞게 정합니다. CGP는 상태·촉각 예측에서 학습한 매핑으로 정하고, GraspSense는 미리 계산한 힘 지도로 정합니다. 하나는 닫힌 루프 학습, 하나는 열린 루프 사전 지식입니다. |
| **FlashDexRetarget** (`flashdexretarget/`) | 같은 Isaac Sim 계열의 dexterous 손 시뮬레이션입니다. FlashDexRetarget의 접촉 보상은 물체가 얼마나 약한지를 고려하지 않으므로, 허용 하중 대비 접촉력 비율을 벌점 항으로 넣을 수 있습니다. |
| **ZeroDex** (`zerodex/`) | 둘 다 언어/VLM → 물체 인식 → 메쉬 기반 파지라는 zero-shot 파이프라인이고, 물체 메쉬를 어디서 얻는지가 핵심 가정입니다. ZeroDex는 메쉬 출처를 밝히지 않았고, GraspSense는 SAM 3D 단일 이미지 복원으로 얻습니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **변형체 파지 시뮬레이션** (예: DefGraspSim, Isaac Gym FEM으로 파지 중 응력·변형을 측정) | 힘 지도를 검증할 수 있는 바로 그 도구입니다. 같은 컵을 FEM으로 눌러 실제 항복 하중과 비교하면 순환 평가 문제가 풀립니다. | 사전 지식, 서지 확인 필요 |
| **Grasp force optimization** (최소 내부력으로 물체를 유지하는 고전 문제) | 손가락별 힘 상한(허용 하중)을 제약으로 넣으면 GraspSense의 강성 규칙보다 원리적인 힘 분배가 됩니다. | 사전 지식, 서지 확인 필요 |
| **얇은 판·셸의 점하중 공식** (Roark류 핸드북) | 허용 하중이 대략 σ·t²에 비례한다는 차원 분석의 근거입니다(`03` §5). F_base의 함수 꼴을 정할 때 쓸 수 있습니다. | 사전 지식 |
| **촉각 기반 미끄러짐 감지·파지력 조절** | 저자들이 향후 과제로 든 "닫힌 루프 촉각 피드백"의 기존 해법입니다. | 사전 지식 |

## 5. 관련 GitHub (2026-10-28 `gh api` 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| eliizavetas/GraspSense | GraspSense 파이프라인의 인식·장면 준비 부분 (Qwen, YOLO-World, SAM, SAM 3D, Isaac Lab USD). 힘 지도는 placeholder | 논문에 링크 없음. 소유자 이름이 1저자와 같고 README가 같은 연구실 장비를 가리킴 → 저자 저장소로 추정 | MIT | 0 | 2026-06-01 | `16dc5fb` |
| facebookresearch/sam-3d-objects | SAM 3D Objects (단일 이미지 → 3D) | 구성 요소 공식 | 비표준(SAM License) | 7,602 | 2026-06-02 | `f91db41` |
| facebookresearch/segment-anything | SAM | 구성 요소 공식 | Apache-2.0 | 55,073 | 2024-09-18 | `dca509f` |
| AILab-CVC/YOLO-World | YOLO-World | 구성 요소 공식 | GPL-3.0 | 6,593 | 2025-02-26 | `4f70adb` |
| ultralytics/ultralytics | 공개 코드가 실제로 쓰는 YOLO-World 구현(`yolov8s-worldv2`) | 제3자 패키지 | AGPL-3.0 | 62,935 | 2026-10-28 | `bb3e155` |
| PKU-EPIC/DexGraspNet | 파지 후보 생성기 | 구성 요소 공식 | 표기 없음 | 478 | 2025-01-06 | `bd1b13d` |
| adithyamurali/TaskGrasp | 기능 필터 평가기 | 구성 요소 공식 | MIT | 100 | 2023-03-14 | `c0bad3b` |
| clear-nus/GRaCE | 다기준 파지 순위 | 관련 연구 공식 | MIT | 4 | 2023-09-16 | `075f555` |
| QwenLM/Qwen2.5-VL (현재 Qwen3-VL로 이동) | 공개 코드의 과제 이해 모델 | 구성 요소 공식 | Apache-2.0 | 20,133 | 2026-01-30 | `9658872` |
| Pointcept/SegmentAnything3D | 논문이 [17]로 **잘못** 인용한 SAM3D(3D 장면 분할) | 관련 없음 | MIT | 1,392 | 2024-04-21 | `2aaa8c9` |
| PKU-EPIC/UniDexGrasp | 범용 dexterous 파지 [4] | 관련 연구 공식 | 표기 없음 | 276 | 2024-05-25 | `36c9bfc` |
| graspnet/AnyDexGrasp | 손 형태 간 파지 [5] | 관련 연구 공식 | 표기 없음 | 227 | 2025-03-22 | `c9c4a43` |

참고: 같은 이름의 `jussihh/graspsense`(2021)는 무관한 프로젝트입니다. 라이선스 조합상 GraspSense를 상업적으로 쓰려면 SAM 3D(SAM License)와 YOLO-World(GPL-3.0)/ultralytics(AGPL-3.0)의 조건을 따로 확인해야 합니다.

**한계:** 4절의 서지 정보는 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 원문을 직접 확인해야 합니다.
