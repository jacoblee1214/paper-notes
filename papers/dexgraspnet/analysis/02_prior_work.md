# 02. 선행연구와 관련 자료: DexGraspNet

확인 날짜 2026-11-22. GitHub 정보는 GitHub REST API(`api.github.com`)로, 학회 표기는 Crossref와 프로젝트 페이지로 확인했습니다. 논문 참고문헌의 나머지 항목은 논문 목록을 옮긴 것이고 각 논문을 열어 검증하지는 않았습니다.

## 1. 계보

```
[손가락 기반 파지의 분석적 방법]
  힘 폐쇄 (Ferrari & Canny 1992 [63])  ·  GraspIt! (Miller & Allen 2004 [10])
        │
        ├─ GraspIt!로 만든 DDGdata (논문 [10], DDG [9]의 데이터)
        │       └─ 한계: 고유 벡터(eigengrasp) 공간을 탐색 → 포즈가 좁게 분포, 관절이 한계에 몰림 (논문 §II-C, 그림 6)
        │
        └─ 미분 가능한 힘 폐쇄 추정기 (Liu 외 [19]: 이 논문이 "원 알고리즘"으로 부르는 것)
                └─ 한계: 성공률 3%, 수렴 느림, 두께 없는 메시에서 관통 에너지가 0
                        │
                        ▼
[이 논문: DexGraspNet]  초기화 개선 + 역 관통 에너지 + 자기 관통·관절 한계 항 + 단순 최적화 + Isaac Gym 검증
                        │
                        ▼
[후속]  이 분석에서 확인하지 않은 후속작은 §3에 적음 (DexGraspNet 2.0은 다른 분석 문서에서 CoRL 2024로 표기됨)
```

- 이 논문에서 "DDG"는 두 가지 뜻입니다. 데이터셋(DDGdata, GraspIt!로 만든 6.9K 파지)과 학습 기법(DDG, Liu 외 [9]의 미분 가능 Q1 손실을 쓰는 네트워크)입니다. 표 III에서는 기법을 가리킵니다.

## 2. 구성 요소별 출처

| 구성 | 논문이 쓰는 출처 | 이 논문에서의 역할 | 비고 |
|---|---|---|---|
| 미분 가능 힘 폐쇄 | Liu 외 [19] (원 알고리즘) | 에너지 \(E_{fc}\) (Eq. 1)의 기반 | 최적화 세부는 바뀜 (§III-B.3) |
| 손 모델 | ShadowHand [8] | 파지 대상 손 (22 관절) | MANO [20], Allegro도 같은 파이프라인 |
| 물체 | ShapeNetCore·ShapeNetSem [52, 53], YCB [54], BigBIRD [55], Grasp [56], KIT [57], Google Scanned Objects [58] | 5355 물체 | §III-A |
| 메시 정리 | ManifoldPlus [59], CoACD [60] | 매니폴드화, 볼록 분해 | 저장소 `asset_process/`에 같은 도구 |
| 시뮬레이션 검증 | Isaac Gym [13] (PhysX) | 100 스텝 6방향 검증 | NVIDIA GPU 전용 |
| Q1 지표 | Ferrari & Canny [63] | 품질 지표 (Eq. 7) | |
| 벤치마크 기법 | DDG [9], GraspTTA [35] (Jiang 외) | 데이터셋 효과 검증 | 이 저장소에 학습 코드 없음 |
| 파지 분류 | Feix 외 [65] (grasp taxonomy) | 한계 논의 | |
| 기존 데이터셋 비교 | ObMan [14], HO3D [15], DexYCB [16], ContactDB [17], ContactPose [18] | 표 I | 사람 손 (MANO) 기반, 실물 또는 촬영 |

## 4. GitHub와 자료 표 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 푸시 · 커밋 | 이 분석에서 |
|---|---|---|---|---|---|
| **공식 코드** | https://github.com/PKU-EPIC/DexGraspNet | 파일 없음 (GitHub 감지 없음). README는 "이 작업과 데이터셋은 CC BY-NC 4.0" | 483 / 50 | 2025-01-06 · `bd1b13d` (main) | `repo/`에 얕은 clone, 정적 읽기만 |
| 공식 코드 분기 | `allegro`, `mano`, `mano-table`, `table` (`git ls-remote`로 확인) | 미확인 | - | `allegro` `6a48bc4`, `mano` `a4cdbc8`, `mano-table` `ffa03cc`, `table` `2880336` | 클론하지 않음 (MANO·Allegro 확인 필요 시 추가) |
| 프로젝트 페이지 | https://pku-epic.github.io/DexGraspNet/ | 페이지 하단: CC BY-NC 4.0 | - | - | 본문·영상·인용 확인. 데이터 다운로드 링크는 `mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/`(크기 미확인, 받지 않음) |
| 데이터셋 미러 | `mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/` | CC BY-NC 4.0 (페이지 표기) | - | - | 받지 않음 (데이터 다운로드 금지 조건) |
| TorchSDF (Kaolin 변형) | https://github.com/wrc042/TorchSDF | 미확인 | 미확인 | 미확인 | 저장소 README가 요구. 받지 않음 |
| pytorch_kinematics (수정본) | UM-ARM-Lab/pytorch_kinematics의 수정본을 저장소에 포함 | 미확인 (동봉 파일에 LICENSE 없음) | - | - | `repo/thirdparty/pytorch_kinematics/`에 있음 |
| CoACD, ManifoldPlus | 저장소 README가 외부 설치 요구 | 미확인 | - | - | 받지 않음 |
| Isaac Gym | NVIDIA 공식 | NVIDIA 라이선스 | - | - | 받지 않음 (Linux + NVIDIA GPU 필요) |

- 공식 코드 라이선스 불일치: 저장소 루트에 LICENSE 파일이 없습니다. "update license" 커밋(2024-05-18)은 README의 라이선스 문단을 고친 것으로 보입니다. 코드의 재사용 조건은 확정할 수 없습니다.
- `repo/data/dataset/`의 `.npy` 다섯 개는 저장소에 들어 있는 샘플입니다. pickle 형식이라 로드하지 않았습니다(임의 코드 실행 위험).
- 공식 저장소의 마지막 커밋이 2025-01-06이고 그 뒤 이슈·업데이트는 확인하지 않았습니다.

## 5. 이 분석에서 확인하지 않은 것

- DexGraspNet 2.0의 세부 내용과 저자 관계. 다른 분석 문서의 표기만 옮겼습니다.
- DDG [9]와 GraspTTA [35] 원 논문의 학습 설정. 본 논문의 재구현(§V-A)이 원 설정과 얼마나 다른지 판단할 수 없습니다.
- 참고문헌 [19]의 학회·저널. 논문 목록에는 제목("Synthesizing diverse and physically stable grasps with arbitrary hand structures...")만 있어 학회 표기는 다른 분석 문서의 표기에 기대었습니다.
