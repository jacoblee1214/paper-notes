# 02. 선행연구와 관련 저장소

## 1. 계보

PhysCoRe는 두 흐름을 잇습니다 (논문 §2.1–2.3).

**(가) 물리 기반 변형체 모델링과 재료 식별.** 스프링-질점, FEM, PBD, MPM으로 객체를 표현하고 재료 파라미터를 관측에서 맞춥니다.
- MLS-MPM [1] 과 MPM 교과서 [13]이 시뮬레이터의 뿌리입니다. APIC 전달 [60].
- 영상에서 재료를 맞추는 계열: PhysTwin [2] (ICCV 2025), PhysDreamer [22], DreamPhysics [23], Physics3D [24], PhysGM [25] (CVPR 2026), PAC-NeRF [19], gradSim [17], Virtual Elastic Objects [20], Yang 외 [21] (IJRR 2025, 탄소성 재료의 미분 가능 식별).
- 학습으로 재료를 추정하는 계열: 신경 구성 법칙 [16], EDO-Net [43], 온라인 적응 [42] AdaptiGraph, [44] 케이블 GNN.

**(나) 학습 기반 동역학.** 그래프 네트워크 [3, 4, 27, 28], 파티클 기반 시뮬레이터 [40, 41], RoboCraft [33], RoboCook [31], 확산 모델 [36].

**(다) 잔차 물리.** 실제와 시뮬레이터의 차이를 학습 보정으로 메움: 로봇 [73], 밀기 [74], 소프트 로봇 [75], 변형체 [59]. PhysCoRe의 RfD가 이 계열입니다.

**(라) 세계 모델.** 픽셀 기반 비디오 모델과 구조화된 3D 모델 [45–50]. PhysCoRe는 물리 시뮬레이터를 세계 모델의 중심에 둔다는 점에서 [2, 6, 57–59] 쪽입니다.

## 2. 가까운 대안 (표 1 기준)

| 방법 | 시뮬레이션 | 온라인 정제 | 전방 추론 | 일반화 | 코드 |
|---|---|---|---|---|---|
| PGND [5] (RSS 2025) | × | × | ✓ | × | 공개 |
| PhysTwin [2] (ICCV 2025) | ✓ | × | × | × | 공개 (MIT) |
| EMPM [6] (RA-L 2026) | ✓ | ✓ | × | × | 논문 표 기준 코드 없음 |
| **PhysCoRe** | ✓ | ✓ | ✓ | ✓ (논문 주장) | 공개 (MIT, 제3자 조항 있음) |

가장 직접 비교되는 것은 **PhysTwin**입니다. 같은 MPM 계열이고 재료를 객체별 최적화로 맞춥니다. PhysCoRe의 주장 중 "식별 시간 81배 단축"과 "CD 30–44% 감소"는 PhysTwin과의 비교입니다.

## 4. 공식 코드와 저장소

GitHub API와 Hugging Face API로 2026-11-21에 확인했습니다. 설치하거나 실행하지 않았습니다.

| 저장소 | 관계 | 라이선스 (API) | 별 / 포크 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [lunarlab-gatech/PhysCoRe](https://github.com/lunarlab-gatech/PhysCoRe) | 논문 공식 코드 (MfM, RfD, 학습, 검증, 증강) | NOASSERTION (LICENSE 파일은 MIT + 제3자 저작권 줄, `04` §5) | 17 / 1 | 2026-09-17 (커밋 `ba7ddf7`, 2026-09-13, "initial open-source release") | **얕은 복제 `repo/`** (정적 읽기) |
| [GeorgiaTech/PhysCoRe](https://huggingface.co/datasets/GeorgiaTech/PhysCoRe) | 데이터셋과 체크포인트 (실제 12개 + PhysTwin 14개) | `other` (혼합: MIT와 CC BY 4.0) | 다운로드 366 | 2026-09-13 | **받지 않음** (`data/` 7.8 GB, `04` §5) |
| [Jianghanxiao/PhysTwin](https://github.com/Jianghanxiao/PhysTwin) | 기준선 PhysTwin 공식 구현 (이 논문의 원천 데이터 포함) | MIT | 491 / 67 | 2026-08-24 | 링크만 |
| [kywind/pgnd](https://github.com/kywind/pgnd) | 기준선 PGND 공식 구현 (RSS 2025) | MIT | 95 / 9 | 2026-09-03 | 링크만 |
| 프로젝트 페이지 [lunarlab-gatech.github.io/PhysCoRe-website](https://lunarlab-gatech.github.io/PhysCoRe-website/) | 영상과 설명 | (사이트) | - | - | 본문 확인 |

- 공식 코드 저장소는 릴리스가 1회(커밋 1개)이고, 2026-09-17에 한 번 더 push되었습니다. 이후 버전 관리 기록은 없습니다.
- EMPM 코드는 논문에 "unavailable"로 적혀 있고, 확인 가능한 저장소를 찾지 못했습니다.
- OmniPhysGS (MPM 코어의 원천으로 LICENSE에 적힌 제3자)의 공식 저장소는 특정하지 못했습니다. 검색으로 MIT 포크가 나왔지만 공식인지 확인하지 못했습니다.
