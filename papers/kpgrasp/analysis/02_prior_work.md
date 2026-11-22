# 02. 선행연구와 관련 자료

## 0. 요약

KPGrasp는 **Dexonomy 데이터 계열 (PKU, He Wang 연구실)**의 학습 쪽 후속입니다. 데이터는 Dexonomy와 DexGraspNet (DGN 객체)에서 왔고, 평가는 Dexonomy 지표와 DexGrasp Anything (DGA) 프로토콜을 씁니다. 모델은 DiT (흐름 정합)와 MinkUNet (희소 합성곱)을 조합합니다. 새로운 것은 **출력 공간을 키포인트로 바꾼 것**과 **그 위에서 표준 흐름 정합 손실을 쓴 것**입니다.

## 1. 계보

| 연도 | 연구 | 역할 | 관계 |
|---|---|---|---|
| 2022 | DexGraspNet [10] (arXiv:2210.02697; ICRA 2023) | 시뮬레이션 대규모 파지 데이터, DGN 객체 | KPGrasp 실로봇 학습의 DGN 5k 객체, Dexonomy 객체 집합의 한쪽 |
| 2023 | UniDexGrasp [19] | 제안 생성과 목표 조건 정책, 학습 기반 기준선 | 표 1, 표 2의 기준선 |
| 2021–2025 | GraspTTA [15] (ICCV 2021), DexGraspAnything [18] (CVPR 2025) | 접촉 손실과 테스트 시점 정제 | KPGrasp가 피하려는 방식의 대표 |
| 2024 | UGG [21] (ECCV 2024) | 통합 생성 파지 | 표 2 기준선 |
| 2025 | Dexonomy [8] (arXiv:2504.18829) | 파지 분류 체계 전체 (31 유형), 9.5M 파지 분석 파이프라인 | 데이터와 평가 지표의 출처. 분류기 있는 변형도 표 1 기준선 |
| 2025 | FlowMap / LSD, Boffi 등 [54] (arXiv:2505.18825) | 1단계 흐름 지도 증류 | 부록 D의 1단계 기준선 |
| 2026 (6월) | KPGrasp v1 (arXiv:2606.09314) | 이 논문 | CoRL 2026 스포트라이트 |

## 2. 구성 요소별 출처

| 구성 | 논문 위치 | 출처 | 라이선스 (저장소 NOTICE.md 기재) |
|---|---|---|---|
| DiT 블록 (자기·교차 주의, AdaLN) | §4.2 [44] | Peebles & Xie, DiT (Meta) | CC BY-NC 4.0 (`licenses/DiT-CC-BY-NC-4.0.txt`) |
| 시간 임베딩 | §4.2 | DiT, GLIDE (OpenAI) | DiT 부분 CC BY-NC, GLIDE MIT |
| 희소 MinkUNet 물체 인코더 | §4.2 [42] | MinkowskiEngine (NVIDIA, Choy) | MIT |
| FPS 128 토큰 | §4.2 [43] | PointNet++ (Qi 등, 2017). 저장소 구현은 Pointnet_Pointnet2_pytorch 기반 | MIT (`licenses/PointNet2-MIT.txt`) |
| Muon 최적화 | §4.2 [45] | Liu 등, "Muon is scalable for LLM training" (2025). 저장소 코드는 Keller Jordan 구현을 적음 | MIT (`licenses/Muon-MIT.txt`) |
| 흐름 정합 (선형 경로, 속도 MSE) | §3, 식 (1) [37] | Lipman 등, Flow Matching (arXiv:2210.02747) | 저장소는 `flow_matching` 패키지 사용 (라이선스 미확인) |
| 가능도 계산 (Hutchinson) | §3, 식 (2) [38, 39, 37] | CNF 관례 | 논문 설명 |
| SVD 강체 정합 | §4.1 [40] | Umeyama (1991) | 논문 설명 |
| 손 IK (Mink, quadprog) | §4.1 [41], 부록 A | Zakka, Mink (MuJoCo 기반 IK, github.com/kevinzakka/mink) | 저장소의 `kinematics/ik.py`는 DexGraspBench 구현에서 가져옴 (NOTICE 기재) |
| Shadow Hand 모델 | 저장소 `assets/hand/shadow/` | MuJoCo Menagerie, Shadow Robot | Apache 2.0 |
| Dexonomy 평가 (GSR, OSR, CDC, PD, 다양성) | §5.1 [8] | Dexonomy, MuJoCo [46] | 평가 코드는 KPGrasp 저장소에 없음 (`04` D9) |
| DGA 프로토콜 (IsaacGym 평가) | §5.2 [18], [49] | DexGrasp Anything | 평가 코드는 저장소에 없음 (`04` D10) |
| 실로봇 객체 분할 | §5.4 [53] | GroundedSAM | 논문 설명 |

## 4. 공식 코드와 자료 (GitHub, 데이터, 체크포인트)

| 저장소 / 자료 | 관계 | 라이선스 | 상태 (2026-11-22 확인) | 대조 기준 |
|---|---|---|---|---|
| [Yuansen-Huang/KPGrasp](https://github.com/Yuansen-Huang/KPGrasp) | 공식 구현 (논문 저자) | LICENSE 파일 **CC BY-NC 4.0** (GitHub API는 NOASSERTION로 표시) | 별 7, 포크 0. 생성 2026-09-22, 마지막 push 2026-10-18 | 얕은 복제 `89716dd` (2026-10-18 "Update Final_base.yaml"). 전체 이력 미확인. `repo/` (gitignore) |
| [KPGrasp 프로젝트 페이지](https://yuansen-huang.github.io/KPGrasp-page/) | 논문 소개, 3D 파지 시각화, 영상 | 명시 없음 | "CoRL 2026 (Spotlight)" 표기 | 페이지 HTML 확인 |
| [JYChen18/Dexonomy](https://github.com/JYChen18/Dexonomy) | 데이터 생성 파이프라인과 데이터 원출처 | 라이선스 필드 없음 (SPDX null) | 별 170, 마지막 push 2026-03-26 | 이 분석에서 복제하지 않음 |
| [JiayiChenPKU/Dexonomy](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy) (데이터셋) | 학습 데이터 원자료 | 라이선스 필드 없음, 비공개 아님 | 파일 9개, 합계 약 14.8 GiB (파일 목록 기준: 객체 처리본 6.1 GiB, 파지 `Dexonomy_GRASP_shadow` 2.7 GiB 등) | 메타데이터만 확인. 내려받지 않음 |
| 체크포인트 `Final_base` | 표 1의 76.3%에 해당 | 저장소 기준 CC BY-NC 4.0 | **공개 URL 미지정** (`docs/CHECKPOINTS.md`), 256.5 MiB, SHA-256 기록됨 | 받지 않음 |
| DexGraspBench (JYChen18) | 물리 평가와 IK의 원본 구현 | 미확인 | 저장소 README가 참조. 이 분석에서 확인하지 않음 | 확인 안 함 |

- 저장소 README는 "76.3% 성공은 역사적 벤치마크 결과"라고 적고, 릴리스 준비 중 전체 벤치마크를 돌리지 않았다고 씁니다 (`docs/VALIDATION.md`). 논문 표 1을 이 저장소만으로 다시 얻을 수는 없습니다.
- NOTICE.md가 `docs/EXPERIMENTS.md`를 가리키지만 저장소에 그 파일이 없습니다 (`04` D8).

## 5. 가까운 대안 (표 1, 2 기준)

| 방법 | 출력 | 추론 | 주된 특징 | 논문 수치 |
|---|---|---|---|---|
| Dexonomy 분류기 있음 [8] | 손목 + 관절, 분류기 | 생성 후 선택 2단계 | 추가 학습 데이터 필요 | GSR 63.9% |
| UniDexGrasp [19] | 제안 + 목표 정책 | 정책 | 두 단계 | GSR 24.5% |
| DexGrasp Anything [18] | 손목 + 관절 | 확산 + 물리 안내 (2.8 s) | 접촉 손실과 정제 | GSR 21.8%, DGA 평균 58.5% |
| UGG [21] | 통합 생성 | 생성 | 판별기 | DGA 평균 44.7% |
| **KPGrasp** | 키포인트 (J = 21 또는 28, 논문과 저장소 불일치) | 흐름 5단계 + 가능도 순위 | 접촉 손실 없음 | GSR 76.3%, DGA 평균 63.8% |

## 6. 이 분석에서 알 수 있는 것과 없는 것

- **알 수 있는 것:** 데이터 계보, 구성 요소의 출처, 라이선스 경계, 공식 코드가 무엇을 포함하고 무엇을 빼는지.
- **알 수 없는 것:** 기준선 (UniDexGrasp, DGA) 재학습 세부, Dexonomy 평가기의 정확한 구현, 시드별 분산. 논문이 이 정보를 주지 않았고 저장소에도 없습니다.
