# 02. 선행연구와 자료 계보: Dexonomy

## 1. 선행 계보 (논문 II절 기준)

| 계열 | 대표 연구 (논문 참고문헌) | 논문이 지적하는 한계 | Dexonomy와의 관계 |
|---|---|---|---|
| 해석적 파지 합성 (힘 폐쇄 지표) | Dex-Net [32], Dex-Net 3.0 [33], GraspIt! [35], Ferrari–Canny [21], 품질 지표 리뷰 [40] | 유형을 지정하기 어렵고 손이 힘 폐쇄에 치우침 | 지표 정의는 III절에서 [51]의 변형을 씀 |
| 미분 기반 최적화 | DexGraspNet [47], FRoGGeR [25], SpringGrasp [8], Grasp'D [43], Fast-Grasp'D [44], BODex [7] | 시작 자세가 쉬는 자세라 반복이 많고 지역 최소, 비자연 자세 | 기준선으로 씀 (Table II, III). 시작을 템플릿으로 바꿈 |
| 기능 파지 전이 | OakInk [55], LHFG [48], CCFG [52], AffordPose [23], 의미 표현 [61] | 형상이 비슷한 물체나 축 정렬 물체에만 됨 | 코드가 없어서 그림 수준으로만 비교 (Fig. 6) |
| 학습 기반 파지 생성 | UniDexGrasp [53], DexDiffuser [50], DexGraspNet 2.0 [60], BODex 학습 모델 [7] | 데이터 품질에 의존 | 데이터 공급원. 학습 모델은 BODex 구조 + 유형 코드북 (VI절) |
| 데이터 수집 | 원격 조작 [38, 11, 56], 사람 손 전이 [3, 37, 30, 5, 55, 10], 강화학습 [59, 46, 13] | 사람 노력, 형태 차이, 학습 비용 | 사람 주석은 유형당 템플릿 하나 |
| 시뮬레이션 기반 정제 | 대부분 맞춤 에너지 함수 [24, 47, 8, 7] | 센티미터 단위 관통, 튜닝 필요 | MuJoCo 전치 야코비안 정제 (IV-C) |

**논문이 주장하는 위치.** 유형을 **지정할 수 있으면서** 물체에 제한이 없고 템플릿 하나만 필요한 방법입니다 (II-B 끝). 이 위치를 정량으로 보이는 기준선은 BODex뿐이고, 나머지는 해석적 방법입니다. 기능 파지 전이는 정성 비교만 있습니다.

## 2. 구성 요소별 출처

논문이 쓰는 외부 도구와 자료입니다. 라이선스는 이 분석에서 확인하지 않은 항목이 많아서 "미확인"으로 적었습니다.

| 구성 요소 | 출처 | 논문 내 역할 | 확인 상태 |
|---|---|---|---|
| 흐름 모델 (Möbius 정규화 흐름) | Liu et al., CVPR 2023 [28] | 유형 조건부 파지 생성 (VI절) | 논문 참고만 |
| 희소 3D 합성곱 | MinkowskiEngine [12] | 점군 인코더 | 미확인 (CUDA 빌드 필요, 논문 VI절) |
| 최근접점 미분 가능 질의 | NVIDIA Warp [31] | 전역 정렬의 최근접점과 손실 (IV-B) | 논문 참고만 |
| 물리 엔진 | MuJoCo [42] | 지역 정제, 검증 (IV-C, D) | 공식 코드가 MjSpec API를 씀 (MuJoCo 3.2 이상 필요, 코드 확인) |
| 배치 QP 풀이 | ReLU-QP [2], BODex 방식 | 힘 QP 배치 처리 (부록 F) | 논문 참고만 |
| 볼록 분해 | CoACD [49] | 물체 전처리 (부록 B) | 논문 참고만 |
| 내부 면 제거 | OpenVDB [36] | 물체 전처리 (부록 B) | 논문 참고만 |
| 메시 간소화 | ACVD [45] | 물체 전처리 (부록 B) | 논문 참고만 |
| 손 자세 궤적 계획 | cuRobo [41] | 실물 실험 (VII-F) | 논문 참고만 |
| 점군 분할 | SAM2 [39] | 실물 실험 (VII-F) | 논문 참고만 |
| 유형 선택 (실물) | GPT-4o [22] | 물체당 유형 3개 선택 (VII-F) | 논문 참고만 |
| 물체 자산 | DexGraspNet 5,697 [47], Objaverse 5,000 [18] | 데이터와 평가 | DGN은 라이선스 확인 필요 |
| 파지 분류 | GRASP 분류 [20] (33 유형) | 유형 정의 | 논문 참고만 |
| 손 모델 | MuJoCo Menagerie [58] (Shadow, Allegro 등) | MJCF 메시 | 공식 저장소에서 submodule, 이 분석에서 받지 않음 |

## 4. GitHub, 데이터, 프로젝트 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 포크 | 마지막 푸시 · 커밋 | 비고 |
|---|---|---|---|---|---|
| **공식 코드** | https://github.com/JYChen18/Dexonomy | 파일 없음 (GitHub API: null). **README는 "이 작업과 데이터셋은 CC BY-NC 4.0"** | 170 / 20 | 2026-03-26 · `d561ff9` ("move BODex lfs to hf") | 얕은 복제 `repo/`, 정적 읽기만. 실행과 설치 안 함 |
| 학습 코드 DexLearn | https://github.com/JYChen18/DexLearn | API 라이선스 필드 없음 | 51 / 18 | 2026-10-19 | 복제하지 않음. README에 KPGrasp 후속 소개 |
| 공식 웹 저장소 (추정) | https://github.com/JYChen18/DexonomyWeb | 미확인 | 0 / - | 2025-05-16 | 이름으로 보아 웹 소스. 읽지 않음 |
| 제3자 저장소 | https://github.com/EthanMaTam/Dexonomy_gen | 미확인 | 0 / - | 2026-11-10 | 공식이 아님. 읽지 않음 |
| 프로젝트 페이지 | https://pku-epic.github.io/Dexonomy | 명시 없음 | - | - | 확인함. 영상 `videos/rss2025.mp4`, 인용 정보는 "Robotics: Science and Systems 2025" |
| 데이터셋 (Hugging Face) | https://huggingface.co/datasets/JiayiChenPKU/Dexonomy | 카드: **cc-by-nc-4.0** | - | 2025-10-21 | 파일 9개 합계 약 **15.9 GB** (아래 표). 받지 않음 |
| 공식 학습 가중치 (HF) | `dexonomy_learning_ckpt.tar.gz` | cc-by-nc-4.0 (카드) | - | - | 0.095 MB. 받지 않음 |
| BODex 코드 (submodule) | https://github.com/JYChen18/BODex | 미확인 | - | - | 저장소의 submodule. 받지 않음 (궤적 계획 선택 사항) |
| 손 모델 | https://github.com/google-deepmind/mujoco_menagerie | 미확인 (보통 Apache 2.0, 확인 필요) | - | - | submodule. 저장소의 `right.xml`이 `third_party/mujoco_menagerie/...`의 메시를 가리킴 (복제 없음이라 실행 불가) |
| 공개 학회 목록 | https://roboticsproceedings.org/rss21/p105.html | 공식 | - | - | RSS XXI (2025), 논문 p105 확인 |

**데이터셋 파일 크기 (HF 트리, 메타데이터만):**

| 파일 | 크기 |
|---|---|
| `Dexonomy_GRASP_shadow.tar.gz` (Shadow 파지) | 2.907 GB |
| `DGN_5k_processed.tar.gz` | 1.555 GB |
| `DGN_5k_vision.tar.gz` | 2.332 GB |
| `objaverse_5k_processed.tar.gz` | 6.547 GB |
| `objaverse_5k_vision.tar.gz` | 1.863 GB |
| `allegro_fingertip.tar.gz` | 0.605 GB |
| `dexonomy_learning_ckpt.tar.gz` | 0.095 MB |
| 합계 | 약 15.9 GB |

**공식 저장소의 커밋과 구성.** 얕은 복제 커밋은 `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`입니다. `repo/`는 약 16 MB이고, `.git`은 5.9 MB입니다. 파이썬 파일 30개 (`dexonomy/`), 주석 YAML (`assets/hand/*/raw_anno`) 36개, 설정 YAML 20여 개가 있습니다. 튜토리얼 세 편(`getting_started/`)이 있습니다.

**후속 연구.** 저장소의 `dexonomy/op/gen_grasp_admm.py`와 `getting_started/3_ADMM.md`는 GraspADMM (arXiv:2603.13832, "GraspADMM: Improving Dexterous Grasp Synthesis via ADMM Optimization")의 구현입니다. 이 분석은 논문에 있는 파이프라인과 대조했고 GraspADMM은 범위 밖으로 두었습니다. 다만 공식 저장소의 주된 변경 중 하나이므로 `04` 절에 확인 사항으로 적었습니다.
