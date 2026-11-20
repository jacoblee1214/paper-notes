# 02. 선행연구와 코드 위치

## 1. 계보: 어디에서 무엇을 가져왔나

WorldArena는 새 지표를 만들기보다 기존 지표를 모으고 조정했다. 가져온 출처는 논문 본문과 부록 A에 적혀 있다.

| 구성 요소 | 출처 (논문 인용) | 이 논문의 변형 |
|---|---|---|
| 영상 지표 묶음 (Subject, Background, Dynamic, Image, Aesthetic) | VBench (Huang et al., 2024, CVPR) | 동적 게이트 `min(1, Sdyn/γ)`를 Subject·Background·Photometric에 추가 (A.7–A.9) |
| 정적 영상 지름길 문제 | 논문 자체 주장 (A.7) | VBench 계열에는 없는 보정 |
| JEPA Similarity (MMD, 다항 커널) | Luo et al., 2024 "Beyond FVD" (arXiv:2410.05203). 특징 추출은 V-JEPA (Bardes et al., 2023) | α = 40 스케일 (식 3) |
| Photometric (AEPE 역수) | WorldScore (Duan et al., 2025) | 동적 게이트 적용 (식 13) |
| Motion Smoothness (VFI, SSIM·ln(1+diff) 가중) | VFI-Mamba (Zhang et al., 2024) 와 WorldScore | 가중치 ln(1+diff) 추가 (식 7) |
| Trajectory Accuracy (NDTW) | EWMBench (Yue et al., 2025) | 궤적을 SAM3 팔 박스 중심으로 얻음 |
| Action Following (CLIP 특징 1−cos) | EWMBench (Yue et al., 2025) | 지시 3개에 대한 평균 |
| Depth Accuracy (Depth-Anything, median 정렬, AbsRel) | Depth-Anything v2 (Yang et al., 2024a), Liang et al., 2025 (WorldLens) | |
| Interaction, Perspectivity, Instruction Following | Qwen3-VL-8B (Bai et al., 2025a) 를 판정기로 사용 | 1–5 리커트 척도를 [0,1]로 정규화 (A.10) |
| Semantic Alignment | Qwen2.5-VL 설명문 + CLIP 텍스트 인코더 | |
| 임베디드 과제 (데이터 엔진) | VPP (Hu et al., 2024) 의 확산 정책 헤드 | 세계 모델 특징을 조건으로 사용 |
| 행동 계획 (IDM) | VPP 의 IDM 설계 | |
| 정책 학습 | π0.5 (Intelligence et al., 2025) | RoboTwin 2.0 데이터 |
| 평가 환경 | RoboTwin 2.0 (Chen et al., 2025) | 50개 과제, Clean-50 |

**가장 가까운 대안**: WoW-World-Eval (Fan et al., 2026, arXiv:2601.04137) 은 Table 1에서 WorldArena와 가장 비슷하다. 임베디드 평가 항목을 다섯 개 차원으로 나눴다. 두 벤치마크가 같은 시기에 나왔으므로, 어느 쪽이 먼저인지와 세계 모델 목록의 겹침은 논문에서 다루지 않는다.

**다른 임베디드 벤치마크**: EWMBench (Yue et al., 2025), WorldEval (Li et al., 2025b, 정책 평가), World-in-World (Zhang et al., 2025, 폐루프 평가). 본문은 이들이 "한 가지 임베디드 역할에 집중"하거나 "텍스트 조건 모델에 치우쳤다"고 한다 (§2.2).

**일반 세계 모델 벤치마크**: WorldModelBench (Li et al., 2025a), WorldSimBench (Qin et al., 2024), WorldScore (Duan et al., 2025), 4DWorldBench (Lu et al., 2025).

**모델 출처**: CogVideoX (Yang et al., 2024b), Wan 2.2 과 Wan 2.6 (Wan et al., 2025), Veo 3.1 (Google, 웹 접근), Genie Envisioner (Liao et al., 2025), GigaWorld-0 (Team et al., 2025), TesserAct (Zhen et al., 2025), Cosmos-Predict 2.5 (Gu, 2025), WoW (Chi et al., 2025), RoboMaster (2025, 웹 공개 데이터), Vidar (Feng et al., 2025), IRASim (Zhu et al., 2024a), CtrlWorld (Guo et al., 2025, arXiv:2510.10125).

**후속 연구**: WorldArena 2.0 (arXiv:2605.17912, 2026-05). 프로젝트 페이지는 2.0을 소개한다. 시각-촉각 평가, 온라인 RL 환경, RoboTwin 2.0과 LIBERO, AgileX 실로봇을 추가했다. 이 폴더의 분석은 1.0 (arXiv:2602.08971 v2) 에만 해당한다.

## 2. 이 논문이 기대는 가정

- **RoboTwin 2.0이 임베디드 세계 모델을 평가할 만큼 현실적이다.** 시뮬레이터 기반 성공률이 실제 로봇을 대신한다고 본다. 정책 평가자 검사(Figure 4)가 이 가정의 근거다. 그러나 시뮬레이터 성공률 자체의 신뢰 구간은 없다.
- **VLM 판정기가 사람 판단을 대신한다.** 사람 평가와의 상관 r = 0.825 (14개 모델)로 뒷받침한다. 판정기 간 일치도는 보고되지 않았다.
- **EWMScore의 동일 가중치 평균이 의미 있다.** 16개 지표를 같은 비중으로 더한다. 가중치 민감도는 분석하지 않았다.

## 3. 코드와 프로젝트 위치

| 구분 | 위치 | 상태 (2026-11-20 확인) | 라이선스 | 비고 |
|---|---|---|---|---|
| **1.0 공식 코드** | https://github.com/tsinghua-fib-lab/WorldArena | 스타 274, 포크 14, 생성 2026-02-11, 마지막 push 2026-07-22 | 저장소 루트에 LICENSE 없음 (GitHub API `license: null`). 하위 `embodied_task/LICENSE`, `worldarena_track2/LICENSE`, `third_party` 개별 라이선스만 있음 | **shallow clone 완료: `repo/`, 커밋 `2da2ae253b` (2026-07-22). 정적 읽기만 했고 실행하지 않음** |
| 2.0 공식 코드 | https://github.com/WorldArena2/WorldArena-2.0 | 스타 40, 포크 5, 생성 2026-05-05, 마지막 push 2026-09-26, 커밋 `5978ce5` ("Initial release") | MIT | 2.0 범위라 클론하지 않음 |
| 1.0 프로젝트 페이지 | https://world-arena.ai/v1/ | CC BY-SA 4.0 (사이트 라이선스) | 논문 라이선스(CC BY 4.0)와 별개 | 리더보드, 제출, 사람 평가 링크 |
| 2.0 프로젝트 페이지 | https://world-arena.ai/ | 같은 사이트 | | 2.0 표 값 (예: Data Engine 1/35 등) |
| 데이터셋 (1.0) | https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0 | 미다운로드 | | 정책 평가 GT 패키지 (영상 500개) 포함 |
| 리더보드 | https://huggingface.co/spaces/WorldArena/WorldArena | 미접근 | | |
| 사람 평가 | 프로젝트 페이지의 Human Evaluation 링크 | 미접근 | | 원자료는 공개되지 않음 |

**클론한 코드에서 확인한 것** (`04` 참조): 정규화 경계(`video_quality/WorldArena/__init__.py`, Table 6과 값 일치), 동적 게이트 상수 0.1213 (세 파일 공통), 정책 평가 상관 스크립트 (`calculate_policy_pearson_r.py`, 기본 시뮬레이터 점수 5개), VLM 판정 JSON 키. EWMScore 계산 코드는 저장소에 없다. 집계 스크립트 `aggregate_results.py`는 CSV만 만든다.

**인용과 관련해 주의할 점**: 1.0 저장소의 README 인용 블록은 `journal={arXiv preprint arXiv:2602.08971}`이다. 학회 표기는 없다.


## 5. 한계

- 선행연구 계보는 논문의 인용과 README로 확인했다. 원논문 전체를 대조하지 않았다.
- WorldArena 2.0 본문은 읽지 않았다. 프로젝트 페이지의 요약 수치만 확인했다.
