# 02. 선행연구, 가까운 대안, 공식 자료

## 1. 계보

| 단계 | 연구 | V-JEPA 2와의 관계 |
|---|---|---|
| 개념 | LeCun 2022 (JEPA 제안) | 잠재 공간 예측의 원리. V-JEPA 2가 이를 실행 |
| 이미지 | I-JEPA (Assran 2023) | 마스크 잠재 예측의 원형. 동일 목적을 이미지에 적용 |
| 영상 | V-JEPA (Bardes 2024) | 직접 선행. 데이터 2M, ViT-H, 16프레임. V-JEPA 2는 데이터 22M, 1B, 연속 학습 |
| 영상 생성형 세계모델 | Cosmos (Agarwal 2025), Ctrl-World, IRASim | 표 2–3의 대조군(Cosmos). 픽셀 또는 생성 잠재 공간 |
| 로봇 VLA | Octo (Octo Model Team 2024) | 표 2의 목표 이미지 BC 대조군 |
| 세계모델 계획 | TD-MPC2, DreamerV3 | 보상 기반 잠재 세계모델. 계획 방식이 다름(§3) |

논문의 새 기여는 (a) 스케일링 레시피(§2.1 4가지 요소), (b) 프레임 인과 블록 예측기로 62시간 로봇 데이터에서 이미지 목표 계획을 한 것입니다. 목적 함수(마스크 잠재 L1 예측)와 CEM은 기존 것입니다(논문 §3.2가 CEM을 Rubinstein 1997로 인용).

## 2. 이 저장소에서 V-JEPA 2를 쓰거나 비교하는 분석

| 폴더 | 논문 | V-JEPA 2와의 관계 |
|---|---|---|


## 3. 계획 방식 비교 (V-JEPA 2-AC 대 가까운 대안)

| 항목 | V-JEPA 2-AC | Cosmos (표 3 대조) | TD-MPC2 |
|---|---|---|---|
| 세계모델 공간 | 고정 V-JEPA 2 특징 (16×16×1408) | 확산 영상 생성 (잠재 확산) | 잠재 벡터 |
| 계획 목적 | 목표 이미지 특징과의 L1 | 목표 이미지 잠재 L1 (같은 형식) | 예측 보상 + 가치 |
| 시간당 행동 계산 | 16초 (RTX 4090) | 4분 (RTX 4090) | — |
| 보상 | 없음 | 없음 | 있음 |

## 4. 공식 자료와 저장소 (GitHub 및 공식 페이지 표)

조사일 2026-11-21. GitHub API로 얻은 값입니다.

| 자료 | 주소 | 라이선스 | 별 | 최근 push | 비고 |
|---|---|---|---|---|---|
| 공식 코드 (V-JEPA 2, V-JEPA 2-AC, V-JEPA 2.1) | https://github.com/facebookresearch/vjepa2 | MIT (Apache 2.0 일부 파일) | 4,909 | 2026-10-29 | 생성 2025-04-25. 클론 HEAD `204698b` (2026-03-23, "Fix figure (#143)"). `vjepa2/repo/`에 얕은 클론(정적 읽기만) |
| 프로젝트 페이지 | https://ai.meta.com/vjepa/ | — | — | — | 접속 확인(200). 학회 표기 없음 |
| 블로그 | https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks | — | — | — | 논문 README에 링크. 접속 확인(200) |
| 체크포인트 모음 | https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6 | 저장소 기준 | — | — | HF 컬렉션 (공식 README 링크). 이번에는 열지 않음 |
| V-JEPA 2-AC ViT-g 체크포인트 | https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt | 저장소 README에 별도 명시 없음 | — | — | HTTP 헤더 크기 약 11.8 GB (내려받지 않음) |
| 인코더 ViT-g384 체크포인트 | https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt | 같음 | — | — | 헤더 크기 약 16.5 GB (내려받지 않음) |
| 인코더 ViT-L 체크포인트 | https://dl.fbaipublicfiles.com/vjepa2/vitl.pt | 같음 | — | — | 헤더 크기 약 5.1 GB (내려받지 않음). 추론용 가중치만 쓰면 더 작을 수 있음 |
| HF Transformers 구현 | `transformers/models/vjepa2/modeling_vjepa2.py` (main, 2026-11-21 조회) | Apache 2.0 | — | — | 어텐션 마스크 확인용 (`04` §5). 이 저장소 버전과 고정 버전은 다를 수 있음 |
| V-JEPA 2 arXiv 페이지 | https://arxiv.org/abs/2506.09985 | CC BY 4.0 | — | — | abs 페이지의 라이선스 링크 확인 (`01`과 README 참조) |

- 재현용 학습 데이터 DROID와 Octo, Cosmos는 이 분석에서 받지 않았습니다.
- 공식 저장소에는 **계획 루프 코드가 있지만 로봇 인터페이스가 없습니다.** `notebooks/utils/mpc_utils.py`(CEM)와 `world_model_wrapper.py`, 그리고 에너지 지형 예제 노트북 `energy_landscape_example.ipynb`와 궤적 파일 `franka_example_traj.npz`가 전부입니다. 카메라 입력, 목표 이미지 획득, OSC 제어 코드는 없습니다(`04` D8).

## 5. 버전과 관련 릴리스

- arXiv: v1(2025-06-11)만 있습니다(2026-11-21 기준).
- 저장소: V-JEPA 2.1(2026-03-16 공지)이 추가되었습니다(`configs/train_2_1`, `app/vjepa_2_1`). 이 분석은 V-JEPA 2를 다루고, 2.1은 범위 밖입니다. 2.1 학습 코드는 `is_causal`과 `pred_is_causal`을 기본값 False로 읽습니다(`app/vjepa_2_1/train.py` 103–104행). 2.1의 인코더 어텐션 구조는 이 분석에서 더 보지 않았습니다.
- 공식 저장소 README는 V-JEPA 2-AC를 V-JEPA 2와 함께 소개합니다.

## 6. 학회 표기

- arXiv 댓글: "48 pages, 19 figures"만 있고 학회 표기는 없습니다.
- 공식 저장소 인용: `@article{assran2025vjepa2 ... journal={arXiv preprint arXiv:2506.09985}`.
- 논문 본문, 프로젝트 페이지, 블로그에 학회 표기가 없고, 저장소 `conferences/`에도 없습니다.
- 따라서 **학회 표기 없음 (arXiv)**입니다.
