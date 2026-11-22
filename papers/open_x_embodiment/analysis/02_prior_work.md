# 선행연구와 관련 자료 (Open X-Embodiment)

## 1. 계보

- **단일 로봇 데이터셋 시대:** 파지, 밀기, 원격조작 시연 같은 공개 데이터는 대부분 같은 종류의 로봇 데이터입니다. 논문은 RoboNet(2019)만 여러 로봇 데이터라는 예외로 둡니다(II, 두 번째 단락).
- **임베디먼트 전이 방법:** 시뮬레이션 전이(Devin 2017, Chen 2018, Huang 2020 등)와 실로봇 전이(Hardware-conditioned, ViNT, GNM, RoboCat, Polybot 등)가 있습니다. 대부분 임베디먼트 차이를 줄이는 구조를 넣습니다. 이 논문은 그런 장치 없이 데이터를 그냥 섞는 것을 시험합니다(II, 첫 단락).
- **인간 영상에서 배우기:** 임베디먼트 차이를 번역하거나 표현을 맞추는 방법(II, 세 번째 단락). 논문과 직접 경쟁하지는 않습니다.
- **언어 조건 정책과 VLA 흐름:** RT-1(2022), RT-2(2023), BC-Z, Bridge, CLIP류 VLM. RT-2를 PaLI-X로 구현한 것이 RT-2-PaLI-X이고, 이 논문의 RT-2-X입니다(IV-B).

## 2. 가까운 대안 (같은 문제를 다루는 다른 방법)

| 연구 | 연도·학회 | 무엇이 다른가 | 이 논문과의 관계 |
|---|---|---|---|
| RoboNet | 2019 CoRL | 여러 로봇 데이터 (논문이 말하는 유일한 예외) | 임베디먼트 통합 데이터의 선례 |
| RoboCat | 2023 arXiv | 자기 개선 파운데이션 에이전트 (논문 제목) | 같은 시기 X-embodiment 학습 (논문 II) |
| Polybot | 2023 arXiv | 여러 로봇에 정책 하나, 로봇 차이를 받아들이는 설계 | 같은 방향의 실로봇 연구 |
| Octo | 2024 RSS | 27M, 93M 확산 정책, OXE 25개 데이터셋 혼합 | 가장 가까운 후속 작업. 기준선으로 RT-1-X 사용 |
| OpenVLA | 2024 CoRL | 7B VLA, OXE 큐레이션 970k 에피소드 | OXE를 주 학습 데이터로 씀 |
| RT-2 | 2023 CoRL | 같은 RT-2 계열의 단일 로봇 판 | RT-2-X의 원형. 창발 평가 프로토콜 공유 |

## 4. 공식 코드와 자료 (확인일 2026-11-22)

| 항목 | 링크 | 라이선스 | 스타 | 최근 push | 메모 |
|---|---|---|---|---|---|
| 공식 코드 (모델, 콜랩) | github.com/google-deepmind/open_x_embodiment | Apache-2.0 (코드), CC BY 4.0 (자료), README 기준 | 2,101 | 2025-11-05 | 얕은 복제 `repo/` (커밋 `9eeb68b`, 2025-11-05 15:45 -0800), 45 MB. 정적 읽기만 함 |
| 프로젝트 페이지 | robotics-transformer-x.github.io | — | — | — | 학회 표기 없음. "RT-1-X ... by 50% in the small-data domain", "RT-2-X ... by 3x" 문구 |
| 데이터셋 스프레드시트 | Google Sheet (gid=0) | — | — | — | 메타데이터 CSV 스냅샷 `analysis/meta/` (72행, 1,059행 파일). 개별 데이터셋 라이선스 열은 없음 |
| RT-1-X JAX 체크포인트 | gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax | 미확인 | — | — | 다운로드하지 않음. 크기 미확인 |
| RT-2-X | 공개되지 않음 | — | — | — | 가중치와 코드 없음 |
| RLDS 형식 | github.com/google-research/rlds | Apache-2.0 | 492 | 2024-09-26 | 보관(archived) 상태 |

- 공식 저장소에는 **모델 정의(`models/`: rt1.py, efficientnet.py, film_conditioning.py, token_learner.py, 추론 예제)와 콜랩 3개**만 있습니다. 학습 루프, 혼합 설정, 평가 코드는 없습니다.
- 콜랩 `Open_X_Embodiment_Datasets.ipynb`의 DATASETS 목록은 53개 이름입니다. 논문과 스프레드시트의 60개와 다릅니다. 콜랩이 오래되었거나 일부만 담은 목록으로 보입니다.
- 데이터셋 재배포는 개별 원 데이터셋 라이선스를 따릅니다. 스프레드시트에는 라이선스 열이 없습니다. 이 분석은 재배포하지 않습니다.

## 5. 이 논문의 기여 위치

- **데이터 기여:** 대규모 공개 데이터의 "표준 형식 통합"입니다. 뒤에 나온 대규모 공개 데이터의 출발점입니다. 예를 들어 DROID는 스프레드시트에서 v1.1로 추가되었습니다.
- **모델 기여:** 새로운 구조가 아니라 기존 RT-1과 RT-2를 같은 혼합으로 학습한 것입니다. 저자도 이렇게 씁니다("Our aim is not to innovate in terms of the particular architectures", I).
- **실험 기여:** 긍정 전이의 방향을 실로봇 평가 여러 개로 보였습니다. 셀 크기와 시드 정보는 없습니다.
