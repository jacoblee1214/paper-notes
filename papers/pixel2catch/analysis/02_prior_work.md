# Pixel2Catch 선행연구와 공식 저장소

분석일 2026-11-17. 논문 인용 번호 [n]은 arXiv v1의 참고문헌 번호입니다(`01_paper_analysis.md`와 같은 기준).

## 1. 선행연구 계보

논문 §II는 세 갈래로 관련 연구를 나눕니다. 각 갈래에서 이 논문이 무엇을 바꿨는지를 중심으로 정리했습니다.

**(가) 물체 운동을 3D로 추정하는 던지기·잡기**

- RGB-D 카메라로 3D 좌표를 얻는 방법: Huang et al. [7] (Dynamic handover, bimanual, CoRL 2023), Zhang et al. [6] (Catch it!, mobile dexterous hand, arXiv:2409.10319). 저자들의 가장 가까운 비교 대상은 [6]입니다. [6]은 두 단계 RL을 쓰고, 이 논문은 한 단계 MARL을 씁니다(§II-B).
- 모션캡처로 3D를 얻는 방법: Chen et al. [2] (CushionCatch), Tassi et al. [4] (IMA-Catcher, IJRR 2025).
- 이 논문의 차이는 3D를 얻지 않고 화면 위 바운딩 박스의 중심, 너비, 높이와 그 시간 차분을 쓰는 것입니다(§II-A, §IV-B).

**(나) 손 달린 잡기와 다중 에이전트**

- DexCatch [5] (arXiv:2310.08809): 임의 물체를 손으로 잡는 학습. 이 논문은 같은 방향의 손 잡기를 다루지만 비교하지 않습니다.
- Kim et al. [9] (arXiv:2502.11437): 양손 잡기를 위한 **적대-협력형 이질 에이전트 RL**. 이 논문의 이질적 MARL과 가장 가까운 선행연구인데, 이 논문은 인용만 하고 비교하지 않습니다. 이 점은 §5에서 다시 봅니다.
- 다중 로봇 MARL에 대한 인용 [11]–[15]은 대부분 동종 로봇입니다. 이 논문의 주장은 "하나의 로봇을 역할로 나누는 이질 분해"입니다(§II-B).

**(다) 도구**

- MAPPO [11] (Yu et al., NeurIPS 2022): 이 논문의 학습 알고리즘입니다.
- SAM 2 [10] (arXiv:2408.00714): 실로봇 분할에 씁니다.
- Isaac Lab [36] (arXiv:2511.04831), skrl [38] (JMLR 2023): 시뮬레이션과 학습 프레임워크입니다.

**선행연구의 빈틈 (우리 관점)**

- 픽셀 특징이 3D 추정보다 얼마나 나쁜지를 보여 주는 비교가 없습니다. 프로젝트 페이지에는 오라클 3D 위치 행(3D Pos)이 있지만, 실로봇에서는 3D 위치 기준선이 없습니다.
- 픽셀 특징과 크기 변화가 거리 정보를 주는지를 시뮬에서 검사하지 않습니다. 물체 크기가 다양한데도 크기 변화로 거리를 얻는다는 주장은 `03_analysis.md` §3의 토이 결과와 연결됩니다.

## 2. GitHub와 코드 저장소

조회일 2026-11-17. 별 수, 마지막 push, 라이선스는 GitHub API 응답을 그대로 옮겼습니다.

| 저장소 | 관계 | 라이선스 (API) | 별 / 포크 | 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [seongdrgn/pixel2catch-github](https://github.com/seongdrgn/pixel2catch-github) | 공식 코드 (시뮬 학습 환경과 설정) | **없음** (LICENSE 파일 없음, README에 "choose a license" TODO) | 10 / 3 | 2026-11-16 08:13 UTC | `9fed000e6c7e8916bd48acfaab93a6f336d823d5` (2026-11-16, "check entire code") | 정적 읽기만 (`repo/`, 얕은 복제). 실행, 설치, import 안 함 |
| [seongdrgn.github.io/pixel2catch](https://seongdrgn.github.io/pixel2catch/) | 프로젝트 페이지 | — | — | — | 수치 일부가 v1과 다름 (`01` §0) | 텍스트 추출만 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 학습 환경 (Pixel2Catch 공식 코드의 의존성) | API: NOASSERTION (저장소 LICENSE 별도 확인 필요) | 8,665 | 2026-11-17 07:13 UTC | 미확인 | 접근 안 함 |
| [Toni-SM/skrl](https://github.com/Toni-SM/skrl) | MAPPO 구현 (`skrl==1.4.3`) | MIT | 1,104 | 2026-11-16 14:34 UTC | 미확인 | 접근 안 함 |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | 분할 모델 (실로봇 파이프라인) | Apache-2.0 | 20,167 | 2026-05-30 02:29 UTC | 미확인 | 접근 안 함 |
| IEEE Xplore 문서 11563623 | RA-L 게재본 (README 링크) | — | — | — | 접근하지 않음 (지시에 따름) | 확인 불가 |

**공식 코드의 라이선스 해석.** LICENSE 파일도 `pyproject.toml`도 없고 README에는 라이선스를 정하지 않았다는 TODO가 있습니다. 저작권법 기준으로는 권리가 저자에게 남아 있다고 보는 것이 안전합니다. 따라서 이 저장소의 코드는 **복사하지 않고**, 파일 경로와 설정 이름을 가리키는 정도로만 인용합니다. 논문 본문은 CC BY-NC-ND 4.0입니다(`01` §0, 권리 판단은 `04_code_verification.md` §3).

**코드 공개 범위.** 공식 저장소에는 시뮬레이션 환경(`catchpolicy/`), MAPPO 설정(`agents/pixel2catch.yaml`), 시뮬 자산(USD, STL)이 있습니다. 실로봇 브리지, SAM2 파이프라인, 시스템 식별 스크립트, 평가 스크립트, 기준선 코드, 체크포인트는 없습니다.

## 4. 정리

- 이 논문은 "3D 추정 없이 픽셀 특징으로 잡기"와 "팔과 손의 이질 MARL"을 한 번에 주장합니다. 두 주장을 분리해 검증한 실험은 없습니다. S-A RL과의 비교는 알고리즘(PPO 대 MAPPO), 관측 구성, 보상을 함께 바꿨습니다(`03_analysis.md` §4, C4).
- 가장 가까운 선행연구 [9]와 [6]에 대한 직접 비교가 없습니다. 실험 설계를 평가할 때 이 점을 먼저 짚어야 합니다.
- 공식 코드는 학습 환경의 정적 기술로는 충분히 쓸모 있지만, 논문 표를 재현할 평가 절차가 없습니다.
