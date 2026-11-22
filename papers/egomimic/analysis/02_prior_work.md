# EgoMimic 선행연구와 공식 저장소

## 1. 선행연구 계보

| 갈래 | 대표 연구 | EgoMimic와의 관계 |
|---|---|---|
| 픽셀→행동 모방 학습 | ACT (Zhao et al., 2023, ALOHA), Diffusion Policy (Chi et al., 2023), RT-1/RT-2 | EgoMimic는 ACT를 기반으로 하고 CVAE, 트랜스포머 구조를 그대로 씀 (§III-C, 부록 D) |
| 사람 영상으로 고수준 계획 | Mimicplay (Wang et al., 2023) | 직접 비교 기준선. 사람 데이터로 고수준 계획을 만들고 저수준 정책은 로봇 데이터로만 학습. EgoMimic는 같은 정책이 두 데이터를 함께 학습 |
| 사람 영상의 표현 학습 | R3M, VIP 등 (본문 [15]–[17]) | 사람 데이터를 표현에만 쓰는 방식 |
| 장치와 수집 | ALOHA(1, 2), GELLO(3), UMI(4), AirExo(30), ACE(31) | 로봇 리더-팔 장치 또는 손에 드는 집게. EgoMimic는 로봇 쪽도 사람 팔 모양에 맞춤 |
| 교차 구현(embodiment) 학습 | HPT (Wang et al., 2024, [53]) | 저자들이 향후 과제로 언급. 로봇 간 정렬 방법 |
| 손 추적과 Aria | Project Aria (Engel et al., 2023, [7]), Aria MPS | 사람 데이터 수집 장치. 저자는 MPS SLAM으로 장치 포즈를 얻음 |

## 2. 가까운 대안과 차이

- **Mimicplay:** 같은 Transformer로 다시 구현하고 목표 조건을 뺀 버전을 씀 (§IV-A). 장면 일반화에서 Mimicplay는 4점, EgoMimic는 63점 (Fig. 7b).
- **사람 데이터만 쓰는 방법:** 논문 §IV-B에서 "EgoMimic (0% human)"으로 같은 구조를 비교하여 아키텍처 효과와 데이터 효과를 나눔.

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 | 최근 push | 기준 커밋 | 스타 / 포크 | 상태 |
|---|---|---|---|---|---|---|
| [SimarKareer/EgoMimic](https://github.com/SimarKareer/EgoMimic) | 공식 학습·데이터 처리 코드 | MIT | 2024-11-10 | `6d63e9a3cd1dcc1d24860a11d7d9c5b2d5cba645` (2024-11-10, "aria_to_robomimic fix, sam imports") | 231 / 21 | 얕은 복제 `repo/` (정적 읽기 전용) |
| [SimarKareer/robomimic](https://github.com/SimarKareer/robomimic) | 공식 저장소가 쓰는 robomimic 포크 (submodule) | 원본 robomimic 라이선스 확인 필요 | - | `3d65f03e76dfbc2f2b813b2ed1d04ba4e7afa3a6` (2024-12-23) | - | 얕은 복제 `repo/external/robomimic/` (정적 읽기 전용). 정규화 코드를 확인하는 데만 씀 |
| [SimarKareer/EgoMimic-Eve](https://github.com/SimarKareer/EgoMimic-Eve) | 하드웨어(Eve) ROS 코드 | MIT | 2025-05-04 | 복제하지 않음 | 29 / 4 | 메타데이터만 확인 |
| [gatech/EgoMimic (HF dataset)](https://huggingface.co/datasets/gatech/EgoMimic) | 샘플 데이터 6개 HDF5 | 확인 필요 | 2024-11-01 | - | 좋아요 4 / 다운로드 10,475 | **받지 않음.** 합계 약 243 GB (`bowlplace`, `groceries`, `smallclothfold`의 human/robot 각각 40–46 GB, robot groceries 23 GB) |
| [egomimic.github.io](https://egomimic.github.io/) | 프로젝트 페이지 (보충 자료 PDF, 영상, 워크숍 안내) | - | - | - | - | 확인 |

- **라이선스 불일치:** 저장소 최상위 `LICENSE`는 MIT입니다. robomimic 포크의 라이선스는 이 분석에서 확인하지 않았습니다.
- **공식 코드 실행 여부:** 실행하지 않았습니다. 설치도 하지 않았습니다. 정적 읽기만 했습니다.
- **공식 저장소 README의 실행 지시:** `environment.yaml`, `pip install -e`, `pl_train.py`, `eval_real`. 이 지시를 따르지 않았습니다.

## 5. 관계 정리

EgoMimic 논문의 방법은 (사람 데이터 정렬, 정규화, 마스킹, 공유 정책)의 네 부분입니다. 이 중 **재현 가능한 부분**은 정규화와 마스킹이 전부 코드로 존재하는 점입니다. 실행 환경 (CUDA 고정)과 데이터 용량 (243 GB) 때문에 그 밖의 재현은 어렵습니다 (`04` §4).
