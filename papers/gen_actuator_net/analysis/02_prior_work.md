# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 논문이 기대는 출처 | 이 논문에서의 위치 |
|---|---|---|
| 액추에이터 네트워크(구동 특성을 신경망으로 학습) | Hwangbo et al. 2019 [23] (사족 보행, 토크 측정을 라벨로 사용, 성긴 이력 stride 4). 이후 [16, 24, 25, 26] | 기본 아이디어. 이력을 현재값 대비 차이로 바꾸고 stride 1을 쓰는 점이 다름(§3.1, 부록 C) |
| 토크 센서 없는 액추에이터 망 | Fey et al. RSS 2025 [29] (UAN: RL로 궤적 재현 토크 학습, 모터+PD 전제) | 유일한 기존 비교 대상. 이 논문은 전체 토크를 예측하게 바꾸고 결합을 보도록 구조를 바꿔 비교(§4.1.2, 부록 A.3) |
| 힘줄·유압 구동 학습 모델 | Yuryev & Hughes 2026 [28] (1자유도 손가락 힘줄 힘), Spinelli et al. 2024 [27] (굴착기 유압) | 액추에이터 모델 학습이 보행 밖에서도 쓰인 사례 |
| 근육 구동 로봇의 sim-to-real | Tao et al. 2025 [30] (단일 관절, 시스템 식별 + 무작위화), Biyajima et al. 2025 [31] (1자유도 타악기), Wang et al. 2025 [32] (DeLaN [33], 근육 하나) | 모두 단순한 시스템과 도달 과제(§2.2) |
| 근육을 소프트웨어로 흉내 | Schumacher et al. 2024 [34] (모터 구동 사족) | 근육 동역학을 정확히 알지만 실제 근육의 순응성은 포기(§2.2) |
| 근육 로봇에서 시뮬 + 실물 혼합 학습 | Büchler et al. 2022 [10] (공만 시뮬레이션, 로봇은 실물), Guist et al. 2023 [22] (Hindsight States), Guist et al. 2024 [3] (PAMY2 설계) | 실물 로봇 시간이 많이 드는 대안. Fig. 13의 비교 대상(HySR) |
| PAM 정적 모델 | Chou & Hannaford 1996 [38], Tondu 2012 [19] | SysID 기준선의 근육 힘 식(부록 F.4) |
| 앙상블과 무작위 구성원 | Janner et al. 2019 [35] (MBPO) | 시뮬 스텝마다 무작위 구성원 선택(§3.2) |
| 도메인 무작위화 | Peng et al. 2018 [17], Muratore et al. 2022 [18], Tiboni et al. 2024 [21] | 근육 로봇은 간극이 커서 무작위화만으로는 부족하다는 논거(§1) |
| 시뮬레이션과 학습 도구 | MuJoCo/MJX [37], skrl [40], PPO [41] | MJX 기울기를 직접 통과시키거나 식 2로 대체(§3.1) |
| 과제와 장비 | ball-in-a-cup [39], AIMY 발사기 [42], 시각 공 추적 [46], 반발 모델 [47, 10] | 과제 정의와 탁구 환경(부록 A.4) |
| 능동 탐색(향후 과제) | Pathak et al. 2019 [43], Shyam et al. 2019 [44], Schneider et al. 2022 [45] | 한계(§5)에서 자유도 증가 대응책으로 언급 |

가까운 대안(논문에 비교가 없는 것): 토크 라벨을 얻는 대신 미분 가능한 시뮬레이터로 시스템 식별을 하는 접근과 같은 선상에 있습니다. SysID 기준선(부록 F)이 이 계열의 한 예이며, GenAN은 그 해석적 구조 자리에 신경망을 넣은 형태로 읽을 수 있습니다. 근육 로봇 전체를 학습된 세계 모델로 대체하는 방식은 논문의 비교에 없습니다. 논문은 강체 부분을 해석적으로 두는 이유를 "장면을 바꿔도 재학습이 필요 없다"고 설명합니다(§3.2).

## 2. 코드와 저장소 (GitHub 표)

기준일 2026-11-25. 스타 수와 마지막 push는 GitHub API 값입니다.

| 저장소 | 관계 | 라이선스 | 스타 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [intelligent-soft-robots/genan](https://github.com/intelligent-soft-robots/genan) | **공식 구현** (데이터 수집, GenAN 학습, 정책 학습, 전이 평가, 실험 로그) | LICENSE 파일 없음, GitHub API license: null, `pyproject.toml`에도 라이선스 표기 없음 | 2 | 2026-10-13 | 커밋 `c443b8b` (Initial commit, 2026-10-13). 정적 읽기만 |
| 프로젝트 페이지 [webdav.tuebingen.mpg.de/genan](https://webdav.tuebingen.mpg.de/genan/) | 논문, 영상, 코드, 데이터 링크 | 페이지 하단에 CC BY-SA 4.0 표기(사이트 템플릿 서술) | - | - | HTML만 읽음 |
| [intelligent-soft-robots/learning_table_tennis_from_scratch](https://github.com/intelligent-soft-robots/learning_table_tennis_from_scratch) | 같은 그룹의 탁구 학습(Büchler et al. 2022 [10]) | BSD-3-Clause | 21 | 2026-09-15 | 비교 참고 |
| [intelligent-soft-robots/o80](https://github.com/intelligent-soft-robots/o80) | 프로세스 동기화(로봇 프레임워크, 공식 README가 설치 전제로 지목) | BSD-3-Clause | 10 | 2026-09-22 | 의존성 |
| [intelligent-soft-robots/pam_documentation](https://github.com/intelligent-soft-robots/pam_documentation) | PAM 로봇 설치와 문서 | BSD-3-Clause | 2 | 2026-10-27 | 의존성 문서 |
| [intelligent-soft-robots/pam_mujoco](https://github.com/intelligent-soft-robots/pam_mujoco) | PAM 로봇 MuJoCo 모델(`launch_pam_mujoco`) | BSD-3-Clause | 2 | 2025-05-09 | 의존성 |
| [intelligent-soft-robots/ball_launcher_beepy](https://github.com/intelligent-soft-robots/ball_launcher_beepy) | AIMY 공 발사기 제어 [42] | BSD-3-Clause | 7 | 2026-10-22 | 장비 |
| [jan1854/skrl](https://github.com/jan1854/skrl) (브랜치 `genan`) | `pyproject.toml`이 지정하는 skrl 포크 | MIT | 0 | 2026-10-07 | 의존성 |
| [Toni-SM/skrl](https://github.com/Toni-SM/skrl) | 원 skrl [40] | MIT | 1,105 | 2026-11-21 | 참고 |
| [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | MuJoCo, MJX [37] | Apache-2.0 | 18,459 | 2026-11-24 | 참고 |

- 공식 저장소는 코드와 함께 **실험 폴더 `exps/`와 학습 로그 `logs/`를 Git LFS로** 담고 있습니다. 프로젝트 페이지의 "Datasets" 링크는 `exps/`를 가리킵니다. LFS 포인터에서 읽은 크기 합계는 약 9.8 GB입니다(`exps` 약 8.1 GB, `logs` 약 1.7 GB). 받지 않았습니다(`04` §1).
- 데이터 수집과 실로봇 전이는 PAM 로봇 프레임워크(o80, `launch_pam_mujoco`)가 필요합니다. README는 python 3.10을 요구합니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "근육·힘줄 로봇은 모델링이 어렵다"는 서술은 PAM 히스테리시스 문헌([6, 7, 8])과 Tondu의 리뷰 [19]에 기댑니다. 이 분석에서 해당 문헌을 새로 확인하지는 않았습니다.
- 최초성 주장("first successful sim-to-real transfer for a four-degrees-of-freedom muscle-actuated robot arm")은 §2.2에서 인용한 단일 관절·단일 근육 연구와의 비교에 기댑니다. 이 분석은 논문이 인용한 범위를 넘어선 문헌 검색을 하지 않았습니다.
