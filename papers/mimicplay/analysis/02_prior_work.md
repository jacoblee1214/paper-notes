# 02. 선행연구와 관련 저장소

## 1. 계보

| 축 | 선행연구 (논문 참고문헌 번호) | MimicPlay와의 관계 |
|---|---|---|
| 행동 복제, 목표 조건부 | GC-BC (Mandlekar 2021, [20]), robomimic | 기준선으로 쓴다. 저수준 정책의 기반 코드도 robomimic이다 |
| 계층 모방 | TACO [4], Mandlekar 2020 [3], Xu 2018 [28] | 계층 구조는 같다. 차이는 상위 계획을 로봇 시연이 아니라 사람 play에서 학습한다 |
| 로봇 play 학습 | LMP (Lynch 2020, [5]), C-BeT (Cui 2022, [6]), TACO-RL (Rosete-Beas 2022, [7]) | 같은 "play" 개념을 쓰되 play를 사람 손으로 모은다. play 비용이 4.5–6시간에서 10분으로 준다는 주장의 기준선이다 |
| 사람 영상 활용 | R3M [40], MVP [41], Hansen 2022 [43], 영역 차이를 줄이는 방법 [29, 44, 34, 35] | R3M-BC를 기준선으로 쓴다. MimicPlay는 표현 사전학습이 아니라 3D 손 궤적을 계획으로 쓴다 |
| 3D 궤적 목표 | Shan 2020 (hand_object_detector, [49]) | 손 검출과 3D 삼각측량에 쓴다 |
| 혼합 분포 출력 | Bishop 1994 MDN [50], BeT [53] | 정책과 계획기 모두 GMM을 쓴다 |

가장 가까운 대안은 **같은 해(2023) 발표된 play 기반 계층 학습(LMP, C-BeT)**과 **사람 영상 표현 사전학습(R3M, MVP)**이다. MimicPlay는 둘 사이에서 "사람 영상으로 계획을, 로봇 시연으로 제어를" 나눈다.

## 2. 논문 밖 자료 확인

- 논문 PDF: arXiv v1과 v2 모두 받아 확인했다(`paper/`).
- 발표 확인: PMLR v229 공식 목록(`proceedings.mlr.press/v229/wang23a.html`), 프로젝트 페이지(mimic-play.github.io, "CoRL 2023 (Oral)").
- 프로젝트 페이지의 라이선스 표기: 없음. 데이터와 체크포인트는 공개 여부를 페이지에 밝히지 않는다(Google Drive 링크만 있음, README 기준).
- 이후 후속 연구의 인용 관계는 확인하지 않았다(웹 검색 한도).

## 4. 저장소와 코드 표

| 저장소 | 관계 | 라이선스 (GitHub API) | 별 | 마지막 push | 이 분석의 상태 |
|---|---|---|---|---|---|
| [j96w/MimicPlay](https://github.com/j96w/MimicPlay) | 공식 구현 (학습, 시뮬 평가, 사람 데이터 처리) | MIT (LICENSE 파일 있음) | 311 | 2024-04-23 (커밋 `334ad94`, 같은 날) | 얕은 복제(`repo/`), 정적 읽기만. `04_code_verification.md` |
| [mimic-play.github.io](https://mimic-play.github.io) | 프로젝트 페이지 | - | - | - | 영상 링크(YouTube), 라이선스 표기 없음 |
| [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | 기반 학습 프레임워크 (`mimicplay-libero` 브랜치 필요) | MIT | 1608 | 2026-08-09 | 읽지 않음 (설치 안 함) |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 시뮬 벤치마크 (표 4) | MIT | 2564 | 2025-03-15 | 읽지 않음 |
| [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) | 시뮬 엔진 래퍼 (`v1.4.1_libero` 태그 필요) | API 판정 NOASSERTION (LICENSE 파일 확인 필요) | 2721 | 2026-11-19 | 읽지 않음 |
| [ddshan/hand_object_detector](https://github.com/ddshan/hand_object_detector) | 사람 손 검출 (사람 데이터 처리) | MIT | 303 | 2023-10-12 | 읽지 않음. PyTorch 1.0.1 / CUDA 10.0 전제(공식 README) |

- 공식 코드 본체의 라이선스는 MIT다(LICENSE 파일의 저작권: 2023 Chen Wang).
- 실로봇 데이터 처리 코드, Deoxys, RoboTurk는 MimicPlay 저장소에 없다. 실로봇 재현에는 이 세 가지가 필요하다.
- 체크포인트와 데이터: Google Drive 링크(README)만 있다. 크기는 확인하지 않았다(다운로드하지 않음).
