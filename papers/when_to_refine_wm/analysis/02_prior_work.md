# 02. 선행 연구와 자료

## 1. 이 논문의 자리

이 논문은 "계획 세계모델에서 탐색 깊이를 언제 멈출까"를 묻습니다. 고정 세계모델(LeWM)과 고정 CEM 계획기는 그대로 두고, 한 solve 안에서 5회째 게이트 하나만 추가합니다(`01` §3).

| 질문 | 가까운 선행 연구 | 이 논문과의 차이 (본문 출처) |
|---|---|---|
| 세계모델로 행동을 고르는 기본 방식 | PETS [4], TD-MPC/TD-MPC2 [10, 11], Dreamer [8, 9], Finn & Levine [7], DINO-WM [29] | 이 논문은 LeWM(픽셀에서 공동 학습, SIGReg 정규화 [17, 1])을 씀(§2.1) |
| CEM 계획의 효율 | iCEM [20], CEM-GD [12], MPPI [24] | 계획기 자체를 바꾸지 않고 반복 수만 정함(§2.2) |
| 예산을 모델-과제 쌍에 맞추기 | SufficientPlan [14] (쌍별 폐루프 평가로 예산 선택, Static-Context Reuse) | 이 논문은 solve 단위 결정 (§2.2) |
| 계획기를 얼마나 자주 다시 부를지 | Zhang 외 [28] (이벤트 트리거), Lu & Maciejowski [15], Magnino 외 [18] (학습된 예산 정책), Labiosa & Hanna [13] (반응 정책 대 계획) | 이벤트 트리거가 아니라 solve 내부 게이트 (§2.3) |
| **CEM을 조기 종료** | **Zhang 외 [27] (LeWM 위의 조기 종료, 평균 엘리트 비용의 상대 감소 기준)** | 가장 가까운 기준선. 이 논문은 엘리트 비용과 첫 묶음 이동·분산을 함께 씀. 저자도 비교 기준으로 지목함(§6) |
| 샘플링 계획과 PushT | Zeer 외 [26] (빔 탐색 호출), LePlanner [2] (고정 횟수 학습 계획 갱신) | LePlanner는 고정 횟수 갱신, Zeer는 빔 탐색 |
| 난수 공통화 | Yadav 외 [25] (공통 난수로 행동 가치 비교), Salmon 외 [22] (counter-based RNG) | 패딩으로 난수 스트림을 맞춤(§3.4). counter 기반은 향후 과제(§6) |
| 환경 | PushT [3] (Diffusion Policy), OGBench-Cube [19], stable-worldmodel [16] | 같은 환경 도구 사용 |
| LeWM의 행동 기하 | Vasilev & Nayak [23] | 전이 질문(§5)의 배경 |

## 2. 구성 요소별 출처

| 구성 | 출처 |
|---|---|
| 비용 = 예측 끝 임베딩과 목표 임베딩의 제곱 거리 | LeWM [17], DINO-WM [29] |
| LeWM 구조와 SIGReg | LeWM [17], LeJEPA [1] |
| CEM 계획기 | de Boer 외 [5], Rubinstein [21] |
| 조기 종료 기준선 | Zhang 외 [27] |
| 탐색 깊이 수익 계산 방법 (종점 혼합) | 이 논문의 식 (11) 자체 정의. 선행 문헌의 표준 방법이 아님 |
| 공통 난수 | Yadav 외 [25] |
| 패딩 난수 | 이 논문 §3.3 |

## 3. 이 저장소의 관련 분석 (`01` §8)

| 폴더 | 논문 | 관계 |
|---|---|---|

## 4. GitHub, 프로젝트 페이지, 코드 (확인 2026-11-22)

이 논문의 코드는 찾지 못했습니다. 검색한 곳: arXiv abs·HTML의 "Code, Data and Media" 항목(없음), GitHub 저장소 검색 "Selective CEM Refinement", 제목, 저자 이름 "Jingni Huang"(결과 없음), 웹 검색(제목과 방법명, 결과 없음). 프로젝트 페이지도 찾지 못했습니다.

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **이 논문의 공식 코드** | 찾지 못함 | - | - | `04_claim_verification.md`로 대체 |
| 기반 코드 LeWM (공식) | https://github.com/lucas-maes/le-wm | MIT | 4654 / 2026-05-26 | `repo/`에 얕은 clone (커밋 `8edfeb3`, 2026-05-22). 정적 읽기만. CEM 기본값과 PushT 설정 대조에만 사용 |
| 환경·계획 라이브러리 stable-worldmodel | https://github.com/galilai-group/stable-worldmodel (LeWM README 링크) | MIT | 2315 / 2026-10-02 | 이 논문의 PushT·Cube 환경(`swm/PushT-v1`, `swm/OGBCube-v0`)을 제공. 복제하지 않음 |
| 학습 프레임워크 stable-pretraining | https://github.com/galilai-group/stable-pretraining | MIT | 352 / 2026-09-29 | LeWM 학습용. 복제하지 않음 |
| LeWM 체크포인트 (PushT) | https://huggingface.co/quentinll/lewm-pusht | 미확인 | - | LeWM README가 링크. 받지 않음. 논문 표 2의 `native pusht/weights.pt`와 같은지 미확인 |
| 프로젝트 페이지 | 찾지 못함 | - | - | - |
| arXiv 소스 | `arxiv.org/src/2611.24470` | - | - | 받지 않음 |

- 기반 코드의 "CEM 기본값 30회, 샘플 300, 엘리트 30"은 이 논문과 맞는 부분이 많습니다. 깊이 15는 이 논문이 고른 값입니다(`04` §3).
- 기반 코드의 `device: "cuda"`가 평가 스크립트에 하드코딩되어 있습니다(`eval.py`, `cem.yaml`). 맥에서 쓰려면 바꿔야 합니다(`04` §4).

## 5. 파생 조사 메모

- 같은 LeWM 공식 체크포인트를 쓰는 다른 논문(ACTUS, Counterfactual WM, 이 논문)은 체크포인트 로드와 평가 설정을 공유할 가능성이 큽니다. 맥에서 한 번 구축하면 여러 분석에 재사용할 수 있습니다.
- 이 논문의 "같은 작업량에서의 비교"는 다른 폴더의 탐색 예산 실험에도 그대로 쓸 수 있습니다(`03` §2).
