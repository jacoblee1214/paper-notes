# 02. 선행연구, 코드 저장소, 관련 분석

## 0. 학회 상태

**학회 표기 없음 (arXiv).** arXiv 제출 기록의 Comments에는 학회명이 없습니다("39 pages, 13 figures, 20 tables. Code and data: … Extends arXiv:2611.02699 with robot-level experiments"). 저자 저장소 README는 이 논문을 AIR 2027 투고 논문으로 적지만, **채택 결과는 아직 없습니다.** 공식 채택 목록은 확인하지 않았습니다(목록이 없습니다). 따라서 `venue: "학회 표기 없음 (arXiv)"`로 둡니다.

## 1. 선행 계보

논문이 스스로 그리는 계보입니다. 이 분석에서 원문은 열지 않았고, 아래는 **이 논문 참고문헌과 본문의 설명**입니다.

```
RWM  (Li, Krause, Hutter; arXiv:2501.10100 [1])   ──┐ 감사 대상 (ANYmal D, 자기 롤아웃 8스텝 학습)
RWM-U (arXiv:2504.16680 [2])                        ├─ 앙상블 불일치로 오프라인 정책 학습
Right Ranking, Wrong Scale (arXiv:2611.02699 [3]) ──┘ 저자의 선행 프리프린트 (재구현 + 불확실성)
이 논문 (arXiv:2611.23118)                            ── 로봇 수준 시험과 사전 배포 검사 추가
```

| 주제 | 논문이 인용한 계열 (참고문헌 번호) | 이 논문이 한 일 |
|---|---|---|
| 학습된 시뮬레이터의 복합 오차 | Venkatraman 2015 [6], Talvitie 2017 [7], Lambert 2022 [9]; 해결책은 다단계 학습 [6–8, 10], 짧은 롤아웃 [5] | 기존 1스텝 검사와 8스텝 이상 검사의 차이를 보임 |
| 목표 불일치 | Lambert 2020 [11] | 1스텝 정확도가 제어 성능과 어긋남을 실험으로 보임 |
| 교사 강제 | Williams & Zipser 1989 [12], Bengio 2015 [8] | 두 방법을 같은 데이터와 시드로 비교 |
| 앙상블 불확실성과 오프라인 RL | Chua 2018 [17], Lakshminarayanan 2017 [18], MOPO [13], MOReL [14], COMBO [15], RAMBO [16]; 비교 설계는 Lu 2022 [19] | 불확실성을 순위(ranking)와 크기(scale)로 나누어 측정 |
| 보정 | Kuleshov 2018 [20], Malik 2019 [21], 분할 보정 [22] | 헤드별 보정 실패와 구간별 배수를 측정 |
| 헤드 분산 손실의 문제 | Seitzer 2022 [23] | 저자 측정: 분산이 0으로 무너짐 |
| 순환 모델의 안정성 | Pascanu 2013 [27], Miller & Hardt 2019 [28], Engelken 2023 [29], Mikhaeil 2022 [30], Hess 2023 [31], Manek & Kolter 2019 [32] | 최대 유한시간 리야푸노프 지수(Benettin 방법 [26])를 측정 |
| 물리적 일관성 검사 | Isermann 2005 [34], Schäfer 2026 [33] (운동학 대 동역학) | 운동학 잔차를 여러 시뮬레이터 순위와 경보 검사에 씀 |
| 압축 | Han 2016 [35], Liu 2019 [36], Jacob 2017 [37], 세계 모델 양자화 Fu 2026 [38], Pavlova 2026 [39], Shin 2026 [52] | 같은 압축을 모든 호라이즌에서 재고, 2017 CPU 비용을 측정 |
| 통계와 등록 | Cameron 2008 [40], Agarwal 2021 [41], Henderson 2018 [42], Nosek 2018 [43] | 독립 단위를 궤적으로 세고 사전 등록 규칙을 씀 |
| 로봇 시뮬레이션 | Hwangbo 2019 [45], Rudin 2021 [47], Mittal 2023 [48] (Orbit/Isaac Lab), MuJoCo [44] | 저자가 만든 MuJoCo ANYmal D로 정책 학습 |

**가장 가까운 대안 (논문이 직접 비교한 것):** Lu 2022 [19]가 오프라인 MBRL의 불확실성 패널티를 "오차 추적"과 "크기"로 나눠 비교한 것이 §6의 가장 가까운 선행 연구라고 논문이 말합니다. Schäfer 2026 [33]은 운동학 검사를 한 모델에서 진단했고, 이 논문은 여러 시뮬레이터에 적용해 순위를 봅니다. Fu 2026 [38]과 Pavlova 2026 [39]은 세계 모델 양자화를 과제 성공이나 한 개의 오차 성장률로 평가하지만, 호라이즌별 오차 추가와 CPU 비용은 보지 않았다고 논문이 말합니다(§2).

## 2. 원 RWM 저장소와 프로젝트 페이지

- **원 RWM 저장소:** `leggedrobotics/robotic_world_model` (GitHub API: 별 737, 마지막 push 2026-10-29, Apache-2.0, 홈페이지 `sites.google.com/view/roboticworldmodel`). 이번 분석에서 복제하지 않았습니다.
- **RWM-U 프로젝트 페이지:** `sites.google.com/view/uncertainty-aware-rwm` (`robotic_world_model_lite`의 홈페이지 필드). 열지 않았습니다.
- **이 논문의 프로젝트 페이지:** 없습니다. 논문 본문과 Comments의 두 GitHub 링크가 실질적 페이지입니다.

## 3. 이 논문의 공식 코드와 관련 저장소

| 저장소 | 역할 | 라이선스 (API) | 별 | 마지막 push (API) | 기록 | 이 분석의 처리 |
|---|---|---|---|---|---|---|
| [joyjeet-singh/rwm](https://github.com/joyjeet-singh/rwm) | 저자의 재구현과 감사 (`PAPER.md`는 2611.02699 판) | Apache-2.0 | 0 | 2026-11-17 (기본 가지 HEAD 2026-11-03) | 2026-08-16 생성 | 얕은 복제 `repo/rwm_author/` (정적 읽기) |
| [joyjeet-singh/rwm-robot-level](https://github.com/joyjeet-singh/rwm-robot-level) | 로봇 수준 실험, 결과 파일, MuJoCo 모델, 등록 기록 | Apache-2.0 | 0 | 2026-11-17 | 2026-11-17 생성 (커밋은 2026-11-01부터 96개) | 얕은 복제 `repo/rwm_robot_level_author/` |
| [leggedrobotics/robotic_world_model_lite](https://github.com/leggedrobotics/robotic_world_model_lite) | 원 RWM 경량 코드, 데이터 1개, 체크포인트 1개 (논문 고정 커밋 `13a798e9`) | Apache-2.0 | 254 | 2026-04-08 | — | 얕은 복제 `repo/upstream_rwm_lite/` (정적 읽기) |
| [leggedrobotics/rsl_rl_rwm](https://github.com/leggedrobotics/rsl_rl_rwm) | 학습 라이브러리, `system_dynamics.py` (논문 고정 커밋 `18eebcdd`) | LICENSE 본문: ETH Zurich, NVIDIA (BSD 계열) | — | — | — | 부분 복제 `repo/rsl_rl_rwm_upstream/` (정적 읽기, 가중치 없음) |
| [leggedrobotics/robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) | 원 RWM·RWM-U 논문 저장소 | Apache-2.0 | 737 | 2026-10-29 | — | 복제하지 않음 |
| [nikodemseb/adaptive-truncation-world-models](https://github.com/nikodemseb/adaptive-truncation-world-models) | 검색으로 찾은 관련 저장소 (GRU 세계 모델의 적응 호라이즌 절단, 다리 로봇) | (미확인) | 1 | 2026-09-20 | — | 열지 않음. 관련 주제만 기록 |

- **논문과 저장소의 관계:** 논문 Comments의 두 저장소 링크는 저자 저장소입니다. "공식 코드"라는 말은 저자의 감사 코드를 가리키고, 원 RWM 학습 코드는 ETH 저장소입니다.
- **라이선스 참고:** 저자 저장소와 ETH 저장소는 모두 Apache-2.0입니다(API 값). rsl_rl_rwm은 LICENSE 본문 머리글이 ETH Zurich와 NVIDIA 저작권을 표시하고 BSD 계열 문구입니다. 논문은 BSD 3-Clause라고 씁니다.
- **재배포 금지:** 논문 "Code and data availability" 절은 원 RWM의 코드, 데이터, 체크포인트를 "수정하거나 재배포하지 않고 고정 커밋에서 읽는다"고 씁니다. 이 분석도 원 데이터와 가중치를 복사하지 않았습니다.


## 4. 구현 가능성과 선행 자료의 한계

- 원 RWM(2501.10100)과 RWM-U(2504.16680)는 이번 분석에서 받지 않았습니다. 두 논문의 주장은 이 논문의 설명으로만 씁니다. 원 논문과의 불일치 주장(예: 2,500 반복)은 확인되지 않은 상태입니다.
- 선행 프리프린트 2611.02699의 텍스트는 `repo/rwm_author/PAPER.md`에 있으나, 이 분석에서는 본문을 읽지 않고 저장소 목록과 README만 참고했습니다.
