# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 출처 (논문 참고문헌 번호) | 이 논문이 한 것 |
|---|---|---|
| Deep unfolding, 알고리즘 펼침 | Hershey 외 2014 [1], Monga 외 2021 [2], LISTA(Gregor와 LeCun 2010) [8] | 정의와 동기를 그대로 씀. 로봇 실시간 예산 문제로 옮김 |
| 학습된 QP·원뿔 solver: 웜스타트 | Sambharya 외 [13, 14] (OSQP, SCS의 고정점 반복을 펼침), Xiong 외 [17] (SCS의 선형계 풀이를 경사 한 걸음으로 바꾼 DR-GD) | 웜스타트 MLP를 함께 학습하되 선형계 풀이는 그대로 둠 |
| 학습된 hyperparameter | Sambharya와 Stellato [18], Saravanos 외 [5] (OSQP와 분산 OSQP에 대한 피드백 정책, 조밀한 AᵀA 암묵 미분) | 같은 정책 계열을 SOC와 PSD로 확장하고, 조밀 행렬 없이 미분 |
| SCP에 DU 적용, 정책 파라미터화 | Oshin 외, Deep FlexQP [6] | 정책 구조(로짓 증분 재귀, 영 초기화)를 가져와 비분리 원뿔용 평균화(rectification)만 추가 |
| COSMO와 ADMM | Garstka 외 [19], Boyd [7], OSQP [15], SCS [16] | COSMO의 간접 모드와 갱신식(3a)–(3e)을 그대로 따름 |
| 선형계의 GPU 간접(CG) 풀이 | Schubiger 외 [23] | 간접 모드를 펼침 구간 전체에서 쓰도록 선택 |
| 암묵 미분 | OptNet [37], cvxpylayers [38], Blondel 외 [40], Amos [24, 41] | 식 5–6은 M(ρ)의 구조(aᵢaᵢᵀ)를 이용한 특수화. 최적해의 미분이 아니라 풀이 궤적을 미분하는 점이 다르다고 서술 |
| eigh 역전파의 불안정성 | Ionescu 외 [45], Wang 외 [46] | 같은 문제를 PSD 투영의 위치에서 다룸 |
| Daleckii–Krein 표현 | Higham [21], Bhatia [44], Engin 외 [20, 정리 1] (SPD 표현 학습) | PSD 투영 역전파 규칙에 쓴 것이 새로운 점 |
| 공분산 조종과 SCP | Goulart 외 [26], Balci와 Bakolas [27], Ridderhof 외 [28, 29] | 아핀 외란 피드백 볼록 형식과 신뢰영역 SCP를 가져옴 |
| 자기지도 손실 | 로그 수렴률 아이디어(Nocedal와 Wright [25]) | 초기 잔차로 정규화한 log 잔차 합(식 13) |
| 벤치마크 문제 | Goemans–Williamson [48], Lovász [49], CVXPY robust KF 예제 [50], Tedrake 교재 [47] | 7개 문제군 정의(부록 F) |
| 기준선 solver | COSMO.jl v0.8.9 [19], Clarabel v0.11.1 [22] | 표 2, 표 6, 그림 1의 비교 대상 |

새로운 점으로 논문이 내세우는 것은 둘입니다. 하나는 M(ρ) = P + σI + Σρⱼ aⱼaⱼᵀ의 구조를 이용해 암묵 미분의 메모리를 행렬 없이 O(n + nnz(A))로 낮춘 것(식 6)이고, 다른 하나는 이미 알려진 DK 표현을 PSD 투영 역전파에 쓴 것입니다. 정책 파라미터화와 학습 손실은 선행 연구[6]를 확장한 것입니다.

## 2. 코드와 저장소 (GitHub 표)

2026-11-26 `gh api`로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 커밋 |
|---|---|---|---|---|---|
| [alexoshin/deep-conic-optimization](https://github.com/alexoshin/deep-conic-optimization) | 공식 구현(논문 각주 1, 저장소 README가 논문과 CoRL 2026을 명시) | Apache-2.0(LICENSE.md, API도 Apache-2.0) | 0 (포크 0) | 2026-11-09 | `cf16834997c3b7d89e877df8cbb1f4edc1238fb9` (2026-10-12, "Initial commit", 커밋 하나, 90 KB) |
| [oxfordcontrol/COSMO.jl](https://github.com/oxfordcontrol/COSMO.jl) | 펼치는 대상 solver, 표 2와 표 6의 기준선 | Apache-2.0 | 320 | 2026-10-05 | 읽지 않음 |
| [oxfordcontrol/Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) | 표 2, 표 6, 그림 1의 interior-point 기준선(저장소는 Rust, 파이썬 바인딩 사용) | Apache-2.0 | 635 | 2026-04-13 | 읽지 않음 |
| [osqp/osqp](https://github.com/osqp/osqp) | 잔차 균형 ρ 규칙의 출처 | Apache-2.0 | 2234 | 2026-01-12 | 읽지 않음 |
| [cvxgrp/scs](https://github.com/cvxgrp/scs) | 논문이 "적용 가능"이라 언급한 다른 ADMM solver(실증 없음) | MIT | 639 | 2026-11-26 | 읽지 않음 |
| 프로젝트 페이지 | arXiv 페이지, 논문, 저장소 README에서 링크를 찾지 못함 | – | – | – | – |
| Deep FlexQP 저장소 | 논문 [6]. GitHub 검색에서 확인하지 못함 | – | – | – | – |

- 논문 본문 각주와 OpenReview(`CSYA6oHdlE`)가 공식 연결입니다. 저장소 README의 인용 URL도 같은 OpenReview 글입니다.
- 공식 저장소는 `repo/`에 얕은 복제로 두고 정적으로만 읽었습니다(`04`).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **DK 도함수의 안정성**은 Daleckii–Krein 정리(Higham [21], Bhatia [44], Engin 외 [20, 정리 1])에서 나옵니다. f(x) = max(x, 0)의 나눈 차분은 [0, 1]이라 간격이 0으로 가도 유한합니다(식 10 아래 서술). 이 논문의 새로운 점은 이 표현을 ADMM 펼침에 쓴 데 있고 정리 자체는 아닙니다.
- **log 수렴률 손실의 퇴화 방지**(분모를 detach)는 이 논문이 선행 정책 학습[6]에서 가져온 설계입니다. 식 13의 정규화는 초기 잔차에 대한 상대 감소를 보게 합니다.
- **"학습된 ρ가 이론적 수렴을 해치지 않는다"**는 이 논문이 증명하지 않습니다. ρ와 α를 반복마다 바꾸는 ADMM은 일반 수렴 이론이 따로 필요하고, 논문은 한계 문단에서도 이를 다루지 않습니다.
