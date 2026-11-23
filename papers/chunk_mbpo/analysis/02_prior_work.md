# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다(`paper/paper.txt`). 각 논문을 따로 열어 검증하지는 않았고, 계보는 논문 본문의 인용과 공식 저장소를 기준으로 정리했습니다.

## 1. 계보

CMPO는 모델 기반 RL의 세 계열과 행동 청킹이 만나는 자리에 있습니다.

**(a) 모델을 데이터 생성기로 쓰는 계열(Dyna).** Dyna [12], MBPO [13], 모델 앙상블 TRPO [32], 이론 보장 프레임워크 [33]는 모델 롤아웃으로 합성 전이를 만들어 모델 없는 학습자에 섞습니다. 오프라인 쪽은 MOPO [14], MOReL [34], COMBO [35]가 불확실성 벌점, 비관적 MDP, 보수적 가치로 분포 밖 질의를 막습니다(§2).

**(b) 계획 계열(MPC).** PETS 계열 [20, 36]의 임의 슈팅과 CEM, TD-MPC [17]과 TD-MPC2 [18]의 짧은 지평 잠재 궤적 최적화와 종료 가치, PlaNet 계열 [19]이 있고, BMPC [21]는 MPC 전문가 모방으로 정책을 부트스트랩하며 TD-M(PC)² [37]는 정책 제약으로 분포 밖 질의를 줄입니다(§2). **CMPO의 기반은 TD-MPC2이고, 공식 코드 README는 BMPC 구현과 TD-MPC2 구현 위에 지었다고 적습니다**(감사 문구).

**(c) 모델 기울기 계열.** SVG [38]가 확률적 동역학을 통과하는 역전파의 틀을, Amos 외 [22]가 SVG와 SAC의 결합을, MAAC [23]가 경로 미분의 단조 향상 보장을 냈습니다. Dreamer 계열 [5, 15, 16]은 상상 궤적의 반환을 액터에 역전파하고, Amigo 외 [24]는 병렬 시뮬레이터로 궤적을 펼쳐 모델 기울기로 오차 누적을 줄입니다(§2). CMPO는 이 계열에 속하되 **재귀 호출 자체를 없앴다**는 점이 차이입니다.

**(d) 행동 청킹.** 모방 학습에서 ACT [25], Diffusion Policy [26], π0 [27]가 시간적 일관성을 위해 쓰고, RL에서는 Q-Chunking [28]과 Decoupled Q-Chunking [29]이 청크 행동 Q로 편향 없는 n스텝 TD를 얻습니다. **오프라인 모델 기반 RL의 MAC [30]**가 청크 정책과 장기 동역학 모델을 함께 학습합니다. 논문은 이들이 "가치 학습"에서만 청킹을 썼고, 모델 기울기 정책 최적화는 남았다고 봅니다(§1, §3.2). 동적 청크 길이는 [40–42](모방 학습)를 향후 방향으로 듭니다(한계).

**이 논문의 위치.** (c)의 가치 확장 목적을 (d)의 청크 정책과 청크 동역학으로 한 번에 계산하고, (b)의 TD-MPC2 구조에서 MPC를 떼어 낸 형태입니다. 가장 가까운 선행은 **MAC [30]**(오프라인, 청크 정책 + 장기 모델)과 **TD-MPC2 [18]**(구조)입니다. 차이는 온라인 설정에서 (H−1)스텝 가치 확장의 행동 기울기를 단일 호출로 얻는 점입니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 역할 | 가까운 대안 |
|---|---|---|---|
| 잠재 동역학, SimNorm, two-hot 회귀, 일관성 손실 | TD-MPC2 [18] | 모델 학습 틀(식 6), 구조 그대로 | Dreamer 계열의 재구성 손실 [15] |
| 보상·가치 two-hot CE, 한 스텝 TD | TD-MPC2 [18] | 식 6, 식 7 | Q-Chunking의 청크 행동 Q [28] |
| 모델 기반 가치 확장 | MVE [39], SVG [38], MAAC [23] | 정책 목적(식 8) | 전체 H스텝 확장(종료 정책 재호출) |
| SAC식 엔트로피 | SAC [8] | 식 8의 β항 | 없음(결정적 정책) |
| 청크 정책 | ACT [25], Q-Chunking [28], MAC [30] | 행동 H개를 한 번에 | 단일 행동 정책 |
| 청크 동역학(출력 L·H) | MAC [30]의 장기 모델과 유사 | 한 번 호출로 z_{1:H} | 재귀 한 스텝 모델 |
| 청크의 첫 행동만 실행 | 논문이 "[29]와 비슷하게"라고 서술(§3.3) | 폐루프 응답성 | 청크 전체 실행(개루프) |
| MPPI 계획(결합 실험) | TD-MPC2 [18] | 표 2의 w MPC | 계획 없음(주 실험) |
| 환경 | HumanoidBench [31] | 19과제, H1hand, 행동 61차원 | DMControl, Meta-World(코드에는 있으나 논문 미사용) |
| 기준선 | PPO [9], SAC [8], DreamerV3 [15], TD-MPC2 [18] (HumanoidBench 보고 결과), MAAC [23] (자체 구현) | 그림 2, 7 | Dreamer 계열, TD-MPC2 변형 |
| 실로봇 | Unitree Go2, Isaac Gym, 잔차 관절 위치 PD | 두 발 서기 | 다른 로봇 |

## 4. GitHub와 코드 자료 (확인 2026-11-23)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (CMPO)** | https://github.com/song-wx/CMPO | MIT | 별 10, 포크 0, 커밋 2개, 최신 `8cebdd2` (2026-11-16) | `repo/`에 얕게 복제, 정적 읽기만. 단일 과제 온라인 학습 코드(HumanoidBench, DMControl, Meta-World 어댑터). 실로봇, 기준선, 절제, 그림 코드는 없음 |
| 프로젝트 페이지 | https://song-wx.github.io/cmpo.github.io/ | - | 저장소 song-wx/cmpo.github.io 별 0, 2026-10-27 | 발표 영상, 표 1, 그림 요약, BibTeX. 저장소 라이선스 없음 |
| TD-MPC2 | https://github.com/nicklashansen/tdmpc2 | MIT | 별 1006, 2026-10-20 | CMPO가 구조를 가져온 기반 |
| BMPC | OpenReview forum i7jAYFYDcM (공식 README가 가리킴) | 미확인 | 미확인 | 공식 코드 LICENSE의 저작권 표기는 "Yuhang Wang (2025)"이고, 논문 참고문헌 [21]의 첫 저자 이니셜(Y. Wang)과 일치 |
| HumanoidBench | https://github.com/carlosferrazza/humanoid-bench | NOASSERTION | 별 807, 2025-09-18 | 원 벤치마크. 기준선 곡선의 출처 |
| HumanoidBench 포크 | https://github.com/windigal/humanoid-bench (고정 `da87196`) | NOASSERTION | 별 0, 2025-03-08 | 공식 CMPO README가 요구하는 의존성. 설명은 "BMPC 알고리즘의 pip 설치용 적응". 원본과의 환경 차이는 확인하지 못함 |
| 논문 PDF 사본 | https://github.com/song-wx/cmpo.github.io (assets/cmpo-paper.pdf) | - | - | 공식 README의 "Paper" 링크. 받지 않음 |
| 실로봇·Isaac Gym 코드, 기준선 코드, 체크포인트 | 저장소에 없음 | - | - | 공개되지 않음 (`04` D6, D15) |

참고: 이 표의 별과 날짜는 `gh api`로 2026-11-23에 읽은 값입니다. TD-MPC2 저장소의 최신 변경(2026-10-20)이 CMPO 구현이 가져온 시점의 코드와 같은지는 확인하지 않았습니다.
