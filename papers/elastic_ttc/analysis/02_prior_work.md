# 02. 선행연구와 관련 저장소

번호는 논문 참고문헌 번호입니다(`paper/paper.txt`의 References). 각 관계 설명은 논문이 직접 쓴 문장(§2)을 근거로 하고, 논문 밖의 판단은 "본 분석"으로 표시했습니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| ELASTIC의 요소 | 출처 | 관계 |
|---|---|---|
| 생성형 제어 정책 | Diffusion Policy [1] (Chi 2023), π0.5 [2] (Black 2025), OpenVLA [3], Octo [4] | 기본 정책. 확산 정책 6과제는 [1], VLA는 π0.5 (§1, §3, §5) |
| 디노이징/플로 | DDPM [5], DDIM [6], Flow Matching [7], SDE [25] | 순차 축의 기반. 전진 폭 Δτ를 바꾸는 것이 순차 스케일링 (§3) |
| 병렬 스케일링 | Large Language Monkeys [8], Setlur 외 [14], 확산 모델 추론 스케일링 [15], Damani 외 [16], Qi 외 [17] | Best-of-N과 검증기. ELASTIC은 완전히 디노이징한 N개가 아니라 디노이징 도중 개입한다고 구분 (§2) |
| 로봇 검증기와 탐색 | RoboMonkey [18], SAILOR [19], V-GPS [20], 정책 조향 [21, 22] | V-GPS가 VLA 실험의 검증기(§5.3, §7.5.2). RoboMonkey는 16샘플에서 기본 정책 지연의 4배 이상이라고 인용(§2) |
| 순차 축 적응 | Consistency Policy [9], DPM-Solver [10], AdaFlow [11], 적응 시간 스텝 플로 매칭 [12], D3P [13] | 기본 정책을 고정한 채 상태별 디노이징 스케줄을 바꾸는 계열은 [11–13]. ELASTIC은 순차만으로 부족하다고 주장 (§2) |
| 병렬 축 효율화 | FASTER [23] | 조기 중단으로 병렬 비용을 줄이려는 시도. 논문은 FASTER의 노이즈 인지 검증기가 실제로는 초기 노이즈 선택에만 쓰인다고 평가 (§2) |
| 순차-병렬 공동 할당 | ZIP-RC [24], Qi 외 [17] | LLM에서 같은 문제를 메타 MDP로 푼 선행. ZIP-RC는 다음 토큰 로짓 자기 검증에 의존하는데 GCP에는 대응물이 없다고 구분 (§2) |
| 샘플 집합 전체를 디노이징 | DPPO [26] | 합동 디노이징 과정으로 보는 공식화 (§4.1 각주) |
| 종료 가치 보상 | Ng 외 [27] | 보상 변환 불변성. 종료 가치로 신용 할당 (§4.2) |
| RL 알고리즘 | SAC [28], 하이브리드 RL [30], FiLM [29] | 메타 정책 학습과 조건화 (§4.3) |
| 벤치마크 | Robomimic [31], LIBERO [32] | Square, Can, LIBERO-10 |
| 검증기 후보 | Robometer [33], VLM 가치 학습자 [34], RoboReward [35], RoboDopamine [36] | 한계 절에서 "검증기가 좋아질 것"의 근거 (§6) |

**가까운 대안과의 차이 (논문 주장).** [11–13]은 순차 축만, [18–21]은 병렬 축만 조정합니다. [24]는 공동 할당이지만 LLM이고 자기 검증 신호가 필요합니다. ELASTIC의 새로움은 (i) 샘플별 stride라는 하나의 행동 공간으로 두 축과 가지치기를 함께 다루고, (ii) 환경 수준 Q를 종료 가치로 써서 RL 신용 할당을 쉽게 만든 것입니다.

**본 분석.** 논문은 순차 단독 기준선으로 "같은 계산량의 Sequential"과 기본 DP/VLA만 두었고, [11–13] 같은 기존 순차 적응 방법과는 직접 비교하지 않습니다(`04` C9). 병렬 쪽도 BoN 하나입니다. FASTER [23], RoboMonkey [18]와의 정면 비교도 없습니다.

## 2. 코드와 저장소 (GitHub 표)

2026-11-22 확인. **ELASTIC 공식 코드와 프로젝트 페이지는 찾지 못했습니다.**

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 상태 |
|---|---|---|---|---|
| (없음) | ELASTIC 공식 코드 | — | — | arXiv abs 페이지에 링크 없음, arXiv HTML과 PDF에 URL 없음, GitHub 검색(논문 제목, 저자 이름, `elastic` 키워드) 결과 없음, 웹 검색 1회 결과 없음 |
| [nakamotoo/V-GPS](https://github.com/nakamotoo/V-GPS) | 실험의 검증기 (§5.3, §7.5.2) | LICENSE 파일 없음 (API `null`) | 60 / 2025-04-28 | 복제 안 함 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 체크포인트와 추론 서버 (`infer()` 왕복 지연, §7.4) | Apache-2.0 | 14,729 / 2026-08-24 | 복제 안 함 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 확산 정책 체크포인트(PushT, Square) | MIT | 4,738 / 2024-12-24 | 복제 안 함 |
| [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | Square, Can 과제 | MIT | 1,608 / 2026-08-09 | 복제 안 함 |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | LIBERO-10 | MIT | 2,565 / 2025-03-15 | 복제 안 함 |

공식 코드가 없으므로 `repo/`는 만들지 않았고 `04_claim_verification.md`로 대신합니다. 논문이 쓴 외부 도구(openpi, V-GPS, diffusion_policy, robomimic, LIBERO)는 모두 공개돼 있어서 **재구현 시 의존성은 확인된 상태**입니다. 빠진 것은 메타 정책 학습 코드, 합성 반사실 전이 생성기, 과제별 검증기 학습 코드, Can Paired와 Can Reverse 체크포인트와 데이터 변형입니다.

## 4. 확인하지 못한 것

- 공식 코드, 프로젝트 페이지, 보충 영상: 없음(2026-11-22 기준). 저자 한 명(Gokul Swamy)의 소셜 게시글이 있다는 웹 검색 결과 한 줄이 있었으나 링크를 직접 확인하지 못했습니다.
- OpenReview 심사 기록: 접근이 막혀(403) 확인하지 못했습니다. 따라서 심사 중 지적과 저자 답변은 이 분석에 반영하지 않았습니다.
- 참고문헌 [12], [13], [17], [23], [24], [35], [36]은 arXiv 프리프린트입니다. 이 문서는 ELASTIC 본문이 그 논문들에 대해 말한 내용만 옮겼고, 해당 논문을 직접 열어 확인하지는 않았습니다.
