# 02. 선행연구와 관련 자료: Action Chunk Scheduling (Armory)

## 1. 계보

```
행동 청킹 (ACT, Zhao 외 2023, [7])  ── 한 번의 추론으로 H개 행동 → 추론 빈도 감소
   │
   ├─ 비동기 실행: 추론과 실행을 겹침 (SmolVLA 계열 "naive async")
   │      ├─ Real-Time Chunking (RTC, Black 외 2025, [8]): 이미 확정된 행동 고정 + inpainting
   │      ├─ Training-time RTC ([10]), VLASH ([9]: 실행 시점 상태를 입력)
   │      └─ 분산 RTC 블로그 ([12]): 실행 인덱스의 단조 증가 요건
   │
   └─ 단일 로봇의 서빙 최적화
          ├─ VLA 추론 병목 프로파일링 ([6] vla-perf), 커널 최적화 ([21]), 한 번에 더 빨리 합성 ([22])
          └─ 검증 기반 추론 ([20])

ML 서빙 (Clipper [25], InferLine [26], Clockwork [27])  ── 지연 SLO, 동적 배치
   └─ LLM 서빙: 반복 단위 스케줄링 (Orca [28]), PagedAttention ([29]), SGLang ([30]),
      청크 프리필 ([11]), 분리 실행 (DistServe [31], Splitwise [37])

실시간 스케줄링 (Liu & Layland 1973, [32]: EDF)
   └─ 클러스터 스케줄링: plan-ahead (TetriSched [33]), 이질성 인식 (Gavel [34])

다중 로봇 서빙 (동시 연구): Kairos [35] (종단 지연 최소화, 가변 실행 지평), ROSA [36] (정적 프로비저닝, 동기 실행 가정)
   └─ Armory (이 논문): 로봇 상태 거울 + MDP + 이질성 인식 가중 Lookahead
```

(§5의 서술과 참고문헌 번호를 따랐습니다.)

## 2. 구성 요소별 출처와 가까운 대안

| 구성 요소 | 출처 | 이 논문이 바꾼 것 (저자 서술 / 우리 판단) |
|---|---|---|
| EDF 기준선 | Liu & Layland 1973 [32] | 마감 = 거울이 예측한 "행동 큐가 바닥나는 시각". 코드의 EDF는 둘이다: 단순 채움(`max-batch`)과 지연 적합(`greedy-deadline`, `04` D1) |
| Round Robin 기준선 | 고전 공정성 정책 | 변경 없음 |
| 동적 배치 | Clipper 등 [25–27] | 배치 크기를 후보로 두고 보상/GPU 시간으로 고름 (A.4: LA는 평균 3.3) |
| 계획 앞서기(plan-ahead) | TetriSched [33] | L 에폭 시뮬레이션. 실험은 L = 1이라 사실상 "한 걸음 모델 기반 점수" |
| 서버 쪽 상태 거울 | (명시 출처 없음) | 로봇 큐·진행 중 청크·지연(EMA)을 서버가 재현하는 디지털 트윈 형태. 핵심 기여로 보임 (`04` D3) |
| 가중치 w_j | 우선순위 스케줄링 | 빠른 로봇 계층에만 적용. 후보 배치를 가중치 계층별 EDF 접두 조합으로 가지치기(`04` D4) |
| 푸시 기반 통신 | OpenPI 서빙 스택 변형 (README 사사) | 로봇이 매 스텝 관측을 보냄 |
| 비동기 청크 이음 | naive async (RTC, VLASH가 아님) | 부록 A.8은 RTC·VLASH를 쓰면 결과가 달라질 수 있다고 인정 |

코드에는 논문에 나오지 않는 스케줄러도 있습니다: `dynamic-action`(α-공정 서비스 부채), `starvation`(관측 간격 필터), `random`(`04` D1). 논문의 비교 대상은 RR, EDF, LA 세 가지뿐입니다.

## 4. GitHub와 코드 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **Armory (공식)** | https://github.com/GaTech-RL2/armory | Apache-2.0 | 27 | 2026-11-15 (다른 브랜치. `main`의 마지막 커밋 `e876202`는 2026-08-04) | 얕은 클론을 `repo/`에 받아 정적으로만 읽음(1.6 MB). 실행·import·설치 안 함. `repo/`는 `.gitignore` |
| Armory 브랜치 `paper_sweep_experiments` | 위 저장소 | 같음 | - | 2026-07-31 (`e4a6206`) | 시뮬 스윕 설정(`configs/client/sim_sweep/*`)이 있음. API로 설정 파일 한 개만 읽음(`04` D8) |
| Armory 브랜치 `rohan-real-0513` | 위 저장소 | 같음 | - | 2026-05-24 | 실로봇 관련 가능성. 파일 목록 일부만 봄. 내용은 읽지 않음 |
| 프로젝트 페이지 | https://gatech-rl2.github.io/actionchunkscheduling/ | - | - | - | 영상(EDF, RR, LA 10대 동시 시연), 방법 요약. 템플릿 흔적으로 VideoMimic 링크가 남아 있음 |
| rohan-bansal/LIBERO | https://github.com/rohan-bansal/LIBERO | MIT | 0 | 2026-05-16 | 서브모듈(수정판 LIBERO) |
| DavidHe137/openpi | https://github.com/DavidHe137/openpi | Apache-2.0 | 4 | 2026-08-11 | 서브모듈(수정판 OpenPI) |
| rohan-bansal/Isaac-GR00T_armory_compat | https://github.com/rohan-bansal/Isaac-GR00T_armory_compat | Apache-2.0 | 0 | 2026-07-22 | 서브모듈, 브랜치 `armory-n1.7-py311` |
| Physical-Intelligence/openpi | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 14,729 | 2026-08-24 | 서빙 스택 출처 (README 사사) |
| NVIDIA/Isaac-GR00T | https://github.com/NVIDIA/Isaac-GR00T | Apache-2.0 | 8,403 | 2026-11-17 | 두 번째 백엔드 |
| Lifelong-Robot-Learning/LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT | 2,565 | 2025-03-15 | 평가 벤치마크 원본 |
| mit-han-lab/vlash | https://github.com/mit-han-lab/vlash | Apache-2.0 | 526 | 2026-04-22 | 비교 대상인 비동기 보정 방법 |

Kairos [35]와 ROSA [36]의 공개 코드는 GitHub 검색(`gh search repos`)에서 찾지 못했습니다. 이 논문은 두 방법과 정량 비교를 하지 않습니다(§5는 정성 서술).

## 5. 학회 확인 근거

- arXiv 코멘트 비어 있음, PDF 첫쪽·마지막쪽에 학회 표기 없음, 프로젝트 페이지 본문에도 학회 표기를 찾지 못했습니다.
- **corl.org 공식 채택 목록**("CoRL 2026 accepted papers", 수집 2026-11-14)에서 제목 완전 일치로 **포스터**(세션 Poster post-p2, 11/10 화 17:00–18:30, 포스터 16번)를 확인했습니다. 따라서 학회는 CoRL 2026(포스터)입니다.
