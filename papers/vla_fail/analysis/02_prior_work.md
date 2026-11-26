# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다(`paper/paper.txt`, v2). 각 논문을 따로 열어 검증하지는 않았고, 계보는 논문 본문의 인용과 서술을 기준으로 정리했습니다.

## 1. 계보

VLA-FAIL은 세 줄기가 만나는 곳에 있습니다.

**(a) VLA와 생성형 정책의 런타임 실패 검출.** SAFE [7]는 VLA 마지막 층 특징으로 성공·실패 롤아웃을 모두 써서 검출기를 학습합니다(실패 데이터 필요). RC-NF [14]는 성공 시연만으로 조건부 정규화 흐름을 학습하지만 외부 객체 인식에 기댑니다. CycleVLA [15], FailSafe [16]는 VLM 추론으로 실패 롤아웃을 판별합니다(§2). 생성형 모방 정책 쪽에서는 FIPER [13]가 관측 임베딩의 랜덤 네트워크 증류(RND)와 여러 행동 샘플의 행동 청크 엔트로피(ACE)를 결합하고, Sentinel [17]은 겹치는 연속 청크의 분포를 비교하는 STAC에 VLM 영상 질의응답을 더합니다. FAIL-Detect [18, 34]는 관측 임베딩 위의 flow-matching 밀도(logpZO)가 가장 낫다고 보고하고, FIDeL [19]은 최적 수송으로 시각 이상을 잡고 VLM으로 거릅니다. 순방향 동역학 모델 기반 이상 검출 [20–22]은 ACC와 겉보기에 비슷하지만 추가 모델이 필요합니다. 동시기 연구인 Rewind-IL [23]도 겹치는 행동 청크의 불일치를 실패 신호로 씁니다. 로봇 게이트 DAgger [27]에서는 앙상블 분산 [8], 확산 손실(Diff-DAgger) [9]이 실패·개입 트리거로 제안되었습니다.

**(b) 일반 OOD 검출.** 분류기 마지막 층 특징의 마할라노비스 거리(Lee 외 [28])와 후속 개선 [29, 30]이 LLMD의 직접 조상입니다. 논문은 마지막 층 마할라노비스가 last-layer Laplace 근사 [31]와 가깝다고 짚고, 연속 행동을 내는 flow-matching 모델과 잡음이 낀 특징으로 확장했다고 말합니다(§2).

**(c) 지표.** SAFE [7]의 균형 정확도 대 탐지 시각 곡선과 FIPER [13]의 시간 가중 정확도가 탐지 속도를 평가에 넣은 선행 지표입니다. 둘 다 임계값 의존이라 AUCPDT를 "임계값 무관"으로 제안합니다(부록 A).

**이 논문의 위치.** 행동 샘플링이 필요한 ACE, Diff, STAC와 달리 순전파 한 번과 이미 만든 청크만 쓰고, SAFE와 달리 실패 롤아웃 없이 성공 롤아웃 20개로 임계값을 정합니다. 대신 LLMD는 미세조정 데이터에 다시 접근해야 하고 ACC는 수신 지평 제어의 청크 겹침이 필요합니다(§6).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 역할 | 가까운 대안 |
|---|---|---|---|
| 마지막 층 마할라노비스 | Lee 외 [28], 개선 [29, 30] | LLMD 점수의 형태(식 3) | LL-RND, LL-logpZO(표 7), 마지막 층 Laplace [31] |
| 고정 prior 잡음 특징 | 이 논문(flow matching의 t = 0, 식 1) | 잡음 변동 제거 | 모든 시각·잡음 평균(LL-RND에서 비교, 부록 C.1) |
| 토큰별 통계 | 이 논문(표 4) | 위치마다 분포가 다르다는 가정 | 전역 평균·공분산 |
| 겹침 청크 불일치 | STAC [17], Rewind-IL [23](동시기) | ACC(식 4–6): STAC의 속도 정규화 단일 샘플 추정으로 해석 | STAC(분포 비교, 32 샘플), 앙상블 분산 [8] |
| 속도 정규화와 EMA | 이 논문(식 4, 6) | 느린 정밀 동작에서의 민감도와 재계획 완충 | 정규화 없음(표 5, 6) |
| conformal 임계값 | FIPER [13] | 시간 불변 임계값(식 7, 부록 D) | 시간 의존 밴드 [34, 13, 7] |
| ACE 기준선 | FIPER [13] | 행동 청크 엔트로피(32 샘플) | – |
| Diff 기준선 | Diff-DAgger [9] | 확산 손실을 flow-matching 손실로 바꿈, 512 대신 32 샘플 | – |
| STAC 기준선 | Sentinel [17] | MMD + RBF 커널(중앙값 휴리스틱), 32 샘플 | – |
| RAPT 기준선 | RAPT [40] | 고유감각 순환 예측(표 8) | – |
| VLA | π0.5 [2], X-VLA [11] | 평가 대상 | GR00T N1 [3], OpenVLA [6] 등(적용 가능하다고만 서술) |
| 벤치마크 | LIBERO [37], LIBERO-Plus [12] | 시뮬 평가 | – |

## 4. GitHub와 자료 표 (2026-11-26 확인)

| 항목 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| 논문 공식 코드 저장소 | 공식 | – | – | – | **확인하지 못했습니다**(2026-11-26). 클론 없음 |
| [프로젝트 페이지](https://vla-fail.github.io/vla-fail-2026) | 공식. 논문 PDF, 실제 로봇 정성 영상(Blocks, Cups, Kitchen, Stack T, Drawer의 ACC·LLMD 점수 곡선과 성공 에피소드) | 페이지 템플릿 CC BY-SA 4.0 표기 | – | – | 열람함 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 구현(논문의 π0.5는 공개 버전, 카메라 2대만 지원, 표 15) | Apache-2.0 | 14,772 | 2026-08-24 | 읽지 않음 |
| [2toinf/X-VLA](https://github.com/2toinf/X-VLA) | X-VLA 구현 | Apache-2.0 | 763 | 2026-10-20 | 읽지 않음 |
| [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | LIBERO-Plus 벤치마크 | 라이선스 표기 없음 | 504 | 2026-01-21 | 읽지 않음 |
| [learnsyslab/fiper](https://github.com/learnsyslab/fiper) | ACE 기준선의 출처(FIPER, NeurIPS 2025 공식 코드) | MIT | 61 | 2025-11-03 | 읽지 않음 |
| [agiachris/sentinel](https://github.com/agiachris/sentinel) | STAC 기준선의 출처(Sentinel, CoRL 2024) | MIT | 38 | 2025-02-06 | 읽지 않음 |
| [sean1295/DiffDAgger](https://github.com/sean1295/DiffDAgger) | Diff 기준선의 출처(Diff-DAgger) | 라이선스 식별 안 됨(NOASSERTION) | 22 | 2025-03-10 | 읽지 않음 |

- 논문 라이선스: arXiv abs 페이지 license 링크가 `creativecommons.org/licenses/by/4.0/`(CC BY 4.0)입니다(2026-11-26 확인, README 라이선스 절).
- 코드를 확인하지 못해 논문 ↔ 코드 대조는 하지 못했고, 04에서 주장 대조로 대신했습니다. 아래는 코드가 공개되면 가장 먼저 볼 항목입니다.
  1. ACC 평활 α의 정의(0.9 대 0.1)와 EMA 식의 가중 방향(04 C8).
  2. ACC가 쓰는 겹침 길이(표 15·16의 "ACC Overlap")와 Libero-Plus의 실행 길이 R(표 17에 없음).
  3. LLMD의 a*₀ 시드, λ의 단위, 토큰별 공분산을 몇 개 프레임으로 추정하는지(03 §10).
  4. AUCPR의 계산 규칙(보간 여부, 동점 처리, 재현율 끝점)과 AUCPDT의 파레토 처리(03 §8).
  5. Libero-Plus 체크포인트가 공식인지 LIBERO-Plus 데이터로 다시 미세조정한 것인지(04 C16).
- 논문에서 쓴 기준선(ACE, Diff, STAC)은 저자가 flow-matching VLA용으로 다시 구현했습니다(부록 F). 위 세 저장소는 원 논문의 공식 코드이고 이 구현과 같은지는 확인하지 못했습니다.
