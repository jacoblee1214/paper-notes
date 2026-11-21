# 02. 선행연구와 자료

조사일 2026-11-21. 논문 본문 참고문헌 43편과 웹·GitHub 검색으로 확인했습니다.

## 1. 계보 한눈에

```
NAVSIM v1 (PDMS, 비반응 4 s) [8] ──┐
NAVSIM v2 navhard (EPDMS) [34] ───┤ 평가 틀 (PGSD가 그대로 씀)
                                   │
WorldDrive [10] ─ 세계모델 + FAR 재순위 (PGSD의 동결 플래너와 FAR)
   ├─ 미래 잠재 특징을 FAR로 증류 (세계모델 추론 없이 점수) [10, 부록 B]
   └─ PGSD: 그 위에 실행 검증 선택(EAS) + 풀 밖 수정 학생(FCRS) 추가
                                   │
교사-학생 증류 계열 [31 일관성 모델, 32 Hydra-MDP, 33 DistillDrive, 30 Diffusion Policy]
   └─ PGSD의 CAST→FCRS 구조는 "탐색 교사 → 한 번 예측 학생"
```

## 2. 구성 요소별 출처

| 구성 요소 (PGSD) | 가까운 선행연구 | PGSD가 더한 것 | 출처 |
|---|---|---|---|
| 세계모델 기반 궤적 점수 | WorldDrive (FAR), WoTE, Drive-WM, World4Drive, LAW | 점수 대신 실행 여유로 후보를 거름 | [4, 6, 10, 21, 23] |
| 후보 선택의 실행 검증 | NAVSIM PDMS 지표 (DAC, EP, TTC) | 선택 단계에서 실행 추적 발자국과 도로 여유를 직접 계산 | [8] |
| 풀 밖 수정 (beyond-pool) | 궤적 정제 (VAD v2, ResAD, GuideFlow) | 횡방향 두 매개변수 Dlat과 여유 최대화 탐색 | [11, 27, 28] |
| 증류 학생 | Hydra-MDP, DistillDrive, 일관성 모델 | 교사 목표를 여유 기준으로 고름, 1회 검증 | [31, 32, 33] |
| 목표 여유 대 행동 오차 | (논문이 직접 인용한 선행 없음) | 작은 행동 오차가 실행 실패를 막지 못함을 실험으로 보임 | §4.2 |

- **논문 스스로 밝힌 차별점** (§2): 예측 특징은 "어느 수정이 허용 오차에 견디는지"를 직접 알려주지 않으므로, 실행 여유 기준 목표로 교사를 고른다는 것입니다.
- 목표 여유를 교사 선택 기준으로 쓰는 생각은 제어·안전 쪽의 여유 기반 설계와 닮았습니다. 논문은 이 연결을 인용하지 않았습니다 (이 분석은 문헌을 더 찾지 않았습니다).

## 4. GitHub와 프로젝트 자료

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| **PGSD 공식 코드** | 찾지 못함 | - | - | - | GitHub 검색 "Physics-Guided Search Distillation" 결과 0건. 웹 검색 0건. 논문 PDF와 arXiv HTML에 코드 링크 없음 |
| **PGSD 프로젝트 페이지** | 찾지 못함 | - | - | - | 논문 PDF, arXiv abs·HTML에 없음 |
| WorldDrive (기준 방법, 공식 코드) | https://github.com/TabGuigui/WorldDrive | Apache-2.0 (코드). 논문 arXiv 2603.14948은 CC BY-NC-SA 4.0 | 85 | 2026-11-18 | PGSD가 동결해서 쓰는 플래너와 FAR의 원 코드. **클론하지 않음** (PGSD 코드가 없어 정적 대조 대상 없음) |
| NAVSIM (평가 벤치마크) | https://github.com/autonomousvision/navsim | Apache-2.0 | 1,147 | 2025-10-27 | 평가 틀. **클론하지 않음** |
| PFD (Privileged Foresight Distillation) | https://github.com/PengchengFang-cs/PFD | NOASSERTION (표기 없음) | 11 | 2026-06-05 | 개념상 유사 (훈련 때만 미래 정보를 쓰는 교사). 다른 논문이며 PGSD의 코드가 아님 |

- 공식 코드가 없으므로 `repo/`는 만들지 않았습니다. 실행, import, 설치는 하지 않았습니다.
- WorldDrive 코드는 Apache-2.0이지만 논문 라이선스는 CC BY-NC-SA입니다. PGSD를 쓰려면 WorldDrive 체크포인트와 코드를 함께 쓰는 것이므로, 비영리 조건이 체크포인트에 붙는지 별도 확인이 필요합니다 (미확인).

## 5. 웹 검색으로 확인하지 못한 것

- PGSD 저자 저장소, 프로젝트 사이트, 학회 공식 채택 목록에서의 이름. 공식 목록에서 확인하려면 학회 목록을 전부 봐야 하는데, 이 논문은 학회 표기가 없어 목록 대조를 하지 않았습니다.
- PGSD를 인용한 후속 연구. 찾지 못했습니다 (검색 시점 기준, 공개된 인용 정보가 적음).
