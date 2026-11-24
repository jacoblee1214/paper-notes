# 02. 선행연구와 코드 저장소

## 1. 계보

| 구성 요소 | 출처 | 이 논문에서의 역할 |
|---|---|---|
| 안전한 anytime-valid 추론(SAVI), e-process | Ramdas, Grünwald, Vovk, Shafer [17]; Ramdas et al. [42] | 쌍별 p-process가 임의 정지 시각에서 유효하다는 근거(정리 1 증명) |
| 로봇 정책 쌍 비교 순차 검정 N-SCORE | Snyder et al. [44] ("Beyond binary success...", 2026) | 쌍별 가설의 p-process. 부분 진행 점수 지원 |
| 그래프 다중 검정 | Bretz, Maurer, Brannath, Posch [18] (Stat. Med. 2009) | 위험 예산 재배분 식 (2). Holm, fixed-sequence가 특수한 경우 |
| E-값 기반 FWER 제어 | Hartog & Lei [19] | FWER 개념 인용 |
| Holm, fixed sequence | Holm [52], [49] | γ=0, γ=∞ 극한 |
| 실로봇 정책 평가 통계 | Vincent et al. [46] (RA-L 2024), Kress-Gazit et al. [1], Barreiros et al. [5] (LBM, Science Robotics 2026) | 시행 수, Bonferroni 쌍 비교의 선행 사례 |
| 분산 실세계 평가와 BT 순위 | RoboArena [2] (CoRL 2025) | 비교 대상 기준선(BT-Davidson, EM)과 데이터 |
| 시뮬·세계모델 평가 | SIMPLER [3], Ctrl-World [11], Veo 평가 [12], PlayWorld [10], 프록시 평가 [15] | 사전(prior)의 출처 |
| 선호 기반 순위 | Bradley-Terry [9, 50] | 휴리스틱 기준선 |

가까운 대안: (i) 쌍별 Bonferroni + 순차 검정(LBM 평가 [5], [44]), (ii) 부트스트랩 신뢰구간 + Bonferroni, (iii) 순위 선택(best-arm identification) 계열(논문은 인용하지 않음). 새로운 부분은 "그래프 재배분을 순차·능동 표본 선택과 결합해도 FWER가 유지된다"는 논증(부록 A.3)과 추이 폐포(부록 B)입니다.

## 2. 코드와 프로젝트 자료

| 항목 | 주소 | 라이선스 | 스타 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 공식 코드 | https://github.com/irom-princeton/AnyRank | MIT | 1 | 2026-10-27 | 커밋 `4d16a97`(단일 커밋 shallow clone, 20 MB, 파일 280개). 논문의 "github.com/anyrank"는 존재하지 않는 조직 주소(2026-11-24 확인, 404) |
| 프로젝트 페이지 | https://abadithela.github.io/anyrank-page/ | 저장소 라이선스 없음 | 0 | 2026-10-24 | 소스 저장소 abadithela/anyrank-page. 페이지 링크는 arXiv, 위 코드만 |
| 의존 라이브러리 1 | https://github.com/TRI-ML/statistical_comparison_core | GitHub 라이선스 판별 안 됨(코드 저장소 README: CC BY-NC 4.0) | 1 | 2026-09-05 | AnyRank 저장소에 포함되지 않음 |
| 의존 라이브러리 2 | https://github.com/TRI-ML/sequentialized_barnard_tests | 판별 안 됨(README: CC BY-NC 4.0) | 15 | 2026-09-15 | 같은 이유 |
| nscore | PyPI 패키지(`nscore`) | 미확인 | - | - | N-SCORE 구현. 저장소에서 `nscore>=0.1.2,<0.2`로 의존. 이 분석에서 내려받지 않음 |

핵심 쌍별 검정 N-SCORE 구현이 저장소 밖(비상업 라이선스 의존)에 있어, 정적 대조로는 p-value 계산 자체를 확인하지 못했습니다(`04` §1).

## 3. 이 저장소의 관련 분석

| 폴더 | 논문 | 관계 |
|---|---|---|
