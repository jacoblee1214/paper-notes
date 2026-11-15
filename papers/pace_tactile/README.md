# PACE (arXiv:2610.16308, CoRL 2026) 분석 노트

*PACE: Temporal-Role Tactile Conditioning for Chunk-Based Insertion Policies* (CoRL 2026 구두, arXiv v1)을 표 재계산, 통계 검정, 통계 단위 점검, 소형 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv non-exclusive 배포 조건이며 CC BY가 아닙니다). 공식 코드와 데이터는 아직 공개되지 않았습니다.

## 결론 (요약)

- 표 수치(90.7%, 88.7%, 절제 평균, Wilson 구간)는 재계산과 맞습니다.
- Event 제거 손실은 보정 후에도 유의합니다. Pred. 제거 효과는 유의하지 않습니다.
- 기계적 분석의 ANOVA와 Pearson 통계는 보고된 30 롤아웃 단위로 나오지 않습니다.
- 신호 대체(F/T, 관절 전류)의 "같음"은 검정으로 증명되지 않았습니다.
- 구조(FiLM, 게이트 잔차, 학습 전용 헤드)는 ACT 위에 소형으로 구현할 수 있습니다. 논문 수치의 재현은 실로봇과 GPU가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, 빠진 정보 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 조직과 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 통계 단위 점검, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 판정 C1–C27, 빠진 정보, 저자 질문, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산, Wilson 구간, Fisher와 Holm 검정, ANOVA와 Pearson 역산 (scipy, 수 초) |
| `analysis/toy_routing.py` | 토이 1: 상태와 사건 분리 대 단일 특징 (numpy, 수 초, 설계 가정 의존) |
| `analysis/toy_power.py` | 토이 2: 과제당 30회 비교의 검출력과 부풀림 (numpy, scipy, 수십 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/toy_routing.py
python3 analysis/toy_power.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive 배포 조건입니다(arXiv abs 페이지 확인, 2026-11-15). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2610.16308에서 받을 수 있습니다.
- 논문의 표 수치는 인용과 재계산을 위해 옮겼습니다. 문장과 그림은 옮기지 않았습니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다.
