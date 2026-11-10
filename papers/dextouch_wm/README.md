# DexTouch-WM (arXiv:2609.20649) 분석 노트

*DexTouch-WM: Learning Action-Conditioned Tactile World Models from Human Touch for Dexterous Robot Manipulation* (Qin et al.)을 표 재계산, 정책 평가 통계 점검, numpy 토이 두 개로 분석한 자료입니다. 이 논문은 IROS 2026 워크숍(RoBoWoMo) 라이트닝 토크이며 본 트랙 논문이 아닙니다. 논문 원문과 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 논문이 답하는 것과 답하지 않는 것 |
| `analysis/02_prior_work.md` | 선행연구 계보, 관련 저장소 표, 공개 코드 조사 |
| `analysis/03_analysis.md` | 표 재계산, 정책 평가 통계, 유도, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C25, 재현에 필요한 정보, 구현 가능성 판정 |
| `analysis/table_check.py` | Table I–IV 재계산 (numpy, 1스레드) |
| `analysis/transfer_toy.py` | 사람 데이터 전이의 부호 토이 (numpy, 1스레드, 약 1초) |
| `analysis/sparse_metric_toy.py` | 희소 접촉 지표와 MSE의 역설 토이 (numpy, 1스레드, 약 0.1초) |
| `analysis/plot_figures.py` | 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/table_check.py
python3 analysis/transfer_toy.py
python3 analysis/sparse_metric_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 공식 코드와 프로젝트 페이지는 찾지 못했습니다(2026-11-10 조사). 그래서 코드 대조 대신 표 재계산과 주장 검증을 했습니다.
- 그래프와 분석 문서, 스크립트는 직접 만들었습니다.
