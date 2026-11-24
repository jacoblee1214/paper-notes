# Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation (arXiv:2606.18960, CoRL 2026) 분석 노트

*Mem-World* (arXiv v3, 2026-09-15)를 표와 그림 수치 재계산, 그림 판독, 주장 검증, 공개 VMem 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 논문 수치를 우리가 다시 계산해 그린 것과 토이 결과입니다. 공식 코드와 프로젝트 페이지는 찾지 못했습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 불가능합니다.** 영상 지표는 8 H100 약 2일의 학습과 코드·평가 궤적 목록이 필요하고, 정책 평가·개선은 실로봇과 4 H100 후학습이 필요합니다.
- **검색 모듈 단독 구현은 가능합니다**(추정 3–5일, 측정 전). 소형 세계모델 축소판은 조건부입니다(추정 2–4주).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표·그림 재계산(V1–V12), 토이, 일관성 검사(K1–K12) |
| `analysis/04_claim_verification.md` | 주장 판정(C1–C10), VMem 공개 코드 정적 대조(U1–U7), 구현 가능성 |
| `analysis/recompute.py` | 표·그림 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_retrieval.py` | 토이: 검색 규칙별 커버리지 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_retrieval.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-24). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드는 찾지 못했습니다. W-VMem의 출발점인 VMem(https://github.com/runjiali-rl/vmem, MIT)은 정적으로만 읽었고 실행하거나 복사하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼고, 그림 4와 5의 값은 그림 판독 값입니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
