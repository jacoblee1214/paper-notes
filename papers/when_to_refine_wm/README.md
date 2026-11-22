# When to Refine: Selective Search for Frozen Visual World Models (arXiv:2611.24470) 분석 노트

*When to Refine* (arXiv v1, 2026-11-18, 단독 저자) 논문을 표 재계산과 주장 검증으로 읽은 자료입니다. 학습과 환경 실행은 하지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv non-exclusive 라이선스). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **게이트 식은 구현 가능합니다.** CPU로 수 시간이면 됩니다(추정).
- **논문 수치의 재현은 이 맥에서 완전하지 않습니다.** 게이트 계수와 임계값을 논문이 공개하지 않았고, 기반 LeWorldModel 체크포인트 다운로드와 환경 설치가 필요합니다. 기반 코드는 평가 시 CUDA를 하드코딩합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 식, 실험, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 산술과 표 일관성, 종점 혼합, 짝지은 검정, 부트스트랩, Welch, 시드별 역산 |
| `analysis/04_claim_verification.md` | 주장 검증, 기반 코드 설정 대조, 구현 가능성, 저자에게 물을 질문 |
| `analysis/recompute.py` | 표 재계산과 검사 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 기반 코드 LeWorldModel(https://github.com/lucas-maes/le-wm, MIT)은 정적으로만 읽었습니다. 실행하거나 복사하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
