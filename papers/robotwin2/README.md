# RoboTwin 2.0 (arXiv:2506.18088, ICML 2026) 분석 노트

*RoboTwin 2.0* (arXiv v2, 2025-08-27)의 표를 원자료에서 다시 계산하고, 공식 코드(MIT)를 정적으로 읽어 평가 절차를 대조한 분석입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 원자료에서 그린 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 설치 스크립트가 CUDA 기반 cuRobo를 빌드하고, 정책 학습과 실세계 실험에는 GPU와 실로봇이 필요합니다.
- 재계산과 정적 대조는 완료했습니다. 시뮬레이터 평가 한 과제는 cuRobo 의존성을 먼저 확인해야 합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, 과제 군집 부트스트랩 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(D1–D17), 평가 절차, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 부트스트랩 (numpy, scipy, 1초 안팎) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
pdftotext -layout paper/paper.pdf .scratch/paper_v2.txt
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/RoboTwin-Platform/RoboTwin)는 MIT 라이선스입니다. 이 분석에서는 실행, import, 설치를 하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치, 데이터셋, 에셋 zip은 받지 않았습니다.
