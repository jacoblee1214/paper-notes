# DreamGen (arXiv:2505.12705, CoRL 2025) 분석 노트

*DreamGen: Unlocking Generalization in Robot Learning through Video World Models* (arXiv v2, 2025-06-17)를 표 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 세 장은 논문 표 숫자로 우리가 그린 것입니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 논문 실험은 재현할 수 없습니다.** 공개 코드는 CUDA 환경과 외부 저장소를 전제하고, 실로봇 데이터, RoboCasa 신경 궤적 240k, IDM·LAPA 가중치가 공개되지 않았습니다.
- 이 폴더는 표 재계산, 검정, 일관성 점검과 코드 대조 결과만 담습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 쌍체 검정, 실로봇 검정력, 표 사이 일관성 점검 |
| `analysis/04_code_verification.md` | 논문과 공식 코드 대조, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0으로 공개되어 있습니다(abs 페이지 "view license", 2026-11-20 확인). PDF에는 회사 저작권 표기가 없습니다. 프로젝트 페이지 하단에는 회사 저작권 표기가 있어, 논문 그림은 싣지 않았습니다.
- 공식 코드 `NVIDIA/GR00T-Dreams`는 Apache-2.0입니다. 이 분석에서는 실행하거나 설치하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
