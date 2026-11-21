# HandUMI (arXiv:2611.22146) 분석 노트

*HandUMI: A Modular Hand-Worn Parallel-Jaw Interface for Robot-Free Bimanual Data Capture* (arXiv v1, 2026-11-17)을 표 재계산, 통계 검정, 공식 저장소 대조로 분석한 자료입니다. 학회 표기는 없습니다(arXiv). 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.22146에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥에서 불가능합니다.** ACT 학습과 π0.5 미세 조정은 GPU가 필요하고, 실로봇 평가는 로봇이 필요합니다.
- **시뮬 재타깃팅 일부는 설치 뒤 조건부로 가능합니다.** 저장소 설치와 데이터 다운로드가 필요합니다. 이 자료에서는 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 두 출처의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 (성공률, 검정, 검출력), 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 성공률 구간, Fisher 검정, 검출력, 데이터 산술 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다. 논문 그림은 장치와 실로봇 사진이라서 넣지 않았습니다.
- 공식 저장소는 `github.com/murobotics-ai`에 있습니다. `handumi-sw`와 `handumi-hw`는 Apache-2.0이고, `handumi-quest-app`은 Apache-2.0이며 APK 기반 재구성에 대한 예외 조항이 있습니다. 이 자료에는 저장소 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 자료의 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.
