# T3 (arXiv:2406.13640, CoRL 2024) 분석 노트

*Transferable Tactile Transformers for Representation Learning Across Diverse Sensors and Tasks* (arXiv v3, 2024-10-06)을 그림 수치 판독, 검정, 공식 체크포인트 크기 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2406.13640에서 받을 수 있습니다). 그래프는 모두 이 저장소 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **사전학습 재현은 이 맥에서 불가입니다.** GPU와 392.5 GB 데이터가 필요합니다.
- **삽입 실로봇 재현은 불가입니다.** Franka와 촉각 센서가 필요합니다.
- **공식 체크포인트 추론은 조건부입니다.** tiny(91 MB)와 small(341 MB)은 설치와 다운로드 승인 후 가능합니다. 이 자료에서는 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 학회 확인, 계보, 공식 저장소 표, 라이선스 |
| `analysis/03_analysis.md` | 그림 판독 재계산, 검정, 파라미터 수 대조, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py` | 데이터셋 합계, 체크포인트 대조, 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). PDF에 충돌하는 저작권 표기는 찾지 못했습니다. 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/alanzjl/t3`에 있으며 MIT입니다. 이 자료에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 데이터셋(FoTa)은 구성 데이터셋의 라이선스를 따르며, 이 자료에서는 받지 않았습니다.
- 표와 본문 수치는 인용과 재계산을 위해 옮겼습니다.
