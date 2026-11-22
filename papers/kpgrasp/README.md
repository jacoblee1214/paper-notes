# KPGrasp (arXiv:2606.09314, CoRL 2026 스포트라이트) 분석 노트

*KPGrasp: Scalable Keypoint Flow Matching for Dexterous Grasp Generation* (arXiv v2, 2026-10-10)을 표 재계산, 그림 판독 검산, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2606.09314에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **학습 재현은 이 맥에서 불가입니다.** 논문은 CUDA GPU 8장에서 약 2일 걸리는 학습을 씁니다.
- **공식 가중치는 공개 URL이 없습니다.** 추론 재현도 현재 불가입니다.
- **방법 소형 재구현은 가능합니다 (2–3주, 추정).** 주장 검증이 아니라 방법 이해용입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소의 출처와 라이선스, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 그림 판독 검산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 그림 판독 검산 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 두 장을 PDF에서 자르기 (pymupdf) |
| `report/figures/` | 그래프 세 장 (우리가 그림)과 논문 그림 두 장 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
python3 analysis/recompute.py
python3 analysis/plot_figures.py
python3 report/crop_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 라이선스 페이지 확인). 원문 PDF에 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장 (그림 3, 5)은 저작자 표기와 함께 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2606.09314"). 물체 렌더, 점군 장면, 실로봇 사진 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Yuansen-Huang/KPGrasp`에 있고 LICENSE는 CC BY-NC 4.0입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.
