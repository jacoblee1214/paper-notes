# The Flip Test: Video World Models Fail to Learn the World State (arXiv:2611.19822 v1) 분석 노트

*The Flip Test*(v1, 2026-11-15)를 표 재계산과 원문 내부 검사로 읽은 자료입니다. 학회 표기는 없고, 공식 코드도 아직 없습니다(주장 검증으로 정리). 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 불가능합니다.** 11개 영상 모델 생성과 상용 판정기가 필요합니다.
- **MuJoCo 기반 축소판 쌍 벤치마크는 CPU로 가능합니다.** 노력 2–4주 추정.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 결과, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 우연 보정, 구간과 검정, 판독 긴장 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C27, 구현 가능성, 저자에게 물을 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지에서 CC BY 4.0으로 표기되어 있습니다(2026-11-21 확인). 원문 PDF에 별도 권리 고지는 없습니다. 다만 그림 대부분에 제3자 모델의 생성 영상과 사진이 들어 있어 논문 그림은 싣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치, 데이터셋, 판정 API는 사용하지 않았습니다.
