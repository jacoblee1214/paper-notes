# VSpark: Speculative Decoding for Autoregressive VLA (arXiv:2611.22666) 분석 노트

*VSpark* (arXiv v1, 2026-11-17, 학회 표기 없음)를 표 재계산, 공식 코드 정적 대조, 그래프로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 표의 재현은 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 불가능합니다.** 공식 코드는 CUDA 전제이고, π0 한 타깃의 학습 롤아웃이 약 551 GB입니다.
- **트리 초안, 연속 재사용, 정확 검증의 아이디어는 CPU에서 자작 코드로 구현해 볼 수 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법(식 1–6), 실험, 주장과 증거의 틈, 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소의 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산과 일관성(V1–V22), 속도 분해, 통계 검정, BF16/FP32 해석 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조표, 차이 D1–D13, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/suho-yoo/VSpark, Apache-2.0)는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
