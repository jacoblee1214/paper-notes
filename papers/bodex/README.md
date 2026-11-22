# BODex (arXiv:2412.16490, IEEE ICRA 2025) 분석 노트

*BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization* (arXiv v3, 2025-09-03)을 표 재계산, 통계 검정, 공식 코드의 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2412.16490에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 우리가 그린 그래프 세 장만 넣었습니다.

## 구현 가능성 (요약)

- **공식 합성 파이프라인은 CUDA GPU가 있어야 돕니다.** 맥 (CUDA 없음)에서는 실행하지 않았습니다.
- **식과 3단계 최적화의 소형 재구현은 가능합니다 (추정 2–3주).** 이 노트는 재구현을 하지 않았습니다.
- **표 II의 차이는 통계적으로 확실합니다.** 표 III의 작은 차이 (1.8 pp)는 시도 수가 본문에 없어 확정하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 (v1·v2·v3) |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소의 출처, 공식 저장소 표, 데이터셋 크기 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (일치·차이·확인 불가), 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 라이선스 페이지 확인). 원문 PDF에 저작권 표시가 없습니다.
- 논문 그림은 넣지 않았습니다. 공식 저장소 README가 "NVIDIA 라이선스 부분을 제외한 나머지는 CC BY-NC 4.0"이라고 밝혀 그림의 조건을 확신할 수 없기 때문입니다.
- 공식 코드는 `github.com/JYChen18/BODex`에 있습니다 (cuRobo 기반). `LICENSE`는 NVIDIA License이고 README는 CC BY-NC 4.0을 씁니다. 이 노트에 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트와 분석 문서는 이 노트의 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다 (합계 약 18 GB, 크기만 확인).
