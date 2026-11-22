# X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA Model (arXiv:2510.10274, ICLR 2026) 분석 노트

*X-VLA* (arXiv v1, 2025-10-11)를 표 재계산, 공식 코드 정적 대조, 그래프 두 장으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다 (라이선스 CC BY-NC-ND 4.0). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 사전학습은 64 A100 4일입니다. 미세조정과 Soft-Fold 학습도 GPU가 필요합니다.
- **LIBERO LoRA 추론 확인은 조건부로 가능합니다.** 체크포인트 (3.5 GB, Apache-2.0)와 LIBERO 설치, MPS 포팅이 필요합니다. 노력은 2–3일로 추정했고, 측정하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 기록 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 모델·데이터 자료 표 |
| `analysis/03_analysis.md` | 표 재계산 (평균, 정수성, 차분, CALVIN, NAVSIM 정의 검산, 신뢰구간) |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치와 빈칸, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY-NC-ND 4.0 라이선스로 공개되어 있습니다 (abs 페이지 license 링크, 확인 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드 (https://github.com/2toinf/X-VLA)는 Apache-2.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
