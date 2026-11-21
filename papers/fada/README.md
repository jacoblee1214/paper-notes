# FADA (arXiv:2606.28476, CoRL 2026 포스터) 분석 노트

*FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control* (arXiv v2, 2026-10-09)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.28476에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **학습과 시뮬 평가는 이 맥에서 재현할 수 없습니다.** IsaacSim과 CUDA GPU를 전제합니다. 학생 체크포인트도 공개되어 있지 않습니다.
- **표 재계산과 코드 대조는 가능하고 완료했습니다.**
- **역할 분리 대조의 CPU 토이는 아직 하지 않았습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그림은 우리가 그린 것 두 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조와 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 그래프 두 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스입니다(`nonexclusive-distrib/1.0`). CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/LeCAR-Lab/FADA`에 있습니다(Apache-2.0). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.
