# HuRo (arXiv:2609.10706, CoRL 2026 포스터) 분석 노트

*HuRo: Robotizing Human Videos for Scalable VLA Pretraining* (arXiv v5, 2026-09-25)를 표 수치 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.10706에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다. 논문 라이선스가 arXiv 비독점 1.0이라 CC BY가 아닙니다.

## 구현 가능성 (요약)

- **논문 결과도 공개 파이프라인도 이 맥(Apple M4 Pro)에서 재현할 수 없습니다.** CUDA 24 GB 이상과 RT 코어 GPU, Isaac Sim이 필요하고 학습·평가 코드는 공개되지 않았습니다.
- **통계 재계산과 코드 정적 대조는 완료했습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_alignment_invariance.py` | 정렬 변환 불변성 toy (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_alignment_invariance.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2609.10706, arXiv 비독점 배포 라이선스 1.0 (CC BY 아님). 논문 그림은 넣지 않았습니다.
- 공식 코드: `github.com/3587jjh/HuRo` (커밋 `082fb40`, 자체 코드 Apache-2.0, 실행에는 비상업 제한 의존성 포함). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 정적 읽기만 했습니다.
- 표 수치와 본문 인용은 분석을 위해 옮겼습니다. 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
