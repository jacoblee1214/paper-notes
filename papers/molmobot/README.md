# MolmoB0T (arXiv:2603.16861, CoRL 2026 구두 발표) 분석 노트

*MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation* (arXiv v3, 2026-11-08)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2603.16861에서 받을 수 있습니다). 논문 라이선스가 arXiv 비독점 배포라서 논문 그림도 넣지 않았고, 그래프는 모두 논문의 표를 다시 계산해 그린 것입니다.

## 구현 가능성 (요약)

- **논문 수준의 학습과 실로봇 재현은 불가능합니다.** 학습은 H100 8–64장, 데이터는 약 10.3 TB이고 실로봇 평가가 필요합니다.
- **공식 코드 정적 대조와 표 재계산은 끝냈습니다.** `analysis/04_code_verification.md`, `analysis/03_analysis.md`.
- **작은 정책(MolmoBot-SPOC, 체크포인트 약 0.4 GB)의 추론 확인은 조건부로 가능합니다.** 설치와 다운로드가 필요하고 속도는 측정하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, GitHub와 자원 표 |
| `analysis/03_analysis.md` | 표 재계산(V), 일관성 검사(K), 주장 판정(C) |
| `analysis/04_code_verification.md` | 공식 코드 대조(D), 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/recompute_results.json` | 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 |
| `report/figures/` | 우리가 그린 그래프 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 비독점 배포 라이선스로 공개되어 있습니다(arXiv 라이선스 링크 확인). 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/allenai/MolmoBot`(Apache-2.0)에 있습니다. 코드는 이 저장소에 넣지 않았고 실행하거나 설치하지 않았습니다. 분석한 커밋은 `be82891b50437fb081589fe3be353567241053da`입니다.
- 가중치와 데이터셋(Hugging Face, ODC-BY)은 받지 않았고 메타데이터만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
