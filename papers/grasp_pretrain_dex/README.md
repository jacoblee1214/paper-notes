# From Grasps to Dexterity (arXiv:2606.30749, CoRL 2026 포스터) 분석 노트

*From Grasps to Dexterity: Large-Scale Grasp Pretraining for Dexterous Manipulation* (arXiv v2, 2026-10-26)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 원문 PDF는 넣지 않았습니다(arXiv:2606.30749에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스). 그래프 세 장은 표 값으로 우리가 그렸습니다.

## 구현 가능성 (요약)

- **공식 수치 재현은 이 맥에서 불가입니다.** 학습 설정이 CUDA로 고정되어 있고, 실로봇 장비가 필요합니다.
- **G2D-Pretrain 355k 궤적은 공개 자료에서 확인되지 않았습니다.** Dexonomy 자산으로 다시 만들어야 합니다.
- **재계산과 정적 대조는 완료했습니다.** 어댑터 매핑 검사는 CPU에서 가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 공식 저장소 표, 선행 계보, 학회 확인 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 주장별 판정 (C1–C19) |
| `analysis/04_code_verification.md` | 공식 코드 대조 (D1–D22), 재현 설정, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스(nonexclusive-distrib/1.0)로 올라 있습니다. CC BY가 아니므로 논문 그림은 싣지 않았습니다.
- 공식 코드 두 저장소(`xinyu1Liu/G2D-Policy`, `YingYuan0414/DexCraft`)에는 라이선스 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 정적으로 읽은 결과만 적었습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 재계산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.
