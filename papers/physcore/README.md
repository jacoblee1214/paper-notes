# PhysCoRe (arXiv:2607.20653, CoRL 2026 포스터) 분석 노트

*PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics* (arXiv v3, 2026-10-09)을 표 재계산과 공식 코드 정적 읽기로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.20653에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다. 그래프는 모두 표 값으로 우리가 그렸습니다.

## 구현 가능성 (요약)

- **공식 코드 실행과 학습은 이 맥(CUDA 없음)에서 불가입니다.** CUDA 확장과 GPU를 전제합니다.
- **기준선 (PhysTwin, PGND), 계획 실험, 시각 지표는 공개 코드로 재현할 수 없습니다.**
- **체크포인트 추론은 포팅과 데이터 (4.2 GB) 뒤 조건부입니다.** 2D MLS-MPM 토이는 가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 기준선 표 |
| `analysis/03_analysis.md` | 표 재계산, 통계 검정 가능성, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 차이 목록, 구현 가능성, 라이선스 |
| `analysis/recompute.py` | 표 수치 재계산과 검정 가능 범위 (수 초, 표준 라이브러리와 scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 라이선스 페이지 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 공식 코드 `github.com/lunarlab-gatech/PhysCoRe`의 LICENSE는 MIT에 제3자 저작권 줄 (OmniPhysGS, PhysTwin)을 더한 형태입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 데이터셋 `GeorgiaTech/PhysCoRe` (Hugging Face)는 혼합 라이선스입니다. 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
