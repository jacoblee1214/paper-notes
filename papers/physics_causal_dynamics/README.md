# PICR (arXiv:2611.19251) 분석 노트

*Physics-Informed Causal Reasoning for Nonstationary Robot Dynamics* (arXiv v1, 2026-11-14)을 표 재계산과 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.19251에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **논문 표 재현은 이 맥에서 불가입니다.** 데이터셋이 공개되지 않았고, 데이터 생성에 NVIDIA GPU가 필요합니다.
- **구조 재구현은 가능합니다.** FiLM-Lagrangian, 문맥 인코더, 대조 손실을 PyTorch로 쓸 수 있습니다(노력 1–2주 추정, 측정 전).
- **표 재계산은 완료했습니다.** 학습 밖 1.6–3.8배, 3 시드 구간, 파라미터 수를 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 기준선 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 차이 목록, 구현 가능성, 라이선스 |
| `analysis/recompute.py` | 표 재계산과 검정 (수 초, 표준 과학 계산 패키지만 필요) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `analysis/recompute_results.json` | 재계산 결과 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 non-exclusive(CC BY가 아님)입니다. 그래서 논문 그림은 넣지 않았습니다. 본문 인용은 표 값과 문장 수준입니다.
- 공식 코드는 `github.com/jungbbal/PICR`에 있고 라이선스 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 데이터셋과 체크포인트는 받아서 쓰지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프와 분석 문서, 스크립트는 이 노트의 작성자가 만들었습니다.
