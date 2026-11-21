# Right Goal, Wrong Body (arXiv:2611.20244) 분석 노트

*Right Goal, Wrong Body: Re-grounding Frozen VLAs under Actuation Shift* (arXiv v1, 2026-11-15)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 학회는 SPAIS 워크숍 @ CoRL 2026 spotlight입니다(워크숍 공식 채택 목록에서 확인). 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.20244에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **논문 실험 재현은 이 맥에서 불가입니다.** 공식 코드가 CUDA를 전제하고, π0.5 3B와 LIBERO 설정이 필요합니다.
- **지연 추정, 잔차 모니터, 추적 MPC의 작은 토이는 가능합니다.** 노력은 1–2일로 추정합니다(측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 저장소 PDF 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 코드와 프로젝트 자료 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 부호 검정, 회복률, 주장 대조 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 (D1–D20), 라이선스, 구현 가능성 |
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

- 논문 라이선스는 arXiv non-exclusive(CC BY 아님)입니다. 원문 PDF에 저작권 표시는 없습니다. 그래서 논문 그림은 싣지 않았습니다.
- 공식 코드(github.com/umersiddique94/ReGround)에는 라이선스 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.
