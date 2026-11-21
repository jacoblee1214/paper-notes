# ImpedanceMimic (arXiv:2611.19153) 분석 노트

*ImpedanceMimic: Learned Impedance as Action Space for Force-Accurate Motion Imitation* (arXiv v1, 2026-11-14)의 표 수치를 다시 계산하고, 주장을 논문 표와 부록에 대조한 자료입니다. 학회 표기는 없고 ("Preprint. Under review."), 공식 코드는 아직 공개되지 않았습니다 (2026-11-21 확인).

## 구현 가능성 (요약)

- **전체 학습은 이 맥에서 불가능합니다.** 2×10⁹ 스텝, 8,192 병렬 환경, 약 550 GPU-h이고 CUDA를 전제합니다.
- **단일 관절 막대 예제는 CPU로 시험할 수 있습니다.** 이 노트에는 그 실험을 넣지 않았습니다.
- **체크포인트 없이 확인할 수 없는 주장이 있습니다.** 활동 식별 재현율, 그림 주장, 로봇 전이의 세부 값.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 결과, 한계, 버전과 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 그래프 설명 |
| `analysis/04_claim_verification.md` | 주장 검증, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 파싱, 재계산, Welch 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
# 원문 PDF(arXiv:2611.19153)를 pdftotext -layout으로 옮긴 텍스트를 paper/paper.txt로 저장
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 라이선스 페이지 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 이 노트는 논문 그림을 쓰지 않았습니다. 표 수치는 인용과 재계산을 위해 옮겼고, 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 공식 코드 저장소 (`github.com/ImpedanceMimic/ImpedanceMimic`)는 README와 웹사이트 파일만 있고 라이선스가 없습니다. 이 노트에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.
