# Qwen-RobotWorld (arXiv:2606.17030) 분석 노트

*Qwen-RobotWorld Technical Report: Unifying Embodied World Modeling through Language-Conditioned Video Generation* (Qwen Team, arXiv v3, 2026-06-17)을 표 재계산과 주장 검증으로 분석한 자료입니다. 기술 보고서이며 학회 채택 표기가 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.17030에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **재현할 수 없습니다.** 공식 코드, 가중치, 데이터 목록이 공개되어 있지 않습니다. 형제 저장소 Qwen-RobotManip, Qwen-RobotNav도 가중치 공개 계획이 없다고 명시합니다.
- **모델 학습은 이 맥에서 불가능합니다.** 20B MMDiT와 Qwen2.5-VL-7B 규모이며 CUDA 다중 GPU를 전제합니다.
- **재계산은 가능합니다.** `analysis/recompute.py`로 표 2–5의 합계를 CPU에서 수 초에 다시 계산했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 조회 결과 |
| `analysis/03_analysis.md` | 합계 재계산, 일관성 검사, 1위 차이 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C22, 구현 가능성 |
| `analysis/recompute.py` | 표 2–5 합계 재계산 (표준 라이브러리만 사용) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지의 license 링크(`creativecommons.org/licenses/by/4.0/`)에 따라 CC BY 4.0입니다. 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림은 제3자 데이터 프레임을 포함하므로 게시본에 넣지 않았습니다. 게시한 그래프는 논문 표 값으로 우리가 그린 것입니다.
- 논문 표 값은 인용과 재계산을 위해 옮겼습니다.
- 가중치와 데이터셋은 받지 않았습니다.
