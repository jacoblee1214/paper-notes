# Self Forcing (arXiv:2506.08009, NeurIPS 2025) 분석 노트

*Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion* (arXiv v2, 2025-11-10)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2506.08009에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 논문 수치로 다시 그렸습니다.

## 구현 가능성 (요약)

- **학습 재현은 이 맥에서 불가능합니다.** 14B 교사와 H100 64장을 씁니다.
- **공개 1.3B 추론은 조건부입니다.** 가중치 다운로드, CUDA 호출의 MPS 포팅, 속도 측정이 남아 있습니다.
- **논문 표의 재계산은 완료했습니다.** 결과와 판정은 `analysis/03_analysis.md`에 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 이항검정 (표준 라이브러리만 사용) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY-NC-SA 4.0으로 공개되어 있습니다(arXiv abs 페이지 확인). PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림은 넣지 않았습니다. 공개 산출물은 논문 수치로 우리가 그린 그래프와 텍스트 요약입니다.
- 공식 코드는 `github.com/guandeh17/Self-Forcing`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 정적으로만 읽었습니다.
- 체크포인트와 데이터셋은 받지 않았습니다.
