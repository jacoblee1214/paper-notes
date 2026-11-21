# DepthWorld (arXiv:2610.08780, CoRL 2026 포스터) 분석 노트

*DepthWorld: 3D World Model for Robot Manipulation* (arXiv v1, 2026-10-06)을 표 재계산, 일관성 검사, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.08780에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **학습은 이 맥에서 불가능합니다.** H200 두 노드에서 약 2일이 걸리고 저장소 설정도 CUDA를 전제합니다.
- **보정 최적화기가 공개되지 않았습니다.** 핵심 기여를 공개 코드로 확인할 수 없고, 새로 짜야 합니다.
- **세계 모델의 구조(공간 타일링, 점지도 헤드)는 구현할 수 있습니다.** 공개 코드는 MIT입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 기록 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 체크포인트 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 논문 표 값 재계산 (표준 라이브러리, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF의 저작권 표시는 찾지 못했습니다. 논문 그림 일부에는 실로봇과 데이터셋 프레임이 있어 싣지 않았습니다.
- 공식 코드는 `github.com/Jai2500/depthworld`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- DROID-3D 보정 자세(CC BY 4.0)는 수동 승인 대상이고, pseudo-label(CC BY-NC 4.0)과 VGGT 초기화 체크포인트(CC BY-NC 4.0)는 비상업 연구용입니다. 이 패키지에는 데이터와 가중치를 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
