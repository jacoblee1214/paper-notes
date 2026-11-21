# WiTac (arXiv:2611.21057) 분석 노트

*WiTac: Wrist-Informed Tactile Representations for Contact-Rich Manipulation* (arXiv v2, 2026-11-17)을 보고 수치 재계산과 통계 검정으로 분석한 자료입니다. 학회 표기는 없습니다(arXiv 프리프린트). 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.21057에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **재계산과 통계 검증은 가능하고 완료했습니다.** 표준 라이브러리만 사용합니다.
- **논문 수치의 재현은 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 불가능합니다.** 사전학습과 VLA 미세조정은 GPU가 필요하고, 실로봇 Sharpa 하드웨어가 없습니다.
- **공식 코드와 WiTac 가중치는 찾지 못했습니다.** 구조 골격은 식과 부록 표로 만들 수 있지만 세부 구현은 추정입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 확인 |
| `analysis/02_prior_work.md` | 선행 계보와 가까운 대안 |
| `analysis/03_analysis.md` | 보고 수치 재계산, 검정, 검출력, 일관성 점검 |
| `analysis/04_claim_verification.md` | 주장 검증표, 권리, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산, Fisher 검정, Wilson 구간, 검출력 (표준 라이브러리) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py   # matplotlib 필요
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 초록 페이지의 라이선스 링크로 확인). 원문 PDF에 저작권 표시는 찾지 못했습니다.
- 논문 그림은 실로봇 사진과 카메라 화면을 포함하므로 게시본에 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋 SharpaDex v1.0은 CC BY-NC 4.0입니다. 이 노트에서 데이터를 받지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다. 공식 코드는 없습니다. 프로젝트 사이트 저장소의 데이터 파일은 표 값을 대조하는 데만 썼습니다.
