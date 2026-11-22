# DexCap (arXiv:2403.07788, RSS 2024) 분석 노트

*DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation* (arXiv v2, 2024-07-04)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2403.07788에서 받을 수 있습니다). 논문 그림은 쓰지 않았고, 그래프는 논문 수치를 다시 그린 것입니다.

## 구현 가능성 (요약)

- **이 맥에서 전체 재현은 불가능합니다.** 장비 (장갑, 카메라 세 대, 라이더), 두 팔 로봇, 교정 코드가 필요합니다. 공식 저장소에 교정과 로봇 실행 코드, 하드웨어 설계가 없습니다.
- **공개 데이터는 두 과제 (약 108 GB)뿐입니다.** 데이터 처리 파이프라인 일부는 CPU에서 확인할 수 있습니다.
- **학습 재현은 GPU가 필요해 하지 않았습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
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

- 논문의 arXiv 라이선스 페이지는 비배타 배포 라이선스 (`arxiv.org/licenses/nonexclusive-distrib/1.0/`)입니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/j96w/DexCap` (MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터와 가중치는 받지 않았습니다.
