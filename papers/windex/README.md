# WinDEX (arXiv:2611.21143 v1) 분석 노트

*WinDEX: Metric 3D Hand Motion Reconstruction in the Wild for Dexterous Teleoperation and Policy Learning* (2026-11-16 arXiv 제출)을 표 재계산, 일관성 검사, 주장 검증으로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.21143에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥에서 전체 재현은 불가능합니다.** 학습은 GPU 여러 장을 전제하고, 1,000시간 규모의 라벨과 스테레오 장치 데이터가 필요합니다.
- **공식 코드와 가중치를 찾지 못했습니다.** 프로젝트 페이지와 GitHub 검색 모두 확인했습니다.
- **재계산과 검정은 가능하고 완료했습니다.** 표 수치와 초록 숫자를 다시 계산했습니다(`analysis/recompute.py`, 표준 라이브러리만 사용).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 정보 |
| `analysis/02_prior_work.md` | 선행 계보, 관련 저장소와 라이선스 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사 (표 II, III, IV 불일치 포함) |
| `analysis/04_claim_verification.md` | 공식 코드 검색 결과, 주장 검증, 구현 가능성 판정 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스는 arXiv의 비독점 배포 1.0입니다(CC BY가 아닙니다). 그래서 논문 그림을 넣지 않았고, 수치는 인용과 재계산을 위해 옮겼습니다.
- 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
