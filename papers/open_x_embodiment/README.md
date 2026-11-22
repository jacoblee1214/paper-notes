# Open X-Embodiment (arXiv:2310.08864, IEEE ICRA 2024) 분석 노트

*Open X-Embodiment: Robotic Learning Datasets and RT-X Models* (Open X-Embodiment Collaboration). arXiv v9(2025-05-14)를 분석했습니다. 재현(학습, 추론 실행)은 하지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 네 장은 보고된 수치와 메타데이터로 우리가 그린 것입니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** RT-2-X 55B는 공개되지 않았고, 혼합 학습은 약 2.0 TB 데이터와 클러스터가 필요하며, 실로봇 평가 3600회가 필요합니다.
- **조건부로 가능한 것:** 공개 RT-1-X 체크포인트 추론(설치와 다운로드 승인 후), 작은 데이터셋 한 개의 오프라인 행동 예측 검사.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, v1과 v9 차이 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(평균, 캡션 대조, n 가정 검정, 스프레드시트 재집계), 그래프 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조(D1–D10), 구현 가능성 판정 |
| `analysis/recompute.py` | 재계산과 일관성 검사 (scipy, 1초 안팎) |
| `analysis/recompute_results.json` | 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |
| `analysis/meta/oxe_dataset_sheet_2026-11-22.csv` | 공개 데이터셋 시트 스냅샷(메타데이터 CSV) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

스냅샷 CSV는 `analysis/meta/`에 있으며 `recompute.py`가 그 파일을 읽습니다.

## 출처와 라이선스

- 논문: arXiv:2310.08864 (abs 페이지의 라이선스는 CC BY 4.0, 2026-11-22 확인). 학회: IEEE ICRA 2024 (Crossref DOI 10.1109/ICRA57147.2024.10611477).
- 공식 코드: github.com/google-deepmind/open_x_embodiment (Apache-2.0, 커밋 9eeb68b). 이 폴더에 복제하지 않았습니다. 정적으로 읽은 내용만 04에 있습니다.
- 데이터셋 시트: 공개 Google Sheet (메타데이터만). 개별 데이터셋 라이선스는 시트에 없습니다. 원 데이터셋은 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
