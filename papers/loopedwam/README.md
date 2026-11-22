# LoopedWAM: Looped World Action Models for Elastic Inference (arXiv:2611.25595) 분석 노트

*LoopedWAM* (arXiv v1, 2026-11-19)을 표 재계산, 공식 코드 정적 대조, 그래프 두 장으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **학습과 지연 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 학습은 H100 4장과 큰 잠재 캐시가 필요합니다.
- **체크포인트 로드와 단일 질의 추론은 조건부로 가능합니다.** 가중치 약 2.3 GB 다운로드와 MPS 포팅이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 증거의 틈, 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 직접 비교 대안, 공식 저장소와 학회 표기 확인 |
| `analysis/03_analysis.md` | 표 재계산 V1–V11: 평균, 구간, 검정, 실로봇 Fisher 검정 |
| `analysis/04_code_verification.md` | 논문-코드 대조 K1–K19 (일치), D1–D16 (차이·누락), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산 (표준 라이브러리, 1초 안) |
| `analysis/plot_figures.py` | 그래프 두 장 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 넣지 않았습니다.
- 공식 코드(https://github.com/anhdao69/LoopedWAM)는 LICENSE 파일 기준 MIT입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
