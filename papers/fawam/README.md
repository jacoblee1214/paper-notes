# FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation (arXiv:2606.08555 v3, CoRL 2026) 분석 노트

FAWAM (arXiv v3, 2026-10-27)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 결과는 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 재현할 수 없습니다.** 설정 기준 학습은 A100급 GPU 여러 장으로 수일이 걸리고, 실로봇 네 과제에는 힘 센서와 텔레오퍼레이션 장비가 필요합니다.
- **공개 코드에 잔차 보정이 없습니다.** 저장소 README는 보정 코드를 "정리 중"이라고 적었습니다. 힘 조건 행동 모델의 학습과 추론만 있습니다.
- 예측 힘 오차 보정을 작은 접촉 모델에서 비교하는 실험은 새 설치 없이 numpy로 가능하지만 이 자료에는 포함되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 차이, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 기준선, GitHub와 자료 표, 라이선스 메모 |
| `analysis/03_analysis.md` | 표 재계산, 쌍별 Fisher 검정, Wilson 구간, 검정력 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조표, 불일치와 빈칸, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-21). 원문은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/HaotianHehaha/FAWAM)는 MIT 라이선스입니다. 파일 일부는 Apache-2.0 헤더를 갖고 있습니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
