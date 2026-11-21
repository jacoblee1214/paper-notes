# WAMotor: Accelerating World Action Models via Speculative Inference under Dual Nonuniform Denoising Dynamics (arXiv:2611.25413) 분석 노트

*WAMotor* (arXiv v1, 2026-11-19 제출)의 표 재계산, 공식 저장소 정적 대조, 작은 CPU 토이 분석입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 성공률과 지연의 재현은 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 불가능합니다.** 백본은 Wan2.2-TI2V-5B 구조이고, 평가에는 CUDA와 LIBERO·LIBERO-Plus·RoboTwin 시뮬레이터가 필요합니다. 공식 저장소에는 LIBERO 평가 스크립트가 빠져 있습니다.
- **재계산과 토이는 맥에서 가능하고 완료했습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 근거, 버전 |
| `analysis/02_prior_work.md` | 선행 계보와 GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 검출 한계, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(불일치와 누락) |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, 1초 미만) |
| `analysis/toy_speculative.py` | 토이: 균일 검증 대 시간 간격 가중 검증 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_speculative.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/Overdue-Lin/WAMotor)에는 라이선스 파일이 없습니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
