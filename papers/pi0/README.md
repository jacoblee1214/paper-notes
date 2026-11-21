# π0: A Vision-Language-Action Flow Model for General Robot Control (arXiv:2410.24164, RSS 2025) 분석 노트

*π0* (arXiv v4, 2026-01-08)를 그림 판독 재계산, 공식 코드(openpi) 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **학습과 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 사전학습 데이터와 레시피가 공개되지 않았고, 미세조정은 CUDA 기기가 필요합니다.
- **추론 메모리(3.2B bf16, 약 7 GB)는 맞지만 속도는 측정하지 않았습니다.**
- **구조 재구현과 CPU 검증은 가능합니다(추정 2–4일).**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 증거의 틈, 버전 이력 |
| `analysis/02_prior_work.md` | 선행 계보, 비교 기준선, GitHub 표 |
| `analysis/03_analysis.md` | 그림 판독 재계산, 객체 단위 검정, 본문과 그림 대조, 추론 시간, 데이터 규모 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 차이와 빈칸, 구현 가능성 |
| `analysis/recompute.py` | 판독값 검정, 추론 시간 합계, 데이터 규모, 파라미터 수, 시간 분포 KS (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-21). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/Physical-Intelligence/openpi)는 Apache-2.0입니다. 이 분석에서는 실행하거나 설치하지 않고 정적으로만 읽었습니다.
- 그림 판독값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
