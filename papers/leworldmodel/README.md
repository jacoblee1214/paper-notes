# LeWorldModel: Stable End-to-End JEPA from Pixels (arXiv:2603.19312, NeurIPS 2026) 분석 노트

*LeWorldModel* (arXiv v3, 2026-06-03)을 표 재계산, 공식 코드 정적 대조, 그래프로 분석한 자료입니다. 논문 원문은 넣지 않았습니다. 논문 그림 두 장(그림 3, 그림 6)은 CC BY 4.0이라 출처를 표기해 게시본에 실었고, 이 폴더의 나머지 그래프는 우리가 만들었습니다.

## 구현 가능성 (요약)

- **모델 구조, 두 항 손실, CEM 계획은 이 맥에서 옮길 수 있습니다.**
- **공개 체크포인트로 Push-T와 TwoRoom 추론을 확인하는 것은 조건부입니다.** 설치, 평가 데이터 13.1 GB와 3.4 GB, CUDA 패치가 필요합니다.
- **처음부터의 학습과 기준선 재현은 맥에서 어렵습니다.** 측정하지 않은 추정으로 한 에포크가 수 시간 이상입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 그림 6 판독값의 대비 검정, 평균 ± SD 격자 검사, SIGReg 검산 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치 목록, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 약 13초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 게시본의 논문 그림(그림 3, 그림 6)은 저자 저작물로 출처를 표기했습니다.
- 공식 코드(https://github.com/lucas-maes/le-wm)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 설치하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
