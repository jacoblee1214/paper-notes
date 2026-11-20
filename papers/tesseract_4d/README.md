# TesserAct: Learning 4D Embodied World Models (arXiv:2504.20995, ICCV 2025) 분석 노트

*TesserAct* (arXiv v1, 2025-04-29)를 표 재계산과 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 CC BY 4.0이지만 그림 안에 제3자 데이터셋 화면이 있어서 뺐습니다). 이 폴더의 그래프 두 장은 우리가 재계산한 값으로 그렸습니다.

## 구현 가능성 (요약)

- **학습과 수치 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 다중 GPU, 미공개 4D 데이터, RLBench 시뮬레이터가 필요합니다.
- **공개 체크포인트 추론은 불가에 가깝습니다.** 저장소 표는 25–29 GB를 요구합니다(추정, 미측정).
- **4D 재구성 최적화 축소판은 조건부로 가능합니다.** 공식 코드에 없으므로 직접 구현해야 합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 증거의 틈, 버전 차이, 권리 요약 |
| `analysis/02_prior_work.md` | 학회 확인, 선행 계보, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 1–4 재계산과 검정, 본문·부록 일관성 검사 |
| `analysis/04_code_verification.md` | 논문과 공식 코드 대조, 불일치 목록, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-20). 원문과 논문 그림은 이 폴더에 넣지 않았습니다.
- 공식 코드(https://github.com/UMass-Embodied-AGI/TesserAct)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
