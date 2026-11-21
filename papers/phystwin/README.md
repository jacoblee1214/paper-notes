# PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos (arXiv:2503.17973, ICCV 2025) 분석 노트

*PhysTwin* (arXiv v1, 2025-03-23)을 표 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 실험의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 시뮬레이터가 `cuda:0`을 하드코딩하고, 렌더링은 CUDA 래스터라이저를 쓰며, 데이터 처리는 GPU 모델에 의존합니다.
- **공개 결과 파일로 평가 스크립트만 CPU에서 돌리는 것은 조건부로 가능합니다.** 데이터와 결과 파일 다운로드가 먼저 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산과 일관성 검사, 개선량, 유의성 역산(시나리오 SD 상한), 부호 검정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(차이 D1–D23), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검사 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-21). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/Jianghanxiao/PhysTwin, MIT)는 정적으로만 읽었습니다. 실행, 설치, 복사하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
