# SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation (arXiv:2602.16863, RSS 2026) 분석 노트

arXiv v3(2026-10-06) 원문을 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가능합니다.** IsaacGym과 IsaacSim 학습은 NVIDIA GPU와 24,576 환경이 필요하고, 실로봇은 Sharpa 손과 KUKA iiwa가 필요합니다.
- **NumPy 부품(Task Progress 계산, 프리미티브 생성, 키포인트 보상)은 이 맥에서 구현할 수 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 II 재계산, 전체 평균과 구간, 37%의 해석과 검정력 토이, 실패 비율 검사 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치 목록 |
| `analysis/recompute.py` | 표 II 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/toy_gap_power.py` | 토이: SD 가정에 따른 5회 반복 검정력 (numpy, scipy, 몇 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_gap_power.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-17). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/tylerlum/simtoolreal)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 사전학습 정책과 DexToolBench 데이터는 받지 않았습니다.
