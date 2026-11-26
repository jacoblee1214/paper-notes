# GazeVLA: Learning Human Intention for Robotic Manipulation (arXiv:2604.22615, CoRL 2026) 분석 노트

*GazeVLA* (arXiv v3, 2026-11-02)를 표·그림 수치 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공개 코드는 AV-ALOHA 시뮬레이션 학습 설정 하나이고 3.3B급 모델(추정)과 CUDA를 가정합니다. 사전학습(8 GPU, 1,344 GPU시간), 평가, 실로봇은 공개되지 않았거나 하드웨어가 필요합니다.
- **소형 개념 검증은 조건부로 가능합니다.** 합성 목표 과제에서 시선 점 CoT와 직접 행동을 비교하는 정도이며 논문 수치와는 비교할 수 없습니다(노력 1–2주 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표·그림 재계산(V1–V14): 평균, 셀 검정, 풀링, 검출력, 단위 점검 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(D1–D14), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/code_evidence.py` | 공식 코드 근거 줄 추출(정적), 입력 크기 산술 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
# code_evidence.py는 공식 저장소의 얕은 clone이 필요합니다:
#   git clone --depth 1 https://github.com/lichy2004/GazeVLA repo && git -C repo checkout 0a834e8748b8d799166712a098fff91263109c43
#   git clone --depth 1 https://github.com/lichy2004/GazeVLA-Data data/GazeVLA-Data
python3 analysis/code_evidence.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-26). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/lichy2004/GazeVLA)는 LICENSE 파일이 Apache-2.0이고 README에는 다른 표기가 있습니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
