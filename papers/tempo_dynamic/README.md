# TEMPO (arXiv:2609.16864, CoRL 2026 구두 발표) 분석 노트

*TEMPO: Learning Temporal Context for Dynamic Robot Manipulation* (arXiv v1, 2026-09-15)를 표 재계산, 공식 코드 정적 대조, 작은 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 CC BY-NC-SA 4.0). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 실로봇(YAM 쌍팔), GPU 학습, 미공개 데이터와 체크포인트가 필요합니다.
- **축소판은 가능합니다.** 별칭 토이(`analysis/toy_aliasing.py`, 약 20초)와 TempoBench 관절 상태 통계(`analysis/tempobench_aliasing.py`, 수 초)를 끝냈고, MuJoCo 축소판은 새 설치 없이 시작할 수 있습니다(처리량 미측정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/toy_aliasing.py` | 토이: 1차원 상태 별칭 과제와 이력 길이 스윕 (torch CPU, 약 20초) |
| `analysis/tempobench_aliasing.py` | TempoBench 관절 상태 재방문 통계 (numpy, pyarrow, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib torch pyarrow
python3 analysis/recompute.py
python3 analysis/toy_aliasing.py
python3 analysis/tempobench_aliasing.py <TempoBench 저장소의 data 폴더>
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY-NC-SA 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/tempo-robot/TEMPO)는 Apache-2.0(+ Gemma 약관)이고, 공식 데이터(https://github.com/tempo-robot/TempoBench)의 LICENSE는 사용 조건이 정해지지 않은 자리표시자입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로 읽었으며 데이터는 통계만 냈습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 영상 데이터는 받지 않았습니다.
